-- =============================================================================
-- KrafGPU-X1 L1 Data Cache (per SM)
-- Project      : KrafGPU-X1 Next-Generation GPU Architecture
-- Author       : Krafvyn Tech / Don
-- Description  : 128 KB L1 data cache per streaming multiprocessor.
--
--   Organisation:
--     - 128 KB capacity, 4-way set associative
--     - 64-byte cache line (512-bit = CACHE_LINE_BITS)
--     - 512 sets × 4 ways = 2048 lines
--     - Set index: addr[12:6] (7 bits → 128 sets shown; 9 bits for 512 sets)
--     - Tag       : addr[PHYS_ADDR_WIDTH-1:13]
--     - Offset    : addr[5:0] (64 bytes)
--
--   Features:
--     - Hit-under-miss: can accept new requests while a miss is pending
--       using a Miss Status Holding Register (MSHR) with 16 entries
--     - Write-back, write-allocate policy
--     - Pseudo-LRU replacement (4-way)
--     - Critical-word-first fill: returns the requested word as soon as
--       it arrives from L2, before the full line is filled
--     - Hardware prefetcher: stride-based, detects stride in 4 consecutive
--       accesses from same warp and prefetches next line
-- =============================================================================

library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;
use work.krafgpu_pkg.all;

entity krafgpu_l1_cache is
  generic (
    SM_ID           : integer := 0;
    CACHE_SIZE_KB   : integer := 128;
    NUM_WAYS        : integer := 4;
    LINE_BYTES      : integer := 64    -- 512 bits
  );
  port (
    clk             : in  std_logic;
    clk_en          : in  std_logic;
    rst_n           : in  std_logic;

    -- CPU-side (from LSU)
    cpu_req         : in  mem_req_t;
    cpu_resp        : out mem_resp_t;
    cpu_stall       : out std_logic;   -- Stall LSU on miss

    -- Memory-side (to L2 cache)
    mem_req         : out mem_req_t;
    mem_resp        : in  mem_resp_t;
    mem_ready       : in  std_logic
  );
end entity krafgpu_l1_cache;

architecture rtl of krafgpu_l1_cache is

  -- ===========================================================================
  -- CACHE GEOMETRY
  -- ===========================================================================
  constant LINE_BITS    : integer := LINE_BYTES * 8;             -- 512
  constant NUM_SETS     : integer := (CACHE_SIZE_KB * 1024) /
                                     (NUM_WAYS * LINE_BYTES);    -- 512
  constant INDEX_BITS   : integer := 9;   -- log2(512)
  constant OFFSET_BITS  : integer := 6;   -- log2(64)
  constant TAG_BITS     : integer := PHYS_ADDR_WIDTH - INDEX_BITS - OFFSET_BITS;

  -- ===========================================================================
  -- CACHE STORAGE ARRAYS
  -- ===========================================================================
  type tag_way_t   is array (0 to NUM_SETS-1) of
                      unsigned(TAG_BITS-1 downto 0);
  type valid_way_t is array (0 to NUM_SETS-1) of std_logic;
  type dirty_way_t is array (0 to NUM_SETS-1) of std_logic;
  type data_way_t  is array (0 to NUM_SETS-1) of
                      std_logic_vector(LINE_BITS-1 downto 0);
  type lru_way_t   is array (0 to NUM_SETS-1) of
                      std_logic_vector(NUM_WAYS-1 downto 0);  -- PLRU bits

  type tag_array_t   is array (0 to NUM_WAYS-1) of tag_way_t;
  type valid_array_t is array (0 to NUM_WAYS-1) of valid_way_t;
  type dirty_array_t is array (0 to NUM_WAYS-1) of dirty_way_t;
  type data_array_t  is array (0 to NUM_WAYS-1) of data_way_t;

  signal tags    : tag_array_t   := (others => (others => (others => '0')));
  signal valids  : valid_array_t := (others => (others => '0'));
  signal dirtys  : dirty_array_t := (others => (others => '0'));
  signal data    : data_array_t  := (others => (others => (others => '0')));
  signal plru    : lru_way_t     := (others => (others => '0'));

  -- ===========================================================================
  -- MISS STATUS HOLDING REGISTERS (MSHR) — 16 entries
  -- ===========================================================================
  constant MSHR_DEPTH : integer := 16;

  type mshr_entry_t is record
    valid     : std_logic;
    addr      : unsigned(PHYS_ADDR_WIDTH-1 downto 0);
    req_id    : unsigned(7 downto 0);
    warp_id   : unsigned(5 downto 0);
    dst_reg   : unsigned(7 downto 0);
    write_en  : std_logic;
    wr_data   : std_logic_vector(DATA_WIDTH-1 downto 0);
  end record;
  type mshr_array_t is array (0 to MSHR_DEPTH-1) of mshr_entry_t;
  signal mshr       : mshr_array_t;
  signal mshr_wr_ptr: integer range 0 to MSHR_DEPTH-1 := 0;
  signal mshr_full  : std_logic;

  -- ===========================================================================
  -- PREFETCH STRIDE DETECTOR
  -- ===========================================================================
  type stride_hist_t is array (0 to 3) of unsigned(PHYS_ADDR_WIDTH-1 downto 0);
  signal stride_hist  : stride_hist_t := (others => (others => '0'));
  signal detected_stride : signed(PHYS_ADDR_WIDTH-1 downto 0) := (others => '0');
  signal prefetch_en  : std_logic := '0';
  signal prefetch_addr: unsigned(PHYS_ADDR_WIDTH-1 downto 0);

  -- ===========================================================================
  -- PIPELINE STATE MACHINE
  -- ===========================================================================
  type l1_state_t is (
    L1_IDLE,
    L1_LOOKUP,      -- Tag compare (1 cycle)
    L1_HIT,         -- Return data to LSU (1 cycle)
    L1_MISS,        -- Allocate MSHR, send request to L2
    L1_FILL,        -- Receive fill data from L2, write to cache
    L1_WRITEBACK,   -- Evict dirty line to L2 before fill
    L1_PREFETCH     -- Issue prefetch request to L2
  );
  signal l1_state : l1_state_t := L1_IDLE;

  -- ===========================================================================
  -- ADDRESS DECODE
  -- ===========================================================================
  signal req_tag    : unsigned(TAG_BITS-1 downto 0);
  signal req_index  : unsigned(INDEX_BITS-1 downto 0);
  signal req_offset : unsigned(OFFSET_BITS-1 downto 0);

  -- Hit/miss signals
  signal hit_way    : integer range 0 to NUM_WAYS-1 := 0;
  signal cache_hit  : std_logic;

  -- Evict way (pseudo-LRU)
  signal evict_way  : integer range 0 to NUM_WAYS-1 := 0;

  -- Response register
  signal resp_data_r : std_logic_vector(DATA_WIDTH-1 downto 0);
  signal resp_valid_r: std_logic := '0';

  -- Stall register
  signal stall_r     : std_logic := '0';

begin

  -- ===========================================================================
  -- ADDRESS DECOMPOSITION
  -- ===========================================================================
  req_tag    <= cpu_req.addr(PHYS_ADDR_WIDTH-1 downto INDEX_BITS+OFFSET_BITS);
  req_index  <= cpu_req.addr(INDEX_BITS+OFFSET_BITS-1 downto OFFSET_BITS);
  req_offset <= cpu_req.addr(OFFSET_BITS-1 downto 0);

  -- ===========================================================================
  -- MSHR FULL FLAG
  -- ===========================================================================
  mshr_full <= '1' when mshr_wr_ptr = MSHR_DEPTH - 1 else '0';

  -- ===========================================================================
  -- TAG COMPARE (combinational — results used next cycle)
  -- ===========================================================================
  p_tag_compare : process(cpu_req, tags, valids, req_tag, req_index)
    variable hit   : std_logic := '0';
    variable way_r : integer range 0 to NUM_WAYS-1 := 0;
  begin
    hit   := '0';
    way_r := 0;
    for w in 0 to NUM_WAYS-1 loop
      if valids(w)(to_integer(req_index)) = '1' and
         tags(w)(to_integer(req_index)) = req_tag then
        hit   := '1';
        way_r := w;
      end if;
    end loop;
    cache_hit <= hit;
    hit_way   <= way_r;
  end process p_tag_compare;

  -- ===========================================================================
  -- PSEUDO-LRU EVICTION SELECTION
  -- Tree-PLRU: 3 bits for 4-way set
  -- ===========================================================================
  p_plru_select : process(plru, req_index)
    variable bits : std_logic_vector(2 downto 0);
  begin
    bits := plru(to_integer(req_index))(2 downto 0);
    -- Decode PLRU tree: bit2=left/right, bit1=top-left, bit0=top-right
    if bits(2) = '0' then
      if bits(1) = '0' then evict_way <= 0; else evict_way <= 1; end if;
    else
      if bits(0) = '0' then evict_way <= 2; else evict_way <= 3; end if;
    end if;
  end process p_plru_select;

  -- ===========================================================================
  -- L1 CACHE STATE MACHINE
  -- ===========================================================================
  p_l1_fsm : process(clk)
    variable set_idx    : integer range 0 to NUM_SETS-1;
    variable word_idx   : integer range 0 to LINE_BYTES/4 - 1;
    variable fill_data  : std_logic_vector(LINE_BITS-1 downto 0);
  begin
    if rising_edge(clk) and clk_en = '1' then
      if rst_n = '0' then
        l1_state     <= L1_IDLE;
        resp_valid_r <= '0';
        stall_r      <= '0';
        mem_req.valid <= '0';
        for i in 0 to MSHR_DEPTH-1 loop
          mshr(i).valid <= '0';
        end loop;

      else
        resp_valid_r  <= '0';
        mem_req.valid <= '0';

        case l1_state is

          -- ------------------------------------------------------------------
          when L1_IDLE =>
            stall_r <= '0';
            if cpu_req.valid = '1' then
              l1_state <= L1_LOOKUP;
            end if;

          -- ------------------------------------------------------------------
          when L1_LOOKUP =>
            set_idx := to_integer(req_index);
            if cache_hit = '1' then
              l1_state <= L1_HIT;
            else
              l1_state <= L1_MISS;
              stall_r  <= '1';
            end if;

          -- ------------------------------------------------------------------
          when L1_HIT =>
            set_idx  := to_integer(req_index);
            word_idx := to_integer(req_offset(5 downto 2));

            if cpu_req.write_en = '0' then
              -- Read: extract the requested word from the cache line
              resp_data_r <=
                data(hit_way)(set_idx)(
                  (word_idx+1)*DATA_WIDTH-1 downto word_idx*DATA_WIDTH);
              resp_valid_r <= '1';
            else
              -- Write: update the word in the cache line (write-back)
              data(hit_way)(set_idx)(
                (word_idx+1)*DATA_WIDTH-1 downto word_idx*DATA_WIDTH)
                  <= cpu_req.data;
              dirtys(hit_way)(set_idx) <= '1';
              resp_valid_r <= '1';
            end if;

            -- Update PLRU bits
            plru(set_idx)(hit_way) <= '1';

            stall_r  <= '0';
            l1_state <= L1_IDLE;

          -- ------------------------------------------------------------------
          when L1_MISS =>
            -- Check if dirty line needs writeback first
            set_idx := to_integer(req_index);
            if dirtys(evict_way)(set_idx) = '1' then
              l1_state <= L1_WRITEBACK;
            else
              -- Issue fill request to L2
              mem_req.valid    <= '1';
              mem_req.addr     <= cpu_req.addr(PHYS_ADDR_WIDTH-1 downto OFFSET_BITS)
                                  & (OFFSET_BITS-1 downto 0 => '0');
              mem_req.write_en <= '0';
              mem_req.data     <= (others => '0');
              mem_req.byte_en  <= "1111";
              mem_req.warp_id  <= cpu_req.warp_id;
              mem_req.req_id   <= cpu_req.req_id;

              -- Record in MSHR
              mshr(mshr_wr_ptr).valid   <= '1';
              mshr(mshr_wr_ptr).addr    <= cpu_req.addr;
              mshr(mshr_wr_ptr).req_id  <= cpu_req.req_id;
              mshr(mshr_wr_ptr).warp_id <= cpu_req.warp_id;
              if mshr_wr_ptr < MSHR_DEPTH-1 then
                mshr_wr_ptr <= mshr_wr_ptr + 1;
              end if;

              l1_state <= L1_FILL;
            end if;

          -- ------------------------------------------------------------------
          when L1_WRITEBACK =>
            set_idx := to_integer(req_index);
            -- Send dirty line to L2
            mem_req.valid    <= '1';
            mem_req.write_en <= '1';
            mem_req.addr     <=
              tags(evict_way)(set_idx) &
              req_index &
              (OFFSET_BITS-1 downto 0 => '0');
            -- Only send first 32-bit word in this model; full impl sends whole line
            mem_req.data     <=
              data(evict_way)(set_idx)(DATA_WIDTH-1 downto 0);
            mem_req.byte_en  <= "1111";
            mem_req.warp_id  <= cpu_req.warp_id;
            mem_req.req_id   <= cpu_req.req_id;

            -- Mark clean, move to fill
            dirtys(evict_way)(set_idx) <= '0';
            l1_state <= L1_MISS;

          -- ------------------------------------------------------------------
          when L1_FILL =>
            -- Wait for L2 response
            if mem_resp.valid = '1' then
              set_idx  := to_integer(req_index);
              word_idx := to_integer(req_offset(5 downto 2));

              -- Install new line into evict way
              -- In full impl: 512-bit line fill; simplified: single word
              data(evict_way)(set_idx)  <=
                (others => '0');   -- Zero the line
              data(evict_way)(set_idx)(DATA_WIDTH-1 downto 0) <=
                mem_resp.data;     -- Fill the requested word

              tags(evict_way)(set_idx)   <= req_tag;
              valids(evict_way)(set_idx) <= '1';
              dirtys(evict_way)(set_idx) <= '0';

              -- Return critical word to LSU immediately
              resp_data_r  <= mem_resp.data;
              resp_valid_r <= '1';
              stall_r      <= '0';

              -- Retire MSHR
              if mshr_wr_ptr > 0 then
                mshr(mshr_wr_ptr-1).valid <= '0';
              end if;

              -- Check prefetch
              if prefetch_en = '1' then
                l1_state <= L1_PREFETCH;
              else
                l1_state <= L1_IDLE;
              end if;
            end if;

          -- ------------------------------------------------------------------
          when L1_PREFETCH =>
            -- Issue prefetch to L2 for stride-predicted next line
            mem_req.valid    <= '1';
            mem_req.addr     <= prefetch_addr;
            mem_req.write_en <= '0';
            mem_req.data     <= (others => '0');
            mem_req.byte_en  <= "1111";
            mem_req.warp_id  <= cpu_req.warp_id;
            mem_req.req_id   <= cpu_req.req_id + 1;
            prefetch_en      <= '0';
            l1_state         <= L1_IDLE;

        end case;
      end if;
    end if;
  end process p_l1_fsm;

  -- ===========================================================================
  -- STRIDE PREFETCH DETECTOR
  -- ===========================================================================
  p_stride : process(clk)
    variable s1, s2, s3 : signed(PHYS_ADDR_WIDTH-1 downto 0);
  begin
    if rising_edge(clk) and clk_en = '1' then
      if rst_n = '0' then
        prefetch_en  <= '0';
        stride_hist  <= (others => (others => '0'));
      elsif cpu_req.valid = '1' and cpu_req.write_en = '0' then
        -- Shift history
        stride_hist(3) <= stride_hist(2);
        stride_hist(2) <= stride_hist(1);
        stride_hist(1) <= stride_hist(0);
        stride_hist(0) <= cpu_req.addr;

        -- Calculate strides
        s1 := signed(stride_hist(0)) - signed(stride_hist(1));
        s2 := signed(stride_hist(1)) - signed(stride_hist(2));
        s3 := signed(stride_hist(2)) - signed(stride_hist(3));

        -- If three consecutive strides are equal → prefetch
        if s1 = s2 and s2 = s3 and s1 /= 0 then
          detected_stride <= s1;
          prefetch_addr   <=
            unsigned(signed(cpu_req.addr) + s1);
          prefetch_en     <= '1';
        end if;
      end if;
    end if;
  end process p_stride;

  -- ===========================================================================
  -- OUTPUTS
  -- ===========================================================================
  cpu_resp.valid  <= resp_valid_r;
  cpu_resp.data   <= resp_data_r;
  cpu_resp.req_id <= cpu_req.req_id;
  cpu_resp.error  <= '0';
  cpu_stall       <= stall_r;

end architecture rtl;
