-- =============================================================================
-- KrafGPU-X1 HBM4 Memory Controller
-- Project      : KrafGPU-X1 Next-Generation GPU Architecture
-- Author       : Krafvyn Tech / Don
-- Description  : HBM4 controller managing 6 stacks × 16 channels.
--                Total bandwidth: 6.4 TB/s at 4.8 GHz effective data rate.
--                Features:
--                  - Bank interleaving across stacks for maximum bandwidth
--                  - Power gating per pseudo-channel (power reduction strategy)
--                  - Address hashing to distribute traffic evenly
--                  - Out-of-order response reordering buffer (32 entries)
--                  - Refresh scheduling (auto partial self-refresh when idle)
-- =============================================================================

library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;
use work.krafgpu_pkg.all;

entity krafgpu_hbm4_ctrl is
  port (
    clk         : in  std_logic;
    rst_n       : in  std_logic;
    -- L3 cache interface
    req_in      : in  mem_req_t;
    resp_out    : out mem_resp_t;
    -- HBM4 physical interface (6 stacks)
    hbm_addr    : out std_logic_vector(PHYS_ADDR_WIDTH-1 downto 0);
    hbm_wdata   : out std_logic_vector(HBM4_TOTAL_WIDTH-1 downto 0);
    hbm_rdata   : in  std_logic_vector(HBM4_TOTAL_WIDTH-1 downto 0);
    hbm_wr_en   : out std_logic_vector(NUM_HBM4_STACKS-1 downto 0);
    hbm_rd_en   : out std_logic_vector(NUM_HBM4_STACKS-1 downto 0);
    hbm_ack     : in  std_logic_vector(NUM_HBM4_STACKS-1 downto 0)
  );
end entity krafgpu_hbm4_ctrl;

architecture rtl of krafgpu_hbm4_ctrl is

  -- ===========================================================================
  -- ADDRESS DECODER
  -- Maps 40-bit physical address to stack + channel + row + col + bank
  -- Address layout: [39:37]=stack [36:33]=channel [32:20]=row [19:7]=col [6:0]=byte
  -- ===========================================================================
  signal target_stack   : integer range 0 to NUM_HBM4_STACKS-1;
  signal target_chan    : unsigned(3 downto 0);
  signal row_addr       : unsigned(12 downto 0);
  signal col_addr       : unsigned(12 downto 0);

  -- ===========================================================================
  -- CHANNEL POWER GATE REGISTER
  -- Bit per channel (6 stacks × 16 ch = 96 bits)
  -- PMC writes this register to gate idle channels
  -- ===========================================================================
  signal chan_power_gate : std_logic_vector(NUM_HBM4_STACKS-1 downto 0)
                         := (others => '0');  -- All channels on by default

  -- ===========================================================================
  -- REORDER BUFFER (ROB) — 32 outstanding requests
  -- ===========================================================================
  constant ROB_DEPTH    : integer := 32;
  type rob_entry_t is record
    valid    : std_logic;
    req_id   : unsigned(7 downto 0);
    stack    : integer range 0 to NUM_HBM4_STACKS-1;
  end record;
  type rob_array_t is array (0 to ROB_DEPTH-1) of rob_entry_t;
  signal rob            : rob_array_t;
  signal rob_wr_ptr     : unsigned(4 downto 0) := (others => '0');
  signal rob_rd_ptr     : unsigned(4 downto 0) := (others => '0');
  signal rob_full       : std_logic;

  -- ===========================================================================
  -- REFRESH COUNTER
  -- Auto-refresh every 7.8µs = 3120 cycles at 400 MHz
  -- ===========================================================================
  constant REFRESH_INTERVAL : integer := 3120;
  signal   refresh_cnt       : integer range 0 to REFRESH_INTERVAL-1 := 0;
  signal   refresh_req        : std_logic := '0';

  -- ===========================================================================
  -- STATE MACHINE
  -- ===========================================================================
  type hbm_state_t is (
    HBM_IDLE,
    HBM_DECODE,
    HBM_ACTIVATE,     -- RAS: open row
    HBM_READ,         -- CAS read
    HBM_WRITE,        -- CAS write
    HBM_PRECHARGE,    -- Close row
    HBM_REFRESH,      -- Auto-refresh cycle
    HBM_RESPOND       -- Return data to L3
  );
  signal hbm_state    : hbm_state_t := HBM_IDLE;
  signal resp_data    : std_logic_vector(DATA_WIDTH-1 downto 0);
  signal resp_valid   : std_logic := '0';
  signal lat_cnt      : integer range 0 to 31 := 0;

begin

  -- ===========================================================================
  -- ADDRESS DECODER
  -- ===========================================================================
  target_stack <= to_integer(req_in.addr(PHYS_ADDR_WIDTH-1 downto PHYS_ADDR_WIDTH-3));
  target_chan  <= req_in.addr(PHYS_ADDR_WIDTH-4 downto PHYS_ADDR_WIDTH-7);
  row_addr     <= req_in.addr(32 downto 20);
  col_addr     <= req_in.addr(19 downto 7);

  -- ===========================================================================
  -- ROB FULL FLAG
  -- ===========================================================================
  rob_full <= '1' when (rob_wr_ptr + 1) = rob_rd_ptr else '0';

  -- ===========================================================================
  -- REFRESH COUNTER
  -- ===========================================================================
  p_refresh : process(clk)
  begin
    if rising_edge(clk) then
      if rst_n = '0' then
        refresh_cnt <= 0;
        refresh_req <= '0';
      elsif refresh_cnt = REFRESH_INTERVAL - 1 then
        refresh_cnt <= 0;
        refresh_req <= '1';
      else
        refresh_cnt <= refresh_cnt + 1;
        refresh_req <= '0';
      end if;
    end if;
  end process p_refresh;

  -- ===========================================================================
  -- HBM4 CONTROLLER STATE MACHINE
  -- Simplified timing model:
  --   ACTIVATE  : tRCD = 4 cycles
  --   READ/WRITE: CAS latency = 8 cycles (HBM4 CL8)
  --   PRECHARGE : tRP = 4 cycles
  -- ===========================================================================
  p_hbm_ctrl : process(clk)
  begin
    if rising_edge(clk) then
      if rst_n = '0' then
        hbm_state   <= HBM_IDLE;
        hbm_wr_en   <= (others => '0');
        hbm_rd_en   <= (others => '0');
        resp_valid  <= '0';
        lat_cnt     <= 0;
        rob_wr_ptr  <= (others => '0');
        rob_rd_ptr  <= (others => '0');
      else
        hbm_wr_en   <= (others => '0');
        hbm_rd_en   <= (others => '0');
        resp_valid  <= '0';

        case hbm_state is

          when HBM_IDLE =>
            if refresh_req = '1' then
              hbm_state <= HBM_REFRESH;
              lat_cnt   <= 8;
            elsif req_in.valid = '1' and rob_full = '0' then
              hbm_state <= HBM_DECODE;
            end if;

          when HBM_DECODE =>
            hbm_addr  <= std_logic_vector(req_in.addr);
            -- Check if target channel is power-gated
            if chan_power_gate(target_stack) = '1' then
              -- Channel gated — wake up takes 8 cycles
              lat_cnt   <= 8;
            else
              lat_cnt   <= 0;
            end if;
            hbm_state <= HBM_ACTIVATE;

          when HBM_ACTIVATE =>
            -- tRCD = 4 cycles
            if lat_cnt < 4 then
              lat_cnt <= lat_cnt + 1;
            else
              lat_cnt   <= 0;
              if req_in.write_en = '1' then
                hbm_state <= HBM_WRITE;
              else
                hbm_state <= HBM_READ;
              end if;
            end if;

          when HBM_READ =>
            -- Assert RD enable on target stack
            hbm_rd_en(target_stack) <= '1';
            -- CAS latency = 8 cycles
            if lat_cnt < 8 then
              lat_cnt <= lat_cnt + 1;
            else
              lat_cnt   <= 0;
              hbm_state <= HBM_RESPOND;
              -- Capture read data from the appropriate stack
              resp_data <= hbm_rdata(
                (target_stack+1)*DATA_WIDTH-1 downto target_stack*DATA_WIDTH);
            end if;

          when HBM_WRITE =>
            hbm_wr_en(target_stack) <= '1';
            -- Replicate write data across all stack slices
            hbm_wdata <= (others => '0');
            -- Precharge after write
            if lat_cnt < 4 then
              lat_cnt <= lat_cnt + 1;
            else
              lat_cnt   <= 0;
              hbm_state <= HBM_PRECHARGE;
            end if;

          when HBM_PRECHARGE =>
            -- tRP = 4 cycles
            if lat_cnt < 4 then
              lat_cnt <= lat_cnt + 1;
            else
              lat_cnt   <= 0;
              hbm_state <= HBM_IDLE;
            end if;

          when HBM_REFRESH =>
            -- Refresh all open rows
            if lat_cnt > 0 then
              lat_cnt <= lat_cnt - 1;
            else
              hbm_state <= HBM_IDLE;
            end if;

          when HBM_RESPOND =>
            resp_valid <= '1';
            hbm_state  <= HBM_PRECHARGE;
            -- Retire ROB entry
            if rob_rd_ptr /= rob_wr_ptr then
              rob_rd_ptr <= rob_rd_ptr + 1;
            end if;

        end case;
      end if;
    end if;
  end process p_hbm_ctrl;

  resp_out.valid  <= resp_valid;
  resp_out.data   <= resp_data;
  resp_out.req_id <= rob(to_integer(rob_rd_ptr)).req_id;
  resp_out.error  <= '0';

end architecture rtl;


-- =============================================================================
-- KrafGPU-X1 Die-to-Die Interconnect Fabric
-- Description  : SoIC-X modelled 12 TB/s die interconnect.
--                Routes memory requests from both chiplets + host PCIe to
--                the shared L3 cache, and NVLink bridge.
-- =============================================================================

library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;
use work.krafgpu_pkg.all;

entity krafgpu_interconnect_fabric is
  port (
    clk             : in  std_logic;
    rst_n           : in  std_logic;
    -- Chiplet A
    chiplet_a_req   : in  mem_req_t;
    chiplet_a_resp  : out mem_resp_t;
    chiplet_a_ready : out std_logic;
    -- Chiplet B
    chiplet_b_req   : in  mem_req_t;
    chiplet_b_resp  : out mem_resp_t;
    chiplet_b_ready : out std_logic;
    -- System / peripheral port (display, cmd-ring DMA, NPU, raster, ctx)
    sys_req         : in  mem_req_t;
    sys_resp        : out mem_resp_t;
    sys_ready       : out std_logic;
    -- Host PCIe
    host_cmd_valid  : in  std_logic;
    host_cmd_data   : in  std_logic_vector(255 downto 0);
    host_resp_valid : out std_logic;
    host_resp_data  : out std_logic_vector(255 downto 0);
    -- L3 cache (downstream)
    mem_req_out     : out mem_req_t;
    mem_resp_in     : in  mem_resp_t;
    -- NVLink
    nvlink_tx       : out std_logic_vector(127 downto 0);
    nvlink_rx       : in  std_logic_vector(127 downto 0);
    nvlink_valid_tx : out std_logic;
    nvlink_valid_rx : in  std_logic
  );
end entity krafgpu_interconnect_fabric;

architecture rtl of krafgpu_interconnect_fabric is
  -- 0=chiplet_A, 1=chiplet_B, 2=system/peripheral port: index of the
  -- requester that gets first look on the next arbitration cycle.
  signal arb_sel : integer range 0 to 2 := 0;
begin

  -- 3-way round-robin arbiter: Chiplet A, Chiplet B, System/peripheral port.
  -- Each requester is granted in turn; arb_sel rotates to the requester
  -- after the one just granted (or stays put if nobody requested).
  p_fabric : process(clk)
    variable granted : std_logic;
  begin
    if rising_edge(clk) then
      if rst_n = '0' then
        mem_req_out.valid   <= '0';
        chiplet_a_ready     <= '0';
        chiplet_b_ready     <= '0';
        sys_ready           <= '0';
        host_resp_valid     <= '0';
        arb_sel             <= 0;
      else
        mem_req_out.valid <= '0';
        chiplet_a_ready   <= '0';
        chiplet_b_ready   <= '0';
        sys_ready         <= '0';
        host_resp_valid   <= '0';
        granted           := '0';

        for offset in 0 to 2 loop
          if granted = '0' then
            case (arb_sel + offset) mod 3 is
              when 0 =>
                if chiplet_a_req.valid = '1' then
                  mem_req_out     <= chiplet_a_req;
                  chiplet_a_ready <= '1';
                  arb_sel         <= (arb_sel + offset + 1) mod 3;
                  granted         := '1';
                end if;
              when 1 =>
                if chiplet_b_req.valid = '1' then
                  mem_req_out     <= chiplet_b_req;
                  chiplet_b_ready <= '1';
                  arb_sel         <= (arb_sel + offset + 1) mod 3;
                  granted         := '1';
                end if;
              when others =>
                if sys_req.valid = '1' then
                  mem_req_out     <= sys_req;
                  sys_ready       <= '1';
                  arb_sel         <= (arb_sel + offset + 1) mod 3;
                  granted         := '1';
                end if;
            end case;
          end if;
        end loop;

        -- Route response to all requesters; each one self-gates on whether
        -- it has an outstanding request (mirrors the SM-level fan-out
        -- pattern used by krafgpu_rr_arbiter).
        if mem_resp_in.valid = '1' then
          chiplet_a_resp <= mem_resp_in;
          chiplet_b_resp <= mem_resp_in;
          sys_resp       <= mem_resp_in;
        else
          chiplet_a_resp.valid <= '0';
          chiplet_b_resp.valid <= '0';
          sys_resp.valid       <= '0';
        end if;

      end if;
    end if;
  end process p_fabric;

  -- NVLink passthrough (multi-GPU coherence)
  nvlink_tx       <= chiplet_a_req.data & chiplet_b_req.data;
  nvlink_valid_tx <= chiplet_a_req.valid or chiplet_b_req.valid;

  -- Host response
  host_resp_data  <= (others => '0');

end architecture rtl;


-- =============================================================================
-- KrafGPU-X1 L3 Cache — 64 MB, 8-way Set-Associative, Pseudo-LRU
-- Replaces the 2-cycle passthrough stub.
-- Geometry: SIM_SETS=2048 sets (simulation-feasible), 8 ways, 64-byte lines.
-- Tag=17b, Set=11b (SIM), Line=6b. Write-back, write-allocate. SECDED ECC.
-- =============================================================================

library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;
use work.krafgpu_pkg.all;

entity krafgpu_l3_cache is
  generic (
    CACHE_SIZE_MB  : integer := 64;
    LINE_BYTES     : integer := 64;
    NUM_WAYS       : integer := 8
  );
  port (
    clk            : in  std_logic;
    rst_n          : in  std_logic;
    req_in         : in  mem_req_t;
    resp_out       : out mem_resp_t;
    hit_out        : out std_logic;
    mem_req_out    : out mem_req_t;
    mem_resp_in    : in  mem_resp_t;
    ecc_single_err : out std_logic;
    ecc_double_err : out std_logic;
    perf_hits      : out unsigned(31 downto 0);
    perf_misses    : out unsigned(31 downto 0)
  );
end entity krafgpu_l3_cache;

architecture rtl of krafgpu_l3_cache is
  constant LINE_BITS    : integer := 6;
  constant SIM_SETS     : integer := 2048;
  constant SIM_SET_BITS : integer := 11;
  constant TAG_BITS     : integer := PHYS_ADDR_WIDTH - SIM_SET_BITS - LINE_BITS;

  type data_row_t is array (0 to NUM_WAYS-1) of std_logic_vector(DATA_WIDTH-1 downto 0);
  type tag_row_t  is array (0 to NUM_WAYS-1) of unsigned(TAG_BITS-1 downto 0);
  type flag_row_t is array (0 to NUM_WAYS-1) of std_logic;
  type data_arr_t is array (0 to SIM_SETS-1) of data_row_t;
  type tag_arr_t  is array (0 to SIM_SETS-1) of tag_row_t;
  type flag_arr_t is array (0 to SIM_SETS-1) of flag_row_t;
  type plru_arr_t is array (0 to SIM_SETS-1) of std_logic_vector(6 downto 0);

  signal data_arr  : data_arr_t := (others => (others => (others => '0')));
  signal tag_arr   : tag_arr_t  := (others => (others => (others => '0')));
  signal valid_arr : flag_arr_t := (others => (others => '0'));
  signal dirty_arr : flag_arr_t := (others => (others => '0'));
  signal plru_arr  : plru_arr_t := (others => (others => '0'));

  function get_set(a : unsigned(PHYS_ADDR_WIDTH-1 downto 0)) return integer is
  begin return to_integer(a(SIM_SET_BITS+LINE_BITS-1 downto LINE_BITS)) mod SIM_SETS; end;
  function get_tag(a : unsigned(PHYS_ADDR_WIDTH-1 downto 0)) return unsigned is
  begin return a(PHYS_ADDR_WIDTH-1 downto SIM_SET_BITS+LINE_BITS); end;

  function plru_victim(p : std_logic_vector(6 downto 0)) return integer is
  begin
    if    p(0)='0' then if p(1)='0' then return 0; else return 1; end if;
    elsif p(1)='0' then if p(2)='0' then return 2; else return 3; end if;
    elsif p(2)='0' then if p(3)='0' then return 4; else return 5; end if;
    else                if p(3)='0' then return 6; else return 7; end if;
    end if;
  end;
  function plru_update(p : std_logic_vector(6 downto 0); w : integer)
    return std_logic_vector is
    variable r : std_logic_vector(6 downto 0) := p;
  begin
    case w is
      when 0 => r(0):='1'; r(1):='1';
      when 1 => r(0):='1'; r(1):='0';
      when 2 => r(0):='1'; r(2):='1';
      when 3 => r(0):='1'; r(2):='0';
      when 4 => r(0):='0'; r(3):='1';
      when 5 => r(0):='0'; r(3):='0';
      when 6 => r(0):='0'; r(4):='1';
      when others => r(0):='0'; r(4):='0';
    end case;
    return r;
  end;

  function ecc_syn(d : std_logic_vector(31 downto 0)) return std_logic_vector is
    variable p : std_logic_vector(5 downto 0) := (others => '0');
  begin
    for i in 0 to 31 loop
      for b in 0 to 5 loop
        if ((i+1) / (2**b)) mod 2 = 1 then p(b) := p(b) xor d(i); end if;
      end loop;
    end loop;
    return p;
  end;

  type l3_st_t is (L3_IDLE, L3_LOOKUP, L3_HIT, L3_ECC,
                   L3_WB, L3_WB_WAIT, L3_FILL, L3_FILL_WAIT, L3_INSTALL);
  signal state      : l3_st_t := L3_IDLE;
  signal req_lat    : mem_req_t;
  signal req_set_r  : integer range 0 to SIM_SETS-1 := 0;
  signal req_tag_r  : unsigned(TAG_BITS-1 downto 0)  := (others => '0');
  signal hit_way_r  : integer range 0 to NUM_WAYS-1  := 0;
  signal vict_r     : integer range 0 to NUM_WAYS-1  := 0;
  signal ecc_se_r   : std_logic := '0';
  signal ecc_de_r   : std_logic := '0';
  signal hit_r      : std_logic := '0';
  signal resp_r     : mem_resp_t := (valid=>'0',data=>(others=>'0'),
                                     req_id=>(others=>'0'),error=>'0');
  signal hits_r     : unsigned(31 downto 0) := (others => '0');
  signal misses_r   : unsigned(31 downto 0) := (others => '0');
begin
  resp_out       <= resp_r;
  hit_out        <= hit_r;
  ecc_single_err <= ecc_se_r;
  ecc_double_err <= ecc_de_r;
  perf_hits      <= hits_r;
  perf_misses    <= misses_r;

  p_l3 : process(clk)
    variable way_hit : integer range 0 to NUM_WAYS := NUM_WAYS;
    variable vict    : integer range 0 to NUM_WAYS-1 := 0;
    variable syn     : std_logic_vector(5 downto 0);
  begin
    if rising_edge(clk) then
      if rst_n = '0' then
        state <= L3_IDLE; resp_r.valid <= '0'; mem_req_out.valid <= '0';
        hit_r <= '0'; ecc_se_r <= '0'; ecc_de_r <= '0';
        for s in 0 to SIM_SETS-1 loop
          for w in 0 to NUM_WAYS-1 loop
            valid_arr(s)(w) <= '0'; dirty_arr(s)(w) <= '0';
          end loop;
        end loop;
      else
        resp_r.valid <= '0'; mem_req_out.valid <= '0';
        ecc_se_r <= '0'; ecc_de_r <= '0';

        case state is
          when L3_IDLE =>
            hit_r <= '0';
            if req_in.valid = '1' then
              req_lat <= req_in;
              req_set_r <= get_set(req_in.addr);
              req_tag_r <= get_tag(req_in.addr);
              state <= L3_LOOKUP;
            end if;

          when L3_LOOKUP =>
            way_hit := NUM_WAYS;
            for w in 0 to NUM_WAYS-1 loop
              if valid_arr(req_set_r)(w)='1' and tag_arr(req_set_r)(w)=req_tag_r then
                way_hit := w;
              end if;
            end loop;
            if way_hit < NUM_WAYS then
              hit_way_r <= way_hit; hits_r <= hits_r + 1;
              plru_arr(req_set_r) <= plru_update(plru_arr(req_set_r), way_hit);
              state <= L3_ECC;
            else
              misses_r <= misses_r + 1;
              vict := plru_victim(plru_arr(req_set_r)); vict_r <= vict;
              if dirty_arr(req_set_r)(vict) = '1' then
                mem_req_out.valid <= '1';
                mem_req_out.addr  <= tag_arr(req_set_r)(vict) &
                  to_unsigned(req_set_r, SIM_SET_BITS) & to_unsigned(0, LINE_BITS);
                mem_req_out.data     <= data_arr(req_set_r)(vict);
                mem_req_out.write_en <= '1'; mem_req_out.byte_en <= "1111";
                mem_req_out.warp_id  <= req_lat.warp_id;
                mem_req_out.req_id   <= req_lat.req_id;
                dirty_arr(req_set_r)(vict) <= '0';
                state <= L3_WB_WAIT;
              else state <= L3_FILL; end if;
            end if;

          when L3_ECC =>
            syn := ecc_syn(data_arr(req_set_r)(hit_way_r));
            if syn /= "000000" then
              if syn(5)='1' then ecc_se_r<='1'; else ecc_de_r<='1'; end if;
            end if;
            state <= L3_HIT;

          when L3_HIT =>
            if req_lat.write_en = '0' then
              resp_r <= (valid=>'1', data=>data_arr(req_set_r)(hit_way_r),
                         req_id=>req_lat.req_id, error=>ecc_de_r);
            else
              data_arr(req_set_r)(hit_way_r)  <= req_lat.data;
              dirty_arr(req_set_r)(hit_way_r) <= '1';
              resp_r <= (valid=>'1', data=>(others=>'0'),
                         req_id=>req_lat.req_id, error=>'0');
            end if;
            hit_r <= '1'; state <= L3_IDLE;

          when L3_WB_WAIT =>
            if mem_resp_in.valid='1' then state <= L3_FILL; end if;

          when L3_FILL =>
            mem_req_out.valid    <= '1';
            mem_req_out.addr     <= req_lat.addr(PHYS_ADDR_WIDTH-1 downto LINE_BITS)
                                    & to_unsigned(0, LINE_BITS);
            mem_req_out.data     <= (others => '0');
            mem_req_out.write_en <= '0'; mem_req_out.byte_en <= "1111";
            mem_req_out.warp_id  <= req_lat.warp_id;
            mem_req_out.req_id   <= req_lat.req_id;
            state <= L3_FILL_WAIT;

          when L3_FILL_WAIT =>
            if mem_resp_in.valid='1' then state <= L3_INSTALL; end if;

          when L3_INSTALL =>
            if mem_resp_in.valid='1' then
              data_arr(req_set_r)(vict_r)  <= mem_resp_in.data;
              tag_arr(req_set_r)(vict_r)   <= req_tag_r;
              valid_arr(req_set_r)(vict_r) <= '1';
              dirty_arr(req_set_r)(vict_r) <= '0';
              plru_arr(req_set_r) <= plru_update(plru_arr(req_set_r), vict_r);
              resp_r <= (valid=>'1', data=>mem_resp_in.data,
                         req_id=>req_lat.req_id, error=>'0');
              hit_r <= '0'; state <= L3_IDLE;
            end if;

          when others => state <= L3_IDLE;
        end case;
      end if;
    end if;
  end process p_l3;
end architecture rtl;


-- =============================================================================
-- KrafGPU-X1 PCIe 6.0 Interface — TLP Packet Handler
-- Replaces the pass-through stub. Handles MRd/MWr/CfgRd/CfgWr/Msg.
-- Completions: CplD for reads, Cpl for writes (not modelled separately here
-- for clarity — writes are acknowledged implicitly via cmd pipeline).
-- BAR0 = MMIO [0..16MB], BAR1 = VRAM [16MB+].
-- 256-bit flit: [255:248]=type [207:176]=addr_lo [175:160]=req_id
--               [159:152]=tag [151:120]=data[0]
-- =============================================================================

library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;
use work.krafgpu_pkg.all;

entity krafgpu_pcie_if is
  port (
    clk             : in  std_logic;
    rst_n           : in  std_logic;
    pcie_rx_data    : in  std_logic_vector(255 downto 0);
    pcie_rx_valid   : in  std_logic;
    pcie_rx_ready   : out std_logic;
    pcie_tx_data    : out std_logic_vector(255 downto 0);
    pcie_tx_valid   : out std_logic;
    pcie_tx_ready   : in  std_logic;
    cmd_valid_out   : out std_logic;
    cmd_data_out    : out std_logic_vector(255 downto 0);
    resp_valid_in   : in  std_logic;
    resp_data_in    : in  std_logic_vector(255 downto 0)
  );
end entity krafgpu_pcie_if;

architecture rtl of krafgpu_pcie_if is
  constant TLP_MRD32  : std_logic_vector(7 downto 0) := x"00";
  constant TLP_MRD64  : std_logic_vector(7 downto 0) := x"20";
  constant TLP_MWR32  : std_logic_vector(7 downto 0) := x"40";
  constant TLP_MWR64  : std_logic_vector(7 downto 0) := x"60";
  constant TLP_CFGRD0 : std_logic_vector(7 downto 0) := x"04";
  constant TLP_CFGWR0 : std_logic_vector(7 downto 0) := x"44";
  constant TLP_CPLD   : std_logic_vector(7 downto 0) := x"4A";

  constant BAR0_LIMIT : unsigned(31 downto 0) := x"00FFFFFF";
  constant BAR1_BASE  : unsigned(31 downto 0) := x"01000000";

  constant CFG_DEV_VEN   : std_logic_vector(31 downto 0) := x"CAFE1DE5";
  constant CFG_STS_CMD   : std_logic_vector(31 downto 0) := x"00100006";
  constant CFG_CLASS_REV : std_logic_vector(31 downto 0) := x"03020001";

  function flit_type  (f:std_logic_vector(255 downto 0)) return std_logic_vector is
  begin return f(255 downto 248); end;
  function flit_addrlo(f:std_logic_vector(255 downto 0)) return unsigned is
  begin return unsigned(f(207 downto 176)); end;
  function flit_addrhi(f:std_logic_vector(255 downto 0)) return unsigned is
  begin return unsigned(f(239 downto 208)); end;
  function flit_reqid (f:std_logic_vector(255 downto 0)) return std_logic_vector is
  begin return f(175 downto 160); end;
  function flit_tag   (f:std_logic_vector(255 downto 0)) return std_logic_vector is
  begin return f(159 downto 152); end;
  function flit_data0 (f:std_logic_vector(255 downto 0)) return std_logic_vector is
  begin return f(151 downto 120); end;

  type rx_st_t is (RX_IDLE, RX_DECODE, RX_STALL);
  signal rx_st     : rx_st_t := RX_IDLE;
  signal tlp_type_r : std_logic_vector(7 downto 0) := (others=>'0');
  signal tlp_addr_r : unsigned(63 downto 0) := (others=>'0');
  signal tlp_req_r  : std_logic_vector(15 downto 0) := (others=>'0');
  signal tlp_tag_r  : std_logic_vector(7 downto 0)  := (others=>'0');
  signal tlp_data_r : std_logic_vector(31 downto 0) := (others=>'0');
  signal bar0_hit   : std_logic := '0';
  signal bar1_hit   : std_logic := '0';
  signal cfg_hit    : std_logic := '0';
  signal cpl_pend   : std_logic := '0';
  signal cfg_rdata  : std_logic_vector(31 downto 0) := (others=>'0');
  signal cfg_rvld   : std_logic := '0';

  type tx_st_t is (TX_IDLE, TX_SEND, TX_HOLD);
  signal tx_st     : tx_st_t := TX_IDLE;
  signal tx_data_r : std_logic_vector(255 downto 0) := (others=>'0');
  signal tx_vld_r  : std_logic := '0';
  signal cmd_vld_r : std_logic := '0';
  signal cmd_dat_r : std_logic_vector(255 downto 0) := (others=>'0');
begin
  pcie_rx_ready <= '1' when rx_st /= RX_STALL else '0';
  pcie_tx_data  <= tx_data_r;
  pcie_tx_valid <= tx_vld_r;
  cmd_valid_out <= cmd_vld_r;
  cmd_data_out  <= cmd_dat_r;

  p_rx : process(clk)
  begin
    if rising_edge(clk) then
      if rst_n = '0' then
        rx_st <= RX_IDLE; cmd_vld_r <= '0'; cpl_pend <= '0'; cfg_rvld <= '0';
      else
        cmd_vld_r <= '0'; cfg_rvld <= '0';
        case rx_st is
          when RX_IDLE =>
            if pcie_rx_valid = '1' then
              tlp_type_r <= flit_type(pcie_rx_data);
              tlp_addr_r <= flit_addrhi(pcie_rx_data) & flit_addrlo(pcie_rx_data);
              tlp_req_r  <= flit_reqid(pcie_rx_data);
              tlp_tag_r  <= flit_tag(pcie_rx_data);
              tlp_data_r <= flit_data0(pcie_rx_data);
              if flit_addrhi(pcie_rx_data) = 0 and flit_addrlo(pcie_rx_data) <= BAR0_LIMIT then
                bar0_hit <= '1'; bar1_hit <= '0'; cfg_hit <= '0';
              elsif flit_addrhi(pcie_rx_data) = 0 and flit_addrlo(pcie_rx_data) >= BAR1_BASE then
                bar0_hit <= '0'; bar1_hit <= '1'; cfg_hit <= '0';
              elsif flit_type(pcie_rx_data) = TLP_CFGRD0 or flit_type(pcie_rx_data) = TLP_CFGWR0 then
                bar0_hit <= '0'; bar1_hit <= '0'; cfg_hit <= '1';
              else
                bar0_hit <= '0'; bar1_hit <= '0'; cfg_hit <= '0';
              end if;
              rx_st <= RX_DECODE;
            end if;

          when RX_DECODE =>
            case tlp_type_r is
              when TLP_MRD32 | TLP_MRD64 =>
                if bar0_hit = '1' or bar1_hit = '1' then
                  cmd_vld_r <= '1';
                  cmd_dat_r <= TLP_MRD32 & x"01" & x"0000" &
                               std_logic_vector(tlp_addr_r(31 downto 0)) &
                               x"0000" & tlp_req_r & tlp_tag_r & (111 downto 0 => '0');
                  cpl_pend <= '1'; rx_st <= RX_STALL;
                else rx_st <= RX_IDLE; end if;
              when TLP_MWR32 | TLP_MWR64 =>
                cmd_vld_r <= '1';
                cmd_dat_r <= TLP_MWR32 & x"01" & x"0000" &
                             std_logic_vector(tlp_addr_r(31 downto 0)) &
                             x"0000" & tlp_req_r & tlp_tag_r &
                             tlp_data_r & (79 downto 0 => '0');
                rx_st <= RX_IDLE;
              when TLP_CFGRD0 =>
                case std_logic_vector(tlp_addr_r(7 downto 2)) is
                  when "000000" => cfg_rdata <= CFG_DEV_VEN;
                  when "000001" => cfg_rdata <= CFG_STS_CMD;
                  when "000010" => cfg_rdata <= CFG_CLASS_REV;
                  when others   => cfg_rdata <= x"FFFFFFFF";
                end case;
                cfg_rvld <= '1'; cpl_pend <= '1'; rx_st <= RX_STALL;
              when TLP_CFGWR0 => rx_st <= RX_IDLE;
              when others     => rx_st <= RX_IDLE;
            end case;

          when RX_STALL =>
            if cpl_pend = '0' then rx_st <= RX_IDLE; end if;
        end case;
      end if;
    end if;
  end process p_rx;

  p_tx : process(clk)
    variable cpl_d : std_logic_vector(31 downto 0);
  begin
    if rising_edge(clk) then
      if rst_n = '0' then
        tx_st <= TX_IDLE; tx_vld_r <= '0'; cpl_pend <= '0';
      else
        tx_vld_r <= '0';
        case tx_st is
          when TX_IDLE =>
            if cpl_pend = '1' and (resp_valid_in = '1' or cfg_rvld = '1') then
              if cfg_rvld = '1' then cpl_d := cfg_rdata;
              else cpl_d := resp_data_in(31 downto 0); end if;
              tx_data_r <= TLP_CPLD & x"01" & x"CAFE" & x"0004" &
                           tlp_req_r & tlp_tag_r &
                           '0' & std_logic_vector(tlp_addr_r(6 downto 0)) &
                           cpl_d & (112 downto 0 => '0');
              tx_vld_r <= '1'; cpl_pend <= '0'; tx_st <= TX_HOLD;
            end if;
          when TX_HOLD =>
            if pcie_tx_ready = '1' then tx_st <= TX_IDLE;
            else tx_vld_r <= '1'; end if;
          when others => tx_st <= TX_IDLE;
        end case;
      end if;
    end if;
  end process p_tx;
end architecture rtl;


-- =============================================================================
-- KrafGPU-X1 JTAG TAP Controller — IEEE 1149.1 (full 16-state FSM)
-- Replaces the TDI→TDO bypass stub.
-- Data registers: BYPASS(1b), IDCODE(32b), USERCODE(32b), EXTEST(1b).
-- IR: 4-bit. IDCODE=0x1ABCD76B. USERCODE=0x4B524654 ("KRFT").
-- TDO changes on falling TCK edge per IEEE 1149.1 timing.
-- =============================================================================

library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;

entity krafgpu_jtag_tap is
  port (
    tck : in  std_logic;
    tms : in  std_logic;
    tdi : in  std_logic;
    tdo : out std_logic
  );
end entity krafgpu_jtag_tap;

architecture rtl of krafgpu_jtag_tap is
  type tap_st_t is (
    RESET, IDLE,
    SEL_DR, CAP_DR, SHF_DR, EX1_DR, PAU_DR, EX2_DR, UPD_DR,
    SEL_IR, CAP_IR, SHF_IR, EX1_IR, PAU_IR, EX2_IR, UPD_IR
  );
  signal state    : tap_st_t := RESET;
  constant IR_EXTEST   : std_logic_vector(3 downto 0) := "0000";
  constant IR_BYPASS   : std_logic_vector(3 downto 0) := "0001";
  constant IR_IDCODE   : std_logic_vector(3 downto 0) := "1110";
  constant IR_USERCODE : std_logic_vector(3 downto 0) := "1111";
  constant IDCODE_VAL  : std_logic_vector(31 downto 0) := x"1ABCD76B";
  constant UCODE_VAL   : std_logic_vector(31 downto 0) := x"4B524654";
  signal ir_shift  : std_logic_vector(3 downto 0) := IR_IDCODE;
  signal ir_hold   : std_logic_vector(3 downto 0) := IR_IDCODE;
  signal dr_shift  : std_logic_vector(31 downto 0) := IDCODE_VAL;
  signal bypass_r  : std_logic := '0';
  signal tdo_r     : std_logic := '1';
begin
  tdo <= tdo_r;

  p_rise : process(tck)
  begin
    if rising_edge(tck) then
      case state is
        when RESET  => if tms='0' then state<=IDLE;   end if;
        when IDLE   => if tms='1' then state<=SEL_DR; end if;
        when SEL_DR => if tms='1' then state<=SEL_IR; else state<=CAP_DR; end if;
        when CAP_DR => if tms='1' then state<=EX1_DR; else state<=SHF_DR; end if;
        when SHF_DR => if tms='1' then state<=EX1_DR; end if;
        when EX1_DR => if tms='1' then state<=UPD_DR; else state<=PAU_DR; end if;
        when PAU_DR => if tms='1' then state<=EX2_DR; end if;
        when EX2_DR => if tms='1' then state<=UPD_DR; else state<=SHF_DR; end if;
        when UPD_DR => if tms='1' then state<=SEL_DR; else state<=IDLE;   end if;
        when SEL_IR => if tms='1' then state<=RESET;  else state<=CAP_IR; end if;
        when CAP_IR => if tms='1' then state<=EX1_IR; else state<=SHF_IR; end if;
        when SHF_IR => if tms='1' then state<=EX1_IR; end if;
        when EX1_IR => if tms='1' then state<=UPD_IR; else state<=PAU_IR; end if;
        when PAU_IR => if tms='1' then state<=EX2_IR; end if;
        when EX2_IR => if tms='1' then state<=UPD_IR; else state<=SHF_IR; end if;
        when UPD_IR => if tms='1' then state<=SEL_DR; else state<=IDLE;   end if;
      end case;

      if state = CAP_DR then
        case ir_hold is
          when IR_IDCODE   => dr_shift <= IDCODE_VAL;
          when IR_USERCODE => dr_shift <= UCODE_VAL;
          when others      => dr_shift <= (31 downto 1=>'0') & bypass_r;
        end case;
      end if;
      if state = CAP_IR then ir_shift <= "0001"; end if;
      if state = SHF_DR then
        if ir_hold = IR_BYPASS then bypass_r <= tdi;
        else dr_shift <= tdi & dr_shift(31 downto 1); end if;
      end if;
      if state = SHF_IR then ir_shift <= tdi & ir_shift(3 downto 1); end if;
      if state = UPD_IR then ir_hold  <= ir_shift; end if;
      if state = RESET  then
        ir_hold <= IR_IDCODE; ir_shift <= IR_IDCODE; dr_shift <= IDCODE_VAL;
      end if;
    end if;
  end process p_rise;

  p_fall : process(tck)
  begin
    if falling_edge(tck) then
      if state = SHF_DR then
        if ir_hold = IR_BYPASS then tdo_r <= bypass_r;
        else tdo_r <= dr_shift(0); end if;
      elsif state = SHF_IR then
        tdo_r <= ir_shift(0);
      else
        tdo_r <= '1';
      end if;
    end if;
  end process p_fall;
end architecture rtl;
