-- =============================================================================
-- KrafGPU-X1 Command Ring Processor
-- Project      : KrafGPU-X1 Next-Generation GPU Architecture
-- Author       : Krafvyn Tech / Don
-- Description  : DMA-driven command ring buffer processor.
--
--   The command ring is the primary interface between the host CPU driver
--   and the GPU. The driver writes GPU command packets into a circular
--   buffer in system memory (pinned/IOMMU-mapped), then writes the new
--   HEAD pointer to BAR0. The ring processor reads and dispatches commands.
--
--   Command packet format (32-bit header + variable payload):
--     [31:28] = Opcode class (4 bits)
--     [27:16] = Payload DWORDs (12 bits, max 4095 DWORDs = 16 KB per packet)
--     [15:0]  = Opcode (16 bits)
--
--   Supported command classes:
--     0x0 = NOP / sync fence
--     0x1 = SET_REGISTER  (write BAR0 register from ring)
--     0x2 = DRAW          (draw primitive: vertex count, index buf, etc.)
--     0x3 = DISPATCH      (compute dispatch: thread group X/Y/Z)
--     0x4 = COPY          (DMA blit: src/dst/size)
--     0x5 = BARRIER       (pipeline flush + cache invalidation)
--     0x6 = FENCE_SIGNAL  (write value to fence addr in memory)
--     0x7 = INDIRECT      (read another ring address for chained commands)
--     0xF = END_OF_STREAM (ring wrap/empty)
--
--   Ring depth: 65536 DWORDs (256 KB ring buffer)
--   Max outstanding commands: 256 (tracked in flight counter)
-- =============================================================================

library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;
use work.krafgpu_pkg.all;

entity krafgpu_cmd_ring is
  generic (
    RING_DWORDS  : integer := 65536;   -- 256 KB ring (65536 × 32-bit)
    MAX_INFLIGHT : integer := 256
  );
  port (
    clk              : in  std_logic;
    rst_n            : in  std_logic;

    -- BAR0 ring control registers (from MMIO)
    ring_base_addr   : in  unsigned(PHYS_ADDR_WIDTH-1 downto 0);
    ring_size_dwords : in  unsigned(31 downto 0);
    ring_head        : in  unsigned(31 downto 0);  -- Written by driver (new data)
    ring_tail        : out unsigned(31 downto 0);  -- Read by driver (consumed)

    -- DMA read interface (fetch commands from host memory)
    dma_req          : out mem_req_t;
    dma_resp         : in  mem_resp_t;

    -- Decoded command outputs
    -- Draw command
    draw_valid       : out std_logic;
    draw_vtx_count   : out unsigned(31 downto 0);
    draw_inst_count  : out unsigned(31 downto 0);
    draw_idx_buf     : out unsigned(PHYS_ADDR_WIDTH-1 downto 0);
    draw_vtx_buf     : out unsigned(PHYS_ADDR_WIDTH-1 downto 0);
    draw_ready       : in  std_logic;

    -- Compute dispatch command
    dispatch_valid   : out std_logic;
    dispatch_x       : out unsigned(15 downto 0);
    dispatch_y       : out unsigned(15 downto 0);
    dispatch_z       : out unsigned(15 downto 0);
    dispatch_ready   : in  std_logic;

    -- Register write command
    reg_wr_valid     : out std_logic;
    reg_wr_addr      : out unsigned(23 downto 0);
    reg_wr_data      : out std_logic_vector(31 downto 0);

    -- DMA copy command
    copy_valid       : out std_logic;
    copy_src         : out unsigned(PHYS_ADDR_WIDTH-1 downto 0);
    copy_dst         : out unsigned(PHYS_ADDR_WIDTH-1 downto 0);
    copy_size        : out unsigned(31 downto 0);
    copy_ready       : in  std_logic;

    -- Fence signal command
    fence_valid      : out std_logic;
    fence_addr       : out unsigned(PHYS_ADDR_WIDTH-1 downto 0);
    fence_value      : out std_logic_vector(31 downto 0);

    -- Pipeline barrier
    barrier_valid    : out std_logic;
    barrier_flags    : out std_logic_vector(7 downto 0);

    -- Interrupt request (on fence signal)
    irq_fence        : out std_logic;

    -- Status
    ring_idle        : out std_logic;
    inflight_count   : out unsigned(7 downto 0)
  );
end entity krafgpu_cmd_ring;

architecture rtl of krafgpu_cmd_ring is

  -- ===========================================================================
  -- COMMAND OPCODES
  -- ===========================================================================
  constant CMD_NOP         : std_logic_vector(3 downto 0) := x"0";
  constant CMD_SET_REG     : std_logic_vector(3 downto 0) := x"1";
  constant CMD_DRAW        : std_logic_vector(3 downto 0) := x"2";
  constant CMD_DISPATCH    : std_logic_vector(3 downto 0) := x"3";
  constant CMD_COPY        : std_logic_vector(3 downto 0) := x"4";
  constant CMD_BARRIER     : std_logic_vector(3 downto 0) := x"5";
  constant CMD_FENCE       : std_logic_vector(3 downto 0) := x"6";
  constant CMD_INDIRECT    : std_logic_vector(3 downto 0) := x"7";
  constant CMD_EOS         : std_logic_vector(3 downto 0) := x"F";

  -- ===========================================================================
  -- RING PROCESSOR STATE MACHINE
  -- ===========================================================================
  type ring_state_t is (
    RING_FSM_IDLE,      -- No commands pending
    RING_CHECK_NEW,     -- Compare head/tail pointers
    RING_FETCH_HDR,     -- DMA fetch command header DWORD
    RING_WAIT_HDR,      -- Wait for header fetch response
    RING_DECODE,        -- Decode header → command type + payload size
    RING_FETCH_PAYLOAD, -- DMA fetch payload DWORDs
    RING_WAIT_PAYLOAD,  -- Wait for payload fetch
    RING_DISPATCH_CMD,  -- Issue decoded command to downstream
    RING_WAIT_ACCEPT,   -- Wait for downstream to accept command
    RING_ADVANCE_TAIL,  -- Advance tail pointer
    RING_WRAP           -- Handle ring wrap-around
  );
  signal ring_state    : ring_state_t := RING_FSM_IDLE;

  -- ===========================================================================
  -- RING POINTERS
  -- ===========================================================================
  signal tail_r        : unsigned(31 downto 0) := (others => '0');
  signal fetch_addr    : unsigned(PHYS_ADDR_WIDTH-1 downto 0);

  -- ===========================================================================
  -- COMMAND DECODE REGISTERS
  -- ===========================================================================
  signal cmd_header    : std_logic_vector(31 downto 0);
  signal cmd_class     : std_logic_vector(3 downto 0);
  signal cmd_opcode    : std_logic_vector(15 downto 0);
  signal cmd_payload_n : unsigned(11 downto 0);   -- Payload DWORDs count

  -- Payload fetch buffer (max 16 DWORDs for common commands)
  type payload_buf_t is array (0 to 15) of std_logic_vector(31 downto 0);
  signal payload_buf   : payload_buf_t;
  signal payload_cnt   : unsigned(3 downto 0) := (others => '0');
  signal payload_idx   : integer range 0 to 15 := 0;

  -- ===========================================================================
  -- DECODED COMMAND REGISTERS
  -- ===========================================================================
  signal draw_valid_r    : std_logic := '0';
  signal dispatch_valid_r: std_logic := '0';
  signal reg_wr_valid_r  : std_logic := '0';
  signal copy_valid_r    : std_logic := '0';
  signal fence_valid_r   : std_logic := '0';
  signal barrier_valid_r : std_logic := '0';
  signal irq_fence_r     : std_logic := '0';

  signal inflight_r      : unsigned(7 downto 0) := (others => '0');

begin

  ring_tail      <= tail_r;
  inflight_count <= inflight_r;

  -- Compute DMA fetch address from tail pointer + ring base
  fetch_addr <= ring_base_addr +
    resize(tail_r * 4, PHYS_ADDR_WIDTH);  -- tail in DWORDs → bytes

  -- ===========================================================================
  -- RING PROCESSOR STATE MACHINE
  -- ===========================================================================
  p_ring_fsm : process(clk)
    variable new_data    : std_logic;
    variable cmd_cls     : std_logic_vector(3 downto 0);
    variable fetch_byte  : unsigned(PHYS_ADDR_WIDTH-1 downto 0);
  begin
    if rising_edge(clk) then
      if rst_n = '0' then
        ring_state      <= RING_FSM_IDLE;
        tail_r          <= (others => '0');
        dma_req.valid   <= '0';
        draw_valid_r    <= '0';
        dispatch_valid_r<= '0';
        reg_wr_valid_r  <= '0';
        copy_valid_r    <= '0';
        fence_valid_r   <= '0';
        barrier_valid_r <= '0';
        irq_fence_r     <= '0';
        inflight_r      <= (others => '0');
        payload_idx     <= 0;
        ring_idle       <= '1';
      else
        dma_req.valid   <= '0';
        draw_valid_r    <= '0';
        dispatch_valid_r<= '0';
        reg_wr_valid_r  <= '0';
        copy_valid_r    <= '0';
        fence_valid_r   <= '0';
        barrier_valid_r <= '0';
        irq_fence_r     <= '0';

        case ring_state is

          -- ----------------------------------------------------------------
          when RING_FSM_IDLE =>
            ring_idle <= '1';
            ring_state <= RING_CHECK_NEW;
          -- ----------------------------------------------------------------
          when RING_CHECK_NEW =>
            -- New work available when head != tail
            if ring_head /= tail_r then
              ring_idle  <= '0';
              ring_state <= RING_FETCH_HDR;
            else
              ring_idle  <= '1';
              ring_state <= RING_FSM_IDLE;
            end if;

          -- ----------------------------------------------------------------
          when RING_FETCH_HDR =>
            -- DMA fetch the command header DWORD
            dma_req.valid    <= '1';
            dma_req.addr     <= fetch_addr;
            dma_req.write_en <= '0';
            dma_req.data     <= (others => '0');
            dma_req.byte_en  <= "1111";
            dma_req.warp_id  <= (others => '0');
            dma_req.req_id   <= to_unsigned(0, 8);
            ring_state       <= RING_WAIT_HDR;

          -- ----------------------------------------------------------------
          when RING_WAIT_HDR =>
            if dma_resp.valid = '1' then
              cmd_header    <= dma_resp.data;
              cmd_class     <= dma_resp.data(31 downto 28);
              cmd_opcode    <= dma_resp.data(15 downto 0);
              cmd_payload_n <= unsigned(dma_resp.data(27 downto 16));
              ring_state    <= RING_DECODE;
            end if;

          -- ----------------------------------------------------------------
          when RING_DECODE =>
            cmd_cls     := cmd_header(31 downto 28);
            payload_idx <= 0;

            -- End of stream: wrap ring
            if cmd_cls = CMD_EOS then
              ring_state <= RING_WRAP;

            -- NOP: advance tail by 1, no payload
            elsif cmd_cls = CMD_NOP then
              tail_r     <= (tail_r + 1) mod ring_size_dwords;
              ring_state <= RING_CHECK_NEW;

            -- Commands with payload: fetch payload next
            elsif cmd_payload_n > 0 then
              ring_state <= RING_FETCH_PAYLOAD;

            -- Commands with no payload: dispatch immediately
            else
              ring_state <= RING_DISPATCH_CMD;
            end if;

          -- ----------------------------------------------------------------
          when RING_FETCH_PAYLOAD =>
            -- Fetch one payload DWORD per cycle
            fetch_byte := ring_base_addr +
              resize((tail_r + 1 + to_unsigned(payload_idx, 32)) * 4,
                     PHYS_ADDR_WIDTH);
            dma_req.valid    <= '1';
            dma_req.addr     <= fetch_byte;
            dma_req.write_en <= '0';
            dma_req.data     <= (others => '0');
            dma_req.byte_en  <= "1111";
            dma_req.warp_id  <= (others => '0');
            dma_req.req_id   <= to_unsigned(payload_idx, 8);
            ring_state       <= RING_WAIT_PAYLOAD;

          -- ----------------------------------------------------------------
          when RING_WAIT_PAYLOAD =>
            if dma_resp.valid = '1' then
              if payload_idx < 16 then
                payload_buf(payload_idx) <= dma_resp.data;
              end if;
              if to_unsigned(payload_idx, 12) < cmd_payload_n - 1 then
                payload_idx <= payload_idx + 1;
                ring_state  <= RING_FETCH_PAYLOAD;
              else
                payload_idx <= 0;
                ring_state  <= RING_DISPATCH_CMD;
              end if;
            end if;

          -- ----------------------------------------------------------------
          when RING_DISPATCH_CMD =>
            case cmd_header(31 downto 28) is

              when CMD_SET_REG =>
                -- payload[0] = BAR0 offset, payload[1] = value
                reg_wr_valid_r <= '1';
                reg_wr_addr    <= unsigned(payload_buf(0)(23 downto 0));
                reg_wr_data    <= payload_buf(1);
                ring_state     <= RING_WAIT_ACCEPT;

              when CMD_DRAW =>
                -- payload[0]=vtx_count [1]=inst_count [2..3]=vtx_buf [4..5]=idx_buf
                draw_valid_r   <= '1';
                draw_vtx_count <= unsigned(payload_buf(0));
                draw_inst_count<= unsigned(payload_buf(1));
                draw_vtx_buf   <=
                  unsigned(payload_buf(3)) & unsigned(payload_buf(2));
                draw_idx_buf   <=
                  unsigned(payload_buf(5)) & unsigned(payload_buf(4));
                if inflight_r < MAX_INFLIGHT then
                  inflight_r <= inflight_r + 1;
                end if;
                ring_state     <= RING_WAIT_ACCEPT;

              when CMD_DISPATCH =>
                -- payload[0]=X [1]=Y [2]=Z thread groups
                dispatch_valid_r <= '1';
                dispatch_x       <= unsigned(payload_buf(0)(15 downto 0));
                dispatch_y       <= unsigned(payload_buf(1)(15 downto 0));
                dispatch_z       <= unsigned(payload_buf(2)(15 downto 0));
                if inflight_r < MAX_INFLIGHT then
                  inflight_r <= inflight_r + 1;
                end if;
                ring_state <= RING_WAIT_ACCEPT;

              when CMD_COPY =>
                -- payload[0..1]=src_addr [2..3]=dst_addr [4]=size
                copy_valid_r <= '1';
                copy_src     <=
                  unsigned(payload_buf(1)) & unsigned(payload_buf(0));
                copy_dst     <=
                  unsigned(payload_buf(3)) & unsigned(payload_buf(2));
                copy_size    <= unsigned(payload_buf(4));
                ring_state   <= RING_WAIT_ACCEPT;

              when CMD_BARRIER =>
                -- payload[0][7:0] = barrier flags
                barrier_valid_r <= '1';
                barrier_flags   <= payload_buf(0)(7 downto 0);
                ring_state      <= RING_ADVANCE_TAIL;

              when CMD_FENCE =>
                -- payload[0..1]=fence_addr [2]=fence_value
                fence_valid_r <= '1';
                fence_addr    <=
                  unsigned(payload_buf(1)) & unsigned(payload_buf(0));
                fence_value   <= payload_buf(2);
                irq_fence_r   <= '1';
                ring_state    <= RING_ADVANCE_TAIL;

              when others =>
                -- Unknown command: skip
                ring_state <= RING_ADVANCE_TAIL;
            end case;

          -- ----------------------------------------------------------------
          when RING_WAIT_ACCEPT =>
            -- Wait for downstream unit to accept the decoded command
            case cmd_header(31 downto 28) is
              when CMD_DRAW =>
                if draw_ready = '1' then
                  draw_valid_r <= '0';
                  ring_state   <= RING_ADVANCE_TAIL;
                end if;
              when CMD_DISPATCH =>
                if dispatch_ready = '1' then
                  dispatch_valid_r <= '0';
                  ring_state       <= RING_ADVANCE_TAIL;
                end if;
              when CMD_COPY =>
                if copy_ready = '1' then
                  copy_valid_r <= '0';
                  ring_state   <= RING_ADVANCE_TAIL;
                end if;
              when CMD_SET_REG =>
                -- Register write is always accepted in 1 cycle
                reg_wr_valid_r <= '0';
                ring_state     <= RING_ADVANCE_TAIL;
              when others =>
                ring_state <= RING_ADVANCE_TAIL;
            end case;

          -- ----------------------------------------------------------------
          when RING_ADVANCE_TAIL =>
            -- Advance tail by 1 (header) + payload_n DWORDs
            tail_r     <= (tail_r + 1 + cmd_payload_n) mod ring_size_dwords;
            ring_state <= RING_CHECK_NEW;
            -- Decrement inflight on completion
            if inflight_r > 0 and
               (cmd_header(31 downto 28) = CMD_DRAW or
                cmd_header(31 downto 28) = CMD_DISPATCH) then
              inflight_r <= inflight_r - 1;
            end if;

          -- ----------------------------------------------------------------
          when RING_WRAP =>
            -- EOS packet: jump tail back to 0
            tail_r     <= (others => '0');
            ring_state <= RING_CHECK_NEW;

        end case;
      end if;
    end if;
  end process p_ring_fsm;

  -- ===========================================================================
  -- OUTPUT ASSIGNMENTS
  -- ===========================================================================
  draw_valid     <= draw_valid_r;
  dispatch_valid <= dispatch_valid_r;
  reg_wr_valid   <= reg_wr_valid_r;
  copy_valid     <= copy_valid_r;
  fence_valid    <= fence_valid_r;
  barrier_valid  <= barrier_valid_r;
  irq_fence      <= irq_fence_r;

end architecture rtl;
