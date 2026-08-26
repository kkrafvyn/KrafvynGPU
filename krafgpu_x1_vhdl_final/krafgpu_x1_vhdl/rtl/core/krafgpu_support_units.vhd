-- =============================================================================
-- KrafGPU-X1 PMBus Controller
-- Description  : I2C-compatible PMBus master.
--                Sends VOUT_COMMAND (0x21) to PMIC over PMBus.
--                Each island maps to one PMIC channel address (0x20..0x2F).
--                400 kHz I2C fast mode.
-- =============================================================================

library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;
use work.krafgpu_pkg.all;

entity krafgpu_pmbus_ctrl is
  port (
    clk          : in    std_logic;
    rst_n        : in    std_logic;
    -- Command interface
    start        : in    std_logic;
    island_id    : in    unsigned(3 downto 0);    -- 0..15
    voltage_mv   : in    unsigned(11 downto 0);   -- millivolts
    busy         : out   std_logic;
    done         : out   std_logic;
    -- PMBus physical
    scl          : inout std_logic;
    sda          : inout std_logic
  );
end entity krafgpu_pmbus_ctrl;

architecture rtl of krafgpu_pmbus_ctrl is

  -- I2C clock divider for 400 kHz from 400 MHz
  constant SCL_DIV      : integer := 500;   -- 400MHz / (2×500) = 400 kHz
  signal   scl_cnt      : integer range 0 to SCL_DIV-1 := 0;
  signal   scl_toggle   : std_logic := '0';

  type pmbus_state_t is (
    PB_IDLE, PB_START, PB_ADDR, PB_CMD, PB_DATA_H, PB_DATA_L,
    PB_ACK, PB_STOP, PB_DONE
  );

  signal pb_state       : pmbus_state_t := PB_IDLE;
  signal bit_cnt        : integer range 0 to 8 := 0;
  signal shift_reg      : std_logic_vector(7 downto 0);
  signal scl_out        : std_logic := '1';
  signal sda_out        : std_logic := '1';

  -- PMIC base address: 0x40 + island_id (4 PMICs × 4 channels)
  signal pmic_addr      : std_logic_vector(6 downto 0);
  signal voltage_word   : std_logic_vector(15 downto 0);

begin

  pmic_addr    <= "100" & std_logic_vector(island_id);
  -- PMBus Linear11 format: mantissa in lower 11 bits, exponent in upper 5
  voltage_word <= std_logic_vector(resize(voltage_mv, 16));

  -- SCL generation
  p_scl : process(clk)
  begin
    if rising_edge(clk) then
      if rst_n = '0' then
        scl_cnt    <= 0;
        scl_toggle <= '0';
      elsif scl_cnt = SCL_DIV - 1 then
        scl_cnt    <= 0;
        scl_toggle <= not scl_toggle;
      else
        scl_cnt    <= scl_cnt + 1;
      end if;
    end if;
  end process p_scl;

  -- PMBus state machine
  p_pmbus : process(clk)
  begin
    if rising_edge(clk) then
      if rst_n = '0' then
        pb_state  <= PB_IDLE;
        busy      <= '0';
        done      <= '0';
        sda_out   <= '1';
        scl_out   <= '1';
      else
        done <= '0';
        case pb_state is
          when PB_IDLE =>
            busy    <= '0';
            sda_out <= '1';
            scl_out <= '1';
            if start = '1' then
              busy      <= '1';
              pb_state  <= PB_START;
            end if;
          when PB_START =>
            sda_out  <= '0';   -- SDA low while SCL high = START
            if scl_toggle = '1' then
              pb_state <= PB_ADDR;
              bit_cnt  <= 7;
              shift_reg <= pmic_addr & '0';  -- R/W = 0 (write)
            end if;
          when PB_ADDR =>
            sda_out  <= shift_reg(7);
            if scl_toggle = '1' then
              shift_reg <= shift_reg(6 downto 0) & '0';
              if bit_cnt = 0 then
                pb_state <= PB_ACK;
              else
                bit_cnt  <= bit_cnt - 1;
              end if;
            end if;
          when PB_ACK =>
            sda_out  <= 'Z';   -- Release SDA for ACK from PMIC
            if scl_toggle = '1' then
              pb_state <= PB_CMD;
              shift_reg <= x"21";  -- VOUT_COMMAND
              bit_cnt   <= 7;
            end if;
          when PB_CMD =>
            sda_out  <= shift_reg(7);
            if scl_toggle = '1' then
              shift_reg <= shift_reg(6 downto 0) & '0';
              if bit_cnt = 0 then
                pb_state <= PB_DATA_H;
                shift_reg <= voltage_word(15 downto 8);
                bit_cnt   <= 7;
              else
                bit_cnt  <= bit_cnt - 1;
              end if;
            end if;
          when PB_DATA_H =>
            sda_out  <= shift_reg(7);
            if scl_toggle = '1' then
              shift_reg <= shift_reg(6 downto 0) & '0';
              if bit_cnt = 0 then
                pb_state <= PB_DATA_L;
                shift_reg <= voltage_word(7 downto 0);
                bit_cnt   <= 7;
              else
                bit_cnt  <= bit_cnt - 1;
              end if;
            end if;
          when PB_DATA_L =>
            sda_out  <= shift_reg(7);
            if scl_toggle = '1' then
              shift_reg <= shift_reg(6 downto 0) & '0';
              if bit_cnt = 0 then
                pb_state <= PB_STOP;
              else
                bit_cnt  <= bit_cnt - 1;
              end if;
            end if;
          when PB_STOP =>
            sda_out  <= '0';
            if scl_toggle = '1' then
              sda_out  <= '1';   -- SDA rising while SCL high = STOP
              pb_state <= PB_DONE;
            end if;
          when PB_DONE =>
            done     <= '1';
            busy     <= '0';
            pb_state <= PB_IDLE;
        end case;
      end if;
    end if;
  end process p_pmbus;

  scl <= scl_out when scl_out = '0' else 'Z';
  sda <= sda_out when sda_out = '0' else 'Z';

end architecture rtl;


-- =============================================================================
-- KrafGPU-X1 UART Transmitter (8N1)
-- =============================================================================

library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;

entity krafgpu_uart_tx is
  generic (
    CLK_FREQ_HZ : integer := 400_000_000;
    BAUD_RATE   : integer := 115_200
  );
  port (
    clk   : in  std_logic;
    rst_n : in  std_logic;
    send  : in  std_logic;
    data  : in  std_logic_vector(7 downto 0);
    busy  : out std_logic;
    tx    : out std_logic
  );
end entity krafgpu_uart_tx;

architecture rtl of krafgpu_uart_tx is
  constant CLKS_PER_BIT : integer := CLK_FREQ_HZ / BAUD_RATE;
  signal baud_cnt : integer range 0 to CLKS_PER_BIT-1 := 0;
  signal bit_idx  : integer range 0 to 9 := 0;
  signal shift    : std_logic_vector(9 downto 0) := (others => '1');
  signal active   : std_logic := '0';
begin
  busy <= active;
  tx   <= shift(0);

  p_uart : process(clk)
  begin
    if rising_edge(clk) then
      if rst_n = '0' then
        active   <= '0';
        shift    <= (others => '1');
        baud_cnt <= 0;
        bit_idx  <= 0;
      elsif active = '0' and send = '1' then
        -- Load: start bit + 8 data bits + stop bit
        shift    <= '1' & data & '0';
        active   <= '1';
        baud_cnt <= 0;
        bit_idx  <= 0;
      elsif active = '1' then
        if baud_cnt = CLKS_PER_BIT - 1 then
          baud_cnt <= 0;
          shift    <= '1' & shift(9 downto 1);
          if bit_idx = 9 then
            active <= '0';
          else
            bit_idx <= bit_idx + 1;
          end if;
        else
          baud_cnt <= baud_cnt + 1;
        end if;
      end if;
    end if;
  end process p_uart;
end architecture rtl;


-- =============================================================================
-- KrafGPU-X1 Instruction Decoder
-- =============================================================================

library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;
use work.krafgpu_pkg.all;

entity krafgpu_decoder is
  port (
    instr_raw  : in  std_logic_vector(INSTR_WIDTH-1 downto 0);
    instr_out  : out instruction_t
  );
end entity krafgpu_decoder;

architecture rtl of krafgpu_decoder is
  signal opcode_bits : std_logic_vector(7 downto 0);
begin
  opcode_bits <= instr_raw(31 downto 24);

  p_decode : process(instr_raw, opcode_bits)
  begin
    -- Default
    instr_out.dst_reg   <= unsigned(instr_raw(23 downto 16));
    instr_out.src_reg_a <= unsigned(instr_raw(15 downto 8));
    instr_out.src_reg_b <= unsigned(instr_raw(7 downto 0));
    instr_out.src_reg_c <= (others => '0');
    instr_out.imm       <= signed(instr_raw(15 downto 0));
    instr_out.pred_reg  <= unsigned(instr_raw(26 downto 24));
    instr_out.pred_not  <= instr_raw(27);

    case opcode_bits is
      when x"10" => instr_out.opcode <= OP_FADD;
      when x"11" => instr_out.opcode <= OP_FSUB;
      when x"12" => instr_out.opcode <= OP_FMUL;
      when x"13" => instr_out.opcode <= OP_FMAD;
      when x"20" => instr_out.opcode <= OP_IADD;
      when x"21" => instr_out.opcode <= OP_ISUB;
      when x"22" => instr_out.opcode <= OP_IMUL;
      when x"30" => instr_out.opcode <= OP_LD;
      when x"31" => instr_out.opcode <= OP_ST;
      when x"32" => instr_out.opcode <= OP_LDS;
      when x"33" => instr_out.opcode <= OP_STS;
      when x"40" => instr_out.opcode <= OP_BRA;
      when x"41" => instr_out.opcode <= OP_CALL;
      when x"42" => instr_out.opcode <= OP_RET;
      when x"43" => instr_out.opcode <= OP_EXIT;
      when x"50" => instr_out.opcode <= OP_SYNC;
      when x"51" => instr_out.opcode <= OP_BAR;
      when x"60" => instr_out.opcode <= OP_TEX;
      when x"70" => instr_out.opcode <= OP_TENSOR;
      when x"80" => instr_out.opcode <= OP_RT_TRACE;
      when others => instr_out.opcode <= OP_NOP;
    end case;
  end process p_decode;
end architecture rtl;


-- =============================================================================
-- KrafGPU-X1 Load/Store Unit (LSU)
-- =============================================================================

library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;
use work.krafgpu_pkg.all;

entity krafgpu_lsu is
  port (
    clk             : in  std_logic;
    clk_en          : in  std_logic;
    rst_n           : in  std_logic;
    valid_in        : in  std_logic;
    opcode          : in  opcode_t;
    addr_base       : in  std_logic_vector(DATA_WIDTH-1 downto 0);
    store_data      : in  std_logic_vector(DATA_WIDTH-1 downto 0);
    imm_offset      : in  signed(15 downto 0);
    warp_id_in      : in  unsigned(5 downto 0);
    dst_reg_in      : in  unsigned(7 downto 0);
    -- Full 128 KB shared-memory array (replaces the previous single-word stub).
    -- The LSU reads/writes it directly for OP_LDS/OP_STS with zero latency.
    smem            : in  smem_array_t;
    smem_wr_en      : out std_logic;
    smem_wr_addr    : out integer range 0 to SMEM_DEPTH-1;
    smem_wr_data    : out std_logic_vector(DATA_WIDTH-1 downto 0);
    -- Global memory path (through L1 cache → L2 → HBM4)
    mem_req_out     : out mem_req_t;
    mem_resp_in     : in  mem_resp_t;
    result_valid    : out std_logic;
    result_data     : out std_logic_vector(DATA_WIDTH-1 downto 0);
    mem_pending_out : out std_logic
  );
end entity krafgpu_lsu;

architecture rtl of krafgpu_lsu is
  signal pending   : std_logic := '0';
  signal req_id_r  : unsigned(7 downto 0) := (others => '0');
begin
  mem_pending_out <= pending;

  p_lsu : process(clk)
    variable eff_addr : unsigned(PHYS_ADDR_WIDTH-1 downto 0);
    variable smem_idx : integer range 0 to SMEM_DEPTH-1;
  begin
    if rising_edge(clk) and clk_en = '1' then
      if rst_n = '0' then
        pending           <= '0';
        mem_req_out.valid <= '0';
        result_valid      <= '0';
        smem_wr_en        <= '0';
      else
        mem_req_out.valid <= '0';
        result_valid      <= '0';
        smem_wr_en        <= '0';

        if valid_in = '1' and pending = '0' then
          -- Effective byte address (base + sign-extended immediate)
          eff_addr := resize(
            unsigned(addr_base) + unsigned(resize(imm_offset, DATA_WIDTH)),
            PHYS_ADDR_WIDTH);
          -- Shared-memory word index (byte addr / 4, modulo SMEM_DEPTH)
          smem_idx := to_integer(eff_addr(14 downto 2)) mod SMEM_DEPTH;

          case opcode is

            -- ── Global load (LD and LDG are synonyms) ──────────────────────
            when OP_LD | OP_LDG =>
              mem_req_out.valid    <= '1';
              mem_req_out.addr     <= eff_addr;
              mem_req_out.data     <= (others => '0');
              mem_req_out.write_en <= '0';
              mem_req_out.byte_en  <= "1111";
              mem_req_out.warp_id  <= warp_id_in;
              mem_req_out.req_id   <= req_id_r;
              req_id_r             <= req_id_r + 1;
              pending              <= '1';

            -- ── Global store (ST and STG are synonyms) ──────────────────────
            when OP_ST | OP_STG =>
              mem_req_out.valid    <= '1';
              mem_req_out.addr     <= eff_addr;
              mem_req_out.data     <= store_data;
              mem_req_out.write_en <= '1';
              mem_req_out.byte_en  <= "1111";
              mem_req_out.warp_id  <= warp_id_in;
              mem_req_out.req_id   <= req_id_r;
              req_id_r             <= req_id_r + 1;
              -- Stores do not stall the pipeline (fire-and-forget at L1)
              pending              <= '0';

            -- ── Shared memory load (zero latency: read registered here) ────
            when OP_LDS =>
              result_data  <= smem(smem_idx);
              result_valid <= '1';
              pending      <= '0';

            -- ── Shared memory store ─────────────────────────────────────────
            when OP_STS =>
              smem_wr_en   <= '1';
              smem_wr_addr <= smem_idx;
              smem_wr_data <= store_data;
              pending      <= '0';

            when others => null;
          end case;
        end if;

        -- Global memory load response
        if mem_resp_in.valid = '1' and pending = '1' then
          result_data  <= mem_resp_in.data;
          result_valid <= '1';
          pending      <= '0';
        end if;
      end if;
    end if;
  end process p_lsu;
end architecture rtl;


-- =============================================================================
-- KrafGPU-X1 Round-Robin Arbiter
-- =============================================================================

library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;
use work.krafgpu_pkg.all;

entity krafgpu_rr_arbiter is
  generic (
    NUM_PORTS : integer := 256
  );
  port (
    clk       : in  std_logic;
    rst_n     : in  std_logic;
    req_valid : in  std_logic_vector(NUM_PORTS-1 downto 0);
    req_data  : in  mem_req_array_t(0 to NUM_PORTS-1);
    grant_idx : out integer range 0 to NUM_PORTS-1;
    out_req   : out mem_req_t;
    in_resp   : in  mem_resp_t;
    sm_resp   : out mem_resp_array_t(0 to NUM_PORTS-1)
  );
end entity krafgpu_rr_arbiter;

architecture rtl of krafgpu_rr_arbiter is
  signal rr_ptr     : integer range 0 to NUM_PORTS-1 := 0;
  signal grant_reg  : integer range 0 to NUM_PORTS-1 := 0;
begin
  grant_idx <= grant_reg;

  p_arb : process(clk)
    variable winner : integer range 0 to NUM_PORTS-1;
    variable found  : std_logic;
  begin
    if rising_edge(clk) then
      if rst_n = '0' then
        rr_ptr    <= 0;
        grant_reg <= 0;
        out_req.valid <= '0';
      else
        found  := '0';
        winner := rr_ptr;

        -- Scan from rr_ptr
        for i in 0 to NUM_PORTS-1 loop
          if found = '0' then
            if req_valid((rr_ptr + i) mod NUM_PORTS) = '1' then
              winner := (rr_ptr + i) mod NUM_PORTS;
              found  := '1';
            end if;
          end if;
        end loop;

        if found = '1' then
          grant_reg <= winner;
          out_req   <= req_data(winner);
          rr_ptr    <= (winner + 1) mod NUM_PORTS;
        else
          out_req.valid <= '0';
        end if;
      end if;
    end if;
  end process p_arb;

  -- Broadcast response to all SMs; each SM checks req_id
  gen_resp : for i in 0 to NUM_PORTS-1 generate
    sm_resp(i) <= in_resp;
  end generate;

end architecture rtl;
