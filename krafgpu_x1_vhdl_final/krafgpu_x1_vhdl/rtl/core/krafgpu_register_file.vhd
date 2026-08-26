-- =============================================================================
-- KrafGPU-X1 Register File
-- Project      : KrafGPU-X1 Next-Generation GPU Architecture
-- Author       : Krafvyn Tech / Don
-- Description  : Banked register file for the SM.
--
--   Architecture:
--     - 64 warps × 256 registers × 32-bit = 512 KB per SM register file
--     - Split into 4 banks (interleaved by register index bits [1:0])
--       to provide 2 read ports + 1 write port simultaneously
--     - Bank conflict detection: if both reads hit the same bank, stall 1 cycle
--     - Implemented as synchronous BRAM (2-cycle read latency)
--     - Forwarding network: if write and read address match in same cycle,
--       bypass BRAM and forward result directly (eliminates RAW hazard)
--
--   Predicate Registers:
--     - 8 × 1-bit predicate registers per warp (P0..P7)
--     - Written by SETP (set predicate) instruction
--     - Read by branch unit for conditional execution
--
--   Register File Size per SM:
--     64 warps × 256 regs × 32 bits = 524,288 bits = 64 KB
--     × 128 lanes (SIMT) = 8 MB total register state per SM
--     (In silicon: distributed across the SM floorplan as SRAM macros)
-- =============================================================================

library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;
use work.krafgpu_pkg.all;

entity krafgpu_register_file is
  generic (
    SM_ID       : integer := 0;
    NUM_WARPS   : integer := MAX_WARPS_PER_SM;  -- 64
    NUM_REGS    : integer := 256
  );
  port (
    clk         : in  std_logic;
    clk_en      : in  std_logic;
    rst_n       : in  std_logic;

    -- READ PORT A (source operand 1)
    rd_warp_a   : in  unsigned(5 downto 0);
    rd_reg_a    : in  unsigned(7 downto 0);
    rd_data_a   : out std_logic_vector(DATA_WIDTH-1 downto 0);

    -- READ PORT B (source operand 2)
    rd_warp_b   : in  unsigned(5 downto 0);
    rd_reg_b    : in  unsigned(7 downto 0);
    rd_data_b   : out std_logic_vector(DATA_WIDTH-1 downto 0);

    -- READ PORT C (source operand 3 — for FMA / tensor)
    rd_warp_c   : in  unsigned(5 downto 0);
    rd_reg_c    : in  unsigned(7 downto 0);
    rd_data_c   : out std_logic_vector(DATA_WIDTH-1 downto 0);

    -- WRITE PORT
    wr_en       : in  std_logic;
    wr_warp     : in  unsigned(5 downto 0);
    wr_reg      : in  unsigned(7 downto 0);
    wr_data     : in  std_logic_vector(DATA_WIDTH-1 downto 0);

    -- PREDICATE REGISTER PORTS
    pred_wr_en  : in  std_logic;
    pred_warp   : in  unsigned(5 downto 0);
    pred_idx    : in  unsigned(2 downto 0);    -- P0..P7
    pred_val_in : in  std_logic;
    pred_val_out: out std_logic;               -- Read for branch unit

    -- Bank conflict stall
    bank_stall  : out std_logic
  );
end entity krafgpu_register_file;

architecture rtl of krafgpu_register_file is

  -- ===========================================================================
  -- REGISTER FILE STORAGE
  -- 4 banks, each holding registers where reg_idx mod 4 = bank_id
  -- Each bank: 64 warps × 64 regs = 4096 entries × 32 bits
  -- ===========================================================================
  constant NUM_BANKS    : integer := 4;
  constant REGS_PER_BANK: integer := NUM_REGS / NUM_BANKS;  -- 64

  type bank_t is array (0 to NUM_WARPS*REGS_PER_BANK - 1) of
                 std_logic_vector(DATA_WIDTH-1 downto 0);
  type regfile_t is array (0 to NUM_BANKS-1) of bank_t;

  signal regfile : regfile_t := (others => (others => (others => '0')));

  -- ===========================================================================
  -- PREDICATE REGISTERS (8 bits per warp)
  -- ===========================================================================
  type pred_file_t is array (0 to NUM_WARPS-1) of
                      std_logic_vector(7 downto 0);
  signal pred_file : pred_file_t := (others => (others => '0'));

  -- ===========================================================================
  -- BANK ADDRESS CALCULATION
  -- bank_id  = reg_idx[1:0]
  -- bank_row = (warp_id × REGS_PER_BANK) + (reg_idx >> 2)
  -- ===========================================================================
  function bank_id(reg : unsigned(7 downto 0)) return integer is
  begin
    return to_integer(reg(1 downto 0));
  end function;

  function bank_addr(warp : unsigned(5 downto 0);
                     reg  : unsigned(7 downto 0)) return integer is
  begin
    return to_integer(warp) * REGS_PER_BANK +
           to_integer(reg(7 downto 2));
  end function;

  -- ===========================================================================
  -- FORWARDING / BYPASS SIGNALS
  -- ===========================================================================
  signal fwd_en_a : std_logic;
  signal fwd_en_b : std_logic;
  signal fwd_en_c : std_logic;

  -- ===========================================================================
  -- READ DATA REGISTERS (2-cycle BRAM latency)
  -- ===========================================================================
  signal rd_data_a_r1, rd_data_a_r2 : std_logic_vector(DATA_WIDTH-1 downto 0);
  signal rd_data_b_r1, rd_data_b_r2 : std_logic_vector(DATA_WIDTH-1 downto 0);
  signal rd_data_c_r1, rd_data_c_r2 : std_logic_vector(DATA_WIDTH-1 downto 0);

  -- ===========================================================================
  -- BANK CONFLICT DETECTION
  -- Conflict when two read ports address the same bank simultaneously
  -- ===========================================================================
  signal bank_conflict_ab : std_logic;
  signal bank_conflict_ac : std_logic;
  signal stall_r          : std_logic := '0';

begin

  -- ===========================================================================
  -- BANK CONFLICT DETECTION (combinational)
  -- ===========================================================================
  bank_conflict_ab <= '1' when bank_id(rd_reg_a) = bank_id(rd_reg_b) and
                               rd_warp_a = rd_warp_b
                          else '0';

  bank_conflict_ac <= '1' when bank_id(rd_reg_a) = bank_id(rd_reg_c) and
                               rd_warp_a = rd_warp_c
                          else '0';

  bank_stall <= (bank_conflict_ab or bank_conflict_ac) and not stall_r;

  -- ===========================================================================
  -- FORWARDING BYPASS DETECTION
  -- If write matches either read address, bypass BRAM with forwarded data
  -- ===========================================================================
  fwd_en_a <= '1' when wr_en = '1' and
                        wr_warp = rd_warp_a and
                        wr_reg  = rd_reg_a
                   else '0';

  fwd_en_b <= '1' when wr_en = '1' and
                        wr_warp = rd_warp_b and
                        wr_reg  = rd_reg_b
                   else '0';

  fwd_en_c <= '1' when wr_en = '1' and
                        wr_warp = rd_warp_c and
                        wr_reg  = rd_reg_c
                   else '0';

  -- ===========================================================================
  -- REGISTER FILE READ (2-cycle BRAM pipeline)
  -- ===========================================================================
  p_read : process(clk)
  begin
    if rising_edge(clk) and clk_en = '1' then
      if rst_n = '0' then
        rd_data_a_r1 <= (others => '0');
        rd_data_b_r1 <= (others => '0');
        rd_data_c_r1 <= (others => '0');
        rd_data_a_r2 <= (others => '0');
        rd_data_b_r2 <= (others => '0');
        rd_data_c_r2 <= (others => '0');
      else
        -- Cycle 1: BRAM read
        rd_data_a_r1 <=
          regfile(bank_id(rd_reg_a))(bank_addr(rd_warp_a, rd_reg_a));
        rd_data_b_r1 <=
          regfile(bank_id(rd_reg_b))(bank_addr(rd_warp_b, rd_reg_b));
        rd_data_c_r1 <=
          regfile(bank_id(rd_reg_c))(bank_addr(rd_warp_c, rd_reg_c));

        -- Cycle 2: Pipeline register
        rd_data_a_r2 <= rd_data_a_r1;
        rd_data_b_r2 <= rd_data_b_r1;
        rd_data_c_r2 <= rd_data_c_r1;

        stall_r <= bank_conflict_ab or bank_conflict_ac;
      end if;
    end if;
  end process p_read;

  -- ===========================================================================
  -- REGISTER FILE WRITE (single-cycle, priority write)
  -- ===========================================================================
  p_write : process(clk)
  begin
    if rising_edge(clk) and clk_en = '1' then
      if rst_n = '1' and wr_en = '1' then
        regfile(bank_id(wr_reg))(bank_addr(wr_warp, wr_reg)) <= wr_data;
      end if;
    end if;
  end process p_write;

  -- ===========================================================================
  -- PREDICATE REGISTER READ/WRITE
  -- ===========================================================================
  p_pred : process(clk)
  begin
    if rising_edge(clk) and clk_en = '1' then
      if rst_n = '0' then
        pred_file <= (others => (others => '0'));
      elsif pred_wr_en = '1' then
        pred_file(to_integer(pred_warp))(to_integer(pred_idx)) <= pred_val_in;
      end if;
    end if;
  end process p_pred;

  -- ===========================================================================
  -- OUTPUT MUX: Forwarding bypass or BRAM read
  -- ===========================================================================
  rd_data_a <= wr_data  when fwd_en_a = '1' else rd_data_a_r2;
  rd_data_b <= wr_data  when fwd_en_b = '1' else rd_data_b_r2;
  rd_data_c <= wr_data  when fwd_en_c = '1' else rd_data_c_r2;

  -- Predicate read (combinational — used same cycle by branch unit)
  pred_val_out <=
    pred_file(to_integer(pred_warp))(to_integer(pred_idx));

end architecture rtl;
