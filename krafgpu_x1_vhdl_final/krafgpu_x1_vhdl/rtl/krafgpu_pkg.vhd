-- =============================================================================
-- KrafGPU-X1 Package
-- Project      : KrafGPU-X1 Next-Generation GPU Architecture
-- Author       : Krafvyn Tech / Don
-- Description  : Global constants, types, and function definitions
--                for the entire KrafGPU-X1 design
-- Process      : TSMC 2nm N2P
-- =============================================================================

library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;

package krafgpu_pkg is

  -- ===========================================================================
  -- GLOBAL ARCHITECTURE CONSTANTS
  -- ===========================================================================

  -- Shader multiprocessor topology
  constant NUM_SMS_TOTAL       : integer := 512;   -- Total shader multiprocessors
  constant NUM_SMS_PER_ISLAND  : integer := 32;    -- SMs per voltage island
  constant NUM_ISLANDS         : integer := 16;    -- Total voltage islands
  constant CORES_PER_SM        : integer := 128;   -- FP32 cores per SM
  constant TENSOR_UNITS_PER_SM : integer := 16;    -- Tensor cores per SM
  constant RT_CORES_PER_SM     : integer := 2;     -- Ray tracing cores per SM

  -- Warp and thread block dimensions
  constant WARP_SIZE           : integer := 32;    -- Threads per warp
  constant MAX_WARPS_PER_SM    : integer := 64;    -- Max concurrent warps
  constant MAX_THREADS_PER_SM  : integer := 2048;  -- Max threads per SM
  constant MAX_REGS_PER_THREAD : integer := 255;   -- Register file depth

  -- Cache hierarchy
  constant L1_CACHE_KB         : integer := 128;   -- L1 per SM in KB
  constant L2_CACHE_MB         : integer := 256;   -- L2 shared in MB
  constant L3_CACHE_GB         : integer := 1;     -- L3 in GB

  -- Memory subsystem
  constant NUM_HBM4_STACKS     : integer := 6;     -- HBM4 stack count
  constant HBM4_CHANNELS       : integer := 16;    -- Channels per stack
  constant HBM4_CHANNEL_WIDTH  : integer := 64;    -- Bits per channel
  constant HBM4_TOTAL_WIDTH    : integer := 6144;  -- Total memory bus width (bits)

  -- Frequency targets
  constant FREQ_FULL_MHZ       : integer := 2800;  -- Full voltage: 2.8 GHz
  constant FREQ_HALF_MHZ       : integer := 2400;  -- Half voltage: 2.4 GHz
  constant FREQ_LOW_MHZ        : integer := 1800;  -- Low voltage: 1.8 GHz

  -- Voltage levels (in millivolts)
  constant VOLT_FULL_MV        : integer := 1000;  -- 1.000V full perf
  constant VOLT_HALF_MV        : integer := 850;   -- 0.850V balanced
  constant VOLT_LOW_MV         : integer := 750;   -- 0.750V efficiency
  constant VOLT_OFF_MV         : integer := 0;     -- 0V gated

  -- Datapath widths
  constant DATA_WIDTH          : integer := 32;    -- FP32 default width
  constant ADDR_WIDTH          : integer := 48;    -- Virtual address width
  constant PHYS_ADDR_WIDTH     : integer := 40;    -- Physical address width
  constant CACHE_LINE_BITS     : integer := 512;   -- Cache line = 64 bytes

  -- Instruction set
  constant INSTR_WIDTH         : integer := 32;
  constant PC_WIDTH            : integer := 32;
  constant NUM_REGS            : integer := 256;

  -- Shared memory: 128 KB per SM (32768 × 32-bit words).
  -- Defined at package level so krafgpu_lsu can accept the full array
  -- as a port rather than the previous single-word stub.
  constant SMEM_DEPTH : integer := 32768;
  type smem_array_t   is array (0 to SMEM_DEPTH-1) of
                          std_logic_vector(DATA_WIDTH-1 downto 0);

  -- ===========================================================================
  -- VOLTAGE ISLAND STATE TYPES
  -- ===========================================================================

  type island_state_t is (
    ISLAND_OFF,     -- Power gated — 0V
    ISLAND_LOW,     -- 0.75V / 1.8 GHz — efficiency mode
    ISLAND_HALF,    -- 0.85V / 2.4 GHz — balanced mode
    ISLAND_FULL     -- 1.00V / 2.8 GHz — max performance
  );

  type island_state_array_t is array (0 to NUM_ISLANDS-1) of island_state_t;
  type island_voltage_array_t is array (0 to NUM_ISLANDS-1) of unsigned(11 downto 0);  -- mV

  -- Unconstrained companion type: lets a sub-block's port be sized by its
  -- own generic (e.g. NUM_ISLANDS_LOCAL on krafgpu_compute_chiplet) while
  -- krafgpu_x1_top's full-width island_states signal stays an
  -- island_state_array_t(0 to NUM_ISLANDS-1) and is passed in per-chiplet
  -- slices.
  type island_state_array_u_t is array (natural range <>) of island_state_t;

  -- ===========================================================================
  -- WARP STATE TYPES
  -- ===========================================================================

  type warp_state_t is (
    WARP_IDLE,
    WARP_READY,
    WARP_RUNNING,
    WARP_STALLED_MEM,
    WARP_STALLED_DEP,
    WARP_BARRIER,
    WARP_DONE
  );

  type warp_state_array_t is array (0 to MAX_WARPS_PER_SM-1) of warp_state_t;

  -- ===========================================================================
  -- INSTRUCTION TYPES
  -- ===========================================================================

  type opcode_t is (
    OP_FADD, OP_FSUB, OP_FMUL, OP_FMAD,
    OP_IADD, OP_ISUB, OP_IMUL,
    OP_MOV,                               -- Integer move / copy
    OP_LD,  OP_LDG,                       -- Global load (LD and LDG synonyms)
    OP_ST,  OP_STG,                       -- Global store (ST and STG synonyms)
    OP_LDS, OP_STS,                       -- Shared memory load / store
    OP_ATOM,
    OP_BRA, OP_CALL, OP_RET, OP_EXIT,
    OP_SYNC, OP_BAR,
    OP_TEX,
    OP_TENSOR,
    OP_RT_TRACE,
    OP_NOP
  );

  -- ===========================================================================
  -- INSTRUCTION RECORD
  -- ===========================================================================

  type instruction_t is record
    opcode    : opcode_t;
    dst_reg   : unsigned(7 downto 0);
    src_reg_a : unsigned(7 downto 0);
    src_reg_b : unsigned(7 downto 0);
    src_reg_c : unsigned(7 downto 0);
    imm       : signed(15 downto 0);
    pred_reg  : unsigned(2 downto 0);
    pred_not  : std_logic;
  end record;

  -- ===========================================================================
  -- PIPELINE STAGE RECORDS
  -- ===========================================================================

  -- Fetch → Decode
  type fetch_decode_t is record
    valid     : std_logic;
    warp_id   : unsigned(5 downto 0);
    pc        : unsigned(PC_WIDTH-1 downto 0);
    instr_raw : std_logic_vector(INSTR_WIDTH-1 downto 0);
  end record;

  -- Decode → Issue
  type decode_issue_t is record
    valid     : std_logic;
    warp_id   : unsigned(5 downto 0);
    instr     : instruction_t;
    src_a_val : std_logic_vector(DATA_WIDTH-1 downto 0);
    src_b_val : std_logic_vector(DATA_WIDTH-1 downto 0);
    src_c_val : std_logic_vector(DATA_WIDTH-1 downto 0);
  end record;

  -- ===========================================================================
  -- MEMORY REQUEST TYPES
  -- ===========================================================================

  type mem_req_t is record
    valid     : std_logic;
    addr      : unsigned(PHYS_ADDR_WIDTH-1 downto 0);
    data      : std_logic_vector(DATA_WIDTH-1 downto 0);
    write_en  : std_logic;
    byte_en   : std_logic_vector(3 downto 0);
    warp_id   : unsigned(5 downto 0);
    req_id    : unsigned(7 downto 0);
  end record;

  type mem_resp_t is record
    valid     : std_logic;
    data      : std_logic_vector(DATA_WIDTH-1 downto 0);
    req_id    : unsigned(7 downto 0);
    error     : std_logic;
  end record;

  -- ===========================================================================
  -- POWER MANAGEMENT TYPES
  -- ===========================================================================

  type pmc_cmd_t is record
    island_id   : unsigned(3 downto 0);   -- 0..15
    target_state: island_state_t;
    valid       : std_logic;
  end record;

  type telemetry_t is record
    island_id   : unsigned(3 downto 0);
    utilisation : unsigned(6 downto 0);   -- 0-100%
    temperature : unsigned(7 downto 0);   -- degrees C
    power_mw    : unsigned(15 downto 0);  -- milliwatts
  end record;

  type telemetry_array_t is array (0 to NUM_ISLANDS-1) of telemetry_t;

  -- Unconstrained companion type, same rationale as island_state_array_u_t
  -- above: krafgpu_compute_chiplet's telemetry_out port is sized by its own
  -- NUM_ISLANDS_LOCAL generic rather than the full-chip NUM_ISLANDS.
  type telemetry_array_u_t is array (natural range <>) of telemetry_t;

  -- ===========================================================================
  -- UNCONSTRAINED ARRAY TYPES FOR GENERIC-WIDTH MEMORY BUSES
  -- Used by krafgpu_rr_arbiter (and similar N-port muxes) so the port width
  -- can be set by a generic at instantiation time. Concrete signals declare
  -- a constrained subtype, e.g. mem_req_array_t(0 to NUM_SMS-1).
  -- ===========================================================================
  type mem_req_array_t  is array (natural range <>) of mem_req_t;
  type mem_resp_array_t is array (natural range <>) of mem_resp_t;

  -- ===========================================================================
  -- UTILITY FUNCTIONS
  -- ===========================================================================

  function log2_ceil(n : integer) return integer;
  function island_of_sm(sm_id : integer) return integer;
  function to_volt_mv(state : island_state_t) return integer;
  function to_freq_mhz(state : island_state_t) return integer;
  function count_active_islands(states : island_state_array_t) return integer;

end package krafgpu_pkg;

-- =============================================================================
-- Package Body
-- =============================================================================
package body krafgpu_pkg is

  function log2_ceil(n : integer) return integer is
    variable result : integer := 0;
    variable val    : integer := n - 1;
  begin
    while val > 0 loop
      result := result + 1;
      val    := val / 2;
    end loop;
    return result;
  end function;

  function island_of_sm(sm_id : integer) return integer is
  begin
    return sm_id / NUM_SMS_PER_ISLAND;
  end function;

  function to_volt_mv(state : island_state_t) return integer is
  begin
    case state is
      when ISLAND_FULL => return VOLT_FULL_MV;
      when ISLAND_HALF => return VOLT_HALF_MV;
      when ISLAND_LOW  => return VOLT_LOW_MV;
      when ISLAND_OFF  => return VOLT_OFF_MV;
    end case;
  end function;

  function to_freq_mhz(state : island_state_t) return integer is
  begin
    case state is
      when ISLAND_FULL => return FREQ_FULL_MHZ;
      when ISLAND_HALF => return FREQ_HALF_MHZ;
      when ISLAND_LOW  => return FREQ_LOW_MHZ;
      when ISLAND_OFF  => return 0;
    end case;
  end function;

  function count_active_islands(states : island_state_array_t) return integer is
    variable cnt : integer := 0;
  begin
    for i in states'range loop
      if states(i) /= ISLAND_OFF then
        cnt := cnt + 1;
      end if;
    end loop;
    return cnt;
  end function;

end package body krafgpu_pkg;
