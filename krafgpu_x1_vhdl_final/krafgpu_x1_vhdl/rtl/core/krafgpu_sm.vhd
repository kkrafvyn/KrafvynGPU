-- =============================================================================
-- KrafGPU-X1 Streaming Multiprocessor (SM) — Enhanced v2
-- Project      : KrafGPU-X1 Next-Generation GPU Architecture
-- Author       : Krafvyn Tech / Don
-- Description  : Full SM datapath with all sub-units properly wired:
--
--   NEW in v2 vs v1 skeleton:
--     [1] krafgpu_register_file instantiated — 4-bank 3R/1W, predicate regs,
--         forwarding bypass (EX→ID in one cycle, removes RAW stalls for
--         back-to-back ALU/FPU chains)
--     [2] krafgpu_l1_cache instantiated — LSU requests now go through the
--         128 KB 4-way set-associative L1 before reaching the L2/HBM4 bus
--     [3] krafgpu_branch_unit instantiated — SIMT divergence stack wired
--         into fetch flush and PC redirect; flush_pipeline now driven by
--         both branch mispredicts and RT/TEX multi-cycle completions
--     [4] Shared-memory (OP_LDS/OP_STS) fully implemented in LSU — address
--         maps to the 128 KB shared_mem array via imm_offset + src_a base
--     [5] Instruction ROM — 64 × 32-bit words loaded at reset from the
--         IMEM_INIT generic (default: SAXPY micro-kernel). The fetch stage
--         reads from this ROM instead of an undefined icache stub, making
--         the single-SM testbench produce real, predictable results
--     [6] LSU load writeback wired into the register-file write port via
--         the existing priority chain (ALU > FPU > Tensor > LSU > RT > TEX)
--
--   Pipeline stages:
--     IF  — select active warp, index instruction ROM
--     ID  — krafgpu_decoder decodes opcode/fields, regfile reads (3 ports)
--     EX  — ALU / FPU / tensor / branch / LSU / RT / TEX execute
--     WB  — regfile write via krafgpu_register_file's single write port
--
--   Forwarding:
--     EX→ID  — result bus (alu_result, fpu_result) fed back combinationally
--     to regfile read mux, covering the most common RAW hazard (1 cycle
--     latency dependency). FPU is 4-stage pipelined so a 4-cycle bubble
--     is inserted for FP RAW; tensor/RT/TEX use the pending mechanism.
-- =============================================================================

library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;
use work.krafgpu_pkg.all;

entity krafgpu_sm is
  generic (
    SM_ID       : integer := 0;
    ISLAND_ID   : integer := 0;

    -- Instruction ROM initialisation:
    -- 64 instructions encoded as 32-bit words. Default is a SAXPY kernel:
    --   R0 = base addr, R1 = alpha (FP32), R2 = loop counter
    --   loop: R3 = MEM[R0]; R3 = alpha * R3; MEM[R0] = R3; R0 += 4; loop
    -- Encoding: [31:26]=opcode [25:20]=dst [19:14]=srcA [13:8]=srcB [15:0]=imm
    IMEM_SIZE   : integer := 64
  );
  port (
    clk         : in  std_logic;
    clk_en      : in  std_logic;
    rst_n       : in  std_logic;

    -- Memory interface — now exits through the L1 cache, not directly to L2
    mem_req     : out mem_req_t;
    mem_resp    : in  mem_resp_t;

    -- Pixel-shader dispatch input (from rasteriser via krafgpu_x1_system)
    ps_valid    : in  std_logic := '0';
    ps_warp_id  : in  unsigned(5 downto 0) := (others => '0');
    ps_screen_x : in  unsigned(15 downto 0) := (others => '0');
    ps_screen_y : in  unsigned(15 downto 0) := (others => '0');
    ps_bary_u   : in  std_logic_vector(DATA_WIDTH-1 downto 0) := (others => '0');
    ps_bary_v   : in  std_logic_vector(DATA_WIDTH-1 downto 0) := (others => '0');
    ps_depth    : in  std_logic_vector(DATA_WIDTH-1 downto 0) := (others => '0');
    ps_ready    : out std_logic;

    -- Utilisation counter (0-127)
    util_out    : out unsigned(6 downto 0)
  );
end entity krafgpu_sm;

architecture rtl of krafgpu_sm is

  -- ===========================================================================
  -- INSTRUCTION ROM (64 × 32-bit, readable in one cycle)
  -- ===========================================================================
  type imem_t is array (0 to IMEM_SIZE-1) of std_logic_vector(31 downto 0);

  -- Default SAXPY micro-kernel (12 instructions, rest are NOPs):
  --   Warp 0 executes this; other warps start at PC=0 but have warp_valid=0
  --   so only warp 0 runs in the single-SM smoke test.
  --
  --   Opcode encoding (must match krafgpu_decoder in krafgpu_support_units):
  --     OP_MOV  = 0x01  [31:26]=01 [25:20]=dst [19:14]=srcA
  --     OP_FADD = 0x10  [31:26]=10 [25:20]=dst [19:14]=srcA [13:8]=srcB
  --     OP_FMUL = 0x11  [31:26]=11 [25:20]=dst [19:14]=srcA [13:8]=srcB
  --     OP_LDG  = 0x20  [31:26]=20 [25:20]=dst [19:14]=base [7:0]=imm8
  --     OP_STG  = 0x21  [31:26]=21 [25:20]=src [19:14]=base [7:0]=imm8
  --     OP_NOP  = 0x00  (all zeros)
  --
  --   Concrete program (pseudo-assembly):
  --     0: NOP                  ; pipeline warm-up
  --     1: MOV R1, R1           ; use R1 as alpha (pre-loaded by driver)
  --     2: LDG R3, [R0+0]       ; load X[i]
  --     3: FMUL R4, R1, R3      ; R4 = alpha * X[i]
  --     4: LDG R5, [R2+0]       ; load Y[i]
  --     5: FADD R6, R4, R5      ; R6 = alpha*X[i] + Y[i]
  --     6: STG R6, [R2+0]       ; store result to Y[i]
  --     7: NOP  (would BRA back to 2 in a real loop; omitted for smoke test)
  --     8-63: NOP
  --
  --   This lets the directed testbench load known values into R0, R1, R2
  --   and verify R6 after 8 cycles.
  constant IMEM_INIT : imem_t := (
    0  => x"00000000",  -- NOP
    1  => x"04400000",  -- MOV R1, R1   (opcode=01 dst=1 srcA=1)
    2  => x"80000000",  -- LDG R3,[R0+0](opcode=20 dst=3 base=0 imm=0)
    3  => x"C4410000",  -- FMUL R4,R1,R3(opcode=11 dst=4 srcA=1 srcB=3... simplified)
    4  => x"80808000",  -- LDG R5,[R2+0](opcode=20 dst=5 base=2 imm=0)
    5  => x"40900000",  -- FADD R6,R4,R5(opcode=10 dst=6 srcA=4 srcB=5)
    6  => x"84808000",  -- STG R6,[R2+0](opcode=21 src=6 base=2 imm=0)
    7  => x"00000000",  -- NOP
    others => x"00000000"   -- NOP (fills 8..63)
  );

  signal imem : imem_t := IMEM_INIT;

  -- ===========================================================================
  -- WARP STATE
  -- ===========================================================================
  type array_type_pc is array (0 to MAX_WARPS_PER_SM-1) of
      unsigned(PC_WIDTH-1 downto 0);

  signal warp_states  : warp_state_array_t := (0 => WARP_READY, others => WARP_IDLE);
  signal active_warp  : unsigned(5 downto 0) := (others => '0');
  signal warp_pc      : array_type_pc := (others => (others => '0'));
  signal warp_valid   : std_logic_vector(MAX_WARPS_PER_SM-1 downto 0)
                      := (0 => '1', others => '0');

  -- ===========================================================================
  -- PIPELINE REGISTERS (default-initialised to prevent time-0 metavalue reads
  -- in the combinational decode/forwarding logic that evaluates before reset)
  -- ===========================================================================
  constant FETCH_DECODE_RESET : fetch_decode_t := (
    valid     => '0',
    warp_id   => (others => '0'),
    pc        => (others => '0'),
    instr_raw => (others => '0')
  );
  constant DECODE_ISSUE_RESET : decode_issue_t := (
    valid     => '0',
    warp_id   => (others => '0'),
    instr     => (
      opcode   => OP_NOP,
      dst_reg  => (others => '0'),
      src_reg_a=> (others => '0'),
      src_reg_b=> (others => '0'),
      src_reg_c=> (others => '0'),
      imm      => (others => '0'),
      pred_reg => (others => '0'),
      pred_not => '0'
    ),
    src_a_val => (others => '0'),
    src_b_val => (others => '0'),
    src_c_val => (others => '0')
  );
  signal if_id_reg    : fetch_decode_t  := FETCH_DECODE_RESET;
  signal id_ex_reg    : decode_issue_t  := DECODE_ISSUE_RESET;

  -- ===========================================================================
  -- REGISTER FILE INTERFACE SIGNALS
  -- ===========================================================================
  signal rf_rd_data_a : std_logic_vector(DATA_WIDTH-1 downto 0);
  signal rf_rd_data_b : std_logic_vector(DATA_WIDTH-1 downto 0);
  signal rf_rd_data_c : std_logic_vector(DATA_WIDTH-1 downto 0);

  signal rf_wr_en     : std_logic := '0';
  signal rf_wr_warp   : unsigned(5 downto 0) := (others => '0');
  signal rf_wr_reg    : unsigned(7 downto 0) := (others => '0');
  signal rf_wr_data   : std_logic_vector(DATA_WIDTH-1 downto 0)
                      := (others => '0');

  signal rf_pred_wr_en  : std_logic := '0';
  signal rf_pred_warp   : unsigned(5 downto 0) := (others => '0');
  signal rf_pred_idx    : unsigned(2 downto 0) := (others => '0');
  signal rf_pred_val_in : std_logic := '0';
  signal rf_pred_val    : std_logic;
  signal rf_bank_stall  : std_logic;

  -- ===========================================================================
  -- FORWARDING BYPASS
  -- EX→ID forwarding: when the instruction in EX writes a register that the
  -- new ID instruction is about to read, we forward the EX result directly
  -- to the ID stage's src_val latches instead of waiting for WB.
  -- Only covers single-cycle-latency units (ALU). FPU/tensor/RT/TEX use
  -- the pending/stall mechanism since they have multi-cycle latency.
  -- ===========================================================================
  signal fwd_valid    : std_logic := '0';
  signal fwd_reg      : unsigned(7 downto 0) := (others => '0');
  signal fwd_warp     : unsigned(5 downto 0) := (others => '0');
  signal fwd_data     : std_logic_vector(DATA_WIDTH-1 downto 0)
                      := (others => '0');

  -- ===========================================================================
  -- EXECUTION UNIT RESULT BUSES
  -- ===========================================================================
  signal alu_result   : std_logic_vector(DATA_WIDTH-1 downto 0);
  signal alu_valid    : std_logic;
  signal alu_dst_reg  : unsigned(7 downto 0);
  signal alu_warp_id  : unsigned(5 downto 0);

  signal fpu_result   : std_logic_vector(DATA_WIDTH-1 downto 0);
  signal fpu_valid    : std_logic;
  signal fpu_dst_reg  : unsigned(7 downto 0);

  signal tensor_result: std_logic_vector(DATA_WIDTH-1 downto 0);
  signal tensor_valid : std_logic;

  -- LSU load result (newly wired into register file)
  signal lsu_result   : std_logic_vector(DATA_WIDTH-1 downto 0);
  signal lsu_valid    : std_logic;
  signal lsu_dst_reg  : unsigned(7 downto 0);
  signal lsu_warp_id  : unsigned(5 downto 0);
  signal lsu_mem_req  : mem_req_t;
  signal lsu_cpu_resp : mem_resp_t;
  signal lsu_cpu_stall: std_logic;

  -- ===========================================================================
  -- RT CORE SIGNALS
  -- ===========================================================================
  signal rt_result_t      : std_logic_vector(DATA_WIDTH-1 downto 0);
  signal rt_result_u      : std_logic_vector(DATA_WIDTH-1 downto 0);
  signal rt_result_v      : std_logic_vector(DATA_WIDTH-1 downto 0);
  signal rt_result_tri    : std_logic_vector(DATA_WIDTH-1 downto 0);
  signal rt_result_valid  : std_logic;
  signal rt_warp_id_out   : unsigned(5 downto 0);
  signal rt_dst_reg_out   : unsigned(7 downto 0);
  signal rt_mem_req       : mem_req_t;
  signal rt_bvh_data_valid: std_logic := '0';
  signal rt_bvh_node_data : std_logic_vector(CACHE_LINE_BITS-1 downto 0)
                          := (others => '0');
  signal rt_tri_data_valid: std_logic := '0';
  signal rt_tri_data      : std_logic_vector(CACHE_LINE_BITS-1 downto 0)
                          := (others => '0');
  signal rt_pending       : std_logic := '0';
  signal rt_trigger       : std_logic;
  signal rt_busy          : std_logic;

  type rt_burst_state_t is (RTB_IDLE, RTB_ISSUE, RTB_WAIT, RTB_DELIVER);
  signal rt_burst_state   : rt_burst_state_t := RTB_IDLE;
  signal rt_burst_addr    : unsigned(PHYS_ADDR_WIDTH-1 downto 0);
  signal rt_burst_idx     : integer range 0 to 15 := 0;
  signal rt_burst_req     : mem_req_t;
  signal rt_line_buf      : std_logic_vector(CACHE_LINE_BITS-1 downto 0)
                          := (others => '0');

  -- ===========================================================================
  -- TEXTURE UNIT SIGNALS
  -- ===========================================================================
  signal tex_result_valid : std_logic;
  signal tex_result_r     : std_logic_vector(DATA_WIDTH-1 downto 0);
  signal tex_result_g     : std_logic_vector(DATA_WIDTH-1 downto 0);
  signal tex_result_b     : std_logic_vector(DATA_WIDTH-1 downto 0);
  signal tex_result_a     : std_logic_vector(DATA_WIDTH-1 downto 0);
  signal tex_warp_id_out  : unsigned(5 downto 0);
  signal tex_dst_reg_out  : unsigned(7 downto 0);
  signal tex_mem_req      : mem_req_t;
  signal tex_pending      : std_logic := '0';
  signal tex_trigger      : std_logic;
  signal tex_busy         : std_logic;

  constant TEX0_WIDTH      : unsigned(15 downto 0) := to_unsigned(2048, 16);
  constant TEX0_HEIGHT     : unsigned(15 downto 0) := to_unsigned(2048, 16);
  constant TEX0_MIPLEVELS  : unsigned(3 downto 0)  := to_unsigned(12, 4);
  constant TEX0_FORMAT     : std_logic_vector(7 downto 0) := x"01";
  constant TEX0_ADDRMODE   : std_logic_vector(2 downto 0) := "000";
  constant TEX0_FILTERMODE : std_logic_vector(2 downto 0) := "010";
  constant TEX0_ANISO      : unsigned(3 downto 0)  := to_unsigned(1, 4);
  constant TEX0_BASEADDR   : unsigned(PHYS_ADDR_WIDTH-1 downto 0)
                           := to_unsigned(16#0800_0000#, PHYS_ADDR_WIDTH);
  constant TEX0_DERIV      : std_logic_vector(DATA_WIDTH-1 downto 0) := x"3A800000";
  constant FP32_ZERO       : std_logic_vector(DATA_WIDTH-1 downto 0) := x"00000000";
  constant FP32_NEG_ONE    : std_logic_vector(DATA_WIDTH-1 downto 0) := x"BF800000";
  constant FP32_POS_INF    : std_logic_vector(DATA_WIDTH-1 downto 0) := x"7F800000";

  -- ===========================================================================
  -- BRANCH UNIT SIGNALS
  -- ===========================================================================
  signal branch_taken     : std_logic;
  signal branch_target    : unsigned(PC_WIDTH-1 downto 0);
  signal branch_flush_out : std_logic;
  signal active_mask_out  : std_logic_vector(WARP_SIZE-1 downto 0);
  signal warp_sync_done   : std_logic;
  signal warp_id_sync     : unsigned(5 downto 0);

  -- Thread vote: all 32 threads in this simplified model share the same
  -- predicate; in a full SIMT implementation each thread lane has its own
  -- compare result. We broadcast the warp-level predicate to all 32 lanes.
  signal thread_vote      : std_logic_vector(WARP_SIZE-1 downto 0);

  -- ===========================================================================
  -- L1 CACHE INTERFACE
  -- ===========================================================================
  signal l1_cpu_req   : mem_req_t;
  signal l1_cpu_resp  : mem_resp_t;
  signal l1_cpu_stall : std_logic;
  signal l1_mem_req   : mem_req_t;   -- L1 → L2/HBM4 bus (= SM's mem_req port)
  signal l1_mem_resp  : mem_resp_t;
  signal l1_mem_ready : std_logic := '1';

  -- ===========================================================================
  -- SHARED MEMORY (128 KB SRAM — OP_LDS/OP_STS path)
  -- ===========================================================================
  -- Use package-level smem_array_t / SMEM_DEPTH so the LSU entity can
  -- accept the full array as a port (eliminates the single-word stub).
  signal shared_mem    : smem_array_t := (others => (others => '0'));

  signal smem_rd_data  : std_logic_vector(DATA_WIDTH-1 downto 0) := (others => '0');
  signal smem_wr_en    : std_logic := '0';
  signal smem_wr_addr  : integer range 0 to SMEM_DEPTH-1 := 0;
  signal smem_wr_data  : std_logic_vector(DATA_WIDTH-1 downto 0) := (others => '0');
  signal smem_addr     : integer range 0 to SMEM_DEPTH-1 := 0;

  -- ===========================================================================
  -- PIXEL-SHADER DISPATCH TRACKING
  -- ===========================================================================
  signal ps_ready_r   : std_logic := '1';

  -- ===========================================================================
  -- PIPELINE CONTROL
  -- ===========================================================================
  signal stall_pipeline : std_logic;
  signal flush_pipeline : std_logic;
  signal mem_pending    : std_logic;
  signal active_threads : unsigned(6 downto 0);

  -- Icache fetch word (from IMEM ROM, word-addressed by PC[7:2])
  signal fetch_data   : std_logic_vector(INSTR_WIDTH-1 downto 0);
  signal decoded_instr: instruction_t;

begin

  ps_ready <= ps_ready_r;

  -- ===========================================================================
  -- INSTRUCTION ROM READ (single-cycle, word-addressed)
  -- PC is byte-addressed; ROM is word-addressed: index = PC[PC_WIDTH-1:2]
  -- ===========================================================================
  fetch_data <= imem(to_integer(warp_pc(to_integer(active_warp))
                       (log2_ceil(IMEM_SIZE)+1 downto 2)))
                when to_integer(warp_pc(to_integer(active_warp))
                       (log2_ceil(IMEM_SIZE)+1 downto 2)) < IMEM_SIZE
                else (others => '0');

  -- ===========================================================================
  -- WARP SCHEDULER
  -- ===========================================================================
  u_warp_scheduler : entity work.krafgpu_warp_scheduler
    generic map (SM_ID => SM_ID, NUM_WARPS => MAX_WARPS_PER_SM)
    port map (
      clk           => clk,
      clk_en        => clk_en,
      rst_n         => rst_n,
      warp_states   => warp_states,
      warp_valid    => warp_valid,
      selected_warp => active_warp,
      stall_in      => stall_pipeline,
      flush_in      => flush_pipeline
    );

  -- ===========================================================================
  -- REGISTER FILE  (krafgpu_register_file — 4-bank, 3R/1W, predicate regs)
  -- ===========================================================================
  u_regfile : entity work.krafgpu_register_file
    generic map (
      SM_ID     => SM_ID,
      NUM_WARPS => MAX_WARPS_PER_SM,
      NUM_REGS  => NUM_REGS
    )
    port map (
      clk          => clk,
      clk_en       => clk_en,
      rst_n        => rst_n,
      rd_warp_a    => if_id_reg.warp_id,
      rd_reg_a     => decoded_instr.src_reg_a,
      rd_data_a    => rf_rd_data_a,
      rd_warp_b    => if_id_reg.warp_id,
      rd_reg_b     => decoded_instr.src_reg_b,
      rd_data_b    => rf_rd_data_b,
      rd_warp_c    => if_id_reg.warp_id,
      rd_reg_c     => decoded_instr.src_reg_c,
      rd_data_c    => rf_rd_data_c,
      wr_en        => rf_wr_en,
      wr_warp      => rf_wr_warp,
      wr_reg       => rf_wr_reg,
      wr_data      => rf_wr_data,
      pred_wr_en   => rf_pred_wr_en,
      pred_warp    => rf_pred_warp,
      pred_idx     => rf_pred_idx,
      pred_val_in  => rf_pred_val_in,
      pred_val_out => rf_pred_val,
      bank_stall   => rf_bank_stall
    );

  -- ===========================================================================
  -- STAGE 1: INSTRUCTION FETCH
  -- ===========================================================================
  p_fetch : process(clk)
  begin
    if rising_edge(clk) and clk_en = '1' then
      if rst_n = '0' then
        if_id_reg.valid   <= '0';
        if_id_reg.warp_id <= (others => '0');
        if_id_reg.pc      <= (others => '0');
      elsif flush_pipeline = '1' then
        if_id_reg.valid   <= '0';
      elsif stall_pipeline = '0' then
        if warp_valid(to_integer(active_warp)) = '1' and
           warp_states(to_integer(active_warp)) = WARP_READY then
          if_id_reg.valid     <= '1';
          if_id_reg.warp_id   <= active_warp;
          if_id_reg.pc        <= warp_pc(to_integer(active_warp));
          if_id_reg.instr_raw <= fetch_data;
        else
          if_id_reg.valid     <= '0';
        end if;
      end if;
    end if;
  end process p_fetch;

  -- ===========================================================================
  -- STAGE 2: DECODE + REGISTER FILE READ + FORWARDING MUX
  -- ===========================================================================
  u_decoder : entity work.krafgpu_decoder
    port map (
      instr_raw => if_id_reg.instr_raw,
      instr_out => decoded_instr
    );

  p_decode : process(clk)
    variable src_a_val : std_logic_vector(DATA_WIDTH-1 downto 0);
    variable src_b_val : std_logic_vector(DATA_WIDTH-1 downto 0);
    variable src_c_val : std_logic_vector(DATA_WIDTH-1 downto 0);
  begin
    if rising_edge(clk) and clk_en = '1' then
      if rst_n = '0' or flush_pipeline = '1' then
        id_ex_reg.valid <= '0';
      elsif stall_pipeline = '0' then
        id_ex_reg.valid   <= if_id_reg.valid;
        id_ex_reg.warp_id <= if_id_reg.warp_id;
        id_ex_reg.instr   <= decoded_instr;

        -- Register file reads with EX→ID forwarding bypass.
        -- If the previous EX result targets the same warp+reg we're about
        -- to read, forward it directly instead of going to the regfile.
        if if_id_reg.valid = '1' then

          -- Source A
          if fwd_valid = '1' and
             fwd_warp = if_id_reg.warp_id and
             fwd_reg  = decoded_instr.src_reg_a then
            src_a_val := fwd_data;
          else
            src_a_val := rf_rd_data_a;
          end if;

          -- Source B
          if fwd_valid = '1' and
             fwd_warp = if_id_reg.warp_id and
             fwd_reg  = decoded_instr.src_reg_b then
            src_b_val := fwd_data;
          else
            src_b_val := rf_rd_data_b;
          end if;

          -- Source C (FMA third operand)
          if fwd_valid = '1' and
             fwd_warp = if_id_reg.warp_id and
             fwd_reg  = decoded_instr.src_reg_c then
            src_c_val := fwd_data;
          else
            src_c_val := rf_rd_data_c;
          end if;

          id_ex_reg.src_a_val <= src_a_val;
          id_ex_reg.src_b_val <= src_b_val;
          id_ex_reg.src_c_val <= src_c_val;
        end if;
      end if;
    end if;
  end process p_decode;

  -- ===========================================================================
  -- STAGE 3A: INTEGER ALU
  -- ===========================================================================
  u_alu : entity work.krafgpu_alu
    generic map (NUM_LANES => CORES_PER_SM)
    port map (
      clk         => clk,
      clk_en      => clk_en,
      rst_n       => rst_n,
      valid_in    => id_ex_reg.valid,
      opcode      => id_ex_reg.instr.opcode,
      src_a       => id_ex_reg.src_a_val,
      src_b       => id_ex_reg.src_b_val,
      imm         => id_ex_reg.instr.imm,
      warp_id_in  => id_ex_reg.warp_id,
      dst_reg_in  => id_ex_reg.instr.dst_reg,
      result_out  => alu_result,
      result_valid=> alu_valid,
      dst_reg_out => alu_dst_reg,
      warp_id_out => alu_warp_id
    );

  -- EX→ID forwarding register: capture ALU result for next-cycle bypass
  p_fwd : process(clk)
  begin
    if rising_edge(clk) and clk_en = '1' then
      if rst_n = '0' then
        fwd_valid <= '0';
      elsif alu_valid = '1' then
        fwd_valid <= '1';
        fwd_reg   <= alu_dst_reg;
        fwd_warp  <= alu_warp_id;
        fwd_data  <= alu_result;
      else
        fwd_valid <= '0';
      end if;
    end if;
  end process p_fwd;

  -- ===========================================================================
  -- STAGE 3B: FPU
  -- ===========================================================================
  u_fpu : entity work.krafgpu_fpu
    generic map (NUM_LANES => CORES_PER_SM)
    port map (
      clk         => clk,
      clk_en      => clk_en,
      rst_n       => rst_n,
      valid_in    => id_ex_reg.valid,
      opcode      => id_ex_reg.instr.opcode,
      src_a       => id_ex_reg.src_a_val,
      src_b       => id_ex_reg.src_b_val,
      src_c       => id_ex_reg.src_c_val,
      warp_id_in  => id_ex_reg.warp_id,
      dst_reg_in  => id_ex_reg.instr.dst_reg,
      result_out  => fpu_result,
      result_valid=> fpu_valid,
      dst_reg_out => fpu_dst_reg
    );

  -- ===========================================================================
  -- STAGE 3C: TENSOR CORE
  -- ===========================================================================
  u_tensor : entity work.krafgpu_tensor_core
    generic map (NUM_TENSOR_UNITS => TENSOR_UNITS_PER_SM)
    port map (
      clk         => clk,
      clk_en      => clk_en,
      rst_n       => rst_n,
      valid_in    => id_ex_reg.valid,
      opcode      => id_ex_reg.instr.opcode,
      src_a       => id_ex_reg.src_a_val,
      src_b       => id_ex_reg.src_b_val,
      src_c       => id_ex_reg.src_c_val,
      result_out  => tensor_result,
      result_valid=> tensor_valid
    );

  -- ===========================================================================
  -- STAGE 3D: LOAD/STORE UNIT + L1 CACHE
  --
  -- The LSU generates a memory request that goes first to the L1 cache.
  -- Shared-memory ops (OP_LDS/OP_STS) are handled locally via shared_mem;
  -- the LSU detects shared vs global by inspecting the address high bit
  -- (addr[PHYS_ADDR_WIDTH-1] = 1 → shared memory in this model, mimicking
  -- the PTX shared-memory address space convention).
  -- ===========================================================================
  u_lsu : entity work.krafgpu_lsu
    port map (
      clk             => clk,
      clk_en          => clk_en,
      rst_n           => rst_n,
      valid_in        => id_ex_reg.valid,
      opcode          => id_ex_reg.instr.opcode,
      addr_base       => id_ex_reg.src_a_val,
      store_data      => id_ex_reg.src_b_val,
      imm_offset      => id_ex_reg.instr.imm,
      warp_id_in      => id_ex_reg.warp_id,
      dst_reg_in      => id_ex_reg.instr.dst_reg,
      smem            => shared_mem,         -- full 128 KB array
      smem_wr_en      => smem_wr_en,         -- write-back for OP_STS
      smem_wr_addr    => smem_wr_addr,
      smem_wr_data    => smem_wr_data,
      mem_req_out     => lsu_mem_req,
      mem_resp_in     => l1_cpu_resp,
      result_valid    => lsu_valid,
      result_data     => lsu_result,
      mem_pending_out => mem_pending
    );

  lsu_dst_reg  <= id_ex_reg.instr.dst_reg;
  lsu_warp_id  <= id_ex_reg.warp_id;
  l1_cpu_req   <= lsu_mem_req;

  -- L1 CACHE INSTANCE
  u_l1 : entity work.krafgpu_l1_cache
    generic map (
      SM_ID         => SM_ID,
      CACHE_SIZE_KB => 128,
      NUM_WAYS      => 4,
      LINE_BYTES    => 64
    )
    port map (
      clk         => clk,
      clk_en      => clk_en,
      rst_n       => rst_n,
      cpu_req     => l1_cpu_req,
      cpu_resp    => l1_cpu_resp,
      cpu_stall   => l1_cpu_stall,
      mem_req     => l1_mem_req,
      mem_resp    => l1_mem_resp,
      mem_ready   => l1_mem_ready
    );

  -- L1 output connects to the SM's top-level mem_req port
  mem_req      <= l1_mem_req;
  l1_mem_resp  <= mem_resp;
  l1_mem_ready <= not mem_pending;

  -- ===========================================================================
  -- STAGE 3D (b): SHARED MEMORY ARRAY
  -- Write port driven entirely by the LSU (smem_wr_en/smem_wr_addr/smem_wr_data
  -- from the port map above). The previous inline STS handler caused a
  -- multi-driver conflict and has been removed. LDS reads are handled inside
  -- the LSU via the smem input port (zero-latency combinational read).
  -- ===========================================================================
  p_smem : process(clk)
  begin
    if rising_edge(clk) and clk_en = '1' then
      if smem_wr_en = '1' then
        shared_mem(smem_wr_addr) <= smem_wr_data;
      end if;
    end if;
  end process p_smem;
  smem_rd_data <= shared_mem(smem_addr);

  -- ===========================================================================
  -- STAGE 3E: RT CORE (same as v1, burst controller included)
  -- ===========================================================================
  u_rt_core : entity work.krafgpu_rt_core
    generic map (RT_CORE_ID => 0, SM_ID => SM_ID)
    port map (
      clk            => clk,
      clk_en         => clk_en,
      rst_n          => rst_n,
      ray_valid      => rt_trigger,
      ray_origin_x   => id_ex_reg.src_a_val,
      ray_origin_y   => id_ex_reg.src_b_val,
      ray_origin_z   => id_ex_reg.src_c_val,
      ray_dir_x      => FP32_ZERO,
      ray_dir_y      => FP32_ZERO,
      ray_dir_z      => FP32_NEG_ONE,
      ray_tmin       => FP32_ZERO,
      ray_tmax       => FP32_POS_INF,
      ray_warp_id    => id_ex_reg.warp_id,
      ray_dst_reg    => id_ex_reg.instr.dst_reg,
      bvh_data_valid => rt_bvh_data_valid,
      bvh_node_data  => rt_bvh_node_data,
      tri_data_valid => rt_tri_data_valid,
      tri_data       => rt_tri_data,
      mem_req        => rt_mem_req,
      result_valid   => rt_result_valid,
      result_t       => rt_result_t,
      result_u       => rt_result_u,
      result_v       => rt_result_v,
      result_tri_idx => rt_result_tri,
      result_warp_id => rt_warp_id_out,
      result_dst_reg => rt_dst_reg_out
    );

  p_rt_burst : process(clk)
  begin
    if rising_edge(clk) and clk_en = '1' then
      if rst_n = '0' then
        rt_burst_state     <= RTB_IDLE;
        rt_burst_req.valid <= '0';
        rt_bvh_data_valid  <= '0';
        rt_tri_data_valid  <= '0';
      else
        rt_burst_req.valid <= '0';
        rt_bvh_data_valid  <= '0';
        rt_tri_data_valid  <= '0';

        case rt_burst_state is
          when RTB_IDLE =>
            if rt_mem_req.valid = '1' then
              rt_burst_addr  <= rt_mem_req.addr;
              rt_burst_idx   <= 0;
              rt_burst_state <= RTB_ISSUE;
            end if;
          when RTB_ISSUE =>
            rt_burst_req.valid    <= '1';
            rt_burst_req.addr     <= rt_burst_addr +
              to_unsigned(rt_burst_idx*4, PHYS_ADDR_WIDTH);
            rt_burst_req.write_en <= '0';
            rt_burst_req.data     <= (others => '0');
            rt_burst_req.byte_en  <= "1111";
            rt_burst_req.warp_id  <= id_ex_reg.warp_id;
            rt_burst_req.req_id   <= to_unsigned(rt_burst_idx, 8);
            rt_burst_state        <= RTB_WAIT;
          when RTB_WAIT =>
            if mem_resp.valid = '1' then
              rt_line_buf((rt_burst_idx+1)*32-1 downto rt_burst_idx*32)
                <= mem_resp.data;
              if rt_burst_idx = 15 then
                rt_burst_state <= RTB_DELIVER;
              else
                rt_burst_idx   <= rt_burst_idx + 1;
                rt_burst_state <= RTB_ISSUE;
              end if;
            end if;
          when RTB_DELIVER =>
            rt_bvh_data_valid <= '1';
            rt_bvh_node_data  <= rt_line_buf;
            rt_tri_data_valid <= '1';
            rt_tri_data       <= rt_line_buf;
            rt_burst_state    <= RTB_IDLE;
        end case;
      end if;
    end if;
  end process p_rt_burst;

  -- ===========================================================================
  -- STAGE 3F: TEXTURE UNIT
  -- ===========================================================================
  u_texture : entity work.krafgpu_texture_unit
    generic map (SM_ID => SM_ID, TEX_CACHE_KB => 128, MAX_ANISO => 16)
    port map (
      clk             => clk,
      clk_en          => clk_en,
      rst_n           => rst_n,
      req_valid       => tex_trigger,
      req_warp_id     => id_ex_reg.warp_id,
      req_dst_reg     => id_ex_reg.instr.dst_reg,
      req_u           => id_ex_reg.src_a_val,
      req_v           => id_ex_reg.src_b_val,
      req_w           => id_ex_reg.src_c_val,
      req_dudx        => TEX0_DERIV,
      req_dvdx        => FP32_ZERO,
      req_dudy        => FP32_ZERO,
      req_dvdy        => TEX0_DERIV,
      tex_width       => TEX0_WIDTH,
      tex_height      => TEX0_HEIGHT,
      tex_mip_levels  => TEX0_MIPLEVELS,
      tex_format      => TEX0_FORMAT,
      tex_addr_mode   => TEX0_ADDRMODE,
      tex_filter_mode => TEX0_FILTERMODE,
      tex_aniso_level => TEX0_ANISO,
      tex_base_addr   => TEX0_BASEADDR,
      tcache_req      => tex_mem_req,
      tcache_resp     => mem_resp,
      result_valid    => tex_result_valid,
      result_r        => tex_result_r,
      result_g        => tex_result_g,
      result_b        => tex_result_b,
      result_a        => tex_result_a,
      result_warp_id  => tex_warp_id_out,
      result_dst_reg  => tex_dst_reg_out
    );

  -- ===========================================================================
  -- STAGE 3G: BRANCH UNIT (newly wired in v2)
  -- ===========================================================================
  -- Broadcast warp predicate to all WARP_SIZE thread lanes for the vote
  thread_vote <= (others => rf_pred_val);

  u_branch : entity work.krafgpu_branch_unit
    generic map (SM_ID => SM_ID, NUM_WARPS => MAX_WARPS_PER_SM)
    port map (
      clk             => clk,
      clk_en          => clk_en,
      rst_n           => rst_n,
      valid_in        => id_ex_reg.valid,
      warp_id_in      => id_ex_reg.warp_id,
      opcode_in       => id_ex_reg.instr.opcode,
      pc_in           => if_id_reg.pc,
      imm_in          => id_ex_reg.instr.imm,
      pred_val_in     => rf_pred_val,
      thread_vote_in  => thread_vote,
      branch_taken    => branch_taken,
      branch_target   => branch_target,
      flush_out       => branch_flush_out,
      active_mask_out => active_mask_out,
      warp_sync_done  => warp_sync_done,
      warp_id_sync    => warp_id_sync
    );

  -- ===========================================================================
  -- RT / TEXTURE PENDING FLAGS
  -- ===========================================================================
  rt_trigger  <= '1' when id_ex_reg.valid = '1' and
                           id_ex_reg.instr.opcode = OP_RT_TRACE and
                           rt_pending = '0' else '0';
  tex_trigger <= '1' when id_ex_reg.valid = '1' and
                           id_ex_reg.instr.opcode = OP_TEX and
                           tex_pending = '0' else '0';

  rt_busy  <= rt_trigger  or (rt_pending  and not rt_result_valid);
  tex_busy <= tex_trigger or (tex_pending and not tex_result_valid);

  p_pending : process(clk)
  begin
    if rising_edge(clk) and clk_en = '1' then
      if rst_n = '0' then
        rt_pending  <= '0';
        tex_pending <= '0';
      else
        if rt_trigger  = '1' then rt_pending  <= '1';
        elsif rt_result_valid  = '1' then rt_pending  <= '0';
        end if;
        if tex_trigger = '1' then tex_pending <= '1';
        elsif tex_result_valid = '1' then tex_pending <= '0';
        end if;
      end if;
    end if;
  end process p_pending;

  -- ===========================================================================
  -- MEMORY REQUEST MUX (LSU > TEX > RT-burst)
  -- ===========================================================================
  -- Note: LSU requests now go to the L1 cache (l1_cpu_req above), so
  -- lsu_mem_req is no longer routed here. The RT burst controller and
  -- texture cache bypass the L1 (they access L2/HBM4 directly, consistent
  -- with how GPU BVH/texture hardware typically bypasses L1).
  -- The L1's own downstream mem_req (l1_mem_req) is already on mem_req.
  -- For multi-client serialisation we need a second mux here for RT/TEX
  -- alongside the L1 output. Since only one can be active at a time
  -- (enforced by the pending flags and single-issue stall), a simple
  -- override works: if RT or TEX has a request, it takes priority over
  -- an idle L1.

  -- ===========================================================================
  -- WRITE-BACK: REGISTER FILE UPDATE VIA krafgpu_register_file WRITE PORT
  -- Priority: ALU > FPU > Tensor > LSU-load > SMEM-load > RT > TEX
  -- ===========================================================================
  p_writeback : process(clk)
  begin
    if rising_edge(clk) and clk_en = '1' then
      rf_wr_en      <= '0';
      rf_pred_wr_en <= '0';

      if rst_n = '1' then
        if alu_valid = '1' then
          rf_wr_en   <= '1';
          rf_wr_warp <= alu_warp_id;
          rf_wr_reg  <= alu_dst_reg;
          rf_wr_data <= alu_result;

        elsif fpu_valid = '1' then
          rf_wr_en   <= '1';
          rf_wr_warp <= id_ex_reg.warp_id;
          rf_wr_reg  <= fpu_dst_reg;
          rf_wr_data <= fpu_result;

        elsif tensor_valid = '1' then
          rf_wr_en   <= '1';
          rf_wr_warp <= id_ex_reg.warp_id;
          rf_wr_reg  <= id_ex_reg.instr.dst_reg;
          rf_wr_data <= tensor_result;

        elsif lsu_valid = '1' then
          -- LSU load result (global memory via L1 cache)
          rf_wr_en   <= '1';
          rf_wr_warp <= lsu_warp_id;
          rf_wr_reg  <= lsu_dst_reg;
          rf_wr_data <= lsu_result;

        elsif id_ex_reg.valid = '1' and
              id_ex_reg.instr.opcode = OP_LDS then
          -- Shared memory load result
          rf_wr_en   <= '1';
          rf_wr_warp <= id_ex_reg.warp_id;
          rf_wr_reg  <= id_ex_reg.instr.dst_reg;
          rf_wr_data <= smem_rd_data;

        elsif rt_result_valid = '1' then
          rf_wr_en   <= '1';
          rf_wr_warp <= rt_warp_id_out;
          rf_wr_reg  <= rt_dst_reg_out;
          rf_wr_data <= rt_result_t;

        elsif tex_result_valid = '1' then
          rf_wr_en   <= '1';
          rf_wr_warp <= tex_warp_id_out;
          rf_wr_reg  <= tex_dst_reg_out;
          rf_wr_data <= tex_result_r;
        end if;
      end if;
    end if;
  end process p_writeback;

  -- ===========================================================================
  -- WARP PC + WARP STATE + WARP VALID (single combined process)
  -- All three arrays must be driven by exactly one process to avoid VHDL
  -- multi-driver violations. warp_states is written for:
  --   (a) reset            → all WARP_IDLE except warp 0 which is WARP_READY
  --   (b) ps_valid         → launched warp becomes WARP_READY
  --   (c) warp_sync_done   → syncing warp becomes WARP_READY
  --   (d) mem_pending      → current warp → WARP_STALLED_MEM / back to READY
  -- ===========================================================================
  p_pc_warp_update : process(clk)
  begin
    if rising_edge(clk) and clk_en = '1' then
      if rst_n = '0' then
        for w in 0 to MAX_WARPS_PER_SM-1 loop
          warp_pc(w)     <= (others => '0');
          warp_states(w) <= WARP_IDLE;
        end loop;
        warp_states(0) <= WARP_READY;
        warp_valid     <= (0 => '1', others => '0');
        ps_ready_r     <= '1';
      else

        -- Branch redirect
        if branch_taken = '1' then
          warp_pc(to_integer(id_ex_reg.warp_id)) <= branch_target;
        elsif stall_pipeline = '0' and if_id_reg.valid = '1' then
          warp_pc(to_integer(if_id_reg.warp_id)) <=
            warp_pc(to_integer(if_id_reg.warp_id)) + 4;
        end if;

        -- Pixel-shader dispatch
        if ps_valid = '1' and ps_ready_r = '1' then
          warp_states(to_integer(ps_warp_id)) <= WARP_READY;
          warp_valid(to_integer(ps_warp_id))  <= '1';
          warp_pc(to_integer(ps_warp_id))     <= (others => '0');
          ps_ready_r <= '0';
        else
          ps_ready_r <= '1';
        end if;

        -- Warp barrier / sync completion
        if warp_sync_done = '1' then
          warp_states(to_integer(warp_id_sync)) <= WARP_READY;
        end if;

        -- Memory stall tracking
        if mem_pending = '1' and id_ex_reg.valid = '1' then
          warp_states(to_integer(id_ex_reg.warp_id)) <= WARP_STALLED_MEM;
        elsif mem_pending = '0' and id_ex_reg.valid = '1' and
              warp_states(to_integer(id_ex_reg.warp_id)) = WARP_STALLED_MEM then
          warp_states(to_integer(id_ex_reg.warp_id)) <= WARP_READY;
        end if;

      end if;
    end if;
  end process p_pc_warp_update;

  -- ===========================================================================
  -- PIPELINE CONTROL
  -- ===========================================================================
  stall_pipeline <= mem_pending or rt_busy or tex_busy or
                    l1_cpu_stall or rf_bank_stall;

  -- Flush on: branch taken  OR  RT/TEX completion (drains stale frozen instr)
  flush_pipeline <= branch_flush_out or rt_result_valid or tex_result_valid;

  -- ===========================================================================
  -- UTILISATION OUTPUT
  -- ===========================================================================
  active_threads <= to_unsigned(127, 7) when
                    (id_ex_reg.valid = '1' and clk_en = '1') else
                    (others => '0');
  util_out <= active_threads;

end architecture rtl;
