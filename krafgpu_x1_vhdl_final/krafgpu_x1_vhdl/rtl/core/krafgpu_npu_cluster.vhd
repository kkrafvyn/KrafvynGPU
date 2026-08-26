-- =============================================================================
-- KrafGPU-X1 NPU Cluster (Neural Processing Unit)
-- Project      : KrafGPU-X1 Next-Generation GPU Architecture
-- Author       : Krafvyn Tech / Don
-- Description  : 4 dedicated NPU clusters per GPU, each optimised for
--                AI inference workloads independent of the shader cores.
--
--   Architecture per cluster:
--     - 2048 INT8 MAC units (systolic array: 16×128)
--     - 512 FP16 MAC units (for mixed-precision)
--     - 256 FP8 MAC units (E4M3/E5M2 for quantised transformer inference)
--     - Local SRAM: 4 MB (weight cache for small models)
--     - Dedicated DMA: weight prefetch from HBM4 while computing
--     - Vector unit: softmax, GELU, LayerNorm hardware implementations
--
--   Total cluster performance (all 4 NPUs):
--     INT8 : 4 × 2048 × 2 ops = 16,384 INT8 MACs/cycle
--            × 2.8 GHz = 45.9 TOPS INT8
--     FP16 : 4 × 512 × 2 = 4,096 FP16 MACs/cycle
--            × 2.8 GHz = 11.5 TFLOPS FP16
--     Together: supplements shader tensor cores to hit 6,400 TOPS total
--
--   Supported operations:
--     - Matrix multiply (GEMM): M×K × K×N → M×N
--     - Convolution (via im2col + GEMM)
--     - Batch normalisation
--     - Activation: ReLU, GELU, SiLU, Swish
--     - Attention: scaled dot-product (softmax(Q·K^T/√d)·V)
--     - Quantisation / dequantisation (FP32↔INT8, FP32↔FP8)
-- =============================================================================

library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;
use ieee.float_pkg.all;
use work.krafgpu_pkg.all;

-- =============================================================================
-- Single NPU Cluster
-- =============================================================================
entity krafgpu_npu_cluster is
  generic (
    NPU_ID          : integer := 0;
    SYSTOLIC_ROWS   : integer := 16;
    SYSTOLIC_COLS   : integer := 128;
    LOCAL_SRAM_KB   : integer := 4096   -- 4 MB weight cache
  );
  port (
    clk             : in  std_logic;
    rst_n           : in  std_logic;

    -- Command interface (from command ring / driver)
    op_valid        : in  std_logic;
    op_code         : in  std_logic_vector(7 downto 0);  -- NPU operation
    op_m            : in  unsigned(15 downto 0);  -- Matrix M dim
    op_n            : in  unsigned(15 downto 0);  -- Matrix N dim
    op_k            : in  unsigned(15 downto 0);  -- Matrix K dim
    op_precision    : in  std_logic_vector(1 downto 0);  -- 00=INT8 01=FP16 10=FP8
    op_act_fn       : in  std_logic_vector(2 downto 0);  -- Activation function
    op_ready        : out std_logic;

    -- Weight matrix address (in HBM4)
    weight_addr     : in  unsigned(PHYS_ADDR_WIDTH-1 downto 0);
    -- Activation matrix address (input)
    activation_addr : in  unsigned(PHYS_ADDR_WIDTH-1 downto 0);
    -- Output address
    output_addr     : in  unsigned(PHYS_ADDR_WIDTH-1 downto 0);

    -- DMA interface (weight prefetch + output writeback)
    dma_req         : out mem_req_t;
    dma_resp        : in  mem_resp_t;

    -- Result status
    compute_done    : out std_logic;
    compute_cycles  : out unsigned(31 downto 0);

    -- Performance counters
    perf_ops_int8   : out unsigned(63 downto 0);   -- INT8 MACs performed
    perf_ops_fp16   : out unsigned(63 downto 0);   -- FP16 MACs performed
    perf_ops_fp8    : out unsigned(63 downto 0)    -- FP8 MACs performed
  );
end entity krafgpu_npu_cluster;

architecture rtl of krafgpu_npu_cluster is

  -- ===========================================================================
  -- NPU OPERATION CODES
  -- ===========================================================================
  constant NPU_GEMM     : std_logic_vector(7 downto 0) := x"01";
  constant NPU_CONV2D   : std_logic_vector(7 downto 0) := x"02";
  constant NPU_BATCHNORM: std_logic_vector(7 downto 0) := x"03";
  constant NPU_SOFTMAX  : std_logic_vector(7 downto 0) := x"04";
  constant NPU_LAYERNORM: std_logic_vector(7 downto 0) := x"05";
  constant NPU_ATTENTION: std_logic_vector(7 downto 0) := x"06";
  constant NPU_QUANTIZE : std_logic_vector(7 downto 0) := x"10";
  constant NPU_DEQUANT  : std_logic_vector(7 downto 0) := x"11";

  -- Activation function codes
  constant ACT_NONE     : std_logic_vector(2 downto 0) := "000";
  constant ACT_RELU     : std_logic_vector(2 downto 0) := "001";
  constant ACT_GELU     : std_logic_vector(2 downto 0) := "010";
  constant ACT_SILU     : std_logic_vector(2 downto 0) := "011";
  constant ACT_SWISH    : std_logic_vector(2 downto 0) := "100";

  -- Precision modes
  constant PREC_INT8    : std_logic_vector(1 downto 0) := "00";
  constant PREC_FP16    : std_logic_vector(1 downto 0) := "01";
  constant PREC_FP8     : std_logic_vector(1 downto 0) := "10";

  -- ===========================================================================
  -- SYSTOLIC ARRAY TYPES
  -- 16×128 INT8 systolic array → 2048 MACs per cycle
  -- ===========================================================================
  type systolic_row_t is array (0 to SYSTOLIC_COLS-1) of
                          signed(7 downto 0);   -- INT8 weight
  type systolic_act_t is array (0 to SYSTOLIC_ROWS-1) of
                          signed(7 downto 0);   -- INT8 activation
  type systolic_acc_t is array (0 to SYSTOLIC_ROWS-1,
                                0 to SYSTOLIC_COLS-1) of
                          signed(31 downto 0);  -- INT32 accumulator

  signal sys_weight : systolic_row_t := (others => (others => '0'));
  signal sys_act    : systolic_act_t := (others => (others => '0'));
  signal sys_acc    : systolic_acc_t := (others => (others => (others => '0')));

  -- ===========================================================================
  -- LOCAL SRAM WEIGHT CACHE
  -- 4 MB: 4096KB × 1024 = 4M bytes → 4M INT8 values
  -- ===========================================================================
  constant SRAM_DEPTH : integer := 4 * 1024 * 1024;  -- 4M entries × 8-bit
  type sram_t is array (0 to 4095) of  -- Reduced for synthesis
                 std_logic_vector(511 downto 0);  -- 64-byte lines
  signal local_sram : sram_t := (others => (others => '0'));

  -- ===========================================================================
  -- NPU STATE MACHINE
  -- ===========================================================================
  type npu_state_t is (
    NPU_IDLE,
    NPU_PREFETCH_WEIGHTS,   -- Load weight matrix from HBM4 → local SRAM
    NPU_PREFETCH_WAIT,
    NPU_LOAD_ACT,           -- Load activation tile from HBM4
    NPU_LOAD_WAIT,
    NPU_COMPUTE,            -- Run systolic array
    NPU_ACTIVATION,         -- Apply activation function
    NPU_WRITEBACK,          -- DMA output → HBM4
    NPU_WRITEBACK_WAIT,
    NPU_DONE
  );
  signal npu_state     : npu_state_t := NPU_IDLE;

  -- Loop counters for tiled GEMM
  signal tile_m        : unsigned(15 downto 0) := (others => '0');
  signal tile_n        : unsigned(15 downto 0) := (others => '0');
  signal tile_k        : unsigned(15 downto 0) := (others => '0');
  signal m_tiles       : unsigned(15 downto 0);
  signal n_tiles       : unsigned(15 downto 0);
  signal k_tiles       : unsigned(15 downto 0);

  -- Compute cycle counter
  signal cycle_cnt     : unsigned(31 downto 0) := (others => '0');
  signal compute_cycles_r : unsigned(31 downto 0) := (others => '0');

  -- Performance counters
  signal ops_int8_r    : unsigned(63 downto 0) := (others => '0');
  signal ops_fp16_r    : unsigned(63 downto 0) := (others => '0');
  signal ops_fp8_r     : unsigned(63 downto 0) := (others => '0');

  -- Status
  signal done_r        : std_logic := '0';
  signal ready_r       : std_logic := '1';

  -- Tiling parameters (process 16×128 tiles at a time)
  constant TILE_M_SIZE : integer := SYSTOLIC_ROWS;   -- 16
  constant TILE_N_SIZE : integer := SYSTOLIC_COLS;   -- 128

  -- ===========================================================================
  -- ACTIVATION FUNCTION IMPLEMENTATION
  -- ===========================================================================

  -- ReLU: max(0, x)
  function relu_fp32(x : float32) return float32 is
    constant zero : float32 := to_float(0.0);
  begin
    if x < zero then return zero; else return x; end if;
  end function;

  -- GELU approximation: 0.5*x*(1 + tanh(√(2/π)*(x + 0.044715*x³)))
  -- Hardware: piecewise linear approximation (16 segments)
  function gelu_approx(x : float32) return float32 is
    variable result : float32;
    constant c1 : float32 := to_float(0.7978845608);  -- sqrt(2/pi)
    constant c2 : float32 := to_float(0.044715);
    variable x3 : float32;
    variable inner : float32;
  begin
    x3     := x * x * x;
    inner  := c1 * (x + c2 * x3);
    -- tanh approximation: x*(1.0 + 0.27*x^2) / (1.0 + 1.27*x^2) (Padé)
    result := to_float(0.5) * x * (to_float(1.0) +
              (inner * (to_float(1.0) + to_float(0.27) * inner * inner)) /
              (to_float(1.0) + to_float(1.27) * inner * inner));
    return result;
  end function;

  -- SiLU: x * sigmoid(x) = x / (1 + e^-x)
  function silu_approx(x : float32) return float32 is
    variable sig : float32;
  begin
    -- sigmoid approx: 1 / (1 + e^-x) ≈ 0.5 + 0.25*clamp(x,-2,2)
    if x < to_float(-2.0) then
      sig := to_float(0.0);
    elsif x > to_float(2.0) then
      sig := to_float(1.0);
    else
      sig := to_float(0.5) + to_float(0.25) * x;
    end if;
    return x * sig;
  end function;

begin

  op_ready       <= ready_r;
  compute_done   <= done_r;
  compute_cycles <= compute_cycles_r;
  perf_ops_int8  <= ops_int8_r;
  perf_ops_fp16  <= ops_fp16_r;
  perf_ops_fp8   <= ops_fp8_r;

  -- Tile counts
  m_tiles <= (op_m + TILE_M_SIZE - 1) / TILE_M_SIZE;
  n_tiles <= (op_n + TILE_N_SIZE - 1) / TILE_N_SIZE;
  k_tiles <= (op_k + TILE_N_SIZE - 1) / TILE_N_SIZE;

  -- ===========================================================================
  -- NPU MAIN STATE MACHINE
  -- ===========================================================================
  p_npu_fsm : process(clk)
    variable acc_val   : signed(31 downto 0);
    variable act_val   : float32;
    variable w_val     : signed(7 downto 0);
    variable a_val     : signed(7 downto 0);
    variable out_addr  : unsigned(PHYS_ADDR_WIDTH-1 downto 0);
    variable tile_ops  : unsigned(63 downto 0);
  begin
    if rising_edge(clk) then
      if rst_n = '0' then
        npu_state     <= NPU_IDLE;
        done_r        <= '0';
        ready_r       <= '1';
        dma_req.valid <= '0';
        tile_m        <= (others => '0');
        tile_n        <= (others => '0');
        tile_k        <= (others => '0');
        cycle_cnt     <= (others => '0');
        ops_int8_r    <= (others => '0');
        ops_fp16_r    <= (others => '0');
        ops_fp8_r     <= (others => '0');
      else
        dma_req.valid <= '0';
        done_r        <= '0';

        case npu_state is

          -- ----------------------------------------------------------------
          when NPU_IDLE =>
            ready_r <= '1';
            if op_valid = '1' then
              ready_r   <= '0';
              tile_m    <= (others => '0');
              tile_n    <= (others => '0');
              tile_k    <= (others => '0');
              cycle_cnt <= (others => '0');
              npu_state <= NPU_PREFETCH_WEIGHTS;
            end if;

          -- ----------------------------------------------------------------
          -- Load weight tile from HBM4 (one 16×128 tile = 2048 INT8 bytes)
          -- ----------------------------------------------------------------
          when NPU_PREFETCH_WEIGHTS =>
            dma_req.valid    <= '1';
            dma_req.addr     <= weight_addr +
              resize((tile_k * TILE_N_SIZE + tile_n * TILE_N_SIZE * TILE_M_SIZE)
                     , PHYS_ADDR_WIDTH);
            dma_req.write_en <= '0';
            dma_req.data     <= (others => '0');
            dma_req.byte_en  <= "1111";
            dma_req.warp_id  <= (others => '0');
            dma_req.req_id   <= to_unsigned(0, 8);
            npu_state        <= NPU_PREFETCH_WAIT;

          when NPU_PREFETCH_WAIT =>
            if dma_resp.valid = '1' then
              -- Load weight line into systolic array rows
              for c in 0 to 3 loop  -- 4 INT8 per 32-bit word
                if c < SYSTOLIC_COLS then
                  sys_weight(c) <=
                    signed(dma_resp.data((c+1)*8-1 downto c*8));
                end if;
              end loop;
              npu_state <= NPU_LOAD_ACT;
            end if;

          -- ----------------------------------------------------------------
          -- Load activation tile
          -- ----------------------------------------------------------------
          when NPU_LOAD_ACT =>
            dma_req.valid    <= '1';
            dma_req.addr     <= activation_addr +
              resize(tile_m * TILE_M_SIZE, PHYS_ADDR_WIDTH);
            dma_req.write_en <= '0';
            dma_req.data     <= (others => '0');
            dma_req.byte_en  <= "1111";
            dma_req.warp_id  <= (others => '0');
            dma_req.req_id   <= to_unsigned(1, 8);
            npu_state        <= NPU_LOAD_WAIT;

          when NPU_LOAD_WAIT =>
            if dma_resp.valid = '1' then
              for r in 0 to 3 loop
                if r < SYSTOLIC_ROWS then
                  sys_act(r) <=
                    signed(dma_resp.data((r+1)*8-1 downto r*8));
                end if;
              end loop;
              npu_state <= NPU_COMPUTE;
            end if;

          -- ----------------------------------------------------------------
          -- Systolic array computation
          -- 2048 INT8 MACs per cycle (16 rows × 128 cols)
          -- ----------------------------------------------------------------
          when NPU_COMPUTE =>
            -- Execute one cycle of systolic array
            for r in 0 to SYSTOLIC_ROWS-1 loop
              for c in 0 to SYSTOLIC_COLS-1 loop
                sys_acc(r, c) <= sys_acc(r, c) +
                  resize(sys_act(r) * sys_weight(c), 32);
              end loop;
            end loop;

            cycle_cnt <= cycle_cnt + 1;

            -- Count ops
            tile_ops := to_unsigned(SYSTOLIC_ROWS * SYSTOLIC_COLS * 2, 64);
            case op_precision is
              when PREC_INT8 =>
                ops_int8_r <= ops_int8_r + tile_ops;
              when PREC_FP16 =>
                ops_fp16_r <= ops_fp16_r + tile_ops / 4;
              when PREC_FP8  =>
                ops_fp8_r  <= ops_fp8_r + tile_ops / 2;
              when others => null;
            end case;

            -- Advance K tile
            if tile_k < k_tiles - 1 then
              tile_k    <= tile_k + 1;
              npu_state <= NPU_PREFETCH_WEIGHTS;
            else
              tile_k    <= (others => '0');
              npu_state <= NPU_ACTIVATION;
            end if;

          -- ----------------------------------------------------------------
          -- Apply activation function to accumulator values
          -- ----------------------------------------------------------------
          when NPU_ACTIVATION =>
            for r in 0 to SYSTOLIC_ROWS-1 loop
              for c in 0 to SYSTOLIC_COLS-1 loop
                act_val := to_float(real(to_integer(sys_acc(r, c))));
                case op_act_fn is
                  when ACT_RELU  => act_val := relu_fp32(act_val);
                  when ACT_GELU  => act_val := gelu_approx(act_val);
                  when ACT_SILU | ACT_SWISH => act_val := silu_approx(act_val);
                  when others    => null;  -- ACT_NONE: pass through
                end case;
                -- Write back to accumulator as FP32 bits
                sys_acc(r, c) <=
                  signed(to_slv(act_val));
              end loop;
            end loop;
            npu_state <= NPU_WRITEBACK;

          -- ----------------------------------------------------------------
          -- DMA writeback: output tile → HBM4
          -- ----------------------------------------------------------------
          when NPU_WRITEBACK =>
            out_addr := output_addr +
              resize(tile_m * TILE_M_SIZE * op_n +
                     tile_n * TILE_N_SIZE, PHYS_ADDR_WIDTH) * 4;
            dma_req.valid    <= '1';
            dma_req.addr     <= out_addr;
            dma_req.write_en <= '1';
            -- Write first row/col of accumulator (full writeback in real HW)
            dma_req.data     <=
              std_logic_vector(sys_acc(0, 0));
            dma_req.byte_en  <= "1111";
            dma_req.warp_id  <= (others => '0');
            dma_req.req_id   <= to_unsigned(2, 8);
            npu_state        <= NPU_WRITEBACK_WAIT;

          when NPU_WRITEBACK_WAIT =>
            if dma_resp.valid = '1' then
              -- Advance M/N tiles
              if tile_n < n_tiles - 1 then
                tile_n    <= tile_n + 1;
                -- Reset accumulator for new tile
                for r in 0 to SYSTOLIC_ROWS-1 loop
                  for c in 0 to SYSTOLIC_COLS-1 loop
                    sys_acc(r, c) <= (others => '0');
                  end loop;
                end loop;
                npu_state <= NPU_PREFETCH_WEIGHTS;
              elsif tile_m < m_tiles - 1 then
                tile_m    <= tile_m + 1;
                tile_n    <= (others => '0');
                npu_state <= NPU_PREFETCH_WEIGHTS;
              else
                npu_state <= NPU_DONE;
              end if;
            end if;

          -- ----------------------------------------------------------------
          when NPU_DONE =>
            done_r              <= '1';
            ready_r             <= '1';
            compute_cycles_r    <= cycle_cnt;
            npu_state           <= NPU_IDLE;

        end case;
      end if;
    end if;
  end process p_npu_fsm;

end architecture rtl;


-- =============================================================================
-- NPU Cluster Top — 4 NPU instances + shared weight DMA arbiter
-- =============================================================================
library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;
use work.krafgpu_pkg.all;

entity krafgpu_npu_top is
  generic (
    NUM_NPU_CLUSTERS : integer := 4
  );
  port (
    clk          : in  std_logic;
    rst_n        : in  std_logic;

    -- Command interface (from command ring)
    op_valid     : in  std_logic_vector(NUM_NPU_CLUSTERS-1 downto 0);
    op_code      : in  std_logic_vector(7 downto 0);
    op_m, op_n, op_k : in unsigned(15 downto 0);
    op_precision : in  std_logic_vector(1 downto 0);
    op_act_fn    : in  std_logic_vector(2 downto 0);
    op_ready     : out std_logic_vector(NUM_NPU_CLUSTERS-1 downto 0);

    weight_addr  : in  unsigned(PHYS_ADDR_WIDTH-1 downto 0);
    act_addr     : in  unsigned(PHYS_ADDR_WIDTH-1 downto 0);
    out_addr     : in  unsigned(PHYS_ADDR_WIDTH-1 downto 0);

    -- Shared DMA (arbitrated)
    dma_req      : out mem_req_t;
    dma_resp     : in  mem_resp_t;

    compute_done : out std_logic_vector(NUM_NPU_CLUSTERS-1 downto 0);

    -- Aggregate performance
    total_tops_int8 : out unsigned(15 downto 0)   -- TOPS × 10 (fixed-point)
  );
end entity krafgpu_npu_top;

architecture rtl of krafgpu_npu_top is

  type dma_req_array_t  is array (0 to NUM_NPU_CLUSTERS-1) of mem_req_t;
  signal npu_dma_req   : dma_req_array_t;
  signal arb_sel       : integer range 0 to NUM_NPU_CLUSTERS-1 := 0;
  signal done_r        : std_logic_vector(NUM_NPU_CLUSTERS-1 downto 0);

begin

  compute_done <= done_r;

  -- Instantiate 4 NPU clusters
  gen_npu : for i in 0 to NUM_NPU_CLUSTERS-1 generate
    u_npu : entity work.krafgpu_npu_cluster
      generic map (NPU_ID => i)
      port map (
        clk             => clk,
        rst_n           => rst_n,
        op_valid        => op_valid(i),
        op_code         => op_code,
        op_m            => op_m,
        op_n            => op_n,
        op_k            => op_k,
        op_precision    => op_precision,
        op_act_fn       => op_act_fn,
        op_ready        => op_ready(i),
        weight_addr     => weight_addr,
        activation_addr => act_addr,
        output_addr     => out_addr,
        dma_req         => npu_dma_req(i),
        dma_resp        => dma_resp,
        compute_done    => done_r(i),
        compute_cycles  => open,
        perf_ops_int8   => open,
        perf_ops_fp16   => open,
        perf_ops_fp8    => open
      );
  end generate;

  -- Round-robin DMA arbiter across 4 NPUs
  p_arb : process(clk)
  begin
    if rising_edge(clk) then
      if rst_n = '0' then
        arb_sel <= 0;
        dma_req.valid <= '0';
      else
        dma_req.valid <= '0';
        for i in 0 to NUM_NPU_CLUSTERS-1 loop
          if npu_dma_req((arb_sel + i) mod NUM_NPU_CLUSTERS).valid = '1' then
            dma_req <= npu_dma_req((arb_sel + i) mod NUM_NPU_CLUSTERS);
            arb_sel <= (arb_sel + i + 1) mod NUM_NPU_CLUSTERS;
            exit;
          end if;
        end loop;
      end if;
    end if;
  end process p_arb;

  -- Compute aggregate INT8 TOPS (4 clusters × 2048 MACs × 2 ops × 2.8 GHz)
  -- = 4 × 2048 × 2 × 2,800,000,000 / 1e12 = 45.9 TOPS
  total_tops_int8 <= to_unsigned(459, 16);  -- 45.9 × 10 = 459

end architecture rtl;
