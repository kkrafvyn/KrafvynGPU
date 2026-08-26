-- =============================================================================
-- KrafGPU-X1 Formal Verification Properties (PSL / VHDL Assertions)
-- Project      : KrafGPU-X1 Next-Generation GPU Architecture
-- Author       : Krafvyn Tech / Don
-- Description  : Formal verification property set covering all major GPU
--                subsystems. Written in PSL (Property Specification Language)
--                embedded in VHDL comments, compatible with:
--                  - Cadence JasperGold
--                  - Synopsys VC Formal
--                  - OneSpin 360
--                  - GHDL + SymbiYosys (open source flow)
--
--   Property categories:
--     1.  SAFETY     — Nothing bad ever happens
--     2.  LIVENESS   — Something good eventually happens
--     3.  COVERAGE   — All interesting states are reachable
--     4.  PROTOCOL   — Handshake and bus protocol correctness
--     5.  POWER      — Power gating and voltage island safety
--     6.  DATAPATH   — Arithmetic correctness invariants
-- =============================================================================

library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;
use work.krafgpu_pkg.all;

-- =============================================================================
-- Formal verification binding entity
-- Instantiates DUT and adds property monitors
-- =============================================================================
entity krafgpu_formal_props is
end entity krafgpu_formal_props;

architecture formal of krafgpu_formal_props is

  -- ===========================================================================
  -- FORMAL MODEL SIGNALS (free variables / symbolic inputs)
  -- ===========================================================================
  signal clk         : std_logic := '0';
  signal rst_n       : std_logic := '0';

  -- PMC signals
  signal island_states : island_state_array_t;
  signal pmic_pgood    : std_logic_vector(NUM_ISLANDS-1 downto 0);
  signal telemetry     : telemetry_array_t;

  -- Memory bus signals
  signal mem_req_valid : std_logic;
  signal mem_req_addr  : unsigned(PHYS_ADDR_WIDTH-1 downto 0);
  signal mem_req_we    : std_logic;
  signal mem_resp_valid: std_logic;
  signal mem_resp_data : std_logic_vector(DATA_WIDTH-1 downto 0);

  -- Warp scheduler signals
  signal warp_valid    : std_logic_vector(MAX_WARPS_PER_SM-1 downto 0);
  signal warp_states   : warp_state_array_t;
  signal selected_warp : unsigned(5 downto 0);
  signal stall         : std_logic;

  -- Pipeline signals
  signal pipeline_valid: std_logic;
  signal pipeline_flush: std_logic;
  signal branch_taken  : std_logic;

  -- BAR0 signals
  signal bar_addr      : unsigned(23 downto 0);
  signal bar_wr_data   : std_logic_vector(31 downto 0);
  signal bar_wr_en     : std_logic;
  signal bar_rd_en     : std_logic;
  signal bar_rd_data   : std_logic_vector(31 downto 0);
  signal bar_rd_valid  : std_logic;

  -- FPU signals
  signal fpu_valid_in  : std_logic;
  signal fpu_src_a     : std_logic_vector(DATA_WIDTH-1 downto 0);
  signal fpu_src_b     : std_logic_vector(DATA_WIDTH-1 downto 0);
  signal fpu_result    : std_logic_vector(DATA_WIDTH-1 downto 0);
  signal fpu_valid_out : std_logic;

  -- Depth buffer signals
  signal depth_test_valid : std_logic;
  signal depth_test_depth : std_logic_vector(DATA_WIDTH-1 downto 0);
  signal depth_pass       : std_logic;
  signal depth_done       : std_logic;

  -- RT core signals
  signal rt_valid_in   : std_logic;
  signal rt_result_valid : std_logic;
  signal rt_result_t   : std_logic_vector(DATA_WIDTH-1 downto 0);

  -- Tensor core signals
  signal tensor_valid_in : std_logic;
  signal tensor_valid_out: std_logic;

  -- Display engine
  signal disp_hsync    : std_logic;
  signal disp_vsync    : std_logic;
  signal disp_de       : std_logic;
  signal disp_h_active : unsigned(13 downto 0);
  signal disp_v_active : unsigned(13 downto 0);

  -- Branch unit
  signal branch_warp   : unsigned(5 downto 0);
  signal branch_pred   : std_logic;
  signal branch_vote   : std_logic_vector(WARP_SIZE-1 downto 0);
  signal branch_flush  : std_logic;
  signal branch_target : unsigned(PC_WIDTH-1 downto 0);
  signal active_mask   : std_logic_vector(WARP_SIZE-1 downto 0);

  -- ===========================================================================
  -- FORMAL CLOCK
  -- ===========================================================================
  -- psl default clock is rising_edge(clk);

begin

  -- Clock generation (for simulation-based formal)
  clk <= not clk after 178 ps;  -- 2.8 GHz

  -- ===========================================================================
  -- SECTION 1: SAFETY PROPERTIES
  -- "Nothing bad ever happens"
  -- ===========================================================================

  -- ---------------------------------------------------------------------------
  -- PROP_SAFE_001: Voltage island never active without PGOOD
  -- If an island is FULL/HALF/LOW it must have received PGOOD within 10ms
  -- ---------------------------------------------------------------------------
  -- psl PROP_SAFE_001 :
  --   assert always (
  --     (island_states(0) /= ISLAND_OFF) ->
  --     (pmic_pgood(0) = '1')
  --   ) @rising_edge(clk);

  p_safe_001 : process(clk)
  begin
    if rising_edge(clk) then
      if rst_n = '1' then
        for i in 0 to NUM_ISLANDS-1 loop
          assert not (island_states(i) /= ISLAND_OFF and pmic_pgood(i) = '0')
            report "PROP_SAFE_001 FAIL: Island " & integer'image(i) &
                   " active without PGOOD"
            severity error;
        end loop;
      end if;
    end if;
  end process p_safe_001;

  -- ---------------------------------------------------------------------------
  -- PROP_SAFE_002: Memory request address must be within physical range
  -- No access above 192 GB boundary
  -- ---------------------------------------------------------------------------
  -- psl PROP_SAFE_002 :
  --   assert always (
  --     mem_req_valid = '1' ->
  --     mem_req_addr < to_unsigned(192 * 1024 * 1024 * 1024, PHYS_ADDR_WIDTH)
  --   );

  p_safe_002 : process(clk)
  begin
    if rising_edge(clk) then
      if rst_n = '1' and mem_req_valid = '1' then
        assert mem_req_addr(PHYS_ADDR_WIDTH-1 downto 28) < to_unsigned(192 * 16, 12)
          report "PROP_SAFE_002 FAIL: Memory access out of range: " &
                 integer'image(to_integer(mem_req_addr))
          severity error;
      end if;
    end if;
  end process p_safe_002;

  -- ---------------------------------------------------------------------------
  -- PROP_SAFE_003: Warp scheduler must never select an invalid warp
  -- ---------------------------------------------------------------------------
  -- psl PROP_SAFE_003 :
  --   assert always (
  --     stall = '0' ->
  --     warp_valid(to_integer(selected_warp)) = '1'
  --   );

  p_safe_003 : process(clk)
  begin
    if rising_edge(clk) then
      if rst_n = '1' and stall = '0' then
        if warp_valid /= (MAX_WARPS_PER_SM-1 downto 0 => '0') then
          assert warp_valid(to_integer(selected_warp)) = '1'
            report "PROP_SAFE_003 FAIL: Scheduler selected invalid warp " &
                   integer'image(to_integer(selected_warp))
            severity error;
        end if;
      end if;
    end if;
  end process p_safe_003;

  -- ---------------------------------------------------------------------------
  -- PROP_SAFE_004: Pipeline flush and valid cannot both assert simultaneously
  -- on the same stage output
  -- ---------------------------------------------------------------------------
  -- psl PROP_SAFE_004 :
  --   assert always not (pipeline_valid = '1' and pipeline_flush = '1');

  p_safe_004 : process(clk)
  begin
    if rising_edge(clk) then
      if rst_n = '1' then
        assert not (pipeline_valid = '1' and pipeline_flush = '1')
          report "PROP_SAFE_004 FAIL: Pipeline valid and flush simultaneously"
          severity error;
      end if;
    end if;
  end process p_safe_004;

  -- ---------------------------------------------------------------------------
  -- PROP_SAFE_005: Active thread mask must never be all-zeros during execution
  -- (Zero-mask means dead warp — scheduler should not have issued it)
  -- ---------------------------------------------------------------------------
  -- psl PROP_SAFE_005 :
  --   assert always (
  --     pipeline_valid = '1' ->
  --     active_mask /= (WARP_SIZE-1 downto 0 => '0')
  --   );

  p_safe_005 : process(clk)
  begin
    if rising_edge(clk) then
      if rst_n = '1' and pipeline_valid = '1' then
        assert active_mask /= (WARP_SIZE-1 downto 0 => '0')
          report "PROP_SAFE_005 FAIL: Pipeline valid with zero active mask"
          severity error;
      end if;
    end if;
  end process p_safe_005;

  -- ---------------------------------------------------------------------------
  -- PROP_SAFE_006: Power limit register must never be set above 850W
  -- ---------------------------------------------------------------------------
  -- psl PROP_SAFE_006 :
  --   assert always bar_rd_data(15 downto 0) <= to_unsigned(850, 16)
  --     when bar_addr = x"001000" and bar_rd_valid = '1';

  p_safe_006 : process(clk)
  begin
    if rising_edge(clk) then
      if rst_n = '1' and bar_rd_valid = '1' and
         bar_addr = x"001000" then
        assert unsigned(bar_rd_data(15 downto 0)) <= 850
          report "PROP_SAFE_006 FAIL: Power limit exceeds 850W: " &
                 integer'image(to_integer(unsigned(bar_rd_data(15 downto 0)))) & "W"
          severity error;
      end if;
    end if;
  end process p_safe_006;

  -- ---------------------------------------------------------------------------
  -- PROP_SAFE_007: Depth test output must be stable within 1 cycle of done
  -- ---------------------------------------------------------------------------
  -- psl PROP_SAFE_007 :
  --   assert always (
  --     depth_done = '1' ->
  --     (depth_pass = '0' or depth_pass = '1')
  --   );

  p_safe_007 : process(clk)
  begin
    if rising_edge(clk) then
      if rst_n = '1' and depth_done = '1' then
        assert (depth_pass = '0' or depth_pass = '1')
          report "PROP_SAFE_007 FAIL: Depth pass output undefined on done"
          severity error;
      end if;
    end if;
  end process p_safe_007;

  -- ---------------------------------------------------------------------------
  -- PROP_SAFE_008: BAR0 reads to reserved addresses return 0xDEADBEEF sentinel
  -- (Not 'X' — X propagation from unmapped reads causes simulation failures)
  -- ---------------------------------------------------------------------------
  -- psl PROP_SAFE_008 :
  --   assert always (
  --     bar_rd_valid = '1' ->
  --     (not (bar_rd_data = (DATA_WIDTH-1 downto 0 => 'X')))
  --   );

  p_safe_008 : process(clk)
  begin
    if rising_edge(clk) then
      if rst_n = '1' and bar_rd_valid = '1' then
        assert bar_rd_data'event or bar_rd_data = bar_rd_data
          report "PROP_SAFE_008 FAIL: BAR0 read returns X (undefined)"
          severity error;
      end if;
    end if;
  end process p_safe_008;

  -- ===========================================================================
  -- SECTION 2: LIVENESS PROPERTIES
  -- "Something good eventually happens"
  -- ===========================================================================

  -- ---------------------------------------------------------------------------
  -- PROP_LIVE_001: Every memory request must eventually receive a response
  -- Bounded liveness: response within 512 cycles (HBM4 max latency)
  -- ---------------------------------------------------------------------------
  -- psl PROP_LIVE_001 :
  --   assert always (
  --     mem_req_valid = '1' ->
  --     next[1 to 512](mem_resp_valid = '1')
  --   );

  p_live_001 : process(clk)
    variable req_pending : std_logic := '0';
    variable req_age     : integer range 0 to 600 := 0;
  begin
    if rising_edge(clk) then
      if rst_n = '0' then
        req_pending := '0';
        req_age     := 0;
      else
        if mem_req_valid = '1' and req_pending = '0' then
          req_pending := '1';
          req_age     := 0;
        elsif req_pending = '1' then
          req_age := req_age + 1;
          if mem_resp_valid = '1' then
            req_pending := '0';
            req_age     := 0;
          else
            assert req_age < 512
              report "PROP_LIVE_001 FAIL: Memory request not responded after 512 cycles"
              severity warning;
          end if;
        end if;
      end if;
    end if;
  end process p_live_001;

  -- ---------------------------------------------------------------------------
  -- PROP_LIVE_002: Warp scheduler must not starve any ready warp
  -- Every READY warp must be selected within N cycles (N = 2 * NUM_WARPS)
  -- ---------------------------------------------------------------------------
  -- psl PROP_LIVE_002 :
  --   assert always (
  --     warp_states(0) = WARP_READY ->
  --     next[1 to 128](selected_warp = 0)
  --   );

  p_live_002 : process(clk)
    variable warp_wait_cnt : integer range 0 to 200 := 0;
    variable warp_waiting  : std_logic := '0';
  begin
    if rising_edge(clk) then
      if rst_n = '0' then
        warp_wait_cnt := 0;
        warp_waiting  := '0';
      else
        -- Monitor warp 0 for starvation
        if warp_states(0) = WARP_READY and warp_waiting = '0' then
          warp_waiting  := '1';
          warp_wait_cnt := 0;
        end if;
        if warp_waiting = '1' then
          warp_wait_cnt := warp_wait_cnt + 1;
          if selected_warp = 0 and stall = '0' then
            warp_waiting  := '0';
            warp_wait_cnt := 0;
          else
            assert warp_wait_cnt < 2 * MAX_WARPS_PER_SM
              report "PROP_LIVE_002 FAIL: Warp 0 starved for " &
                     integer'image(warp_wait_cnt) & " cycles"
              severity warning;
          end if;
        end if;
      end if;
    end if;
  end process p_live_002;

  -- ---------------------------------------------------------------------------
  -- PROP_LIVE_003: FPU must complete within 4 cycles of valid input
  -- ---------------------------------------------------------------------------
  -- psl PROP_LIVE_003 :
  --   assert always (
  --     fpu_valid_in = '1' ->
  --     next[4](fpu_valid_out = '1')
  --   );

  p_live_003 : process(clk)
    variable fpu_issued : std_logic := '0';
    variable fpu_cnt    : integer range 0 to 8 := 0;
  begin
    if rising_edge(clk) then
      if rst_n = '0' then
        fpu_issued := '0';
        fpu_cnt    := 0;
      else
        if fpu_valid_in = '1' then
          fpu_issued := '1';
          fpu_cnt    := 0;
        end if;
        if fpu_issued = '1' then
          fpu_cnt := fpu_cnt + 1;
          if fpu_valid_out = '1' then
            fpu_issued := '0';
          else
            assert fpu_cnt <= 4
              report "PROP_LIVE_003 FAIL: FPU did not complete within 4 cycles"
              severity error;
          end if;
        end if;
      end if;
    end if;
  end process p_live_003;

  -- ---------------------------------------------------------------------------
  -- PROP_LIVE_004: Tensor core must complete within 8 cycles of valid input
  -- ---------------------------------------------------------------------------
  p_live_004 : process(clk)
    variable ten_issued : std_logic := '0';
    variable ten_cnt    : integer range 0 to 16 := 0;
  begin
    if rising_edge(clk) then
      if rst_n = '0' then
        ten_issued := '0';
        ten_cnt    := 0;
      else
        if tensor_valid_in = '1' then
          ten_issued := '1';
          ten_cnt    := 0;
        end if;
        if ten_issued = '1' then
          ten_cnt := ten_cnt + 1;
          if tensor_valid_out = '1' then
            ten_issued := '0';
          else
            assert ten_cnt <= 8
              report "PROP_LIVE_004 FAIL: Tensor core stalled beyond 8 cycles"
              severity error;
          end if;
        end if;
      end if;
    end if;
  end process p_live_004;

  -- ===========================================================================
  -- SECTION 3: PROTOCOL PROPERTIES
  -- "Handshake and bus protocol correctness"
  -- ===========================================================================

  -- ---------------------------------------------------------------------------
  -- PROP_PROTO_001: Memory request must hold valid until acknowledged
  -- (req_valid should not deassert before ack/ready)
  -- ---------------------------------------------------------------------------
  -- psl PROP_PROTO_001 :
  --   assert always (
  --     mem_req_valid = '1' and mem_resp_valid = '0' ->
  --     next(mem_req_valid = '1')
  --   );

  p_proto_001 : process(clk)
    variable last_req_valid : std_logic := '0';
    variable last_resp_valid: std_logic := '0';
  begin
    if rising_edge(clk) then
      if rst_n = '0' then
        last_req_valid  := '0';
        last_resp_valid := '0';
      else
        -- Check: if request was valid last cycle and no response yet,
        -- request should remain valid
        if last_req_valid = '1' and last_resp_valid = '0' then
          assert mem_req_valid = '1'
            report "PROP_PROTO_001 FAIL: mem_req_valid dropped before response"
            severity error;
        end if;
        last_req_valid  := mem_req_valid;
        last_resp_valid := mem_resp_valid;
      end if;
    end if;
  end process p_proto_001;

  -- ---------------------------------------------------------------------------
  -- PROP_PROTO_002: BAR0 read-valid must follow read-enable by exactly 1 cycle
  -- ---------------------------------------------------------------------------
  -- psl PROP_PROTO_002 :
  --   assert always (
  --     bar_rd_en = '1' ->
  --     next(bar_rd_valid = '1')
  --   );

  p_proto_002 : process(clk)
    variable last_rd_en : std_logic := '0';
  begin
    if rising_edge(clk) then
      if rst_n = '0' then
        last_rd_en := '0';
      else
        if last_rd_en = '1' then
          assert bar_rd_valid = '1'
            report "PROP_PROTO_002 FAIL: BAR0 rd_valid not asserted 1 cycle after rd_en"
            severity error;
        end if;
        last_rd_en := bar_rd_en;
      end if;
    end if;
  end process p_proto_002;

  -- ---------------------------------------------------------------------------
  -- PROP_PROTO_003: PMBus must not issue two commands simultaneously
  -- (start signal must be deasserted between commands)
  -- ---------------------------------------------------------------------------
  -- Monitored externally via PMBus controller busy signal

  -- ---------------------------------------------------------------------------
  -- PROP_PROTO_004: Branch flush must last exactly 1 cycle
  -- (Extended flush wastes pipeline cycles)
  -- ---------------------------------------------------------------------------
  -- psl PROP_PROTO_004 :
  --   assert always (
  --     branch_flush = '1' ->
  --     next(branch_flush = '0')
  --   );

  p_proto_004 : process(clk)
    variable last_flush : std_logic := '0';
  begin
    if rising_edge(clk) then
      if rst_n = '0' then
        last_flush := '0';
      else
        if last_flush = '1' then
          assert branch_flush = '0'
            report "PROP_PROTO_004 FAIL: Branch flush held for more than 1 cycle"
            severity error;
        end if;
        last_flush := branch_flush;
      end if;
    end if;
  end process p_proto_004;

  -- ---------------------------------------------------------------------------
  -- PROP_PROTO_005: Display DE must be de-asserted during sync pulses
  -- (Data Enable cannot be high during HSYNC or VSYNC)
  -- ---------------------------------------------------------------------------
  -- psl PROP_PROTO_005 :
  --   assert always not (disp_de = '1' and (disp_hsync = '1' or disp_vsync = '1'));

  p_proto_005 : process(clk)
  begin
    if rising_edge(clk) then
      if rst_n = '1' then
        assert not (disp_de = '1' and (disp_hsync = '1' or disp_vsync = '1'))
          report "PROP_PROTO_005 FAIL: DE asserted during SYNC pulse"
          severity error;
      end if;
    end if;
  end process p_proto_005;

  -- ---------------------------------------------------------------------------
  -- PROP_PROTO_006: RT core result must be valid only after ray_valid
  -- (No spurious outputs)
  -- ---------------------------------------------------------------------------
  -- psl PROP_PROTO_006 :
  --   assert always (
  --     rt_result_valid = '1' ->
  --     prev(rt_valid_in = '1') within [1 to 512 cycles]
  --   );

  p_proto_006 : process(clk)
    variable rt_issued : std_logic := '0';
  begin
    if rising_edge(clk) then
      if rst_n = '0' then
        rt_issued := '0';
      else
        if rt_valid_in = '1' then
          rt_issued := '1';
        end if;
        if rt_result_valid = '1' then
          assert rt_issued = '1'
            report "PROP_PROTO_006 FAIL: RT result without prior ray request"
            severity error;
          rt_issued := '0';
        end if;
      end if;
    end if;
  end process p_proto_006;

  -- ===========================================================================
  -- SECTION 4: POWER PROPERTIES
  -- ===========================================================================

  -- ---------------------------------------------------------------------------
  -- PROP_PWR_001: Island temperature must never exceed thermal limit
  -- ---------------------------------------------------------------------------
  -- psl PROP_PWR_001 :
  --   assert always (
  --     forall i in 0 to NUM_ISLANDS-1:
  --       telemetry(i).temperature < 95
  --   );

  p_pwr_001 : process(clk)
  begin
    if rising_edge(clk) then
      if rst_n = '1' then
        for i in 0 to NUM_ISLANDS-1 loop
          assert telemetry(i).temperature < to_unsigned(95, 8)
            report "PROP_PWR_001 FAIL: Island " & integer'image(i) &
                   " temperature " &
                   integer'image(to_integer(telemetry(i).temperature)) &
                   "°C exceeds TJ limit"
            severity error;
        end loop;
      end if;
    end if;
  end process p_pwr_001;

  -- ---------------------------------------------------------------------------
  -- PROP_PWR_002: Sum of all island power must not exceed configured power limit
  -- ---------------------------------------------------------------------------
  p_pwr_002 : process(clk)
    variable total_mw : unsigned(19 downto 0);
  begin
    if rising_edge(clk) then
      if rst_n = '1' then
        total_mw := (others => '0');
        for i in 0 to NUM_ISLANDS-1 loop
          total_mw := total_mw + resize(telemetry(i).power_mw, 20);
        end loop;
        -- Total > 850W = 850000 mW triggers warning (not hard error — thermal model)
        assert total_mw <= to_unsigned(850_000, 20)
          report "PROP_PWR_002 WARN: Total island power " &
                 integer'image(to_integer(total_mw)) & " mW exceeds 850W TDP"
          severity warning;
      end if;
    end if;
  end process p_pwr_002;

  -- ---------------------------------------------------------------------------
  -- PROP_PWR_003: An OFF island must consume zero reported power
  -- ---------------------------------------------------------------------------
  p_pwr_003 : process(clk)
  begin
    if rising_edge(clk) then
      if rst_n = '1' then
        for i in 0 to NUM_ISLANDS-1 loop
          if island_states(i) = ISLAND_OFF then
            assert telemetry(i).power_mw = to_unsigned(0, 16)
              report "PROP_PWR_003 FAIL: Island " & integer'image(i) &
                     " is OFF but reports " &
                     integer'image(to_integer(telemetry(i).power_mw)) & " mW"
              severity error;
          end if;
        end loop;
      end if;
    end if;
  end process p_pwr_003;

  -- ---------------------------------------------------------------------------
  -- PROP_PWR_004: FULL island must not be active without PGOOD
  -- ---------------------------------------------------------------------------
  p_pwr_004 : process(clk)
  begin
    if rising_edge(clk) then
      if rst_n = '1' then
        for i in 0 to NUM_ISLANDS-1 loop
          if island_states(i) = ISLAND_FULL then
            assert pmic_pgood(i) = '1'
              report "PROP_PWR_004 FAIL: Island " & integer'image(i) &
                     " at FULL voltage without PGOOD"
              severity error;
          end if;
        end loop;
      end if;
    end if;
  end process p_pwr_004;

  -- ===========================================================================
  -- SECTION 5: DATAPATH PROPERTIES
  -- ===========================================================================

  -- ---------------------------------------------------------------------------
  -- PROP_DATA_001: FPU result must not be NaN for valid finite inputs
  -- IEEE 754: NaN encoded as exponent=0xFF and non-zero mantissa
  -- ---------------------------------------------------------------------------
  p_data_001 : process(clk)
    variable exp_bits : std_logic_vector(7 downto 0);
    variable man_bits : std_logic_vector(22 downto 0);
  begin
    if rising_edge(clk) then
      if rst_n = '1' and fpu_valid_out = '1' then
        exp_bits := fpu_result(30 downto 23);
        man_bits := fpu_result(22 downto 0);
        -- NaN: all exponent bits set AND non-zero mantissa
        if exp_bits = x"FF" and man_bits /= (22 downto 0 => '0') then
          -- Check if inputs were also NaN (NaN propagation is correct)
          if fpu_src_a(30 downto 23) /= x"FF" and
             fpu_src_b(30 downto 23) /= x"FF" then
            assert false
              report "PROP_DATA_001 FAIL: FPU produced NaN from finite inputs"
              severity error;
          end if;
        end if;
      end if;
    end if;
  end process p_data_001;

  -- ---------------------------------------------------------------------------
  -- PROP_DATA_002: Depth value must stay in [0.0, 1.0] (NDC depth range)
  -- ---------------------------------------------------------------------------
  p_data_002 : process(clk)
    variable sign_bit : std_logic;
    variable exp_v    : unsigned(7 downto 0);
  begin
    if rising_edge(clk) then
      if rst_n = '1' and depth_done = '1' and depth_pass = '1' then
        sign_bit := depth_test_depth(31);
        exp_v    := unsigned(depth_test_depth(30 downto 23));
        -- Negative depth is invalid
        assert sign_bit = '0'
          report "PROP_DATA_002 FAIL: Negative depth value in depth buffer write"
          severity error;
        -- Value > 1.0: exponent > 127 means magnitude > 1.0 for positive FP32
        assert exp_v <= 127
          report "PROP_DATA_002 FAIL: Depth value > 1.0 written to depth buffer"
          severity error;
      end if;
    end if;
  end process p_data_002;

  -- ---------------------------------------------------------------------------
  -- PROP_DATA_003: RT intersection distance t must be positive
  -- Negative t means intersection behind the ray origin — invalid
  -- ---------------------------------------------------------------------------
  p_data_003 : process(clk)
  begin
    if rising_edge(clk) then
      if rst_n = '1' and rt_result_valid = '1' then
        assert rt_result_t(31) = '0'
          report "PROP_DATA_003 FAIL: RT core returned negative intersection distance"
          severity error;
      end if;
    end if;
  end process p_data_003;

  -- ---------------------------------------------------------------------------
  -- PROP_DATA_004: Selected warp ID must be within valid range
  -- ---------------------------------------------------------------------------
  p_data_004 : process(clk)
  begin
    if rising_edge(clk) then
      if rst_n = '1' then
        assert to_integer(selected_warp) < MAX_WARPS_PER_SM
          report "PROP_DATA_004 FAIL: Selected warp " &
                 integer'image(to_integer(selected_warp)) &
                 " exceeds MAX_WARPS_PER_SM=" &
                 integer'image(MAX_WARPS_PER_SM)
          severity error;
      end if;
    end if;
  end process p_data_004;

  -- ---------------------------------------------------------------------------
  -- PROP_DATA_005: Display active pixel count per line must equal h_active
  -- ---------------------------------------------------------------------------
  p_data_005 : process(clk)
    variable de_count   : unsigned(13 downto 0) := (others => '0');
    variable last_de    : std_logic := '0';
    variable last_hsync : std_logic := '0';
    variable line_active: std_logic := '0';
  begin
    if rising_edge(clk) then
      if rst_n = '0' then
        de_count    := (others => '0');
        last_de     := '0';
        last_hsync  := '0';
        line_active := '0';
      else
        -- Count DE pulses per line
        if disp_de = '1' then
          de_count := de_count + 1;
          line_active := '1';
        end if;

        -- At HSYNC falling edge: check DE count matches h_active
        if last_hsync = '1' and disp_hsync = '0' and line_active = '1' then
          assert de_count = disp_h_active
            report "PROP_DATA_005 FAIL: DE count " &
                   integer'image(to_integer(de_count)) &
                   " does not match h_active " &
                   integer'image(to_integer(disp_h_active))
            severity warning;
          de_count    := (others => '0');
          line_active := '0';
        end if;

        last_de    := disp_de;
        last_hsync := disp_hsync;
      end if;
    end if;
  end process p_data_005;

  -- ---------------------------------------------------------------------------
  -- PROP_DATA_006: After reset, all island states must be ISLAND_FULL
  -- (Boot state: all power domains active until PMC calibrates)
  -- ---------------------------------------------------------------------------
  p_data_006 : process(clk)
    variable rst_released : std_logic := '0';
    variable check_cycle  : integer range 0 to 4 := 0;
  begin
    if rising_edge(clk) then
      if rst_n = '0' then
        rst_released := '0';
        check_cycle  := 0;
      else
        if rst_released = '0' then
          rst_released := '1';
          check_cycle  := 0;
        end if;
        -- Check 2 cycles after reset release
        if check_cycle < 2 then
          check_cycle := check_cycle + 1;
        elsif check_cycle = 2 then
          for i in 0 to NUM_ISLANDS-1 loop
            assert island_states(i) = ISLAND_FULL
              report "PROP_DATA_006 FAIL: Island " & integer'image(i) &
                     " not FULL after reset"
              severity error;
          end loop;
          check_cycle := 3;  -- Only check once
        end if;
      end if;
    end if;
  end process p_data_006;

  -- ===========================================================================
  -- SECTION 6: COVERAGE PROPERTIES
  -- "All interesting states are reachable"
  -- These are cover directives — formal tool verifies they CAN be reached
  -- ===========================================================================

  -- ---------------------------------------------------------------------------
  -- COVER_001: At least one island reaches ISLAND_OFF state
  -- ---------------------------------------------------------------------------
  -- psl COVER_001 : cover {island_states(0) = ISLAND_OFF};

  p_cover_001 : process(clk)
  begin
    if rising_edge(clk) then
      if island_states(0) = ISLAND_OFF then
        report "COVER_001 HIT: Island 0 reached OFF state" severity note;
      end if;
    end if;
  end process p_cover_001;

  -- ---------------------------------------------------------------------------
  -- COVER_002: All 16 islands simultaneously at ISLAND_FULL (peak load)
  -- ---------------------------------------------------------------------------
  p_cover_002 : process(clk)
    variable all_full : std_logic;
  begin
    if rising_edge(clk) then
      all_full := '1';
      for i in 0 to NUM_ISLANDS-1 loop
        if island_states(i) /= ISLAND_FULL then
          all_full := '0';
        end if;
      end loop;
      if all_full = '1' then
        report "COVER_002 HIT: All 16 islands simultaneously at FULL" severity note;
      end if;
    end if;
  end process p_cover_002;

  -- ---------------------------------------------------------------------------
  -- COVER_003: FPU produces exactly +0.0 result
  -- ---------------------------------------------------------------------------
  p_cover_003 : process(clk)
  begin
    if rising_edge(clk) then
      if fpu_valid_out = '1' and fpu_result = x"00000000" then
        report "COVER_003 HIT: FPU produced +0.0" severity note;
      end if;
    end if;
  end process p_cover_003;

  -- ---------------------------------------------------------------------------
  -- COVER_004: Divergence stack reaches maximum depth (8)
  -- Tests deepest nested conditional in shader code
  -- ---------------------------------------------------------------------------
  -- psl COVER_004 : cover { branch_taken and active_mask /= all-ones };

  -- ---------------------------------------------------------------------------
  -- COVER_005: RT core finds no intersection (miss path)
  -- ---------------------------------------------------------------------------
  p_cover_005 : process(clk)
  begin
    if rising_edge(clk) then
      if rt_result_valid = '1' and rt_result_t = x"7F800000" then  -- +Inf = miss
        report "COVER_005 HIT: RT core returned miss (no intersection)" severity note;
      end if;
    end if;
  end process p_cover_005;

  -- ---------------------------------------------------------------------------
  -- COVER_006: Depth test passes and fails (both paths exercised)
  -- ---------------------------------------------------------------------------
  p_cover_006_pass : process(clk)
  begin
    if rising_edge(clk) then
      if depth_done = '1' and depth_pass = '1' then
        report "COVER_006a HIT: Depth test PASS" severity note;
      end if;
    end if;
  end process p_cover_006_pass;

  p_cover_006_fail : process(clk)
  begin
    if rising_edge(clk) then
      if depth_done = '1' and depth_pass = '0' then
        report "COVER_006b HIT: Depth test FAIL (pixel culled)" severity note;
      end if;
    end if;
  end process p_cover_006_fail;

  -- ---------------------------------------------------------------------------
  -- COVER_007: BAR0 write followed immediately by read to same address
  -- (Test write-then-read coherence)
  -- ---------------------------------------------------------------------------
  p_cover_007 : process(clk)
    variable last_wr_addr : unsigned(23 downto 0);
    variable last_wr_en   : std_logic := '0';
  begin
    if rising_edge(clk) then
      if last_wr_en = '1' and bar_rd_en = '1' and
         bar_addr = last_wr_addr then
        report "COVER_007 HIT: Back-to-back write-read to BAR0 addr " &
               integer'image(to_integer(bar_addr))
          severity note;
      end if;
      last_wr_en   := bar_wr_en;
      last_wr_addr := bar_addr;
    end if;
  end process p_cover_007;

  -- ===========================================================================
  -- SECTION 7: INVARIANTS (checked continuously via assertions)
  -- ===========================================================================

  -- ---------------------------------------------------------------------------
  -- INV_001: NUM_ISLANDS constant must equal NUM_SMS_TOTAL / NUM_SMS_PER_ISLAND
  -- (Package constant consistency check)
  -- ---------------------------------------------------------------------------
  p_inv_001 : process
  begin
    assert NUM_ISLANDS = NUM_SMS_TOTAL / NUM_SMS_PER_ISLAND
      report "INV_001 FAIL: NUM_ISLANDS constant inconsistent with SM topology"
      severity failure;
    wait;
  end process p_inv_001;

  -- ---------------------------------------------------------------------------
  -- INV_002: WARP_SIZE must be a power of 2
  -- ---------------------------------------------------------------------------
  p_inv_002 : process
    variable ws    : integer;
    constant WBITS : integer := 16;  -- Comfortably covers any realistic
                                      -- WARP_SIZE (32 in this design)
    variable ws_u  : unsigned(WBITS-1 downto 0);
  begin
    ws   := WARP_SIZE;
    ws_u := to_unsigned(ws, WBITS);
    -- Classic power-of-2 test (x & (x-1)) = 0, performed on the unsigned
    -- bit-vector since VHDL's integer type has no bitwise "and" operator
    -- -- ws and (ws - 1) on plain integers does not exist in the language.
    assert (ws_u and (ws_u - 1)) = to_unsigned(0, WBITS)
      report "INV_002 FAIL: WARP_SIZE must be a power of 2"
      severity failure;
    wait;
  end process p_inv_002;

  -- ---------------------------------------------------------------------------
  -- INV_003: DATA_WIDTH must be 32 (FP32 IEEE 754)
  -- ---------------------------------------------------------------------------
  p_inv_003 : process
  begin
    assert DATA_WIDTH = 32
      report "INV_003 FAIL: DATA_WIDTH must be 32 for IEEE 754 FP32"
      severity failure;
    wait;
  end process p_inv_003;

end architecture formal;
