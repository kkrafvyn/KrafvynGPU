-- =============================================================================
-- KrafGPU-X1 Compute Chiplet
-- Project      : KrafGPU-X1 Next-Generation GPU Architecture
-- Author       : Krafvyn Tech / Don
-- Description  : Single compute chiplet containing 256 SMs across 8 voltage
--                islands. Each island holds 32 SMs with independent V/F rails.
--                Two of these chiplets make up the full 512-SM GPU.
-- =============================================================================

library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;
use work.krafgpu_pkg.all;

entity krafgpu_compute_chiplet is
  generic (
    CHIPLET_ID        : integer := 0;   -- 0 = Chiplet A, 1 = Chiplet B
    SM_ID_BASE        : integer := 0;   -- First SM index on this chiplet
    ISLAND_ID_BASE    : integer := 0;   -- First island index on this chiplet
    NUM_SMS           : integer := 256;
    NUM_ISLANDS_LOCAL : integer := 8
  );
  port (
    clk               : in  std_logic;
    rst_n             : in  std_logic;
    -- Voltage island states from PMC (sized to NUM_ISLANDS_LOCAL, not the
    -- full-chip NUM_ISLANDS -- the top level passes an 8-element slice of
    -- its own island_state_array_t(0 to NUM_ISLANDS-1) signal here)
    island_states     : in  island_state_array_u_t(0 to NUM_ISLANDS_LOCAL-1);
    -- Memory request/response to interconnect fabric
    mem_req           : out mem_req_t;
    mem_resp          : in  mem_resp_t;
    mem_ready         : in  std_logic;
    -- Telemetry back to PMC (same NUM_ISLANDS_LOCAL sizing rationale)
    telemetry_out     : out telemetry_array_u_t(0 to NUM_ISLANDS_LOCAL-1)
  );
end entity krafgpu_compute_chiplet;

architecture rtl of krafgpu_compute_chiplet is

  -- ===========================================================================
  -- Per-island clock enable signals (derived from voltage island state)
  -- ===========================================================================
  signal island_clk_en    : std_logic_vector(NUM_ISLANDS_LOCAL-1 downto 0);
  signal island_rst_n     : std_logic_vector(NUM_ISLANDS_LOCAL-1 downto 0);

  -- ===========================================================================
  -- Per-SM memory request buses (muxed onto shared chiplet bus)
  -- Types come from krafgpu_pkg (mem_req_array_t / mem_resp_array_t) so the
  -- same types can be shared with krafgpu_rr_arbiter's generic-width ports.
  -- ===========================================================================
  signal sm_mem_req        : mem_req_array_t(0 to NUM_SMS-1);
  signal sm_mem_resp       : mem_resp_array_t(0 to NUM_SMS-1);

  -- ===========================================================================
  -- Per-island telemetry aggregation
  -- ===========================================================================
  type island_util_array_t is array (0 to NUM_ISLANDS_LOCAL-1) of unsigned(6 downto 0);
  type island_temp_array_t is array (0 to NUM_ISLANDS_LOCAL-1) of unsigned(7 downto 0);
  type island_pwr_array_t  is array (0 to NUM_ISLANDS_LOCAL-1) of unsigned(15 downto 0);

  signal island_util       : island_util_array_t;
  signal island_temp       : island_temp_array_t;
  signal island_power_mw   : island_pwr_array_t;

  -- ===========================================================================
  -- Arbitration signals
  -- ===========================================================================
  signal arb_grant         : integer range 0 to NUM_SMS-1 := 0;
  signal arb_req_valid     : std_logic_vector(NUM_SMS-1 downto 0);

begin

  -- ===========================================================================
  -- GENERIC CONSISTENCY CHECK
  -- NUM_SMS sizes sm_mem_req/sm_mem_resp/arb_req_valid and the downstream
  -- arbiter's NUM_PORTS, while gen_islands x gen_sms below actually
  -- instantiates NUM_ISLANDS_LOCAL * NUM_SMS_PER_ISLAND SMs (the latter is
  -- the fixed package constant, not a generic). Both krafgpu_x1_top
  -- instantiations (256 SMs / 8 islands / 32 SMs-per-island = 256, OK)
  -- already satisfy this, but nothing previously caught a mismatched
  -- instantiation at elaboration time -- it would have silently built an
  -- inconsistent design (e.g. unconnected SM indices, or an out-of-range
  -- arb_req_valid index in gen_telemetry's per-island sum below).
  -- ===========================================================================
  assert NUM_SMS = NUM_ISLANDS_LOCAL * NUM_SMS_PER_ISLAND
    report "krafgpu_compute_chiplet: NUM_SMS generic (" &
           integer'image(NUM_SMS) &
           ") must equal NUM_ISLANDS_LOCAL * NUM_SMS_PER_ISLAND (" &
           integer'image(NUM_ISLANDS_LOCAL) & " * " &
           integer'image(NUM_SMS_PER_ISLAND) & " = " &
           integer'image(NUM_ISLANDS_LOCAL * NUM_SMS_PER_ISLAND) & ")"
    severity failure;

  -- ===========================================================================
  -- ISLAND CLOCK ENABLE GENERATION
  -- Derive per-island clock enable from PMC island state
  -- ===========================================================================
  gen_island_ctrl : for i in 0 to NUM_ISLANDS_LOCAL-1 generate
    island_clk_en(i) <= '0' when island_states(i) = ISLAND_OFF else '1';
    island_rst_n(i)  <= '0' when island_states(i) = ISLAND_OFF else rst_n;
  end generate;

  -- ===========================================================================
  -- SM INSTANTIATION
  -- NUM_ISLANDS_LOCAL islands x NUM_SMS_PER_ISLAND SMs per island
  -- (256 SMs across 8 islands / 32 SMs per island in the full design --
  -- see krafgpu_x1_top.vhd's u_chiplet_a/u_chiplet_b instantiations)
  -- ===========================================================================
  gen_islands : for isl in 0 to NUM_ISLANDS_LOCAL-1 generate
    gen_sms : for s in 0 to NUM_SMS_PER_ISLAND-1 generate

      constant SM_IDX : integer := isl * NUM_SMS_PER_ISLAND + s;

    begin

      u_sm : entity work.krafgpu_sm
        generic map (
          SM_ID       => SM_ID_BASE + SM_IDX,
          ISLAND_ID   => ISLAND_ID_BASE + isl
        )
        port map (
          clk         => clk,
          clk_en      => island_clk_en(isl),
          rst_n       => island_rst_n(isl),
          mem_req     => sm_mem_req(SM_IDX),
          mem_resp    => sm_mem_resp(SM_IDX),
          util_out    => open   -- aggregated at island level
        );

    end generate gen_sms;
  end generate gen_islands;

  -- ===========================================================================
  -- MEMORY REQUEST ARBITRATION
  -- Round-robin arbiter across all 256 SMs on this chiplet
  -- ===========================================================================
  u_arbiter : entity work.krafgpu_rr_arbiter
    generic map (
      NUM_PORTS   => NUM_SMS
    )
    port map (
      clk         => clk,
      rst_n       => rst_n,
      req_valid   => arb_req_valid,
      req_data    => sm_mem_req,
      grant_idx   => arb_grant,
      out_req     => mem_req,
      in_resp     => mem_resp,
      sm_resp     => sm_mem_resp
    );

  -- Build req_valid vector
  gen_req_valid : for i in 0 to NUM_SMS-1 generate
    arb_req_valid(i) <= sm_mem_req(i).valid;
  end generate;

  -- ===========================================================================
  -- TELEMETRY AGGREGATION PER ISLAND
  -- Collect utilisation, temperature, power per island and report to PMC
  -- ===========================================================================
  gen_telemetry : for i in 0 to NUM_ISLANDS_LOCAL-1 generate
    process(clk)
      variable util_sum  : unsigned(13 downto 0) := (others => '0');
      variable temp_max  : unsigned(7 downto 0)  := (others => '0');
      variable pwr_accum : unsigned(21 downto 0) := (others => '0');
    begin
      if rising_edge(clk) then
        if rst_n = '0' or island_states(i) = ISLAND_OFF then
          island_util(i)     <= (others => '0');
          island_temp(i)     <= (others => '0');
          island_power_mw(i) <= (others => '0');
        else
          -- Average utilisation across 32 SMs in this island
          util_sum := (others => '0');
          for s in 0 to NUM_SMS_PER_ISLAND-1 loop
            util_sum := util_sum +
              resize(arb_req_valid(i * NUM_SMS_PER_ISLAND + s) &
                     "000000", 14);
          end loop;
          island_util(i) <= util_sum(12 downto 6);  -- divide by 32

          -- Simple thermal model: base temp + load factor
          case island_states(i) is
            when ISLAND_FULL => island_temp(i) <= to_unsigned(75, 8);
            when ISLAND_HALF => island_temp(i) <= to_unsigned(60, 8);
            when ISLAND_LOW  => island_temp(i) <= to_unsigned(45, 8);
            when ISLAND_OFF  => island_temp(i) <= to_unsigned(25, 8);
          end case;

          -- Power estimate based on state
          case island_states(i) is
            when ISLAND_FULL => island_power_mw(i) <= to_unsigned(35000, 16); -- 35W
            when ISLAND_HALF => island_power_mw(i) <= to_unsigned(24000, 16); -- 24W
            when ISLAND_LOW  => island_power_mw(i) <= to_unsigned(14000, 16); -- 14W
            when ISLAND_OFF  => island_power_mw(i) <= (others => '0');
          end case;

        end if;
      end if;
    end process;

    -- Wire telemetry record
    telemetry_out(i).island_id   <= to_unsigned(ISLAND_ID_BASE + i, 4);
    telemetry_out(i).utilisation <= island_util(i);
    telemetry_out(i).temperature <= island_temp(i);
    telemetry_out(i).power_mw    <= island_power_mw(i);

  end generate gen_telemetry;

end architecture rtl;
