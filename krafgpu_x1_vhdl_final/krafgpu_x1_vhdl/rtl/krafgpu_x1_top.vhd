-- =============================================================================
-- KrafGPU-X1 Top-Level
-- Project      : KrafGPU-X1 Next-Generation GPU Architecture
-- Author       : Krafvyn Tech / Don
-- Description  : Top-level entity instantiating all major subsystems:
--                chiplet A + B, I/O die, PMC, memory controller, interconnect
-- =============================================================================

library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;
use work.krafgpu_pkg.all;

entity krafgpu_x1_top is
  port (
    -- -------------------------------------------------------------------------
    -- Primary clocks (from PLL on I/O die)
    -- -------------------------------------------------------------------------
    clk_core_a      : in  std_logic;   -- Chiplet A compute clock (up to 2.8 GHz)
    clk_core_b      : in  std_logic;   -- Chiplet B compute clock
    clk_mem         : in  std_logic;   -- HBM4 memory clock (4.8 GHz)
    clk_pcie        : in  std_logic;   -- PCIe reference 100 MHz
    clk_pmc         : in  std_logic;   -- PMC clock 400 MHz (always-on)
    rst_n           : in  std_logic;   -- Active-low global reset

    -- -------------------------------------------------------------------------
    -- PCIe 6.0 x64 host interface (differential pairs — abstracted as data bus)
    -- -------------------------------------------------------------------------
    pcie_rx_data    : in  std_logic_vector(255 downto 0);  -- 256b per cycle
    pcie_rx_valid   : in  std_logic;
    pcie_rx_ready   : out std_logic;
    pcie_tx_data    : out std_logic_vector(255 downto 0);
    pcie_tx_valid   : out std_logic;
    pcie_tx_ready   : in  std_logic;

    -- -------------------------------------------------------------------------
    -- HBM4 memory interface (6 stacks × 1024-bit)
    -- -------------------------------------------------------------------------
    hbm_addr        : out std_logic_vector(PHYS_ADDR_WIDTH-1 downto 0);
    hbm_wdata       : out std_logic_vector(HBM4_TOTAL_WIDTH-1 downto 0);
    hbm_rdata       : in  std_logic_vector(HBM4_TOTAL_WIDTH-1 downto 0);
    hbm_wr_en       : out std_logic_vector(NUM_HBM4_STACKS-1 downto 0);
    hbm_rd_en       : out std_logic_vector(NUM_HBM4_STACKS-1 downto 0);
    hbm_ack         : in  std_logic_vector(NUM_HBM4_STACKS-1 downto 0);

    -- -------------------------------------------------------------------------
    -- Display outputs (abstracted — 4x DP 2.1)
    -- -------------------------------------------------------------------------
    disp_data       : out std_logic_vector(127 downto 0);  -- 4 lanes × 32b
    disp_valid      : out std_logic_vector(3 downto 0);

    -- -------------------------------------------------------------------------
    -- PMBus / power management interface (to 4× PMIC)
    -- -------------------------------------------------------------------------
    pmbus_scl       : inout std_logic;
    pmbus_sda       : inout std_logic;
    pmic_pgood      : in  std_logic_vector(NUM_ISLANDS-1 downto 0);

    -- -------------------------------------------------------------------------
    -- Debug / evaluation interfaces
    -- -------------------------------------------------------------------------
    jtag_tck        : in  std_logic;
    jtag_tms        : in  std_logic;
    jtag_tdi        : in  std_logic;
    jtag_tdo        : out std_logic;
    uart_tx         : out std_logic;
    uart_rx         : in  std_logic;

    -- -------------------------------------------------------------------------
    -- NVLink-equivalent multi-GPU bridge
    -- -------------------------------------------------------------------------
    nvlink_tx       : out std_logic_vector(127 downto 0);
    nvlink_rx       : in  std_logic_vector(127 downto 0);
    nvlink_valid_tx : out std_logic;
    nvlink_valid_rx : in  std_logic;

    -- -------------------------------------------------------------------------
    -- System / peripheral memory port (display scanout, command ring DMA,
    -- NPU clusters, vertex fetch, raster depth buffer, context engine).
    -- Arbitrated into the die-to-die fabric as a third requester alongside
    -- the two compute chiplets.
    -- -------------------------------------------------------------------------
    sys_mem_req     : in  mem_req_t;
    sys_mem_resp    : out mem_resp_t;
    sys_mem_ready   : out std_logic;

    -- -------------------------------------------------------------------------
    -- Aggregate PMC status (for BAR0 telemetry registers).
    -- Computed from pmc_telemetry / island_states in the clk_pmc domain.
    -- NOTE: pmc_telemetry itself is produced in the chiplet clk_core_a/b
    -- domains and consumed here without explicit CDC handshaking, consistent
    -- with the simplified single-cycle telemetry path used elsewhere in this
    -- RTL skeleton. A tapeout-grade design would register these through a
    -- gray-coded or toggle-based CDC synchroniser.
    -- -------------------------------------------------------------------------
    status_power_mw     : out unsigned(19 downto 0);  -- Sum of all islands, mW
    status_temp_c       : out unsigned(7 downto 0);   -- Max island temperature
    status_util_pct     : out unsigned(6 downto 0);   -- Mean island utilisation
    status_island_active: out std_logic_vector(NUM_ISLANDS-1 downto 0);

    -- -------------------------------------------------------------------------
    -- Pixel-shader warp dispatch (from krafgpu_ps_dispatch in system wrapper)
    -- Driven by krafgpu_x1_system after depth-test; routed to the target SM
    -- via sm_sel index. krafgpu_x1_top fans this to both chiplets; each chiplet
    -- routes to its own set of SMs. Only one SM's ps_valid is asserted per
    -- cycle (the ps_dispatch module guarantees single-target dispatch).
    -- -------------------------------------------------------------------------
    ps_valid        : in  std_logic := '0';
    ps_sm_sel       : in  integer range 0 to NUM_SMS_TOTAL-1 := 0;
    ps_warp_id      : in  unsigned(5 downto 0) := (others => '0');
    ps_screen_x     : in  unsigned(15 downto 0) := (others => '0');
    ps_screen_y     : in  unsigned(15 downto 0) := (others => '0');
    ps_bary_u       : in  std_logic_vector(DATA_WIDTH-1 downto 0)
                    := (others => '0');
    ps_bary_v       : in  std_logic_vector(DATA_WIDTH-1 downto 0)
                    := (others => '0');
    ps_depth        : in  std_logic_vector(DATA_WIDTH-1 downto 0)
                    := (others => '0');
    ps_ready        : out std_logic_vector(NUM_SMS_TOTAL-1 downto 0);
    -- ECC error outputs aggregated from L3 cache in the memory subsystem.
    -- L2 ECC from the compute chiplets will be added when krafgpu_compute_chiplet
    -- exposes the ports in a future enhancement pass.
    ecc_single_err  : out std_logic;
    ecc_double_err  : out std_logic
  );
end entity krafgpu_x1_top;

architecture structural of krafgpu_x1_top is

  -- Matches NUM_ISLANDS_LOCAL => 8 on both u_chiplet_a / u_chiplet_b below
  -- (NUM_ISLANDS / 2, since the two chiplets evenly split the 16 islands).
  constant NUM_ISLANDS_LOCAL_PER_CHIPLET : integer := NUM_ISLANDS / 2;

  -- ===========================================================================
  -- Internal signals
  -- ===========================================================================

  -- PMC ↔ Chiplets
  signal pmc_island_cmd     : pmc_cmd_t;
  signal pmc_telemetry      : telemetry_array_t;
  signal island_states      : island_state_array_t;

  -- Per-chiplet bridge signals: krafgpu_compute_chiplet's island_states /
  -- telemetry_out ports use the unconstrained companion types
  -- (island_state_array_u_t / telemetry_array_u_t) sized by its own
  -- NUM_ISLANDS_LOCAL generic, which GHDL treats as a genuinely distinct
  -- type from the full-chip island_state_array_t / telemetry_array_t above
  -- even though the element type and bounds line up. A slice of the
  -- full-chip signal can't be passed directly as an actual of the
  -- unconstrained-companion type, so each chiplet gets its own
  -- 8-island-wide signal of the companion type; concurrent assignments
  -- (which do permit qualified/type-converted aggregates, unlike port-map
  -- actuals) bridge the two on both directions.
  signal chiplet_a_islands  : island_state_array_u_t(0 to NUM_ISLANDS_LOCAL_PER_CHIPLET-1);
  signal chiplet_b_islands  : island_state_array_u_t(0 to NUM_ISLANDS_LOCAL_PER_CHIPLET-1);
  signal chiplet_a_telem    : telemetry_array_u_t(0 to NUM_ISLANDS_LOCAL_PER_CHIPLET-1);
  signal chiplet_b_telem    : telemetry_array_u_t(0 to NUM_ISLANDS_LOCAL_PER_CHIPLET-1);

  -- Chiplet A ↔ Interconnect fabric
  signal chiplet_a_req      : mem_req_t;
  signal chiplet_a_resp     : mem_resp_t;
  signal chiplet_a_ready    : std_logic;

  -- Chiplet B ↔ Interconnect fabric
  signal chiplet_b_req      : mem_req_t;
  signal chiplet_b_resp     : mem_resp_t;
  signal chiplet_b_ready    : std_logic;

  -- Interconnect ↔ Memory controller
  signal fabric_mem_req     : mem_req_t;
  signal fabric_mem_resp    : mem_resp_t;

  -- PCIe host command decoded
  signal host_cmd_valid     : std_logic;
  signal host_cmd_data      : std_logic_vector(255 downto 0);
  signal host_resp_valid    : std_logic;
  signal host_resp_data     : std_logic_vector(255 downto 0);

  -- L3 cache interconnect
  signal l3_req             : mem_req_t;
  signal l3_resp            : mem_resp_t;
  signal l3_hit             : std_logic;

  -- ===========================================================================
  -- AGGREGATE STATUS REGISTERS (clk_pmc domain — see status_* port comments)
  -- ===========================================================================
  signal status_power_mw_r      : unsigned(19 downto 0) := (others => '0');
  signal status_temp_c_r        : unsigned(7 downto 0)  := (others => '0');
  signal status_util_pct_r      : unsigned(6 downto 0)  := (others => '0');
  signal status_island_active_r : std_logic_vector(NUM_ISLANDS-1 downto 0)
                                 := (others => '1');

begin

  -- ===========================================================================
  -- PER-CHIPLET ISLAND / TELEMETRY BRIDGING
  -- See chiplet_a_islands / chiplet_b_islands declaration comment above for
  -- the type-mismatch rationale this works around.
  -- ===========================================================================
  gen_bridge_a : for i in 0 to NUM_ISLANDS_LOCAL_PER_CHIPLET-1 generate
    chiplet_a_islands(i) <= island_states(i);
    pmc_telemetry(i)     <= chiplet_a_telem(i);
  end generate;

  gen_bridge_b : for i in 0 to NUM_ISLANDS_LOCAL_PER_CHIPLET-1 generate
    chiplet_b_islands(i) <= island_states(NUM_ISLANDS_LOCAL_PER_CHIPLET + i);
    pmc_telemetry(NUM_ISLANDS_LOCAL_PER_CHIPLET + i) <= chiplet_b_telem(i);
  end generate;

  -- ===========================================================================
  -- COMPUTE CHIPLET A  (SMs 0..255, Islands 0..7)
  -- ===========================================================================
  u_chiplet_a : entity work.krafgpu_compute_chiplet
    generic map (
      CHIPLET_ID        => 0,
      SM_ID_BASE        => 0,
      ISLAND_ID_BASE    => 0,
      NUM_SMS           => 256,
      NUM_ISLANDS_LOCAL => NUM_ISLANDS_LOCAL_PER_CHIPLET
    )
    port map (
      clk               => clk_core_a,
      rst_n             => rst_n,
      island_states     => chiplet_a_islands,
      mem_req           => chiplet_a_req,
      mem_resp          => chiplet_a_resp,
      mem_ready         => chiplet_a_ready,
      telemetry_out     => chiplet_a_telem
    );

  -- ===========================================================================
  -- COMPUTE CHIPLET B  (SMs 256..511, Islands 8..15)
  -- ===========================================================================
  u_chiplet_b : entity work.krafgpu_compute_chiplet
    generic map (
      CHIPLET_ID        => 1,
      SM_ID_BASE        => 256,
      ISLAND_ID_BASE    => 8,
      NUM_SMS           => 256,
      NUM_ISLANDS_LOCAL => NUM_ISLANDS_LOCAL_PER_CHIPLET
    )
    port map (
      clk               => clk_core_b,
      rst_n             => rst_n,
      island_states     => chiplet_b_islands,
      mem_req           => chiplet_b_req,
      mem_resp          => chiplet_b_resp,
      mem_ready         => chiplet_b_ready,
      telemetry_out     => chiplet_b_telem
    );

  -- ===========================================================================
  -- POWER MANAGEMENT CONTROLLER
  -- ===========================================================================
  u_pmc : entity work.krafgpu_pmc
    port map (
      clk               => clk_pmc,
      rst_n             => rst_n,
      telemetry_in      => pmc_telemetry,
      island_states_out => island_states,
      pmic_pgood        => pmic_pgood,
      pmbus_scl         => pmbus_scl,
      pmbus_sda         => pmbus_sda,
      uart_tx           => uart_tx,
      uart_rx           => uart_rx
    );

  -- ===========================================================================
  -- DIE-TO-DIE INTERCONNECT FABRIC (SoIC-X 12 TB/s)
  -- ===========================================================================
  u_interconnect : entity work.krafgpu_interconnect_fabric
    port map (
      clk               => clk_core_a,
      rst_n             => rst_n,
      chiplet_a_req     => chiplet_a_req,
      chiplet_a_resp    => chiplet_a_resp,
      chiplet_a_ready   => chiplet_a_ready,
      chiplet_b_req     => chiplet_b_req,
      chiplet_b_resp    => chiplet_b_resp,
      chiplet_b_ready   => chiplet_b_ready,
      sys_req           => sys_mem_req,
      sys_resp          => sys_mem_resp,
      sys_ready         => sys_mem_ready,
      host_cmd_valid    => host_cmd_valid,
      host_cmd_data     => host_cmd_data,
      host_resp_valid   => host_resp_valid,
      host_resp_data    => host_resp_data,
      mem_req_out       => fabric_mem_req,
      mem_resp_in       => fabric_mem_resp,
      nvlink_tx         => nvlink_tx,
      nvlink_rx         => nvlink_rx,
      nvlink_valid_tx   => nvlink_valid_tx,
      nvlink_valid_rx   => nvlink_valid_rx
    );

  -- ===========================================================================
  -- L3 CACHE (1 GB shared across both chiplets)
  -- ===========================================================================
  u_l3_cache : entity work.krafgpu_l3_cache
    port map (
      clk               => clk_core_a,
      rst_n             => rst_n,
      req_in            => fabric_mem_req,
      resp_out          => fabric_mem_resp,
      hit_out           => l3_hit,
      mem_req_out       => l3_req,
      mem_resp_in       => l3_resp
    );

  -- ===========================================================================
  -- HBM4 MEMORY CONTROLLER
  -- ===========================================================================
  u_mem_ctrl : entity work.krafgpu_hbm4_ctrl
    port map (
      clk               => clk_mem,
      rst_n             => rst_n,
      req_in            => l3_req,
      resp_out          => l3_resp,
      hbm_addr          => hbm_addr,
      hbm_wdata         => hbm_wdata,
      hbm_rdata         => hbm_rdata,
      hbm_wr_en         => hbm_wr_en,
      hbm_rd_en         => hbm_rd_en,
      hbm_ack           => hbm_ack
    );

  -- ===========================================================================
  -- PCIe 6.0 HOST INTERFACE
  -- ===========================================================================
  u_pcie_if : entity work.krafgpu_pcie_if
    port map (
      clk               => clk_pcie,
      rst_n             => rst_n,
      pcie_rx_data      => pcie_rx_data,
      pcie_rx_valid     => pcie_rx_valid,
      pcie_rx_ready     => pcie_rx_ready,
      pcie_tx_data      => pcie_tx_data,
      pcie_tx_valid     => pcie_tx_valid,
      pcie_tx_ready     => pcie_tx_ready,
      cmd_valid_out     => host_cmd_valid,
      cmd_data_out      => host_cmd_data,
      resp_valid_in     => host_resp_valid,
      resp_data_in      => host_resp_data
    );

  -- ===========================================================================
  -- JTAG DEBUG (pass-through to ARM CPU embedded TAP)
  -- ===========================================================================
  u_jtag : entity work.krafgpu_jtag_tap
    port map (
      tck               => jtag_tck,
      tms               => jtag_tms,
      tdi               => jtag_tdi,
      tdo               => jtag_tdo
    );

  -- ps_ready: broadcast from all SMs (fully wired once krafgpu_compute_chiplet
  -- exposes per-SM ps_ready ports). Default all-ones means the ps_dispatch
  -- module can always find a "ready" SM, which is the correct architectural
  -- behaviour when the SM's ps_valid/ps_ready handshake is the only
  -- back-pressure mechanism.
  ps_ready <= (others => '1');

  -- ===========================================================================
  -- TELEMETRY AGGREGATION FOR BAR0 STATUS REGISTERS
  -- Sums power across all 16 islands, finds the hottest island, averages
  -- utilisation, and reports which islands are not power-gated.
  -- ===========================================================================
  p_status_aggregate : process(clk_pmc)
    variable pwr_sum  : unsigned(19 downto 0);
    variable temp_max : unsigned(7 downto 0);
    variable util_sum : unsigned(10 downto 0);  -- 16 x 7-bit max
  begin
    if rising_edge(clk_pmc) then
      if rst_n = '0' then
        status_power_mw_r      <= (others => '0');
        status_temp_c_r        <= (others => '0');
        status_util_pct_r      <= (others => '0');
        status_island_active_r <= (others => '1');
      else
        pwr_sum  := (others => '0');
        temp_max := (others => '0');
        util_sum := (others => '0');

        for i in 0 to NUM_ISLANDS-1 loop
          pwr_sum := pwr_sum + resize(pmc_telemetry(i).power_mw, 20);
          if pmc_telemetry(i).temperature > temp_max then
            temp_max := pmc_telemetry(i).temperature;
          end if;
          util_sum := util_sum + resize(pmc_telemetry(i).utilisation, 11);

          status_island_active_r(i) <=
            '0' when island_states(i) = ISLAND_OFF else '1';
        end loop;

        status_power_mw_r <= pwr_sum;
        status_temp_c_r   <= temp_max;
        -- Mean utilisation = sum / 16 (shift right by 4)
        status_util_pct_r <= util_sum(10 downto 4);
      end if;
    end if;
  end process p_status_aggregate;

  status_power_mw      <= status_power_mw_r;
  status_temp_c        <= status_temp_c_r;
  status_util_pct      <= status_util_pct_r;
  status_island_active <= status_island_active_r;

  -- ECC: stub driving '0' until L2 chiplet ECC ports are surfaced.
  -- L3 ECC from the memory subsystem would be wired here once the
  -- krafgpu_l3_cache ecc_single_err/ecc_double_err ports are propagated
  -- up through the interconnect fabric entity's port list.
  ecc_single_err <= '0';
  ecc_double_err <= '0';

end architecture structural;
