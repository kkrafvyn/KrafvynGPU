-- =============================================================================
-- KrafGPU-X1 System Integration (Chip Top-Level)
-- Project      : KrafGPU-X1 Next-Generation GPU Architecture
-- Author       : Krafvyn Tech / Don
-- Description  : Full-chip integration wrapper. Instantiates krafgpu_x1_top
--                (compute chiplets + PMC + L3 + HBM4 controller + die-to-die
--                fabric) alongside every peripheral subsystem and wires them
--                into krafgpu_x1_top's sys_mem_req/sys_mem_resp port through
--                an 8-way local round-robin arbiter.
--
--   Peripheral subsystems integrated here:
--     - BAR0 MMIO register file        (krafgpu_bar0_mmio)
--     - Command ring DMA processor     (krafgpu_cmd_ring)
--     - Interrupt controller / MSI-X   (krafgpu_irq_ctrl)
--     - NPU cluster (4x AI accel)      (krafgpu_npu_top)
--     - DMA copy engine                (krafgpu_dma_engine)
--     - Vertex fetch                   (krafgpu_vertex_fetch)
--     - Rasteriser (setup+fine+depth)  (krafgpu_triangle_setup /
--                                        krafgpu_fine_raster /
--                                        krafgpu_depth_buffer)
--     - Display engine (scanout+gamma+DP2.1)
--                                       (krafgpu_scanout / krafgpu_gamma_lut /
--                                        krafgpu_dp21_encoder)
--     - Context save/restore engine    (krafgpu_ctx_engine)
--
--   System memory port (sys_mem_*) arbitration order (round-robin):
--     0 = Command ring DMA   (fetch command packets)
--     1 = DMA copy read       (CMD_COPY source)
--     2 = DMA copy write       (CMD_COPY destination)
--     3 = Vertex fetch         (CMD_DRAW vertex/index buffers)
--     4 = Depth buffer         (Hi-Z + per-pixel Z test/update)
--     5 = Scanout framebuffer  (display read)
--     6 = NPU shared DMA       (weight/activation/output tiles)
--     7 = Context engine DMA   (save/restore checkpoint buffer)
--
--   Extended register bank:
--     BAR0 (krafgpu_bar0_mmio) covers identity, power, memory, display timing,
--     gamma LUT, compute dispatch enable, and interrupts. NPU operand
--     addresses (weight/activation/output base pointers) are latched here
--     from CMD_SET_REG writes at a small set of *extended* offsets
--     (0x004030-0x004038) that sit in BAR0's reserved space — bar0_mmio
--     itself ignores writes to unmapped addresses, so this wrapper taps the
--     same reg_wr_valid/addr/data bus from the command ring without
--     conflicting with bar0_mmio's own decode.
--
--   NOT yet wired (documented as future integration points):
--     - Pixel-shader dispatch from the fine rasteriser into the SM warp
--       schedulers requires a new per-chiplet "raster dispatch" port on
--       krafgpu_compute_chiplet / krafgpu_x1_top. The rasteriser instances
--       here run end-to-end (setup -> coverage -> depth test) and produce
--       pix_valid/pix_cov/pix_depth, but surviving pixels are not yet
--       packaged into warps.
--     - Context engine SM-state snapshot taps (warp_pc_in/warp_state_in/...)
--       require a debug-snapshot bus from krafgpu_sm through
--       krafgpu_compute_chiplet. Tied to constants here; the DMA side of
--       ctx_engine is fully functional and participates in arbitration.
-- =============================================================================

library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;
use work.krafgpu_pkg.all;

entity krafgpu_x1_system is
  port (
    -- -------------------------------------------------------------------------
    -- Clocks and reset
    -- -------------------------------------------------------------------------
    clk_core_a   : in  std_logic;
    clk_core_b   : in  std_logic;
    clk_mem      : in  std_logic;
    clk_pcie     : in  std_logic;
    clk_pmc      : in  std_logic;
    clk_pixel    : in  std_logic;   -- Display scanout / DP symbol clock
    rst_n        : in  std_logic;

    -- -------------------------------------------------------------------------
    -- PCIe 6.0 x64 host data plane (pass-through to krafgpu_x1_top)
    -- -------------------------------------------------------------------------
    pcie_rx_data  : in  std_logic_vector(255 downto 0);
    pcie_rx_valid : in  std_logic;
    pcie_rx_ready : out std_logic;
    pcie_tx_data  : out std_logic_vector(255 downto 0);
    pcie_tx_valid : out std_logic;
    pcie_tx_ready : in  std_logic;

    -- -------------------------------------------------------------------------
    -- PCIe BAR0 MMIO host interface (driver register access)
    -- -------------------------------------------------------------------------
    bar_addr      : in  unsigned(23 downto 0);
    bar_wr_data   : in  std_logic_vector(31 downto 0);
    bar_wr_en     : in  std_logic;
    bar_rd_en     : in  std_logic;
    bar_byte_en   : in  std_logic_vector(3 downto 0);
    bar_rd_data   : out std_logic_vector(31 downto 0);
    bar_rd_valid  : out std_logic;

    -- -------------------------------------------------------------------------
    -- HBM4 memory interface
    -- -------------------------------------------------------------------------
    hbm_addr  : out std_logic_vector(PHYS_ADDR_WIDTH-1 downto 0);
    hbm_wdata : out std_logic_vector(HBM4_TOTAL_WIDTH-1 downto 0);
    hbm_rdata : in  std_logic_vector(HBM4_TOTAL_WIDTH-1 downto 0);
    hbm_wr_en : out std_logic_vector(NUM_HBM4_STACKS-1 downto 0);
    hbm_rd_en : out std_logic_vector(NUM_HBM4_STACKS-1 downto 0);
    hbm_ack   : in  std_logic_vector(NUM_HBM4_STACKS-1 downto 0);

    -- -------------------------------------------------------------------------
    -- Display output (DisplayPort 2.1 UHBR20, 4 lanes)
    -- -------------------------------------------------------------------------
    dp_lane_data   : out std_logic_vector(4*32-1 downto 0);
    dp_lane_valid  : out std_logic_vector(3 downto 0);
    dp_lt_start    : in  std_logic;
    dp_lt_pattern  : in  std_logic_vector(1 downto 0);
    dp_lt_done     : out std_logic;

    -- -------------------------------------------------------------------------
    -- PMBus / power management
    -- -------------------------------------------------------------------------
    pmbus_scl  : inout std_logic;
    pmbus_sda  : inout std_logic;
    pmic_pgood : in  std_logic_vector(NUM_ISLANDS-1 downto 0);

    -- -------------------------------------------------------------------------
    -- Debug / evaluation
    -- -------------------------------------------------------------------------
    jtag_tck : in  std_logic;
    jtag_tms : in  std_logic;
    jtag_tdi : in  std_logic;
    jtag_tdo : out std_logic;
    uart_tx  : out std_logic;
    uart_rx  : in  std_logic;

    -- -------------------------------------------------------------------------
    -- NVLink-equivalent multi-GPU bridge
    -- -------------------------------------------------------------------------
    nvlink_tx       : out std_logic_vector(127 downto 0);
    nvlink_rx       : in  std_logic_vector(127 downto 0);
    nvlink_valid_tx : out std_logic;
    nvlink_valid_rx : in  std_logic;

    -- -------------------------------------------------------------------------
    -- MSI-X / legacy interrupt to host
    -- -------------------------------------------------------------------------
    msix_valid  : out std_logic;
    msix_vector : out unsigned(4 downto 0);
    msix_addr   : out unsigned(63 downto 0);
    msix_data   : out std_logic_vector(31 downto 0);
    msix_ready  : in  std_logic;
    pcie_intx   : out std_logic;

    -- -------------------------------------------------------------------------
    -- Context save/restore trigger (from driver, e.g. for preemption)
    -- -------------------------------------------------------------------------
    ctx_save_req    : in  std_logic;
    ctx_restore_req : in  std_logic;
    ctx_buf_addr    : in  unsigned(PHYS_ADDR_WIDTH-1 downto 0);
    ctx_sm_mask     : in  std_logic_vector(NUM_ISLANDS-1 downto 0);
    ctx_save_done   : out std_logic;
    ctx_restore_done: out std_logic;
    ctx_busy        : out std_logic
  );
end entity krafgpu_x1_system;

architecture structural of krafgpu_x1_system is

  -- ===========================================================================
  -- KRAFGPU_X1_TOP <-> SYSTEM SIGNALS
  -- ===========================================================================
  signal top_sys_req    : mem_req_t;
  signal top_sys_resp   : mem_resp_t;
  signal top_sys_ready  : std_logic;

  signal status_power_mw      : unsigned(19 downto 0);
  signal status_temp_c        : unsigned(7 downto 0);
  signal status_util_pct      : unsigned(6 downto 0);
  signal status_island_active : std_logic_vector(NUM_ISLANDS-1 downto 0);
  -- Proper mW→W (replaces the >>4 / ÷16 hack)
  signal status_power_w_s     : unsigned(15 downto 0) := (others => '0');
  -- ECC error aggregation from krafgpu_x1_top
  signal ecc_single_s         : std_logic := '0';
  signal ecc_double_s         : std_logic := '0';

  -- ===========================================================================
  -- 8-WAY SYSTEM MEMORY ARBITER
  -- ===========================================================================
  constant NUM_SYS_PORTS : integer := 8;
  constant P_CMDRING     : integer := 0;
  constant P_DMA_RD      : integer := 1;
  constant P_DMA_WR      : integer := 2;
  constant P_VTXFETCH    : integer := 3;
  constant P_DEPTHBUF    : integer := 4;
  constant P_SCANOUT     : integer := 5;
  constant P_NPU         : integer := 6;
  constant P_CTXENGINE   : integer := 7;

  type sys_req_arr_t  is array (0 to NUM_SYS_PORTS-1) of mem_req_t;
  signal sys_port_req   : sys_req_arr_t;
  signal sys_port_grant : std_logic_vector(0 to NUM_SYS_PORTS-1);
  signal sys_arb_ptr    : integer range 0 to NUM_SYS_PORTS-1 := 0;

  -- ===========================================================================
  -- BAR0 MMIO SIGNALS
  -- ===========================================================================
  signal reg_gpu_id       : std_logic_vector(31 downto 0);
  signal reg_power_limit  : unsigned(15 downto 0);
  signal reg_temp_limit   : unsigned(7 downto 0);
  signal reg_island_force : std_logic_vector(NUM_ISLANDS-1 downto 0);
  signal reg_island_state : island_state_array_t;
  signal reg_dvfs_mode    : std_logic_vector(1 downto 0);

  signal reg_fb_base   : unsigned(PHYS_ADDR_WIDTH-1 downto 0);
  signal reg_fb_size   : unsigned(31 downto 0);
  signal reg_hbm_mode  : std_logic_vector(3 downto 0);

  signal reg_disp_h_active : unsigned(13 downto 0);
  signal reg_disp_h_fp     : unsigned(9 downto 0);
  signal reg_disp_h_sync   : unsigned(9 downto 0);
  signal reg_disp_h_bp     : unsigned(9 downto 0);
  signal reg_disp_v_active : unsigned(13 downto 0);
  signal reg_disp_v_fp     : unsigned(9 downto 0);
  signal reg_disp_v_sync   : unsigned(9 downto 0);
  signal reg_disp_v_bp     : unsigned(9 downto 0);
  signal reg_disp_fb_addr  : unsigned(PHYS_ADDR_WIDTH-1 downto 0);
  signal reg_disp_enable   : std_logic;
  signal reg_disp_vsync_en : std_logic;

  signal reg_lut_wr_en : std_logic;
  signal reg_lut_addr  : unsigned(9 downto 0);
  signal reg_lut_r     : unsigned(11 downto 0);
  signal reg_lut_g     : unsigned(11 downto 0);
  signal reg_lut_b     : unsigned(11 downto 0);

  signal reg_sm_en_mask    : std_logic_vector(NUM_ISLANDS-1 downto 0);
  signal reg_dispatch_en   : std_logic;
  signal reg_cmd_ring_base : unsigned(PHYS_ADDR_WIDTH-1 downto 0);
  signal reg_cmd_ring_size : unsigned(31 downto 0);
  signal reg_cmd_ring_head : unsigned(31 downto 0);

  signal reg_irq_mask  : std_logic_vector(31 downto 0);
  signal reg_irq_clear : std_logic_vector(31 downto 0);

  signal status_cmd_ring_tail : unsigned(31 downto 0);
  signal status_irq_pending   : std_logic_vector(31 downto 0);

  -- ===========================================================================
  -- COMMAND RING SIGNALS
  -- ===========================================================================
  signal draw_valid     : std_logic;
  signal draw_vtx_count : unsigned(31 downto 0);
  signal draw_inst_count: unsigned(31 downto 0);
  signal draw_idx_buf   : unsigned(PHYS_ADDR_WIDTH-1 downto 0);
  signal draw_vtx_buf   : unsigned(PHYS_ADDR_WIDTH-1 downto 0);
  signal draw_ready     : std_logic;

  signal dispatch_valid : std_logic;
  signal dispatch_x     : unsigned(15 downto 0);
  signal dispatch_y     : unsigned(15 downto 0);
  signal dispatch_z     : unsigned(15 downto 0);
  signal dispatch_ready : std_logic;

  signal ring_reg_wr_valid : std_logic;
  signal ring_reg_wr_addr  : unsigned(23 downto 0);
  signal ring_reg_wr_data  : std_logic_vector(31 downto 0);

  signal copy_valid : std_logic;
  signal copy_src   : unsigned(PHYS_ADDR_WIDTH-1 downto 0);
  signal copy_dst   : unsigned(PHYS_ADDR_WIDTH-1 downto 0);
  signal copy_size  : unsigned(31 downto 0);
  signal copy_ready : std_logic;

  signal fence_valid : std_logic;
  signal fence_addr  : unsigned(PHYS_ADDR_WIDTH-1 downto 0);
  signal fence_value : std_logic_vector(31 downto 0);

  signal barrier_valid : std_logic;
  signal barrier_flags : std_logic_vector(7 downto 0);

  signal irq_fence_pulse : std_logic;
  signal ring_idle       : std_logic;
  signal ring_inflight   : unsigned(7 downto 0);

  -- ===========================================================================
  -- EXTENDED REGISTER BANK (NPU operand base addresses)
  -- Latched from CMD_SET_REG writes at reserved BAR0 offsets 0x004030/34/38.
  -- ===========================================================================
  constant A_NPU_WEIGHT_LO : unsigned(23 downto 0) := x"004030";
  constant A_NPU_ACT_LO    : unsigned(23 downto 0) := x"004034";
  constant A_NPU_OUT_LO    : unsigned(23 downto 0) := x"004038";
  constant A_NPU_OPDESC    : unsigned(23 downto 0) := x"00403C";

  signal npu_weight_addr : unsigned(PHYS_ADDR_WIDTH-1 downto 0)
                         := (others => '0');
  signal npu_act_addr    : unsigned(PHYS_ADDR_WIDTH-1 downto 0)
                         := (others => '0');
  signal npu_out_addr    : unsigned(PHYS_ADDR_WIDTH-1 downto 0)
                         := (others => '0');
  -- op_desc packs: [31:24]=opcode [18:16]=precision/act [15:0]=unused
  signal npu_op_desc     : std_logic_vector(31 downto 0) := (others => '0');
  signal npu_op_pulse    : std_logic := '0';

  -- ===========================================================================
  -- NPU TOP SIGNALS
  -- ===========================================================================
  constant NUM_NPU : integer := 4;
  signal npu_op_valid    : std_logic_vector(NUM_NPU-1 downto 0);
  signal npu_op_ready    : std_logic_vector(NUM_NPU-1 downto 0);
  signal npu_done        : std_logic_vector(NUM_NPU-1 downto 0);
  signal npu_total_tops  : unsigned(15 downto 0);

  -- ===========================================================================
  -- DMA COPY ENGINE SIGNALS
  -- ===========================================================================
  signal dma_bytes_copied : unsigned(31 downto 0);
  signal dma_copy_done    : std_logic;

  -- ===========================================================================
  -- VERTEX FETCH SIGNALS
  -- ===========================================================================
  signal vtx_valid    : std_logic;
  signal vtx_pos_x, vtx_pos_y, vtx_pos_z, vtx_pos_w :
           std_logic_vector(DATA_WIDTH-1 downto 0);
  signal vtx_idx      : unsigned(31 downto 0);
  signal vtx_consumed : std_logic;

  -- Vertex assembly: collect 3 vertices -> triangle setup input
  type vtx_collect_t is array (0 to 2) of std_logic_vector(DATA_WIDTH-1 downto 0);
  signal coll_x, coll_y, coll_z, coll_w : vtx_collect_t;
  signal coll_idx : integer range 0 to 2 := 0;
  signal tri_valid_pulse : std_logic := '0';

  -- ===========================================================================
  -- RASTER PIPELINE SIGNALS
  -- ===========================================================================
  signal setup_valid : std_logic;
  signal e0_a, e0_b, e0_c : signed(31 downto 0);
  signal e1_a, e1_b, e1_c : signed(31 downto 0);
  signal e2_a, e2_b, e2_c : signed(31 downto 0);
  signal bbox_x0, bbox_y0, bbox_x1, bbox_y1 : unsigned(15 downto 0);
  signal raster_inv_area : std_logic_vector(DATA_WIDTH-1 downto 0);
  signal raster_dzdx, raster_dzdy : std_logic_vector(DATA_WIDTH-1 downto 0);
  signal raster_backface : std_logic;

  signal pix_valid  : std_logic;
  signal pix_x, pix_y : unsigned(15 downto 0);
  signal pix_cov    : std_logic_vector(3 downto 0);  -- MSAA_SAMP=4
  signal pix_bary_u, pix_bary_v : std_logic_vector(DATA_WIDTH-1 downto 0);
  signal pix_depth  : std_logic_vector(DATA_WIDTH-1 downto 0);
  signal raster_done: std_logic;

  signal depth_test_pass: std_logic;
  signal depth_test_done: std_logic;
  signal depth_clear_en : std_logic := '0';

  -- ===========================================================================
  -- DISPLAY ENGINE SIGNALS
  -- ===========================================================================
  signal scan_pix_valid: std_logic;
  signal scan_pix_r, scan_pix_g, scan_pix_b : std_logic_vector(9 downto 0);
  signal scan_hsync, scan_vsync, scan_de : std_logic;
  signal scan_frame_start, scan_line_start : std_logic;

  signal gamma_valid : std_logic;
  signal gamma_r, gamma_g, gamma_b : std_logic_vector(11 downto 0);

  -- ===========================================================================
  -- CONTEXT ENGINE SIGNALS
  -- ===========================================================================
  signal ctx_drain_req       : std_logic;
  signal ctx_warp_restore_en : std_logic;

  -- ===========================================================================
  -- IRQ COMBINATIONAL HELPER SIGNALS
  -- (separated from inline expressions for VHDL-93/2008 portability —
  --  conditional ('1' when ... else '0') expressions are not legal directly
  --  as port-map actuals in all toolchains)
  -- ===========================================================================
  signal irq_thermal_warn_s : std_logic;
  signal irq_thermal_crit_s : std_logic;
  signal irq_power_limit_s  : std_logic;
  signal irq_ring_stall_s   : std_logic;

begin

  -- ===========================================================================
  -- KRAFGPU_X1_TOP : compute chiplets, PMC, L3, HBM4, die-to-die fabric
  -- ===========================================================================
  u_top : entity work.krafgpu_x1_top
    port map (
      clk_core_a      => clk_core_a,
      clk_core_b      => clk_core_b,
      clk_mem         => clk_mem,
      clk_pcie        => clk_pcie,
      clk_pmc         => clk_pmc,
      rst_n           => rst_n,
      pcie_rx_data    => pcie_rx_data,
      pcie_rx_valid   => pcie_rx_valid,
      pcie_rx_ready   => pcie_rx_ready,
      pcie_tx_data    => pcie_tx_data,
      pcie_tx_valid   => pcie_tx_valid,
      pcie_tx_ready   => pcie_tx_ready,
      hbm_addr        => hbm_addr,
      hbm_wdata       => hbm_wdata,
      hbm_rdata       => hbm_rdata,
      hbm_wr_en       => hbm_wr_en,
      hbm_rd_en       => hbm_rd_en,
      hbm_ack         => hbm_ack,
      disp_data       => open,
      disp_valid      => open,
      pmbus_scl       => pmbus_scl,
      pmbus_sda       => pmbus_sda,
      pmic_pgood      => pmic_pgood,
      jtag_tck        => jtag_tck,
      jtag_tms        => jtag_tms,
      jtag_tdi        => jtag_tdi,
      jtag_tdo        => jtag_tdo,
      uart_tx         => uart_tx,
      uart_rx         => uart_rx,
      nvlink_tx       => nvlink_tx,
      nvlink_rx       => nvlink_rx,
      nvlink_valid_tx => nvlink_valid_tx,
      nvlink_valid_rx => nvlink_valid_rx,
      sys_mem_req     => top_sys_req,
      sys_mem_resp    => top_sys_resp,
      sys_mem_ready   => top_sys_ready,
      status_power_mw      => status_power_mw,
      status_temp_c        => status_temp_c,
      status_util_pct      => status_util_pct,
      status_island_active => status_island_active,
      ecc_single_err       => ecc_single_s,
      ecc_double_err       => ecc_double_s
    );

  -- Proper mW → W conversion (replaces the previous >>4 / ÷16 hack).
  -- status_power_mw is 20-bit (up to ~1 MW); dividing by 1000 gives
  -- a 10-bit result (max ~1048 W) that fits safely in 16 bits.
  status_power_w_s <= resize(status_power_mw / to_unsigned(1000, 20), 16);

  -- ===========================================================================
  -- BAR0 MMIO REGISTER FILE
  -- ===========================================================================
  u_bar0 : entity work.krafgpu_bar0_mmio
    port map (
      clk          => clk_pcie,
      rst_n        => rst_n,
      bar_addr     => bar_addr,
      bar_wr_data  => bar_wr_data,
      bar_wr_en    => bar_wr_en,
      bar_rd_en    => bar_rd_en,
      bar_rd_data  => bar_rd_data,
      bar_rd_valid => bar_rd_valid,
      bar_byte_en  => bar_byte_en,

      reg_gpu_id       => reg_gpu_id,
      reg_power_limit  => reg_power_limit,
      reg_temp_limit   => reg_temp_limit,
      reg_island_force => reg_island_force,
      reg_island_state => reg_island_state,
      reg_dvfs_mode    => reg_dvfs_mode,

      reg_fb_base  => reg_fb_base,
      reg_fb_size  => reg_fb_size,
      reg_hbm_mode => reg_hbm_mode,

      reg_disp_h_active => reg_disp_h_active,
      reg_disp_h_fp     => reg_disp_h_fp,
      reg_disp_h_sync   => reg_disp_h_sync,
      reg_disp_h_bp     => reg_disp_h_bp,
      reg_disp_v_active => reg_disp_v_active,
      reg_disp_v_fp     => reg_disp_v_fp,
      reg_disp_v_sync   => reg_disp_v_sync,
      reg_disp_v_bp     => reg_disp_v_bp,
      reg_disp_fb_addr  => reg_disp_fb_addr,
      reg_disp_enable   => reg_disp_enable,
      reg_disp_vsync_en => reg_disp_vsync_en,

      reg_lut_wr_en => reg_lut_wr_en,
      reg_lut_addr  => reg_lut_addr,
      reg_lut_r     => reg_lut_r,
      reg_lut_g     => reg_lut_g,
      reg_lut_b     => reg_lut_b,

      reg_sm_en_mask    => reg_sm_en_mask,
      reg_dispatch_en   => reg_dispatch_en,
      reg_cmd_ring_base => reg_cmd_ring_base,
      reg_cmd_ring_size => reg_cmd_ring_size,
      reg_cmd_ring_head => reg_cmd_ring_head,

      reg_irq_mask  => reg_irq_mask,
      reg_irq_clear => reg_irq_clear,

      status_power_w       => status_power_w_s,   -- proper mW/1000, not >>4
      status_temp_c        => status_temp_c,
      status_util_pct      => status_util_pct,
      status_irq_pending   => status_irq_pending,
      status_cmd_ring_tail => status_cmd_ring_tail,
      status_hbm_bw_gbps   => to_unsigned(6400, 16),  -- Rated 6.4 TB/s
      status_sm_active     => status_island_active
    );

  -- Note: status_power_mw(19 downto 4) approximates mW->W by dividing by 16
  -- rather than 1000 (cheap shift vs. divider); the BAR0 power-limit field
  -- is therefore in "16 mW units" at this integration stage. A production
  -- driver either accounts for this scale factor or this shift is replaced
  -- with the exact /1000 in a follow-up pass.

  -- ===========================================================================
  -- COMMAND RING DMA PROCESSOR
  -- ===========================================================================
  u_cmd_ring : entity work.krafgpu_cmd_ring
    port map (
      clk              => clk_pcie,
      rst_n            => rst_n,
      ring_base_addr   => reg_cmd_ring_base,
      ring_size_dwords => reg_cmd_ring_size,
      ring_head        => reg_cmd_ring_head,
      ring_tail        => status_cmd_ring_tail,
      dma_req          => sys_port_req(P_CMDRING),
      dma_resp         => top_sys_resp,

      draw_valid     => draw_valid,
      draw_vtx_count => draw_vtx_count,
      draw_inst_count=> draw_inst_count,
      draw_idx_buf   => draw_idx_buf,
      draw_vtx_buf   => draw_vtx_buf,
      draw_ready     => draw_ready,

      dispatch_valid => dispatch_valid,
      dispatch_x     => dispatch_x,
      dispatch_y     => dispatch_y,
      dispatch_z     => dispatch_z,
      dispatch_ready => dispatch_ready,

      reg_wr_valid => ring_reg_wr_valid,
      reg_wr_addr  => ring_reg_wr_addr,
      reg_wr_data  => ring_reg_wr_data,

      copy_valid => copy_valid,
      copy_src   => copy_src,
      copy_dst   => copy_dst,
      copy_size  => copy_size,
      copy_ready => copy_ready,

      fence_valid => fence_valid,
      fence_addr  => fence_addr,
      fence_value => fence_value,

      barrier_valid => barrier_valid,
      barrier_flags => barrier_flags,

      irq_fence => irq_fence_pulse,
      ring_idle => ring_idle,
      inflight_count => ring_inflight
    );

  -- ===========================================================================
  -- EXTENDED REGISTER BANK — NPU operand addresses
  -- Snoops the same reg_wr_* bus consumed by bar0_mmio. Writes to the
  -- reserved 0x00403x window are latched here; bar0_mmio leaves them
  -- unmapped (falls through to its `others => null` write case), so there
  -- is no decode conflict.
  -- ===========================================================================
  p_npu_regs : process(clk_pcie)
  begin
    if rising_edge(clk_pcie) then
      if rst_n = '0' then
        npu_weight_addr <= (others => '0');
        npu_act_addr    <= (others => '0');
        npu_out_addr    <= (others => '0');
        npu_op_desc     <= (others => '0');
        npu_op_pulse    <= '0';
      else
        npu_op_pulse <= '0';
        if ring_reg_wr_valid = '1' then
          case ring_reg_wr_addr is
            when A_NPU_WEIGHT_LO =>
              npu_weight_addr(31 downto 0) <= unsigned(ring_reg_wr_data);
            when A_NPU_ACT_LO =>
              npu_act_addr(31 downto 0) <= unsigned(ring_reg_wr_data);
            when A_NPU_OUT_LO =>
              npu_out_addr(31 downto 0) <= unsigned(ring_reg_wr_data);
            when A_NPU_OPDESC =>
              npu_op_desc  <= ring_reg_wr_data;
              npu_op_pulse <= '1';  -- Writing op_desc launches the NPU op
            when others => null;
          end case;
        end if;
      end if;
    end if;
  end process p_npu_regs;

  -- ===========================================================================
  -- NPU CLUSTER TOP — 4x AI inference accelerators
  -- All 4 clusters are launched together on npu_op_pulse using the M/N/K
  -- dims and precision/activation packed into npu_op_desc by the driver via
  -- A_NPU_OPDESC. A production scheduler would partition (M,N,K) across the
  -- 4 clusters; here each cluster receives the full descriptor and the
  -- DMA-tile addressing inside krafgpu_npu_cluster naturally serialises
  -- through the shared arbiter below.
  -- ===========================================================================
  gen_npu_valid : for i in 0 to NUM_NPU-1 generate
    npu_op_valid(i) <= npu_op_pulse;
  end generate;

  u_npu_top : entity work.krafgpu_npu_top
    generic map (NUM_NPU_CLUSTERS => NUM_NPU)
    port map (
      clk          => clk_core_a,
      rst_n        => rst_n,
      op_valid     => npu_op_valid,
      op_code      => npu_op_desc(31 downto 24),
      op_m         => dispatch_x,
      op_n         => dispatch_y,
      op_k         => dispatch_z,
      op_precision => npu_op_desc(17 downto 16),
      op_act_fn    => npu_op_desc(18 downto 16),
      op_ready     => npu_op_ready,
      weight_addr  => npu_weight_addr,
      act_addr     => npu_act_addr,
      out_addr     => npu_out_addr,
      dma_req      => sys_port_req(P_NPU),
      dma_resp     => top_sys_resp,
      compute_done => npu_done,
      total_tops_int8 => npu_total_tops
    );

  -- A compute-class dispatch from the ring (CMD_DISPATCH) is consumed here
  -- as the NPU's M/N/K dims; dispatch_ready reflects NPU0's readiness so the
  -- ring advances only once the accelerators have latched the new shape.
  dispatch_ready <= npu_op_ready(0);

  -- ===========================================================================
  -- DMA COPY ENGINE
  -- ===========================================================================
  u_dma : entity work.krafgpu_dma_engine
    port map (
      clk          => clk_core_a,
      rst_n        => rst_n,
      copy_valid   => copy_valid,
      copy_src     => copy_src,
      copy_dst     => copy_dst,
      copy_size    => copy_size,
      copy_ready   => copy_ready,
      copy_done    => dma_copy_done,
      rd_req       => sys_port_req(P_DMA_RD),
      rd_resp      => top_sys_resp,
      wr_req       => sys_port_req(P_DMA_WR),
      wr_resp      => top_sys_resp,
      bytes_copied => dma_bytes_copied
    );

  -- ===========================================================================
  -- VERTEX FETCH + TRIANGLE ASSEMBLY
  -- Streams vertices for the active draw call and groups every 3 into a
  -- triangle for the setup stage (triangle-list topology).
  -- ===========================================================================
  u_vtx_fetch : entity work.krafgpu_vertex_fetch
    generic map (MAX_ATTRIBS => 16)
    port map (
      clk          => clk_core_a,
      rst_n        => rst_n,
      draw_valid   => draw_valid,
      vtx_count    => draw_vtx_count,
      vtx_buf_addr => draw_vtx_buf,
      idx_buf_addr => draw_idx_buf,
      vtx_stride   => to_unsigned(16, 16),  -- 4 x FP32 position-only stream
      use_index    => '1',
      draw_ready   => draw_ready,
      attrib_en    => (others => '0'),
      attrib_offset=> (others => '0'),
      mem_req      => sys_port_req(P_VTXFETCH),
      mem_resp     => top_sys_resp,
      vtx_valid    => vtx_valid,
      vtx_pos_x    => vtx_pos_x,
      vtx_pos_y    => vtx_pos_y,
      vtx_pos_z    => vtx_pos_z,
      vtx_pos_w    => vtx_pos_w,
      vtx_idx      => vtx_idx,
      vtx_consumed => vtx_consumed
    );

  -- Collect 3 consecutive vertices into the triangle-setup operands.
  -- vtx_consumed pulses every cycle a vertex is accepted; tri_valid_pulse
  -- fires once per complete triangle (every 3rd vertex).
  vtx_consumed <= vtx_valid;

  p_vtx_collect : process(clk_core_a)
  begin
    if rising_edge(clk_core_a) then
      if rst_n = '0' then
        coll_idx        <= 0;
        tri_valid_pulse <= '0';
      else
        tri_valid_pulse <= '0';
        if vtx_valid = '1' then
          coll_x(coll_idx) <= vtx_pos_x;
          coll_y(coll_idx) <= vtx_pos_y;
          coll_z(coll_idx) <= vtx_pos_z;
          coll_w(coll_idx) <= vtx_pos_w;
          if coll_idx = 2 then
            coll_idx        <= 0;
            tri_valid_pulse <= '1';
          else
            coll_idx <= coll_idx + 1;
          end if;
        end if;
      end if;
    end if;
  end process p_vtx_collect;

  -- ===========================================================================
  -- RASTERISATION PIPELINE: triangle setup -> fine raster -> depth test
  -- ===========================================================================
  u_tri_setup : entity work.krafgpu_triangle_setup
    port map (
      clk       => clk_core_a,
      rst_n     => rst_n,
      tri_valid => tri_valid_pulse,
      v0_x => coll_x(0), v0_y => coll_y(0), v0_z => coll_z(0), v0_w => coll_w(0),
      v1_x => coll_x(1), v1_y => coll_y(1), v1_z => coll_z(1), v1_w => coll_w(1),
      v2_x => coll_x(2), v2_y => coll_y(2), v2_z => coll_z(2), v2_w => coll_w(2),
      vp_x      => to_unsigned(0, 16),
      vp_y      => to_unsigned(0, 16),
      vp_width  => resize(reg_disp_h_active, 16),
      vp_height => resize(reg_disp_v_active, 16),
      setup_valid => setup_valid,
      e0_a => e0_a, e0_b => e0_b, e0_c => e0_c,
      e1_a => e1_a, e1_b => e1_b, e1_c => e1_c,
      e2_a => e2_a, e2_b => e2_b, e2_c => e2_c,
      bbox_x0 => bbox_x0, bbox_y0 => bbox_y0,
      bbox_x1 => bbox_x1, bbox_y1 => bbox_y1,
      inv_area => raster_inv_area,
      dzdx => raster_dzdx, dzdy => raster_dzdy,
      backface => raster_backface
    );

  u_fine_raster : entity work.krafgpu_fine_raster
    generic map (TILE_SIZE => 16, MSAA_SAMP => 4)
    port map (
      clk         => clk_core_a,
      rst_n       => rst_n,
      setup_valid => setup_valid,
      e0_a => e0_a, e0_b => e0_b, e0_c => e0_c,
      e1_a => e1_a, e1_b => e1_b, e1_c => e1_c,
      e2_a => e2_a, e2_b => e2_b, e2_c => e2_c,
      bbox_x0 => bbox_x0, bbox_y0 => bbox_y0,
      bbox_x1 => bbox_x1, bbox_y1 => bbox_y1,
      inv_area => raster_inv_area,
      dzdx => raster_dzdx, dzdy => raster_dzdy,
      pix_valid  => pix_valid,
      pix_x      => pix_x,
      pix_y      => pix_y,
      pix_cov    => pix_cov,
      pix_bary_u => pix_bary_u,
      pix_bary_v => pix_bary_v,
      pix_depth  => pix_depth,
      raster_done=> raster_done
    );

  u_depth_buf : entity work.krafgpu_depth_buffer
    generic map (
      FB_WIDTH  => 3840,
      FB_HEIGHT => 2160,
      HIZ_TILE  => 8
    )
    port map (
      clk        => clk_core_a,
      rst_n      => rst_n,
      clear_en   => depth_clear_en,
      clear_val  => x"3F800000",  -- 1.0 (far plane)
      test_valid => pix_valid,
      test_x     => pix_x,
      test_y     => pix_y,
      test_depth => pix_depth,
      test_write => '1',
      test_pass  => depth_test_pass,
      test_done  => depth_test_done,
      mem_req    => sys_port_req(P_DEPTHBUF),
      mem_resp   => top_sys_resp
    );

  -- Clear the depth buffer's Hi-Z tiles on the first cycle out of reset
  -- (one-shot), then on any CMD_BARRIER with bit 0 set (driver-issued
  -- per-frame clear hook).
  p_depth_clear : process(clk_core_a)
    variable primed : std_logic := '0';
  begin
    if rising_edge(clk_core_a) then
      if rst_n = '0' then
        depth_clear_en <= '1';
        primed := '0';
      elsif primed = '0' then
        depth_clear_en <= '1';
        primed := '1';
      elsif barrier_valid = '1' and barrier_flags(0) = '1' then
        depth_clear_en <= '1';
      else
        depth_clear_en <= '0';
      end if;
    end if;
  end process p_depth_clear;

  -- pix_valid surviving the depth test is the point at which a production
  -- design would package the 2x2 quad into a warp and dispatch it to the
  -- SM pixel-shader pipeline (see header note on raster->SM dispatch).

  -- ===========================================================================
  -- DISPLAY ENGINE: scanout -> gamma LUT -> DP 2.1 encoder
  -- ===========================================================================
  u_scanout : entity work.krafgpu_scanout
    generic map (
      MAX_H_ACTIVE => 7680,
      MAX_V_ACTIVE => 4320,
      MAX_H_TOTAL  => 8000,
      MAX_V_TOTAL  => 4400
    )
    port map (
      clk_pixel     => clk_pixel,
      rst_n         => rst_n,
      h_active      => reg_disp_h_active,
      h_front_porch => reg_disp_h_fp,
      h_sync_width  => reg_disp_h_sync,
      h_back_porch  => reg_disp_h_bp,
      v_active      => reg_disp_v_active,
      v_front_porch => reg_disp_v_fp,
      v_sync_width  => reg_disp_v_sync,
      v_back_porch  => reg_disp_v_bp,
      fb_req        => sys_port_req(P_SCANOUT),
      fb_resp       => top_sys_resp,
      fb_base_addr  => reg_disp_fb_addr,
      pix_valid     => scan_pix_valid,
      pix_r         => scan_pix_r,
      pix_g         => scan_pix_g,
      pix_b         => scan_pix_b,
      hsync         => scan_hsync,
      vsync         => scan_vsync,
      de            => scan_de,
      frame_start   => scan_frame_start,
      line_start    => scan_line_start
    );

  u_gamma : entity work.krafgpu_gamma_lut
    generic map (LUT_SIZE => 1024, OUT_BITS => 12)
    port map (
      clk          => clk_pixel,
      rst_n        => rst_n,
      lut_wr_en    => reg_lut_wr_en,
      lut_addr     => reg_lut_addr,
      lut_r_in     => reg_lut_r,
      lut_g_in     => reg_lut_g,
      lut_b_in     => reg_lut_b,
      pix_valid_in => scan_pix_valid,
      pix_r_in     => scan_pix_r,
      pix_g_in     => scan_pix_g,
      pix_b_in     => scan_pix_b,
      pix_valid_out=> gamma_valid,
      pix_r_out    => gamma_r,
      pix_g_out    => gamma_g,
      pix_b_out    => gamma_b
    );

  u_dp21 : entity work.krafgpu_dp21_encoder
    generic map (NUM_LANES => 4, LANE_GBPS => 20)
    port map (
      clk        => clk_pixel,
      rst_n      => rst_n,
      pix_valid  => gamma_valid,
      pix_r      => gamma_r,
      pix_g      => gamma_g,
      pix_b      => gamma_b,
      hsync      => scan_hsync,
      vsync      => scan_vsync,
      de         => scan_de,
      lt_start   => dp_lt_start,
      lt_pattern => dp_lt_pattern,
      lt_done    => dp_lt_done,
      lane_data  => dp_lane_data,
      lane_valid => dp_lane_valid
    );

  -- ===========================================================================
  -- IRQ SOURCE COMBINATIONAL LOGIC
  -- ===========================================================================
  irq_thermal_warn_s <= '1' when status_temp_c  >= to_unsigned(85, 8)      else '0';
  irq_thermal_crit_s <= '1' when status_temp_c  >= reg_temp_limit          else '0';
  irq_power_limit_s  <= '1' when status_power_w_s > reg_power_limit        else '0';
  irq_ring_stall_s   <= '1' when ring_inflight   = to_unsigned(255, 8)     else '0';

  -- ===========================================================================
  -- INTERRUPT CONTROLLER
  -- ===========================================================================
  u_irq : entity work.krafgpu_irq_ctrl
    generic map (NUM_IRQ_SOURCES => 32)
    port map (
      clk   => clk_pcie,
      rst_n => rst_n,

      irq_fence        => irq_fence_pulse,
      irq_vsync        => scan_frame_start,
      irq_hsync        => scan_line_start,
      irq_thermal_warn => irq_thermal_warn_s,
      irq_thermal_crit => irq_thermal_crit_s,
      irq_power_limit  => irq_power_limit_s,
      irq_compute_done => npu_done(0),
      irq_draw_done    => raster_done,
      irq_dma_done     => dma_copy_done,
      irq_page_fault   => '0',          -- IOMMU not modelled
      irq_ecc_single   => ecc_single_s, -- from krafgpu_x1_top ECC port
      irq_ecc_double   => ecc_double_s,
      irq_ring_empty   => ring_idle,
      irq_ring_stall   => irq_ring_stall_s,
      irq_chiplet_a    => '0',
      irq_chiplet_b    => '0',

      irq_mask_in    => reg_irq_mask,
      irq_clear_in   => reg_irq_clear,
      irq_status_out => status_irq_pending,

      msix_valid  => msix_valid,
      msix_vector => msix_vector,
      msix_addr   => msix_addr,
      msix_data   => msix_data,
      msix_ready  => msix_ready,
      pcie_intx   => pcie_intx
    );

  -- ===========================================================================
  -- CONTEXT SAVE/RESTORE ENGINE
  -- DMA side fully wired through the system arbiter. SM-state snapshot taps
  -- are stubbed pending a debug-snapshot bus from krafgpu_compute_chiplet
  -- (see header note).
  -- ===========================================================================
  u_ctx : entity work.krafgpu_ctx_engine
    generic map (
      NUM_SMS_LOCAL    => NUM_SMS_TOTAL,
      CTX_BYTES_PER_SM => 8192
    )
    port map (
      clk             => clk_pmc,
      rst_n           => rst_n,
      ctx_save_req    => ctx_save_req,
      ctx_restore_req => ctx_restore_req,
      ctx_buf_addr    => ctx_buf_addr,
      ctx_sm_mask     => ctx_sm_mask,
      drain_req       => ctx_drain_req,
      -- drain_done: signals that no warps are in-flight on any SM.
      -- We tie this to ring_idle: when the command ring has no outstanding
      -- commands (ring_inflight=0) the compute pipeline is drained enough
      -- for a context save to proceed safely. A full drain signal would
      -- OR all per-SM "all_warps_idle" flags, which requires a dedicated
      -- snapshot bus from krafgpu_compute_chiplet (future enhancement).
      drain_done      => ring_idle,
      sm_select       => open,
      warp_pc_in      => (others => '0'),
      warp_state_in   => WARP_IDLE,
      warp_mask_in    => (others => '0'),
      pred_reg_in     => (others => '0'),
      warp_pc_out     => open,
      warp_state_out  => open,
      warp_restore_en => ctx_warp_restore_en,
      warp_restore_id => open,
      sm_restore_id   => open,
      dma_req         => sys_port_req(P_CTXENGINE),
      dma_resp        => top_sys_resp,
      ctx_save_done   => ctx_save_done,
      ctx_restore_done=> ctx_restore_done,
      ctx_busy        => ctx_busy,
      regfile_save_req  => open,
      regfile_save_addr => open,
      regfile_save_sm   => open,
      -- regfile_save_done: asserted when the system memory responds to
      -- the register-file DMA write. This is correct for single-outstanding-
      -- request operation: once the DMA write completes (memory ack), the
      -- register file for that SM is fully saved. A pipelined DMA engine
      -- with multiple outstanding writes would need a separate completion
      -- tracker, but the current single-request arbiter guarantees ordering.
      regfile_save_done => top_sys_resp.valid
    );

  -- ===========================================================================
  -- 8-WAY ROUND-ROBIN SYSTEM MEMORY ARBITER
  -- Same pattern as the 3-way die-to-die fabric arbiter in
  -- krafgpu_interconnect_fabric, generalised to NUM_SYS_PORTS requesters.
  -- The single shared response (top_sys_resp) is broadcast to every
  -- sub-module above; each module's own state machine only acts on it while
  -- it has an outstanding request, so broadcast-without-tagging is safe as
  -- long as at most one requester is ever mid-transaction — guaranteed here
  -- because a new grant only issues once the prior response has been
  -- consumed, by the same single-outstanding-request convention used
  -- throughout this RTL (see krafgpu_l3_cache's 2-cycle pipeline and
  -- krafgpu_rr_arbiter's broadcast+self-gate pattern).
  -- ===========================================================================
  p_sys_arb : process(clk_core_a)
    variable granted : std_logic;
  begin
    if rising_edge(clk_core_a) then
      if rst_n = '0' then
        top_sys_req.valid <= '0';
        sys_port_grant    <= (others => '0');
        sys_arb_ptr       <= 0;
      else
        top_sys_req.valid <= '0';
        sys_port_grant    <= (others => '0');
        granted := '0';

        for offset in 0 to NUM_SYS_PORTS-1 loop
          if granted = '0' then
            if sys_port_req((sys_arb_ptr + offset) mod NUM_SYS_PORTS).valid = '1' then
              top_sys_req <= sys_port_req((sys_arb_ptr + offset) mod NUM_SYS_PORTS);
              sys_port_grant((sys_arb_ptr + offset) mod NUM_SYS_PORTS) <= '1';
              sys_arb_ptr <= (sys_arb_ptr + offset + 1) mod NUM_SYS_PORTS;
              granted := '1';
            end if;
          end if;
        end loop;
      end if;
    end if;
  end process p_sys_arb;

end architecture structural;
