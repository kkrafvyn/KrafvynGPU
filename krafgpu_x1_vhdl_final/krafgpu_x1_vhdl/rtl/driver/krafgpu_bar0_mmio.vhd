-- =============================================================================
-- KrafGPU-X1 BAR0 MMIO Register Map
-- Project      : KrafGPU-X1 Next-Generation GPU Architecture
-- Author       : Krafvyn Tech / Don
-- Description  : PCIe BAR0 memory-mapped I/O register file.
--                The driver (KrafGPU kernel module) accesses all GPU
--                configuration, status, and control through this interface.
--
--   BAR0 layout (16 MB total):
--
--   Offset 0x000000 – 0x000FFF  : GPU Identification & Capabilities
--   Offset 0x001000 – 0x001FFF  : Power Management & DVFS
--   Offset 0x002000 – 0x002FFF  : Memory Controller
--   Offset 0x003000 – 0x003FFF  : Display Engine (per-head registers)
--   Offset 0x004000 – 0x004FFF  : Compute Engine (SM control)
--   Offset 0x005000 – 0x005FFF  : Command Ring Buffers
--   Offset 0x006000 – 0x006FFF  : Interrupt Control
--   Offset 0x007000 – 0x007FFF  : Debug & Telemetry
--   Offset 0x008000 – 0x00FFFF  : Gamma LUT (R+G+B × 1024 × 12-bit)
--   Offset 0x010000 – 0x01FFFF  : Context Save/Restore Area
--   Offset 0x020000 – 0xFFFFFF  : Reserved for future expansion
-- =============================================================================

library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;
use work.krafgpu_pkg.all;

entity krafgpu_bar0_mmio is
  port (
    clk             : in  std_logic;
    rst_n           : in  std_logic;

    -- PCIe BAR0 access from host CPU (32-bit aligned accesses)
    bar_addr        : in  unsigned(23 downto 0);  -- 16 MB window
    bar_wr_data     : in  std_logic_vector(31 downto 0);
    bar_wr_en       : in  std_logic;
    bar_rd_en       : in  std_logic;
    bar_rd_data     : out std_logic_vector(31 downto 0);
    bar_rd_valid    : out std_logic;
    bar_byte_en     : in  std_logic_vector(3 downto 0);

    -- -------------------------------------------------------------------------
    -- Register outputs → GPU subsystems
    -- -------------------------------------------------------------------------

    -- Identity
    reg_gpu_id      : out std_logic_vector(31 downto 0);

    -- Power management → PMC
    reg_power_limit : out unsigned(15 downto 0);   -- Watt limit (max 850)
    reg_temp_limit  : out unsigned(7 downto 0);    -- TJ limit (°C)
    reg_island_force: out std_logic_vector(NUM_ISLANDS-1 downto 0);
    reg_island_state: out island_state_array_t;
    reg_dvfs_mode   : out std_logic_vector(1 downto 0);  -- 00=auto 01=perf 10=pwr

    -- Memory controller
    reg_fb_base     : out unsigned(PHYS_ADDR_WIDTH-1 downto 0);
    reg_fb_size     : out unsigned(31 downto 0);
    reg_hbm_mode    : out std_logic_vector(3 downto 0);

    -- Display engine
    reg_disp_h_active  : out unsigned(13 downto 0);
    reg_disp_h_fp      : out unsigned(9 downto 0);
    reg_disp_h_sync    : out unsigned(9 downto 0);
    reg_disp_h_bp      : out unsigned(9 downto 0);
    reg_disp_v_active  : out unsigned(13 downto 0);
    reg_disp_v_fp      : out unsigned(9 downto 0);
    reg_disp_v_sync    : out unsigned(9 downto 0);
    reg_disp_v_bp      : out unsigned(9 downto 0);
    reg_disp_fb_addr   : out unsigned(PHYS_ADDR_WIDTH-1 downto 0);
    reg_disp_enable    : out std_logic;
    reg_disp_vsync_en  : out std_logic;

    -- Gamma LUT
    reg_lut_wr_en   : out std_logic;
    reg_lut_addr    : out unsigned(9 downto 0);
    reg_lut_r       : out unsigned(11 downto 0);
    reg_lut_g       : out unsigned(11 downto 0);
    reg_lut_b       : out unsigned(11 downto 0);

    -- Compute engine
    reg_sm_en_mask  : out std_logic_vector(NUM_ISLANDS-1 downto 0);
    reg_dispatch_en : out std_logic;
    reg_cmd_ring_base : out unsigned(PHYS_ADDR_WIDTH-1 downto 0);
    reg_cmd_ring_size : out unsigned(31 downto 0);
    reg_cmd_ring_head : out unsigned(31 downto 0);

    -- Interrupts
    reg_irq_mask    : out std_logic_vector(31 downto 0);
    reg_irq_clear   : out std_logic_vector(31 downto 0);

    -- -------------------------------------------------------------------------
    -- Status inputs ← GPU subsystems
    -- -------------------------------------------------------------------------
    status_power_w       : in  unsigned(15 downto 0);
    status_temp_c        : in  unsigned(7 downto 0);
    status_util_pct      : in  unsigned(6 downto 0);
    status_irq_pending   : in  std_logic_vector(31 downto 0);
    status_cmd_ring_tail : in  unsigned(31 downto 0);
    status_hbm_bw_gbps   : in  unsigned(15 downto 0);
    status_sm_active     : in  std_logic_vector(NUM_ISLANDS-1 downto 0)
  );
end entity krafgpu_bar0_mmio;

architecture rtl of krafgpu_bar0_mmio is

  -- ===========================================================================
  -- REGISTER STORAGE
  -- ===========================================================================

  -- GPU Identity (read-only)
  constant C_GPU_VENDOR_ID    : std_logic_vector(31 downto 0) := x"4B524146"; -- "KRAF"
  constant C_GPU_DEVICE_ID    : std_logic_vector(31 downto 0) := x"58315F56"; -- "X1_V"
  constant C_GPU_REVISION     : std_logic_vector(31 downto 0) := x"00000100"; -- Rev 1.0
  constant C_GPU_SM_COUNT     : std_logic_vector(31 downto 0) :=
    std_logic_vector(to_unsigned(NUM_SMS_TOTAL, 32));
  constant C_GPU_VRAM_MB      : std_logic_vector(31 downto 0) :=
    std_logic_vector(to_unsigned(192 * 1024, 32));  -- 192 GB
  constant C_GPU_CAPS_0       : std_logic_vector(31 downto 0) :=
    "0000" &                          -- [31:28] Reserved
    "1" &                             -- [27] NVLink support
    "1" &                             -- [26] Ray tracing
    "1" &                             -- [25] Tensor cores
    "1" &                             -- [24] Sparse tensor
    "0000" &                          -- [23:20] Reserved
    "1111" &                          -- [19:16] MSAA 16x
    x"00FF";                          -- [15:0] SM island count

  -- Power management registers
  signal r_power_limit  : unsigned(15 downto 0) := to_unsigned(850, 16);
  signal r_temp_limit   : unsigned(7 downto 0)  := to_unsigned(95, 8);
  signal r_island_force : std_logic_vector(NUM_ISLANDS-1 downto 0) := (others => '0');
  signal r_dvfs_mode    : std_logic_vector(1 downto 0) := "00";

  -- Memory controller registers
  signal r_fb_base      : unsigned(PHYS_ADDR_WIDTH-1 downto 0) := (others => '0');
  signal r_fb_size      : unsigned(31 downto 0) := to_unsigned(192*1024*1024, 32);
  signal r_hbm_mode     : std_logic_vector(3 downto 0) := "0000";

  -- Display timing registers (default: 4K@120Hz)
  signal r_h_active     : unsigned(13 downto 0) := to_unsigned(3840, 14);
  signal r_h_fp         : unsigned(9 downto 0)  := to_unsigned(176, 10);
  signal r_h_sync       : unsigned(9 downto 0)  := to_unsigned(32, 10);
  signal r_h_bp         : unsigned(9 downto 0)  := to_unsigned(296, 10);
  signal r_v_active     : unsigned(13 downto 0) := to_unsigned(2160, 14);
  signal r_v_fp         : unsigned(9 downto 0)  := to_unsigned(4, 10);
  signal r_v_sync       : unsigned(9 downto 0)  := to_unsigned(5, 10);
  signal r_v_bp         : unsigned(9 downto 0)  := to_unsigned(56, 10);
  signal r_disp_fb_addr : unsigned(PHYS_ADDR_WIDTH-1 downto 0) := (others => '0');
  signal r_disp_enable  : std_logic := '0';
  signal r_disp_vsync   : std_logic := '1';

  -- Gamma LUT write registers
  signal r_lut_wr_en    : std_logic := '0';
  signal r_lut_addr     : unsigned(9 downto 0) := (others => '0');
  signal r_lut_r        : unsigned(11 downto 0) := (others => '0');
  signal r_lut_g        : unsigned(11 downto 0) := (others => '0');
  signal r_lut_b        : unsigned(11 downto 0) := (others => '0');

  -- Compute registers
  signal r_sm_en_mask   : std_logic_vector(NUM_ISLANDS-1 downto 0) := (others => '1');
  signal r_dispatch_en  : std_logic := '0';
  signal r_cmd_base     : unsigned(PHYS_ADDR_WIDTH-1 downto 0) := (others => '0');
  signal r_cmd_size     : unsigned(31 downto 0) := to_unsigned(65536, 32);
  signal r_cmd_head     : unsigned(31 downto 0) := (others => '0');

  -- Interrupt registers
  signal r_irq_mask     : std_logic_vector(31 downto 0) := (others => '0');
  signal r_irq_clear    : std_logic_vector(31 downto 0) := (others => '0');

  -- Read data register
  signal rd_data_r      : std_logic_vector(31 downto 0) := (others => '0');
  signal rd_valid_r     : std_logic := '0';

  -- ===========================================================================
  -- ADDRESS DECODE CONSTANTS
  -- ===========================================================================

  -- Identity block (0x000000 – 0x000FFF)
  constant A_VENDOR_ID    : unsigned(23 downto 0) := x"000000";
  constant A_DEVICE_ID    : unsigned(23 downto 0) := x"000004";
  constant A_REVISION     : unsigned(23 downto 0) := x"000008";
  constant A_SM_COUNT     : unsigned(23 downto 0) := x"00000C";
  constant A_VRAM_MB      : unsigned(23 downto 0) := x"000010";
  constant A_CAPS_0       : unsigned(23 downto 0) := x"000014";

  -- Power management block (0x001000)
  constant A_PWR_LIMIT    : unsigned(23 downto 0) := x"001000";
  constant A_TEMP_LIMIT   : unsigned(23 downto 0) := x"001004";
  constant A_DVFS_MODE    : unsigned(23 downto 0) := x"001008";
  constant A_ISLAND_FORCE : unsigned(23 downto 0) := x"001010";
  constant A_STATUS_PWR   : unsigned(23 downto 0) := x"001100";  -- RO
  constant A_STATUS_TEMP  : unsigned(23 downto 0) := x"001104";  -- RO
  constant A_STATUS_UTIL  : unsigned(23 downto 0) := x"001108";  -- RO

  -- Memory controller block (0x002000)
  constant A_FB_BASE_LO   : unsigned(23 downto 0) := x"002000";
  constant A_FB_BASE_HI   : unsigned(23 downto 0) := x"002004";
  constant A_FB_SIZE      : unsigned(23 downto 0) := x"002008";
  constant A_HBM_MODE     : unsigned(23 downto 0) := x"00200C";
  constant A_HBM_BW       : unsigned(23 downto 0) := x"002100";  -- RO

  -- Display engine block (0x003000)
  constant A_DISP_EN      : unsigned(23 downto 0) := x"003000";
  constant A_DISP_H_ACT   : unsigned(23 downto 0) := x"003010";
  constant A_DISP_H_FP    : unsigned(23 downto 0) := x"003014";
  constant A_DISP_H_SYNC  : unsigned(23 downto 0) := x"003018";
  constant A_DISP_H_BP    : unsigned(23 downto 0) := x"00301C";
  constant A_DISP_V_ACT   : unsigned(23 downto 0) := x"003020";
  constant A_DISP_V_FP    : unsigned(23 downto 0) := x"003024";
  constant A_DISP_V_SYNC  : unsigned(23 downto 0) := x"003028";
  constant A_DISP_V_BP    : unsigned(23 downto 0) := x"00302C";
  constant A_DISP_FB_LO   : unsigned(23 downto 0) := x"003030";
  constant A_DISP_FB_HI   : unsigned(23 downto 0) := x"003034";

  -- Gamma LUT (0x003100)
  constant A_LUT_CTRL     : unsigned(23 downto 0) := x"003100";
  constant A_LUT_ADDR     : unsigned(23 downto 0) := x"003104";
  constant A_LUT_DATA_R   : unsigned(23 downto 0) := x"003108";
  constant A_LUT_DATA_G   : unsigned(23 downto 0) := x"00310C";
  constant A_LUT_DATA_B   : unsigned(23 downto 0) := x"003110";

  -- Compute engine block (0x004000)
  constant A_DISPATCH_EN  : unsigned(23 downto 0) := x"004000";
  constant A_SM_EN_MASK   : unsigned(23 downto 0) := x"004004";
  constant A_CMD_BASE_LO  : unsigned(23 downto 0) := x"004010";
  constant A_CMD_BASE_HI  : unsigned(23 downto 0) := x"004014";
  constant A_CMD_SIZE     : unsigned(23 downto 0) := x"004018";
  constant A_CMD_HEAD     : unsigned(23 downto 0) := x"00401C";
  constant A_CMD_TAIL     : unsigned(23 downto 0) := x"004020";  -- RO
  constant A_SM_ACTIVE    : unsigned(23 downto 0) := x"004100";  -- RO

  -- Interrupt block (0x006000)
  constant A_IRQ_MASK     : unsigned(23 downto 0) := x"006000";
  constant A_IRQ_STATUS   : unsigned(23 downto 0) := x"006004";  -- RO
  constant A_IRQ_CLEAR    : unsigned(23 downto 0) := x"006008";  -- WO

begin

  -- ===========================================================================
  -- REGISTER WRITE
  -- ===========================================================================
  p_reg_write : process(clk)
  begin
    if rising_edge(clk) then
      if rst_n = '0' then
        r_power_limit  <= to_unsigned(850, 16);
        r_temp_limit   <= to_unsigned(95, 8);
        r_dvfs_mode    <= "00";
        r_island_force <= (others => '0');
        r_disp_enable  <= '0';
        r_dispatch_en  <= '0';
        r_sm_en_mask   <= (others => '1');
        r_irq_mask     <= (others => '0');
        r_irq_clear    <= (others => '0');
        r_lut_wr_en    <= '0';
        r_cmd_head     <= (others => '0');
      else
        r_lut_wr_en <= '0';
        r_irq_clear <= (others => '0');

        if bar_wr_en = '1' then
          case bar_addr is

            -- Power management
            when A_PWR_LIMIT =>
              if bar_byte_en(1 downto 0) = "11" then
                r_power_limit <= unsigned(bar_wr_data(15 downto 0));
              end if;
            when A_TEMP_LIMIT =>
              if bar_byte_en(0) = '1' then
                r_temp_limit <= unsigned(bar_wr_data(7 downto 0));
              end if;
            when A_DVFS_MODE =>
              r_dvfs_mode <= bar_wr_data(1 downto 0);
            when A_ISLAND_FORCE =>
              r_island_force <= bar_wr_data(NUM_ISLANDS-1 downto 0);

            -- Memory controller
            when A_FB_BASE_LO =>
              r_fb_base(31 downto 0) <=
                unsigned(bar_wr_data);
            when A_FB_BASE_HI =>
              r_fb_base(PHYS_ADDR_WIDTH-1 downto 32) <=
                unsigned(bar_wr_data(PHYS_ADDR_WIDTH-33 downto 0));
            when A_FB_SIZE =>
              r_fb_size <= unsigned(bar_wr_data);
            when A_HBM_MODE =>
              r_hbm_mode <= bar_wr_data(3 downto 0);

            -- Display timing
            when A_DISP_EN =>
              r_disp_enable <= bar_wr_data(0);
              r_disp_vsync  <= bar_wr_data(1);
            when A_DISP_H_ACT  => r_h_active <= unsigned(bar_wr_data(13 downto 0));
            when A_DISP_H_FP   => r_h_fp     <= unsigned(bar_wr_data(9 downto 0));
            when A_DISP_H_SYNC => r_h_sync   <= unsigned(bar_wr_data(9 downto 0));
            when A_DISP_H_BP   => r_h_bp     <= unsigned(bar_wr_data(9 downto 0));
            when A_DISP_V_ACT  => r_v_active <= unsigned(bar_wr_data(13 downto 0));
            when A_DISP_V_FP   => r_v_fp     <= unsigned(bar_wr_data(9 downto 0));
            when A_DISP_V_SYNC => r_v_sync   <= unsigned(bar_wr_data(9 downto 0));
            when A_DISP_V_BP   => r_v_bp     <= unsigned(bar_wr_data(9 downto 0));
            when A_DISP_FB_LO  =>
              r_disp_fb_addr(31 downto 0) <= unsigned(bar_wr_data);
            when A_DISP_FB_HI  =>
              r_disp_fb_addr(PHYS_ADDR_WIDTH-1 downto 32) <=
                unsigned(bar_wr_data(PHYS_ADDR_WIDTH-33 downto 0));

            -- Gamma LUT
            when A_LUT_CTRL =>
              r_lut_wr_en <= bar_wr_data(0);
            when A_LUT_ADDR =>
              r_lut_addr  <= unsigned(bar_wr_data(9 downto 0));
            when A_LUT_DATA_R =>
              r_lut_r     <= unsigned(bar_wr_data(11 downto 0));
            when A_LUT_DATA_G =>
              r_lut_g     <= unsigned(bar_wr_data(11 downto 0));
            when A_LUT_DATA_B =>
              r_lut_b     <= unsigned(bar_wr_data(11 downto 0));

            -- Compute engine
            when A_DISPATCH_EN =>
              r_dispatch_en <= bar_wr_data(0);
            when A_SM_EN_MASK =>
              r_sm_en_mask  <= bar_wr_data(NUM_ISLANDS-1 downto 0);
            when A_CMD_BASE_LO =>
              r_cmd_base(31 downto 0) <= unsigned(bar_wr_data);
            when A_CMD_BASE_HI =>
              r_cmd_base(PHYS_ADDR_WIDTH-1 downto 32) <=
                unsigned(bar_wr_data(PHYS_ADDR_WIDTH-33 downto 0));
            when A_CMD_SIZE =>
              r_cmd_size <= unsigned(bar_wr_data);
            when A_CMD_HEAD =>
              r_cmd_head <= unsigned(bar_wr_data);

            -- Interrupts
            when A_IRQ_MASK =>
              r_irq_mask <= bar_wr_data;
            when A_IRQ_CLEAR =>
              r_irq_clear <= bar_wr_data;

            when others => null;
          end case;
        end if;
      end if;
    end if;
  end process p_reg_write;

  -- ===========================================================================
  -- REGISTER READ
  -- ===========================================================================
  p_reg_read : process(clk)
  begin
    if rising_edge(clk) then
      if rst_n = '0' then
        rd_data_r  <= (others => '0');
        rd_valid_r <= '0';
      else
        rd_valid_r <= '0';

        if bar_rd_en = '1' then
          rd_valid_r <= '1';
          case bar_addr is
            -- Identity
            when A_VENDOR_ID   => rd_data_r <= C_GPU_VENDOR_ID;
            when A_DEVICE_ID   => rd_data_r <= C_GPU_DEVICE_ID;
            when A_REVISION    => rd_data_r <= C_GPU_REVISION;
            when A_SM_COUNT    => rd_data_r <= C_GPU_SM_COUNT;
            when A_VRAM_MB     => rd_data_r <= C_GPU_VRAM_MB;
            when A_CAPS_0      => rd_data_r <= C_GPU_CAPS_0;

            -- Power status (RO)
            when A_STATUS_PWR  =>
              rd_data_r <= std_logic_vector(resize(status_power_w, 32));
            when A_STATUS_TEMP =>
              rd_data_r <= std_logic_vector(resize(status_temp_c, 32));
            when A_STATUS_UTIL =>
              rd_data_r <= std_logic_vector(resize(status_util_pct, 32));

            -- Power config readback
            when A_PWR_LIMIT   =>
              rd_data_r <= std_logic_vector(resize(r_power_limit, 32));
            when A_TEMP_LIMIT  =>
              rd_data_r <= std_logic_vector(resize(r_temp_limit, 32));
            when A_DVFS_MODE   =>
              rd_data_r <= (31 downto 2 => '0') & r_dvfs_mode;

            -- Memory
            when A_FB_BASE_LO  =>
              rd_data_r <= std_logic_vector(r_fb_base(31 downto 0));
            when A_FB_SIZE     =>
              rd_data_r <= std_logic_vector(r_fb_size);
            when A_HBM_BW      =>
              rd_data_r <= std_logic_vector(resize(status_hbm_bw_gbps, 32));

            -- Display
            when A_DISP_EN     =>
              rd_data_r <= (31 downto 2 => '0') & r_disp_vsync & r_disp_enable;
            when A_DISP_H_ACT  =>
              rd_data_r <= (31 downto 14 => '0') & std_logic_vector(r_h_active);
            when A_DISP_V_ACT  =>
              rd_data_r <= (31 downto 14 => '0') & std_logic_vector(r_v_active);

            -- Compute
            when A_CMD_TAIL    =>
              rd_data_r <= std_logic_vector(status_cmd_ring_tail);
            when A_SM_ACTIVE   =>
              rd_data_r <= (31 downto NUM_ISLANDS => '0') &
                           status_sm_active;

            -- Interrupts
            when A_IRQ_STATUS  =>
              rd_data_r <= status_irq_pending;
            when A_IRQ_MASK    =>
              rd_data_r <= r_irq_mask;

            when others        => rd_data_r <= x"DEADBEEF";  -- Unmapped
          end case;
        end if;
      end if;
    end if;
  end process p_reg_read;

  -- ===========================================================================
  -- ISLAND STATE DECODE (DVFS mode → island state array)
  -- ===========================================================================
  gen_island_state : for i in 0 to NUM_ISLANDS-1 generate
    reg_island_state(i) <=
      ISLAND_OFF  when r_island_force(i) = '1' and r_dvfs_mode = "10" else
      ISLAND_LOW  when r_dvfs_mode = "10" else
      ISLAND_FULL when r_dvfs_mode = "01" else
      ISLAND_HALF;  -- Default: balanced
  end generate;

  -- ===========================================================================
  -- OUTPUT ASSIGNMENTS
  -- ===========================================================================
  bar_rd_data    <= rd_data_r;
  bar_rd_valid   <= rd_valid_r;

  reg_gpu_id     <= C_GPU_DEVICE_ID;
  reg_power_limit<= r_power_limit;
  reg_temp_limit <= r_temp_limit;
  reg_island_force <= r_island_force;
  reg_dvfs_mode  <= r_dvfs_mode;

  reg_fb_base    <= r_fb_base;
  reg_fb_size    <= r_fb_size;
  reg_hbm_mode   <= r_hbm_mode;

  reg_disp_h_active <= r_h_active;
  reg_disp_h_fp     <= r_h_fp;
  reg_disp_h_sync   <= r_h_sync;
  reg_disp_h_bp     <= r_h_bp;
  reg_disp_v_active <= r_v_active;
  reg_disp_v_fp     <= r_v_fp;
  reg_disp_v_sync   <= r_v_sync;
  reg_disp_v_bp     <= r_v_bp;
  reg_disp_fb_addr  <= r_disp_fb_addr;
  reg_disp_enable   <= r_disp_enable;
  reg_disp_vsync_en <= r_disp_vsync;

  reg_lut_wr_en  <= r_lut_wr_en;
  reg_lut_addr   <= r_lut_addr;
  reg_lut_r      <= r_lut_r;
  reg_lut_g      <= r_lut_g;
  reg_lut_b      <= r_lut_b;

  reg_sm_en_mask    <= r_sm_en_mask;
  reg_dispatch_en   <= r_dispatch_en;
  reg_cmd_ring_base <= r_cmd_base;
  reg_cmd_ring_size <= r_cmd_size;
  reg_cmd_ring_head <= r_cmd_head;

  reg_irq_mask   <= r_irq_mask;
  reg_irq_clear  <= r_irq_clear;

end architecture rtl;
