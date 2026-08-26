-- =============================================================================
-- KrafGPU-X1 Interrupt Controller
-- Project      : KrafGPU-X1 Next-Generation GPU Architecture
-- Author       : Krafvyn Tech / Don
-- Description  : 32-source interrupt aggregation and MSI-X dispatch.
--
--   Interrupt sources (32 total):
--     [0]  = FENCE_SIGNAL      — GPU fence value written
--     [1]  = VSYNC             — Display vertical sync
--     [2]  = HSYNC             — Display horizontal sync (optional)
--     [3]  = THERMAL_WARN      — Island temp > 85°C warning
--     [4]  = THERMAL_CRIT      — Island temp > 95°C critical
--     [5]  = POWER_LIMIT       — Power budget exceeded
--     [6]  = COMPUTE_DONE      — Dispatch complete
--     [7]  = DRAW_DONE         — Draw call complete
--     [8]  = DMA_DONE          — DMA copy complete
--     [9]  = PAGE_FAULT        — IOMMU page fault
--     [10] = ECC_SINGLE        — ECC single-bit correction
--     [11] = ECC_DOUBLE        — ECC double-bit uncorrectable
--     [12] = RING_EMPTY        — Command ring consumed all commands
--     [13] = RING_STALL        — Command ring stall (back-pressure)
--     [14] = CHIPLET_A_ERR     — Chiplet A internal error
--     [15] = CHIPLET_B_ERR     — Chiplet B internal error
--     [31:16] = Reserved
--
--   MSI-X: up to 32 vectors, each mapped to a separate host interrupt
--   MSI:   single aggregated interrupt (legacy compat mode)
--
--   Registers (in BAR0 at 0x006000):
--     0x006000 = IRQ_MASK    (R/W) — 1=masked
--     0x006004 = IRQ_STATUS  (RO)  — pending interrupts
--     0x006008 = IRQ_CLEAR   (WO)  — write 1 to clear
--     0x00600C = IRQ_MODE    (R/W) — 0=MSI, 1=MSI-X
--     0x006010 = MSIX_BASE   (R/W) — MSI-X table base address (PCIe BAR2)
-- =============================================================================

library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;
use work.krafgpu_pkg.all;

entity krafgpu_irq_ctrl is
  generic (
    NUM_IRQ_SOURCES : integer := 32
  );
  port (
    clk             : in  std_logic;
    rst_n           : in  std_logic;

    -- Interrupt source inputs (level-triggered, active high)
    irq_fence       : in  std_logic;
    irq_vsync       : in  std_logic;
    irq_hsync       : in  std_logic;
    irq_thermal_warn: in  std_logic;
    irq_thermal_crit: in  std_logic;
    irq_power_limit : in  std_logic;
    irq_compute_done: in  std_logic;
    irq_draw_done   : in  std_logic;
    irq_dma_done    : in  std_logic;
    irq_page_fault  : in  std_logic;
    irq_ecc_single  : in  std_logic;
    irq_ecc_double  : in  std_logic;
    irq_ring_empty  : in  std_logic;
    irq_ring_stall  : in  std_logic;
    irq_chiplet_a   : in  std_logic;
    irq_chiplet_b   : in  std_logic;

    -- BAR0 register interface
    irq_mask_in     : in  std_logic_vector(NUM_IRQ_SOURCES-1 downto 0);
    irq_clear_in    : in  std_logic_vector(NUM_IRQ_SOURCES-1 downto 0);
    irq_status_out  : out std_logic_vector(NUM_IRQ_SOURCES-1 downto 0);

    -- MSI-X output (to PCIe interface)
    msix_valid      : out std_logic;
    msix_vector     : out unsigned(4 downto 0);    -- Which MSI-X vector
    msix_addr       : out unsigned(63 downto 0);   -- MSI-X message addr
    msix_data       : out std_logic_vector(31 downto 0); -- MSI-X message data
    msix_ready      : in  std_logic;

    -- Legacy INTx (PCIe INTA#)
    pcie_intx       : out std_logic
  );
end entity krafgpu_irq_ctrl;

architecture rtl of krafgpu_irq_ctrl is

  -- ===========================================================================
  -- INTERRUPT PENDING REGISTER
  -- ===========================================================================
  signal irq_raw     : std_logic_vector(NUM_IRQ_SOURCES-1 downto 0);
  signal irq_pending : std_logic_vector(NUM_IRQ_SOURCES-1 downto 0)
                     := (others => '0');
  signal irq_masked  : std_logic_vector(NUM_IRQ_SOURCES-1 downto 0);

  -- Edge detection (convert level to edge for latching)
  signal irq_raw_d1  : std_logic_vector(NUM_IRQ_SOURCES-1 downto 0)
                     := (others => '0');
  signal irq_edge    : std_logic_vector(NUM_IRQ_SOURCES-1 downto 0);

  -- ===========================================================================
  -- MSI-X STATE MACHINE
  -- ===========================================================================
  type msix_state_t is (MX_IDLE, MX_SCAN, MX_SEND, MX_WAIT);
  signal msix_state  : msix_state_t := MX_IDLE;
  signal scan_idx    : integer range 0 to NUM_IRQ_SOURCES-1 := 0;
  signal msix_valid_r: std_logic := '0';
  signal msix_vec_r  : unsigned(4 downto 0) := (others => '0');

  -- MSI-X table (addr + data per vector) — programmed by driver
  type msix_entry_t is record
    addr  : unsigned(63 downto 0);
    data  : std_logic_vector(31 downto 0);
    masked: std_logic;
  end record;
  type msix_table_t is array (0 to NUM_IRQ_SOURCES-1) of msix_entry_t;
  signal msix_table  : msix_table_t := (others => (
    addr   => (others => '0'),
    data   => (others => '0'),
    masked => '0'));

begin

  -- ===========================================================================
  -- INTERRUPT SOURCE AGGREGATION
  -- ===========================================================================
  irq_raw(0)  <= irq_fence;
  irq_raw(1)  <= irq_vsync;
  irq_raw(2)  <= irq_hsync;
  irq_raw(3)  <= irq_thermal_warn;
  irq_raw(4)  <= irq_thermal_crit;
  irq_raw(5)  <= irq_power_limit;
  irq_raw(6)  <= irq_compute_done;
  irq_raw(7)  <= irq_draw_done;
  irq_raw(8)  <= irq_dma_done;
  irq_raw(9)  <= irq_page_fault;
  irq_raw(10) <= irq_ecc_single;
  irq_raw(11) <= irq_ecc_double;
  irq_raw(12) <= irq_ring_empty;
  irq_raw(13) <= irq_ring_stall;
  irq_raw(14) <= irq_chiplet_a;
  irq_raw(15) <= irq_chiplet_b;
  irq_raw(NUM_IRQ_SOURCES-1 downto 16) <= (others => '0');

  -- ===========================================================================
  -- EDGE DETECTION + LATCH
  -- ===========================================================================
  irq_edge <= irq_raw and not irq_raw_d1;

  p_latch : process(clk)
  begin
    if rising_edge(clk) then
      if rst_n = '0' then
        irq_pending <= (others => '0');
        irq_raw_d1  <= (others => '0');
      else
        irq_raw_d1 <= irq_raw;

        -- Set bits on rising edge
        for i in 0 to NUM_IRQ_SOURCES-1 loop
          if irq_edge(i) = '1' then
            irq_pending(i) <= '1';
          end if;
        end loop;

        -- Clear bits when driver writes to IRQ_CLEAR register
        for i in 0 to NUM_IRQ_SOURCES-1 loop
          if irq_clear_in(i) = '1' then
            irq_pending(i) <= '0';
          end if;
        end loop;
      end if;
    end if;
  end process p_latch;

  -- Masked pending = pending AND NOT mask
  irq_masked <= irq_pending and not irq_mask_in;

  -- ===========================================================================
  -- MSI-X DISPATCH STATE MACHINE
  -- Scans for highest-priority unmasked pending interrupt and sends MSI-X
  -- ===========================================================================
  p_msix : process(clk)
  begin
    if rising_edge(clk) then
      if rst_n = '0' then
        msix_state  <= MX_IDLE;
        msix_valid_r<= '0';
        scan_idx    <= 0;
      else
        msix_valid_r <= '0';

        case msix_state is

          when MX_IDLE =>
            if irq_masked /= (NUM_IRQ_SOURCES-1 downto 0 => '0') then
              scan_idx   <= 0;
              msix_state <= MX_SCAN;
            end if;

          when MX_SCAN =>
            -- Find lowest set bit (highest priority)
            if irq_masked(scan_idx) = '1' then
              msix_vec_r <= to_unsigned(scan_idx, 5);
              msix_state <= MX_SEND;
            elsif scan_idx < NUM_IRQ_SOURCES-1 then
              scan_idx   <= scan_idx + 1;
            else
              msix_state <= MX_IDLE;
            end if;

          when MX_SEND =>
            -- Issue MSI-X message if vector not table-masked
            if msix_table(to_integer(msix_vec_r)).masked = '0' then
              msix_valid_r <= '1';
              msix_state   <= MX_WAIT;
            else
              msix_state   <= MX_IDLE;
            end if;

          when MX_WAIT =>
            if msix_ready = '1' then
              msix_valid_r <= '0';
              msix_state   <= MX_IDLE;
            end if;

        end case;
      end if;
    end if;
  end process p_msix;

  -- ===========================================================================
  -- OUTPUTS
  -- ===========================================================================
  irq_status_out <= irq_pending;
  msix_valid     <= msix_valid_r;
  msix_vector    <= msix_vec_r;
  msix_addr      <= msix_table(to_integer(msix_vec_r)).addr;
  msix_data      <= msix_table(to_integer(msix_vec_r)).data;

  -- Legacy INTx: assert if any unmasked interrupt pending
  pcie_intx      <= '1' when irq_masked /= (NUM_IRQ_SOURCES-1 downto 0 => '0')
                        else '0';

end architecture rtl;
