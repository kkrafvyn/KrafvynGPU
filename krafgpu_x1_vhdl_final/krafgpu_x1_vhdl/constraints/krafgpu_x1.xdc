# ==============================================================================
# KrafGPU-X1 Timing Constraints (XDC)
# Project : KrafGPU-X1 Next-Generation GPU Architecture
# Author  : Krafvyn Tech / Don
# Tool    : AMD Vivado 2024.1+  (for FPGA prototype / timing closure)
# Note    : These constraints target the XCVU19P Alveo U280 evaluation board.
#           Final silicon uses TSMC 2nm N2P — constraints will differ.
# ==============================================================================

# ==============================================================================
# PRIMARY CLOCKS
# ==============================================================================

# Core compute clock — 2.8 GHz (FPGA max ~750 MHz; scaled for prototype)
create_clock -name clk_core_a -period 1.333 [get_ports clk_core_a]
create_clock -name clk_core_b -period 1.333 [get_ports clk_core_b]

# HBM4 memory clock — 4.8 GHz (HBM controller uses 600 MHz on FPGA)
create_clock -name clk_mem    -period 1.667 [get_ports clk_mem]

# PCIe reference 100 MHz
create_clock -name clk_pcie   -period 10.000 [get_ports clk_pcie]

# PMC always-on 400 MHz
create_clock -name clk_pmc    -period 2.500  [get_ports clk_pmc]

# ==============================================================================
# CLOCK DOMAIN CROSSINGS
# ==============================================================================

# Core clocks are synchronous (same PLL) — only async crossings need CDC
set_clock_groups -asynchronous \
  -group [get_clocks clk_core_a] \
  -group [get_clocks clk_mem]

set_clock_groups -asynchronous \
  -group [get_clocks {clk_core_a clk_core_b}] \
  -group [get_clocks clk_pcie]

set_clock_groups -asynchronous \
  -group [get_clocks {clk_core_a clk_core_b}] \
  -group [get_clocks clk_pmc]

# ==============================================================================
# INPUT DELAYS (PCIe interface — setup/hold vs ref clock)
# ==============================================================================
set_input_delay  -clock clk_pcie -max 2.0 [get_ports pcie_rx_data*]
set_input_delay  -clock clk_pcie -min 0.5 [get_ports pcie_rx_data*]
set_input_delay  -clock clk_pcie -max 2.0 [get_ports pcie_rx_valid]
set_input_delay  -clock clk_pcie -min 0.5 [get_ports pcie_rx_valid]

# ==============================================================================
# OUTPUT DELAYS (PCIe TX)
# ==============================================================================
set_output_delay -clock clk_pcie -max 2.0 [get_ports pcie_tx_data*]
set_output_delay -clock clk_pcie -min -0.5 [get_ports pcie_tx_data*]
set_output_delay -clock clk_pcie -max 2.0 [get_ports pcie_tx_valid]

# ==============================================================================
# PMBus I2C — 400 kHz, no tight timing constraints needed
# ==============================================================================
set_input_delay  -clock clk_pmc -max 10.0 [get_ports pmbus_sda]
set_input_delay  -clock clk_pmc -max 10.0 [get_ports pmbus_scl]
set_output_delay -clock clk_pmc -max 10.0 [get_ports pmbus_sda]
set_output_delay -clock clk_pmc -max 10.0 [get_ports pmbus_scl]

# ==============================================================================
# PMIC PGOOD — relatively slow signal, relax timing
# ==============================================================================
set_input_delay  -clock clk_pmc -max 5.0 [get_ports pmic_pgood*]

# ==============================================================================
# JTAG — JTAG clock is self-contained, multicycle
# ==============================================================================
set_input_delay  -clock [get_clocks clk_pcie] -max 4.0 [get_ports jtag_*]
set_output_delay -clock [get_clocks clk_pcie] -max 4.0 [get_ports jtag_tdo]

# ==============================================================================
# FALSE PATHS
# ==============================================================================
# Reset is asynchronous, async assertion is OK
set_false_path -from [get_ports rst_n]

# UART is async serial — no timing relationship with core clock
set_false_path -from [get_ports uart_rx]
set_false_path -to   [get_ports uart_tx]

# Display outputs are registered outputs, relax to 2 cycles
set_multicycle_path 2 -setup -to [get_ports disp_data*]
set_multicycle_path 1 -hold  -to [get_ports disp_data*]

# ==============================================================================
# PHYSICAL CONSTRAINTS (Alveo U280 prototype mapping)
# ==============================================================================

# PCIe reference clock on dedicated GCIO
set_property PACKAGE_PIN AM10 [get_ports clk_pcie]
set_property IOSTANDARD LVDS  [get_ports clk_pcie]

# UART on on-board USB UART bridge
set_property PACKAGE_PIN AW25 [get_ports uart_tx]
set_property PACKAGE_PIN AY25 [get_ports uart_rx]
set_property IOSTANDARD LVCMOS18 [get_ports uart_tx]
set_property IOSTANDARD LVCMOS18 [get_ports uart_rx]

# PMIC PGOOD (16 bits → GPIO header)
set_property IOSTANDARD LVCMOS18 [get_ports {pmic_pgood[*]}]

# PMBus I2C (open-drain — use pullup on board)
set_property IOSTANDARD LVCMOS18 [get_ports pmbus_scl]
set_property IOSTANDARD LVCMOS18 [get_ports pmbus_sda]

# ==============================================================================
# TIMING EXCEPTIONS FOR ISLAND STATE REGISTERS
# Island state is updated by PMC at 1ms intervals — ultra-slow, relax fully
# ==============================================================================
set_multicycle_path 400000 -setup \
  -from [get_cells u_pmc/*island_states*] \
  -to   [get_cells u_chiplet_a/*]
set_multicycle_path 400000 -setup \
  -from [get_cells u_pmc/*island_states*] \
  -to   [get_cells u_chiplet_b/*]
