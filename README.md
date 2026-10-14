# KrafGPU-X1

A **VHDL-2008 GPU architecture model** for studying compute pipelines, warp scheduling, memory hierarchy and peripheral integration.

This repository contains RTL, module testbenches, constraints and a GHDL simulation runner. It is an engineering prototype and architecture study. The process node, throughput, memory bandwidth, power and package figures in the extended documentation are **design targets**, not measured silicon results. A completed FPGA implementation or a passing full-chip simulation is not established by this README.

## Start here

- [RTL source](krafgpu_x1_vhdl_final/krafgpu_x1_vhdl/rtl)
- [Module testbenches](krafgpu_x1_vhdl_final/krafgpu_x1_vhdl/sim/unit)
- [Simulation runner](krafgpu_x1_vhdl_final/krafgpu_x1_vhdl/simulate.sh)
- [Extended architecture documentation](DOCUMENTATION.md)
- [Reading list](READING_LIST.md)

## What is in the model

- Compute modules: warp scheduler, register file, arithmetic datapath, tensor and branch units.
- Memory modules: L1/L2 caches and a memory-subsystem model.
- Integration modules: command ring, MMIO, interrupts, display, raster and power-management logic.
- Five unit/smoke testbench files for a single SM, directed SM behavior, ray tracing, texture and warp scheduling.

Source presence does not mean a block has been proven against a production GPU specification. Testbench coverage and simulator output should be reviewed for each claim.

## Run the supplied simulations

Install GHDL and a Bash environment, then run from the RTL project directory:

```bash
git clone https://github.com/kkrafvyn/KrafvynGPU.git
cd KrafvynGPU/krafgpu_x1_vhdl_final/krafgpu_x1_vhdl
bash simulate.sh unit
```

The separate `full` mode attempts the replicated full-chip model:

```bash
bash simulate.sh full
```

The runner documents substantial memory requirements for full-chip elaboration. Unit simulations are the practical first step; analysis, elaboration, simulation, synthesis and hardware validation are different checks.

## Review the engineering

The most useful entry points for an RTL review are:

1. [Shared types and constants](krafgpu_x1_vhdl_final/krafgpu_x1_vhdl/rtl/krafgpu_pkg.vhd).
2. [Streaming multiprocessor integration](krafgpu_x1_vhdl_final/krafgpu_x1_vhdl/rtl/core/krafgpu_sm.vhd).
3. [Warp scheduler](krafgpu_x1_vhdl_final/krafgpu_x1_vhdl/rtl/core/krafgpu_warp_scheduler.vhd).
4. [Testbenches and assertions](krafgpu_x1_vhdl_final/krafgpu_x1_vhdl/sim/unit).
5. [Dependency order and test invocation](krafgpu_x1_vhdl_final/krafgpu_x1_vhdl/simulate.sh).

## Current limits

- The architectural scale and advertised performance targets have not been established on hardware.
- Passing a module test is not a full-chip correctness result.
- Full-chip simulation can exceed available memory.
- Physical implementation, timing closure, FPGA resource fit and silicon validation require separate evidence.
- Local development work is not automatically part of this published branch; benchmark results should identify the exact source revision and stimulus.

## Contributing

For a bug report, include the source revision, simulator/backend version, command, failing assertion or log, and a minimal reproducer. For a behavior change, explain the expected result and update the relevant testbench.
