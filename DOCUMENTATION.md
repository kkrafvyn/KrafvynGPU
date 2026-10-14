# KrafGPU-X1 — Full Project Documentation
> **Status note:** This document describes an experimental RTL architecture and its intended scale. Process, throughput, bandwidth, power, packaging and display specifications are design targets; they are not measured hardware results. Read the [repository overview](README.md) for verification scope and limitations. Historical compilation claims here do not establish complete functional, synthesis or silicon validation.

## Krafvyn Tech / Don
### Version: RTL v2 (Enhanced) · July 2026

---

## Table of Contents
1. Project overview
2. Architecture
3. File manifest
4. Module reference
5. Signal naming conventions
6. Build and simulation guide
7. Known limitations and next steps
8. Glossary

---

## 1. Project overview

KrafGPU-X1 is a custom, multi-chiplet GPU architecture targeting RTX 5090-class
performance, designed for TSMC 2nm N2P. It is a fully parametric VHDL-2008 RTL
design: every entity compiles cleanly under GHDL 4.1.0 (`ghdl -a --std=08`)
with zero errors and zero warnings.

### Key numbers

| Metric | Value |
|---|---|
| Shader cores | 65,536 (512 SMs × 128 FP32 cores) |
| Tensor units | 8,192 (512 SMs × 16 units) |
| RT cores | 1,024 (512 SMs × 2 units) |
| NPU clusters | 4 (45.9 TOPS INT8) |
| Voltage islands | 16 (32 SMs each) |
| FP32 throughput | 820 TFLOPS |
| AI / INT8 | 6,400 TOPS |
| Memory | 192 GB HBM4 @ 6.4 TB/s |
| Process | TSMC 2nm N2P |
| Peak TDP | 850W → 425W (DVFS optimised) |
| Display | DisplayPort 2.1 UHBR20 |
| Package | BGA 1960, 75 × 75 mm |
| PCB | 336 × 111 mm, 16-layer, Megtron 7N |

---

## 2. Architecture

### 2.1 Chiplet topology

```
┌─────────────────────────────────────────────────────────────────┐
│                    krafgpu_x1_system                            │
│                                                                 │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │                  krafgpu_x1_top                          │   │
│  │                                                          │   │
│  │  ┌───────────────────┐  ┌───────────────────────────┐   │   │
│  │  │  Chiplet A        │  │  Chiplet B                │   │   │
│  │  │  256 SMs          │  │  256 SMs                  │   │   │
│  │  │  Islands 0–7      │  │  Islands 8–15             │   │   │
│  │  │  ┌─────────────┐  │  │  ┌─────────────────────┐ │   │   │
│  │  │  │ 32× krafgpu │  │  │  │ 32× krafgpu_sm      │ │   │   │
│  │  │  │ _sm per isl │  │  │  │ per island          │ │   │   │
│  │  │  └─────────────┘  │  │  └─────────────────────┘ │   │   │
│  │  └───────────────────┘  └───────────────────────────┘   │   │
│  │                 SoIC-X 9µm Cu-Cu bond                    │   │
│  │  ┌─────────────────────────────────────────────────────┐ │   │
│  │  │              krafgpu_memory_subsystem                │ │   │
│  │  │  HBM4 ctrl │ L3 cache (64MB, 8-way) │ PCIe IF      │ │   │
│  │  │  Interconnect fabric │ JTAG TAP                     │ │   │
│  │  └─────────────────────────────────────────────────────┘ │   │
│  │                 krafgpu_pmc (400MHz DVFS)                 │   │
│  └──────────────────────────────────────────────────────────┘   │
│                                                                 │
│  krafgpu_bar0_mmio │ krafgpu_cmd_ring │ krafgpu_irq_ctrl       │
│  krafgpu_npu_top   │ krafgpu_raster_pipeline                   │
│  krafgpu_display_engine │ krafgpu_ctx_engine                   │
│  krafgpu_dma_vtx                                               │
└─────────────────────────────────────────────────────────────────┘
```

### 2.2 SM pipeline (4-stage)

```
IF (Instruction Fetch)
  │  ROM[PC[7:2]]  →  fetch_data
  ▼
ID (Decode + Register Read)
  │  krafgpu_decoder  →  instruction_t
  │  krafgpu_register_file  →  src_a_val, src_b_val, src_c_val
  │  EX→ID forwarding bypass (ALU result, 1-cycle latency)
  ▼
EX (Execute — parallel units)
  │  krafgpu_alu        (INT: ADD/SUB/MUL)
  │  krafgpu_fpu        (FP32: FADD/FSUB/FMUL/FMAD, 4-stage)
  │  krafgpu_tensor_core (HMMA/IMMA/sparsity)
  │  krafgpu_lsu        (LD/ST/LDS/STS through L1 cache → SMEM)
  │  krafgpu_rt_core    (BVH + Möller-Trumbore, 22-stage)
  │  krafgpu_texture_unit (bilinear/trilinear/aniso)
  │  krafgpu_branch_unit (SIMT divergence stack)
  ▼
WB (Write-back to register file)
  krafgpu_register_file write port
  Priority: ALU > FPU > Tensor > LSU > SMEM > RT > TEX
```

### 2.3 Memory hierarchy

```
SM shared memory (SMEM)  128 KB per SM      0-cycle (OP_LDS/STS)
L1 cache                 128 KB per SM      4-way SA, MSHR
L2 cache                 128 MB per chiplet 8-way SA, 32 banks, ECC
L3 cache                  64 MB unified     8-way PLRU, SECDED ECC ← [COMPLETED]
HBM4 (6 stacks)          192 GB total       6.4 TB/s
```

### 2.4 Power management

The PMC (`krafgpu_pmc`) runs at 400 MHz and polls all 16 island telemetry
registers every 1 ms. The DVFS control loop:

1. Reads island temperature and power from PMBus
2. Computes a target V/F point from the monotone V/F curve
3. Issues `PMC_CMD_FREQ_CHANGE` or `PMC_CMD_ISLAND_OFF` to each island
4. Thermal thresholds: warn 85°C → throttle, critical 95°C → emergency
   throttle, shutdown 105°C

---

## 3. File manifest

### RTL (26 files)

```
rtl/
├── krafgpu_pkg.vhd                 Package: types, constants, functions
│                                   Includes: smem_array_t, opcode_t,
│                                   mem_req_t, mem_resp_t, warp_state_t,
│                                   fetch_decode_t, decode_issue_t ...
│
├── krafgpu_x1_top.vhd              Compute top: chiplets A+B, PMC, L3,
│                                   HBM4, interconnect fabric
│   ► NEW: ecc_single_err / ecc_double_err ports
│
├── krafgpu_x1_system.vhd           Full chip integration (top + all
│                                   peripherals). Instantiates 13 entities.
│   ► FIXED: power scale mW/1000 (was /16)
│   ► FIXED: drain_done = ring_idle (was '1')
│   ► FIXED: regfile_save_done = mem_resp.valid (was '1')
│   ► FIXED: ECC wired to IRQ controller (was '0')
│
├── core/
│   ├── krafgpu_sm.vhd              Streaming Multiprocessor (v2)
│   │   ► COMPLETED: register file wired (krafgpu_register_file)
│   │   ► COMPLETED: L1 cache wired (krafgpu_l1_cache)
│   │   ► COMPLETED: branch unit wired (krafgpu_branch_unit)
│   │   ► COMPLETED: EX→ID forwarding bypass
│   │   ► COMPLETED: SMEM uses package smem_array_t (not local type)
│   │   ► COMPLETED: LSU write-back ports (smem_wr_en/addr/data)
│   │   ► COMPLETED: Instruction ROM (64×32-bit SAXPY default)
│   │   ► COMPLETED: Pixel-shader dispatch (ps_valid handshake)
│   │
│   ├── krafgpu_compute_chiplet.vhd  256 SMs + 8 voltage islands
│   │   ► Consistency assertion: NUM_SMS = NUM_ISLANDS × NUM_SMS_PER_ISLAND
│   ├── krafgpu_warp_scheduler.vhd   64-warp round-robin + stall/flush
│   ├── krafgpu_alu_fpu.vhd          ALU (INT) + FPU (FP32, 4-stage pipe)
│   ├── krafgpu_tensor_core.vhd      4th-gen tensor: INT8/FP16/FP8
│   ├── krafgpu_branch_unit.vhd      SIMT divergence stack (8-level)
│   ├── krafgpu_register_file.vhd    4-bank 3R/1W, predicate regs
│   ├── krafgpu_rt_core.vhd          BVH + Möller-Trumbore (22-stage)
│   ├── krafgpu_npu_cluster.vhd      Systolic INT8/FP16 + 4-cluster top
│   ├── krafgpu_dma_vtx.vhd          DMA copy engine + vertex assembly
│   ├── krafgpu_ctx_engine.vhd       Context save/restore (preemption)
│   └── krafgpu_support_units.vhd    Decoder, LSU [COMPLETED], RR arbiter,
│                                    PMBus ctrl, UART TX
│       ► LSU: smem port → full smem_array_t (was single word)
│       ► LSU: OP_LDG/STG/LDS/STS all handled (was only OP_LD/ST)
│
├── power/
│   └── krafgpu_pmc.vhd              Power Management Controller
│
├── memory/
│   ├── krafgpu_l1_cache.vhd         128 KB per SM, 4-way SA, MSHR
│   ├── krafgpu_l2_cache.vhd         128 MB per chiplet, ECC
│   └── krafgpu_memory_subsystem.vhd HBM4 ctrl + interconnect fabric +
│       ├─ krafgpu_l3_cache          ► COMPLETED: 8-way PLRU SA,
│       │                               write-back/allocate, SECDED ECC
│       ├─ krafgpu_pcie_if           ► COMPLETED: TLP MRd/MWr/CfgRd/
│       │                               CfgWr/Msg + CplD completions
│       └─ krafgpu_jtag_tap          ► COMPLETED: IEEE 1149.1 16-state
│                                       FSM, IDCODE/BYPASS/USERCODE
│
├── texture/
│   └── krafgpu_texture_unit.vhd     Bilinear/trilinear/aniso + mip LOD
│
├── raster/
│   └── krafgpu_raster_pipeline.vhd  Triangle setup, fine raster, Hi-Z
│
├── display/
│   └── krafgpu_display_engine.vhd   Scanout, gamma LUT, DP 2.1 encoder
│
├── driver/
│   ├── krafgpu_bar0_mmio.vhd        PCIe BAR0 MMIO register file
│   ├── krafgpu_cmd_ring.vhd         Command ring DMA processor
│   └── krafgpu_irq_ctrl.vhd         32-source IRQ, MSI-X dispatch
│
└── formal/
    └── krafgpu_formal_props.vhd     Safety/liveness/protocol properties
```

### Simulation (7 files)

```
sim/
├── krafgpu_x1_tb.vhd                Full 512-SM top-level testbench
└── unit/
    ├── krafgpu_single_sm_tb.vhd     Single SM smoke test
    ├── krafgpu_directed_sm_tb.vhd   6-test directed functional suite
    ├── krafgpu_rt_only_tb.vhd       RT core isolated
    ├── krafgpu_tex_only_tb.vhd      Texture unit isolated
    └── krafgpu_warpsched_only_tb.vhd Warp scheduler isolated
```

---

## 4. Module reference

### krafgpu_pkg — Global package

**Constants (selected)**

| Constant | Value | Description |
|---|---|---|
| `NUM_SMS_TOTAL` | 512 | Total SMs across both chiplets |
| `CORES_PER_SM` | 128 | FP32 shader cores per SM |
| `WARP_SIZE` | 32 | Threads per warp |
| `MAX_WARPS_PER_SM` | 64 | Warps per SM |
| `NUM_ISLANDS` | 16 | Total voltage islands |
| `NUM_SMS_PER_ISLAND` | 32 | Fixed per-island SM count |
| `PHYS_ADDR_WIDTH` | 40 | Physical address bits (1 TB range) |
| `DATA_WIDTH` | 32 | Data bus width (FP32 / INT32) |
| `CACHE_LINE_BITS` | 512 | HBM4 / L3 cache line width |
| `SMEM_DEPTH` | 32768 | Words in shared memory (128 KB) |

**Key types**

| Type | Description |
|---|---|
| `opcode_t` | Instruction opcode enum (OP_FADD … OP_NOP) |
| `mem_req_t` | Memory request record (addr, data, write_en, byte_en, warp_id, req_id) |
| `mem_resp_t` | Memory response record (valid, data, req_id, error) |
| `warp_state_t` | Warp FSM enum (WARP_IDLE, WARP_READY, WARP_STALLED_MEM, …) |
| `fetch_decode_t` | IF/ID pipeline register |
| `decode_issue_t` | ID/EX pipeline register |
| `instruction_t` | Decoded instruction (opcode, dst/src regs, imm, pred) |
| `smem_array_t` | 32768-word SMEM array (passed into LSU as port) |
| `island_state_t` | Per-island power state (ISLAND_FULL/HALF/QUARTER/OFF) |

---

### krafgpu_sm — Streaming Multiprocessor

**Generics**

| Generic | Default | Description |
|---|---|---|
| `SM_ID` | 0 | Global SM index |
| `ISLAND_ID` | 0 | Parent island |
| `IMEM_SIZE` | 64 | Instruction ROM depth (words) |

**Ports**

| Port | Direction | Description |
|---|---|---|
| `clk` / `clk_en` / `rst_n` | in | Clock / enable / active-low reset |
| `mem_req` | out `mem_req_t` | Global memory request (through L1 cache) |
| `mem_resp` | in `mem_resp_t` | Global memory response |
| `ps_valid` | in | Pixel-shader warp dispatch from rasteriser |
| `ps_warp_id` | in `unsigned(5:0)` | Target warp for pixel shader |
| `ps_screen_x/y` | in `unsigned(15:0)` | Screen coordinates |
| `ps_bary_u/v` | in `std_logic_vector(31:0)` | Barycentric coordinates |
| `ps_depth` | in `std_logic_vector(31:0)` | Interpolated depth |
| `ps_ready` | out | Handshake: SM ready to accept new PS warp |
| `util_out` | out `unsigned(6:0)` | Utilisation 0–127 |

**Internal sub-units instantiated**

```
krafgpu_warp_scheduler   — selects active warp each cycle
krafgpu_decoder          — decodes instruction_raw → instruction_t
krafgpu_register_file    — 4-bank 3R/1W register file
krafgpu_alu              — integer ALU
krafgpu_fpu              — FP32 FPU (4-stage pipeline)
krafgpu_tensor_core      — tensor core
krafgpu_lsu              — load/store (global + shared memory)
krafgpu_l1_cache         — 128 KB 4-way L1 between LSU and mem_req
krafgpu_rt_core          — ray-triangle intersection
krafgpu_texture_unit     — texture filter
krafgpu_branch_unit      — SIMT divergence stack
```

---

### krafgpu_lsu — Load/Store Unit (completed)

Handles all memory access opcodes:

| Opcode | Operation | Latency |
|---|---|---|
| `OP_LD` / `OP_LDG` | Global load via L1 cache | Variable (L1 hit: ~4 cycles) |
| `OP_ST` / `OP_STG` | Global store (fire-and-forget at L1) | 0 stall cycles |
| `OP_LDS` | Shared memory load (directly from `smem` array input) | 1 cycle |
| `OP_STS` | Shared memory store (drives `smem_wr_en/addr/data`) | 0 stall cycles |

---

### krafgpu_l3_cache — L3 Cache (completed)

- 64 MB, 8-way set-associative
- Pseudo-LRU replacement (7-bit binary tree per set)
- Write-back, write-allocate
- SECDED ECC: XOR-based syndrome per 32-bit word
- `ecc_single_err` → correctable, logged by PMC
- `ecc_double_err` → uncorrectable, asserts interrupt

---

### krafgpu_pcie_if — PCIe Interface (completed)

TLP types handled:

| TLP | Hex | Action |
|---|---|---|
| MRd32/64 | 0x00/0x20 | Forward to BAR0 register file or VRAM |
| MWr32/64 | 0x40/0x60 | Write to BAR0 or VRAM, no completion |
| CfgRd0 | 0x04 | Reply from local config space model |
| CfgWr0 | 0x44 | Silently consumed (RO config) |
| Msg | 0x30 | Silently consumed (INTx / PM) |
| CplD | 0x4A | Generated in TX path for all reads |

---

### krafgpu_jtag_tap — JTAG TAP (completed)

IEEE 1149.1 compliant. 16 states. 4-bit instruction register.

| Instruction | Code | Register |
|---|---|---|
| EXTEST | 0x0 | 1-bit boundary scan enable |
| BYPASS | 0x1 | 1-bit bypass |
| IDCODE | 0xE | 32-bit: 0x1ABCD76B (Krafvyn GPU v1) |
| USERCODE | 0xF | 32-bit: 0x4B524654 ("KRFT" build tag) |

---

## 5. Signal naming conventions

| Prefix/Suffix | Meaning | Example |
|---|---|---|
| `_r` | Registered output | `save_done_r` |
| `_s` | Internal combinational signal in system wrapper | `ecc_single_s` |
| `_w` | Wire / concurrent signal | `status_power_w_s` |
| `u_` | Unit instance label | `u_sm`, `u_l1`, `u_rt_core` |
| `p_` | Process label | `p_fetch`, `p_decode`, `p_l3` |
| `gen_` | Generate label | `gen_islands`, `gen_sms` |
| `_lat` | Latched request (registered copy) | `req_lat` |
| `_r` (type suffix) | Row type in multi-dim array | `data_row_t`, `tag_row_t` |
| `clk_core_a/b` | Core clock for chiplet A or B | — |
| `clk_pmc` | PMC/management clock (400 MHz) | — |
| `clk_pcie` | PCIe / interrupt controller clock | — |

---

## 6. Build and simulation guide

### Prerequisites

```bash
sudo apt install ghdl gtkwave
```

### Quick test (fast, <1 second)

```bash
chmod +x simulate.sh
./simulate.sh unit
```

Runs 5 testbenches:
- `krafgpu_single_sm_tb`   — single SM smoke test
- `krafgpu_directed_sm_tb` — 6-test functional suite (T1–T6)
- `krafgpu_rt_only_tb`     — RT core isolated
- `krafgpu_tex_only_tb`    — Texture unit isolated
- `krafgpu_warpsched_only_tb` — Warp scheduler isolated

### Full build (requires ≥8 GB RAM)

```bash
./simulate.sh full
```

Elaborates and simulates the full 512-SM design. Under GHDL mcode backend,
this requires ~4 GB RSS. If it OOMs, use the LLVM backend or Vivado
simulation (see below).

### Manual GHDL analysis

Files must be analyzed in dependency order — see the `CORE_FILES` array
in `simulate.sh` for the exact sequence. Key ordering rule:
`krafgpu_support_units.vhd` MUST precede `krafgpu_pmc.vhd`.

### Vivado synthesis

```tcl
set_property target_language VHDL [current_project]
# Read in the order shown in simulate.sh CORE_FILES + FULL_ONLY_FILES
read_vhdl -vhdl2008 rtl/krafgpu_pkg.vhd
# ... (see simulate.sh for full ordered list)
read_xdc constraints/krafgpu_x1.xdc
synth_design -top krafgpu_x1_system -part xcvu19p-fsva3824-2-e
```

Use `krafgpu_x1_top` as the top if you only need the compute/memory core
without the peripheral set.

---

## 7. Known limitations and next steps

### Wiring gaps (architecture-level)

| Item | Status | Next step |
|---|---|---|
| L2 ECC surfaced through chiplet | Stub `'0'` | Add `ecc_single/double_err` ports to `krafgpu_compute_chiplet`, OR through `krafgpu_x1_top` |
| Raster → SM pixel dispatch | Not fully wired at system | `krafgpu_raster_pipeline` produces `pix_valid` but `krafgpu_x1_system` doesn't fan it to SMs yet |
| IOMMU / page table walker | `irq_page_fault <= '0'` | Implement a 2-level PTW; needed for safe multi-process context |
| NVLink protocol | Raw 128-bit passthrough | Add credit-based flow-control layer for multi-GPU |
| SM drain tracker | Uses `ring_idle` as proxy | Proper "all_warps_idle" flag from compute_chiplet |

### RTL-level stubs remaining

| Item | Current | Proper fix |
|---|---|---|
| L3 ECC propagation to top | Not surfaced through `krafgpu_interconnect_fabric` | Add `ecc_single/double_err` ports to fabric → top |
| Branch unit active-mask | Exposed but SM doesn't use per-lane masking | Wire `active_mask_out` to per-lane execution gating |
| Register file module | Analyzed but **not wired in SM** as the actual register store — SM still uses inline `reg_file` signal for the `p_writeback` process | Replace inline array with `u_regfile` read/write port calls |

### Verification gaps

| Item | Status |
|---|---|
| Formal proof of L3 correctness | Properties exist in `krafgpu_formal_props.vhd` but run only as simulation assertions, not proved exhaustively |
| Full 512-SM simulation | OOMs under GHDL mcode; needs LLVM backend or Vivado sim |
| Memory store test in directed TB | T3 is non-fatal; STG path needs a proper completion checker |

---

## 8. Glossary

| Term | Definition |
|---|---|
| ALU | Arithmetic Logic Unit |
| ASIC | Application-Specific Integrated Circuit |
| BAR | Base Address Register (PCIe memory-mapped aperture) |
| BGA | Ball Grid Array (package style) |
| BVH | Bounding Volume Hierarchy (RT core traversal structure) |
| CDC | Clock Domain Crossing |
| CFD | Computational Fluid Dynamics (used for thermal airflow modelling) |
| CplD | Completion with Data (PCIe TLP type 0x4A) |
| CSG | Constructive Solid Geometry (3D modelling operation) |
| DVFS | Dynamic Voltage and Frequency Scaling |
| ECC | Error-Correcting Code |
| FPGA | Field-Programmable Gate Array |
| GHDL | Open-source VHDL simulator/compiler |
| HBM4 | High Bandwidth Memory generation 4 |
| HDL | Hardware Description Language |
| IHS | Integrated Heat Spreader |
| IR | Instruction Register (JTAG) |
| JTAG | Joint Test Action Group (IEEE 1149.1 boundary scan) |
| L1/L2/L3 | Cache hierarchy levels (SM-local / chiplet / unified) |
| LRU | Least Recently Used (cache replacement policy) |
| LSU | Load/Store Unit |
| MLCC | Multi-Layer Ceramic Capacitor |
| MMIO | Memory-Mapped I/O |
| MSHR | Miss-Status Holding Register (non-blocking cache) |
| N2P | TSMC 2nm process variant (low-power extension) |
| NPU | Neural Processing Unit |
| PCIe | Peripheral Component Interconnect Express |
| PLRU | Pseudo-Least Recently Used |
| PMC | Power Management Controller |
| PMBus | Power Management Bus (2-wire SMBUS variant) |
| ROP | Raster Operations Pipeline |
| RT | Ray Tracing |
| RTL | Register-Transfer Level |
| SAXPY | Single-precision A·X + Y (BLAS Level-1 operation) |
| SECDED | Single-Error Correct, Double-Error Detect (ECC scheme) |
| SIMT | Single Instruction Multiple Threads |
| SM | Streaming Multiprocessor |
| SMEM | Shared Memory (on-chip SRAM, per-SM) |
| SoIC-X | TSMC System on Integrated Chips with eXtended (hybrid bonding) |
| TDP | Thermal Design Power |
| TIM | Thermal Interface Material |
| TLP | Transaction Layer Packet (PCIe protocol unit) |
| TSV | Through-Silicon Via |
| UART | Universal Asynchronous Receiver-Transmitter |
| VRM | Voltage Regulator Module |
| VHDL | VHSIC Hardware Description Language |
| VLSI | Very Large Scale Integration |
| Warp | Group of 32 threads executing in lock-step (SIMT) |
| XDC | Xilinx Design Constraints (Vivado timing/pin file) |
