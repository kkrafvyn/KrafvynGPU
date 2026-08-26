# VLSI & VHDL Reading List
## Krafvyn Tech / Don — KrafGPU-X1 Project
### Curated July 2026

This list is ordered by what you should read *first*, mapped directly to
what you built in the KrafGPU-X1 RTL. Every book is tied to a specific
module or design decision in the project so the reading stays concrete
rather than abstract.

---

## Tier 1 — Read these first (foundations)

These three books cover everything you touched in the KrafGPU-X1 RTL.
Read them in order. Do the exercises.

---

### 1. Digital Design and Computer Architecture: ARM Edition
**Authors:** David Harris, Sarah Harris  
**Publisher:** Morgan Kaufmann, 2nd edition (2015)  
**ISBN:** 978-0128000564  

The single best starting point. Chapters 1–4 give you Boolean logic,
Karnaugh maps, sequential circuits, and finite state machines — the
conceptual backbone of every process in your RTL. Chapter 5 covers
processor microarchitecture: the 4-stage fetch/decode/execute/writeback
pipeline you implemented in `krafgpu_sm.vhd` is exactly the pipeline
they build step by step, with hazard detection, forwarding bypasses,
and stall insertion. Chapter 7 introduces memory systems: cache
organisation, associativity, hit/miss rates, LRU replacement — the
theory behind your L1/L2/L3 implementations.

**Directly relevant to your project:**
- § 5.3–5.5 → your IF/ID/EX/WB pipeline in `krafgpu_sm.vhd`
- § 5.6 → the EX→ID forwarding bypass you added in `p_fwd`
- § 7.3–7.4 → L1 cache, set-associative organisation, PLRU
- § 8.1–8.3 → memory-mapped I/O (your BAR0 register file)

---

### 2. VHDL Programming by Example
**Author:** Douglas Perry  
**Publisher:** McGraw-Hill, 4th edition (2002)  
**ISBN:** 978-0071400701  

The most practical VHDL reference available. Perry's book covers the
full VHDL-93/2008 language with worked examples for every construct you
used: records, arrays, generics, generate statements, clocked processes,
combinational processes, concurrent signal assignments, and packages.
The chapter on testbenches is the most useful one in print for learning
how `wait for`, `assert`, and `severity failure` work together — which
is exactly how all five of your unit testbenches are structured.

**Directly relevant to your project:**
- Ch. 3 → `std_logic_vector`, `unsigned`, `signed`, arithmetic conversions
- Ch. 5 → records (`mem_req_t`, `decode_issue_t`, `fetch_decode_t`)
- Ch. 6 → arrays (`smem_array_t`, `data_arr_t`, `plru_arr_t`)
- Ch. 7 → packages (`krafgpu_pkg.vhd` structure)
- Ch. 9 → generics (how `SM_ID`, `NUM_WAYS`, `SMEM_DEPTH` flow through)
- Ch. 12 → testbench writing (your `krafgpu_directed_sm_tb.vhd`)
- Ch. 14 → generate statements (`gen_islands`, `gen_sms` in the chiplet)

---

### 3. CMOS VLSI Design: A Circuits and Systems Perspective
**Authors:** Neil Weste, David Harris  
**Publisher:** Pearson, 4th edition (2010)  
**ISBN:** 978-0321547743  

This is the standard VLSI textbook at every major university. It bridges
the gap between the RTL you wrote and what actually happens in silicon.
Part I (Chapters 1–4) covers CMOS transistors, logic gates, and static
timing — you need this to understand what "TSMC 2nm N2P" actually means
for your design. Part II covers sequential circuits and memory at the
circuit level: it explains how SRAM cells work, which directly explains
why your L1/L2 cache "arrays" in RTL are synthesised into SRAM macros by
the tool rather than flip-flops. Part III covers chip design flow from
RTL to GDSII — synthesis, placement, routing, timing closure — which is
what Vivado does with your `krafgpu_x1_system.vhd`.

**Directly relevant to your project:**
- Ch. 5 → timing: setup/hold, clock-to-Q, why `falling_edge(tck)` in
  the JTAG TAP drives TDO on the falling edge per IEEE 1149.1
- Ch. 12 → SRAM arrays: explains why your 128 KB SMEM and L1 cache
  arrays are RTL-level models of what becomes hard-macro SRAM
- Ch. 14 → chip design flow: synthesis, floorplan, power grid —
  what happens after `ghdl -a` and how Vivado uses your XDC constraints
- Ch. 15 → I/O: HBM4 PHY, PCIe SerDes, why the SMBUS/PMBus protocol
  runs at 400 kHz in your PMC

---

## Tier 2 — GPU architecture and parallel computing

Read these after Tier 1. They explain why the GPU pipeline is designed
the way it is, which makes the RTL choices in KrafGPU-X1 obvious rather
than arbitrary.

---

### 4. Programming Massively Parallel Processors
**Authors:** David Kirk, Wen-mei Hwu  
**Publisher:** Morgan Kaufmann, 4th edition (2022)  
**ISBN:** 978-0323912310  

The definitive GPU programming book. Even though it targets CUDA
software, the hardware model it describes is precisely what you built:
SIMT execution, warp scheduling, shared memory, memory coalescing,
divergence and reconvergence. Chapter 4 on CUDA memory model maps
directly to your LSU opcodes (OP_LDS/OP_STS = shared memory,
OP_LDG/OP_STG = global memory through L1). Chapter 6 on performance
explains why you implemented MSHR in L1 and why the warp scheduler
hides memory latency by switching to a ready warp.

**Directly relevant to your project:**
- Ch. 3 → SIMT model: explains `krafgpu_warp_scheduler` scheduling policy
- Ch. 4 → memory hierarchy: maps to your L1/SMEM/L2/L3/HBM4 organisation
- Ch. 6 → latency hiding: why 64 warps per SM in `MAX_WARPS_PER_SM`
- Ch. 9 → tensor operations: what `krafgpu_tensor_core` does in hardware
- Ch. 17 → ray tracing: BVH traversal in `krafgpu_rt_core`

---

### 5. Computer Architecture: A Quantitative Approach
**Authors:** John Hennessy, David Patterson  
**Publisher:** Morgan Kaufmann, 6th edition (2017)  
**ISBN:** 978-0128119051  

Hennessy & Patterson is the graduate-level companion to Harris & Harris.
Where H&H teaches you to build a pipeline, H&P teaches you to analyse
and optimise one. Appendix C (pipeline hazards) gives you the formal
framework for RAW/WAW/WAR hazards that your forwarding bypass and stall
logic address. Chapter 2 on memory hierarchy is the most rigorous
treatment of cache design available outside of research papers: inclusion,
exclusion, victim caches, stream prefetching. Chapter 4 on ILP explains
out-of-order execution — useful background for understanding why your
in-order SM is simpler but your warp scheduler compensates for the
same latency-hiding goal.

**Directly relevant to your project:**
- App. C → hazard classification: RAW/WAW/WAR in `p_decode` forwarding
- Ch. 2 → cache hierarchy: sizing rationale for L1/L2/L3 in KrafGPU-X1
- Ch. 4 → ILP vs TLP: why GPU uses many warps instead of out-of-order
- Ch. 5 → memory systems: HBM4 bandwidth analysis, DRAM timing

---

### 6. Real-Time Rendering
**Authors:** Tomas Akenine-Möller, Eric Haines, Naty Hoffman  
**Publisher:** CRC Press, 4th edition (2018)  
**ISBN:** 978-1138627000  

Explains the graphics pipeline that your raster pipeline and display
engine implement. The Möller-Trumbore algorithm in Chapter 22 is
exactly what `krafgpu_rt_core` implements in its 22-stage pipeline.
Chapter 10 covers BVH construction and traversal — the data structure
your RT core traverses. Chapter 5 explains the texture filtering math
behind `krafgpu_texture_unit`'s bilinear/trilinear/anisotropic modes.

**Directly relevant to your project:**
- Ch. 5 → texture filtering: bilinear/trilinear/aniso in texture unit
- Ch. 10 → BVH: what your RT core traverses in `p_rt_fsm`
- Ch. 22 → Möller-Trumbore: the intersection test in `krafgpu_rt_core`

---

## Tier 3 — Specialised topics

Read specific chapters from these when you hit the corresponding
problem in your design work. You don't need to read them cover to cover.

---

### 7. Digital Integrated Circuits: A Design Perspective
**Author:** Jan Rabaey, Anantha Chandrakasan, Borivoje Nikolic  
**Publisher:** Pearson, 2nd edition (2003)  
**ISBN:** 978-0130909961  

The power and timing chapters (Ch. 5, 9, 11) explain the physics behind
your PMC's DVFS control loop. Chapter 9 on dynamic power gives you the
formula P = α · C · V² · f that your PMC implements in the `p_dvfs`
process. Chapter 11 on low-power design explains clock gating, voltage
islands, and power domains — the 16-island architecture in KrafGPU-X1.

**Read:** Ch. 5, 9, 11 for the PMC / DVFS / power island design.

---

### 8. High-Speed Signal Propagation: Advanced Black Magic
**Author:** Howard Johnson, Martin Graham  
**Publisher:** Prentice Hall, 2003  
**ISBN:** 978-0130844088  

PCB signal integrity for the 336×111mm KrafGPU-X1 evaluation board.
Explains why Megtron 7N substrate with Dk=3.37 was chosen, how
differential impedance is controlled to 100 Ω ±5% for PCIe 6.0 and
HBM4 traces, and what the -25 dB insertion-loss budget in the XDC file
means for trace length and via design.

**Read:** Ch. 3 (transmission lines), Ch. 7 (differential signalling),
Ch. 11 (via modelling) for PCB layout decisions.

---

### 9. IEEE Std 1149.1-2013 (JTAG Standard)
**Publisher:** IEEE  
**Free access:** Via IEEE Xplore with institutional login  

The normative reference for `krafgpu_jtag_tap`. Your implementation
covers Sections 6 (TAP controller state machine), 8 (IDCODE register
format), and 9 (BYPASS instruction). Reading the standard explains
exactly why TDO changes on the falling edge of TCK (Section 6.4.2) and
what the capture-IR "01" pattern means (Section 7.2.1).

**Read:** Sections 6, 7, 8, 9 — these map directly to your TAP FSM.

---

### 10. PCIe 6.0 Base Specification  
**Publisher:** PCI-SIG  
**Access:** Free download at pcisig.com (registration required)

The normative reference for `krafgpu_pcie_if`. Your TLP parser
implements a subset of the transaction layer described in Chapters 2–4.
The FLIT mode encoding your 256-bit interface uses is defined in
Chapter 5 of PCIe 6.0 specifically. Chapter 6 covers completions
(CplD/Cpl) including the status field and byte-count rules in your
completion generator.

**Read:** Ch. 2 (TLP types), Ch. 4 (completion rules), Ch. 5 (FLIT mode).

---

### 11. Synthesis and Optimization of Digital Circuits
**Authors:** Giovanni De Micheli  
**Publisher:** McGraw-Hill, 1994  
**ISBN:** 978-0070163331  

The theoretical foundation for what Vivado does with your RTL.
Explains Boolean minimisation, technology mapping, retiming, and the
difference between area and timing optimisation. Particularly relevant
when you run synthesis and find that your L3 cache's PLRU logic or the
PCIe TLP state machine isn't meeting timing at the 1.6 GHz core clock.

**Read:** Ch. 5 (technology mapping), Ch. 7 (retiming), Ch. 9 (FSM
optimisation) — these explain what to do when timing fails.

---

### 12. Constraining Designs for Synthesis and Timing Analysis
**Author:** Sridhar Gangadharan, Sanjay Churiwala  
**Publisher:** Springer, 2013  
**ISBN:** 978-1461432074  

Explains Synopsys Design Constraints (SDC) / XDC format in detail.
Your `constraints/krafgpu_x1.xdc` file needs to define clock periods,
input/output delays, false paths (between clock domains: `clk_core` vs
`clk_pmc` vs `clk_pcie`), and multi-cycle paths (for the L3 PLRU
victim-select logic which spans more than one cycle). This book explains
how to write those constraints correctly.

**Read:** Ch. 3 (clock definitions), Ch. 5 (exceptions: false/multi-cycle
paths), Ch. 7 (CDC constraints) — for `krafgpu_x1.xdc`.

---

## Quick-reference map: project file → book

| KrafGPU-X1 file / feature | Primary book | Chapter(s) |
|---|---|---|
| 4-stage SM pipeline (`krafgpu_sm.vhd`) | Harris & Harris [1] | 5.3–5.6 |
| EX→ID forwarding bypass | Harris & Harris [1] | 5.5 |
| VHDL processes, records, arrays | Perry [2] | 3, 5, 6, 9 |
| Package (`krafgpu_pkg.vhd`) | Perry [2] | 7 |
| Testbenches (`krafgpu_directed_sm_tb`) | Perry [2] | 12 |
| Generate statements (chiplet) | Perry [2] | 14 |
| CMOS transistors, gate-level | Weste & Harris [3] | 1–4 |
| SRAM arrays (L1/SMEM in silicon) | Weste & Harris [3] | 12 |
| Synthesis / Vivado flow | Weste & Harris [3] | 14 |
| JTAG TAP timing (falling-edge TDO) | Weste & Harris [3] | 5 |
| SIMT, warp scheduling | Kirk & Hwu [4] | 3, 6 |
| LSU opcode map (LDS/LDG/STS/STG) | Kirk & Hwu [4] | 4 |
| Tensor core operations | Kirk & Hwu [4] | 9 |
| BVH + ray tracing | Kirk & Hwu [4] | 17 |
| Cache hierarchy sizing | Hennessy & Patterson [5] | 2 |
| Hazard analysis (RAW/WAW/WAR) | Hennessy & Patterson [5] | App. C |
| Texture filtering math | Real-Time Rendering [6] | 5 |
| Möller-Trumbore (RT core) | Real-Time Rendering [6] | 22 |
| DVFS power formula P=αCV²f | Rabaey [7] | 9 |
| Power islands, clock gating | Rabaey [7] | 11 |
| PCB signal integrity, Megtron 7N | Johnson & Graham [8] | 3, 7 |
| JTAG (normative) | IEEE 1149.1-2013 [9] | 6–9 |
| PCIe TLP / FLIT / completions | PCIe 6.0 Spec [10] | 2, 4, 5 |
| Timing closure, retiming | De Micheli [11] | 5, 7, 9 |
| XDC constraints (CDC, multi-cycle) | Gangadharan [12] | 3, 5, 7 |

---

## Suggested reading order (12-week plan)

| Week | Books | Focus |
|---|---|---|
| 1–2 | Harris & Harris Ch. 1–4 | Logic, sequential circuits, FSMs |
| 3–4 | Harris & Harris Ch. 5, 7, 8 | Pipeline, cache, memory-mapped I/O |
| 5–6 | Perry Ch. 3–7, 9, 12, 14 | VHDL language, testbenches, packages |
| 7 | Weste & Harris Ch. 1–5, 12 | CMOS circuits, SRAM, timing |
| 8 | Kirk & Hwu Ch. 3–4, 6, 9, 17 | GPU architecture, SIMT, tensor, RT |
| 9 | Hennessy & Patterson Ch. 2, App. C | Cache depth, hazard analysis |
| 10 | Real-Time Rendering Ch. 5, 10, 22 | Texture, BVH, ray-triangle |
| 11 | Rabaey Ch. 9, 11 + IEEE 1149.1 §6–9 | Power design + JTAG standard |
| 12 | PCIe 6.0 Ch. 2, 4, 5 + Gangadharan Ch. 3, 5, 7 | PCIe TLP + XDC constraints |

After week 12, Weste & Harris Ch. 14 (chip design flow) and De Micheli
Ch. 5, 7 (synthesis optimisation) fill the gap between your RTL and
actual silicon. Run those in parallel with Vivado synthesis experiments
on the KrafGPU-X1 design.

---

## Free online resources

These cost nothing and are high quality:

| Resource | URL | What it covers |
|---|---|---|
| MIT 6.004 (Computation Structures) | ocw.mit.edu/6-004 | Logic → pipeline → memory, lectures + problem sets |
| Stanford CS316 (Advanced Logic Design) | web.stanford.edu/class/cs316 | VHDL/Verilog, synthesis, timing |
| GHDL documentation | ghdl.readthedocs.io | Everything about your simulator |
| IEEE Std 1149.1 (JTAG) | ieeexplore.ieee.org | Normative TAP controller spec |
| PCI-SIG specs | pcisig.com | PCIe 6.0 base specification |
| TSMC N2P design rule manual | Via NDAs / tape-out kits | Not public; request through foundry partner |
| Xilinx UG901 VHDL Reference | xilinx.com/support | Vivado VHDL synthesis constructs |
| Xilinx UG984 (XDC) | xilinx.com/support | Constraint syntax for your .xdc file |

---

## Notes on this project specifically

After reading Harris & Harris Ch. 5, go back and read `krafgpu_sm.vhd`
from top to bottom. Every design decision in that file will make sense:
the 4-stage structure, the `if_id_reg` / `id_ex_reg` pipeline registers,
the `stall_pipeline` / `flush_pipeline` signals, the forwarding mux in
`p_decode`, the priority chain in `p_writeback`. That file is a graduate
course in CPU microarchitecture compressed into 965 lines of VHDL.

After reading Kirk & Hwu Ch. 3–4, go back and read the LSU and the
warp scheduler together. The question "why does the SM have 64 warps
when WARP_SIZE is 32?" is answered in Ch. 6 of that book: latency
hiding requires enough warps that while one is stalled on L1 miss,
others keep the ALU/FPU busy. The `krafgpu_warp_scheduler`'s
round-robin policy is the simplest correct implementation of this.

After reading Weste & Harris Ch. 12, the reason the `smem_array_t`
array in `krafgpu_pkg.vhd` maps to actual silicon as an SRAM macro
(not 32768 flip-flops) becomes clear. The RTL correctly models the
behaviour; the tool maps it to the right structure. Understanding that
distinction is the core of RTL design at this level.
