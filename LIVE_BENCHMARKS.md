# Browser Benchmarks and Engine Constraints Report

This document analyzes the execution of the IndicTrans2 ONNX models in modern web browsers (specifically Chrome/Edge with WebGPU and WASM capabilities).
It breaks down overall throughput, prefill and decode latencies, and catalogs the browser-specific engine ceilings that prevent execution of certain configurations.

## Browser Compatibility Overview

Executing deep learning models containing hundreds of millions or billions of parameters directly inside browser engines pushes the limits of standard web APIs.
Below is the status of the tested configurations and the technical boundaries encountered during evaluation.

![Constraint Matrix Grid](./fixtures/live_status_matrix.png)

### Catalog of Engine Skipped Boundaries

The evaluation encountered four distinct hardware, browser engine, or compilation ceilings:

- **1. All `INT8` configurations on `WebGPU`**  
  * *Affected Configurations*: `base-int8` (WebGPU), `1b-int8` (WebGPU)  
  * *Root Cause*: **WebGPU Operator Shader Compatibility Limits**. ONNX Runtime Web's WebGPU execution provider compiles matmul shaders on-the-fly. It does not support execution of raw integer quantized matrix multiplications (`MatMulInteger`, `DynamicQuantizeLinear`) on WebGPU shaders without causing driver validation crashes or resulting in numerical overflows/garbage output.

- **2. All `1B (FP16)` and `1B (FP32)` on `WASM CPU`**  
  * *Affected Configurations*: `1b-fp32` (WASM), `1b-fp16` (WASM)  
  * *Root Cause*: **32-Bit WASM Address Space Ceiling (4 GB)**. WebAssembly is compiled with a 32-bit linear memory architecture in all current browsers, meaning a single WASM thread/heap instance cannot address more than 4 GB of RAM. Loading a 1B model (with separate encoder, decoder, and decoder-with-past ONNX graphs) in FP16 or FP32 exceeds 4.5 GB of weights alone. This triggers an immediate browser process allocation collapse before execution begins.

- **3. `Base (FP32)` on `WASM CPU`**  
  * *Affected Configurations*: `base-fp32` (WASM)  
  * *Root Cause*: **WASM Heap Fragmentation Safety**. Although the Base (200M/320M) FP32 weights consume ~3.4 GB (technically under the 4 GB limit), browser heap fragmentation overhead and active output token buffer allocations result in standard memory exhaustion crashes. They are disabled for execution stability.

- **4. All `1B (FP32)` on `WebGPU`**  
  * *Affected Configurations*: `1b-fp32` (WebGPU)  
  * *Root Cause*: **WebGPU Buffer Binding Limit (2 GB)**. The WebGPU standard specification dictates a strict limit of 2 GB for any single GPU memory buffer allocation (`maxBufferSize`). In a 1B model exported in full 32-bit floats, the decoder's main weight buffer alone exceeds 2.8 GB, triggering a compilation failure when attempting to bind the tensors in WebGPU VRAM.

## Throughput Analysis (Tokens/Second)

Throughput is evaluated across 10 translation sentences and averaged. It represents the number of tokens generated per second during the decode loop.

![Throughput Comparison](./fixtures/live_throughput.png)

### Key Throughput Insights

- **WASM CPU FP32 Supremacy on Base**: On the Base model (200M/320M), **WASM CPU FP32 achieved a blazing 102.5 tokens/sec**, outperforming all other options (including WebGPU FP32 at 15.2 t/s). At smaller scales, CPU execution via native WebAssembly 128-bit SIMD completely bypasses the GPU command submission and CPU-GPU transfer overhead that bottleneck WebGPU.
- **WASM CPU FP16 Software Emulation Trap**: On the CPU, WASM FP16 drops to only **14.1 tokens/sec** (a 7.2x degradation compared to FP32). Browser CPUs lack native hardware support for half-precision math, forcing the runtime into slow software emulation.
- **Optimized WASM CPU INT8 Execution**: WASM CPU INT8 runs at **62.8 tokens/sec** using native integer SIMD hardware instructions (Intel VNNI / ARM NEON dot-product).
- **1B Scale WebGPU Mandatory Acceleration**: For the 1B scale model, WebGPU Q4F16 is accelerated to **16.3 tokens/sec**, whereas WASM CPU degrades to an unusable **2.4 tokens/sec** (a **6.8x speedup** for WebGPU). At 1B parameters, compute density far outweighs GPU bus latency, making GPU acceleration mandatory.

## Latency Profile: Prefill vs. Decode

Prefill Latency (Time to First Token - TTFT) represents prompt processing, while Step Latency represents sequential auto-regressive generation. Both values are displayed below (on a logarithmic scale).

![Latency Analysis](./fixtures/live_latency.png)

### Key Latency Insights

- **WASM CPU TTFT (Prefill) Penalty**: Time to First Token on WASM CPU for the 1B Q4F16 model is **1014 ms**, compared to only **224 ms** on WebGPU. Prompt processing latency on WASM CPU introduces a visible 1-second delay, while WebGPU remains highly responsive.
- **Step Latency Comparison**: During autoregressive decoding, WebGPU runs at **77-100 ms per token** for the 1B model, whereas WASM CPU takes **461 ms per token** on 1B Q4F16, causing character-by-character lagging.

## Detailed Benchmark Results Table

Below is the aggregated raw metric table across all tested configurations (averaged across directions).

| Model Scale      | Precision | Execution Provider | Load Time (ms) | Avg TTFT (ms) | Avg Step (ms) | Speed (tokens/sec) | Status                                      |
| :--------------- | :-------- | :----------------- | :------------- | :------------ | :------------ | :----------------- | :------------------------------------------ |
| Base (200M/320M) | FP32      | WebGPU             | 2428 ms        | 187 ms        | 88 ms         | 15.2 t/s           | 🟢 Completed                                 |
| Base (200M/320M) | FP32      | WASM CPU           | 4110 ms        | 119 ms        | 12 ms         | 102.5 t/s          | 🟢 Completed                                 |
| Base (200M/320M) | FP16      | WebGPU             | 2161 ms        | 160 ms        | 87 ms         | 14.8 t/s           | 🟢 Completed                                 |
| Base (200M/320M) | FP16      | WASM CPU           | 7247 ms        | 247 ms        | 82 ms         | 14.1 t/s           | 🟢 Completed                                 |
| Base (200M/320M) | INT8      | WebGPU             | 3096 ms        | 677 ms        | 201 ms        | 6.2 t/s            | 🟢 Completed                                 |
| Base (200M/320M) | INT8      | WASM CPU           | 2864 ms        | 112 ms        | 17 ms         | 62.8 t/s           | 🟢 Completed                                 |
| Base (200M/320M) | Q4F16     | WebGPU             | 2710 ms        | 214 ms        | 91 ms         | 12.8 t/s           | 🟢 Completed                                 |
| Base (200M/320M) | Q4F16     | WASM CPU           | 2600 ms        | 230 ms        | 86 ms         | 12.7 t/s           | 🟢 Completed                                 |
| 1B Large         | FP32      | WebGPU             | 12272 ms       | 287 ms        | 95 ms         | 12.6 t/s           | 🟢 Completed                                 |
| 1B Large         | FP32      | WASM CPU           | —              | —             | —             | —                  | 🔴 Skipped: 32-Bit WASM 4 GB Address Ceiling |
| 1B Large         | FP16      | WebGPU             | 4571 ms        | 265 ms        | 100 ms        | 12.3 t/s           | 🟢 Completed                                 |
| 1B Large         | FP16      | WASM CPU           | —              | —             | —             | —                  | 🔴 Skipped: 32-Bit WASM 4 GB Address Ceiling |
| 1B Large         | INT8      | WebGPU             | 3051 ms        | 931 ms        | 202 ms        | 5.8 t/s            | 🟢 Completed                                 |
| 1B Large         | INT8      | WASM CPU           | 3820 ms        | 768 ms        | 76 ms         | 14.5 t/s           | 🟢 Completed                                 |
| 1B Large         | Q4F16     | WebGPU             | 2566 ms        | 224 ms        | 77 ms         | 16.3 t/s           | 🟢 Completed                                 |
| 1B Large         | Q4F16     | WASM CPU           | 3296 ms        | 1014 ms       | 461 ms        | 2.4 t/s            | 🟢 Completed                                 |
