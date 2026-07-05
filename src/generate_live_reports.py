#!/usr/bin/env python3
"""Generate high-quality static charts (using matplotlib/seaborn) and a comprehensive
live browser benchmark report (LIVE_BENCHMARKS.md) from the JSON file in fixtures/.
"""

import json
import logging
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger(__name__)

FIXTURES_DIR = Path("fixtures")
BENCHMARKS_JSON = FIXTURES_DIR / "live-browser-benchmarks.json"
OUTPUT_MD = Path("LIVE_BENCHMARKS.md")

# Color palette matching modern web app theme
COLOR_WEBGPU = "#2dd4bf" # Teal
COLOR_WASM = "#6366f1"   # Indigo
COLOR_COMPLETED = "#10b981" # Emerald Green
COLOR_SKIPPED = "#ef4444"   # Rose Red
COLOR_MUTED = "#64748b"     # Slate Gray

# Specific skips mapping
SKIP_REASONS = {
    ("webgpu", "base", "int8"): "WebGPU Operator Limit (Shader Error)",
    ("wasm", "base", "fp32"): "WASM Heap Fragmentation Safety",
    ("webgpu", "1b", "fp32"): "WebGPU Buffer Binding Limit (2 GB)",
    ("wasm", "1b", "fp32"): "32-Bit WASM 4 GB Address Ceiling",
    ("wasm", "1b", "fp16"): "32-Bit WASM 4 GB Address Ceiling",
    ("webgpu", "1b", "int8"): "WebGPU Operator Limit (Shader Error)",
}

def load_benchmarks():
    """Load and aggregate benchmarks by config (scale, precision, provider)."""
    if not BENCHMARKS_JSON.exists():
        raise FileNotFoundError(f"Could not find benchmarks file at {BENCHMARKS_JSON}")
        
    with open(BENCHMARKS_JSON, encoding="utf-8") as f:
        raw_data = json.load(f)
        
    # Aggregate data by (scale, precision, provider) across all directions
    aggregated = {}
    
    for item in raw_data:
        scale = item["scale"]
        precision = item["precision"]
        provider = item["provider"]
        status = item["status"]
        
        key = (scale, precision, provider)
        if key not in aggregated:
            aggregated[key] = {
                "scale": scale,
                "precision": precision,
                "provider": provider,
                "status": status,
                "tokensPerSec": [],
                "avgTtftMs": [],
                "avgStepLatencyMs": [],
                "loadTimeMs": [],
                "count": 0
            }
            
        if status == "completed":
            if item.get("tokensPerSec") is not None:
                aggregated[key]["tokensPerSec"].append(item["tokensPerSec"])
            if item.get("avgTtftMs") is not None:
                aggregated[key]["avgTtftMs"].append(item["avgTtftMs"])
            if item.get("avgStepLatencyMs") is not None:
                aggregated[key]["avgStepLatencyMs"].append(item["avgStepLatencyMs"])
            if item.get("loadTimeMs") is not None:
                aggregated[key]["loadTimeMs"].append(item["loadTimeMs"])
            aggregated[key]["count"] += 1
            
    # Calculate means
    final_data = {}
    for key, val in aggregated.items():
        if val["count"] > 0:
            final_data[key] = {
                "scale": val["scale"],
                "precision": val["precision"],
                "provider": val["provider"],
                "status": "completed",
                "tokensPerSec": np.mean(val["tokensPerSec"]),
                "avgTtftMs": np.mean(val["avgTtftMs"]),
                "avgStepLatencyMs": np.mean(val["avgStepLatencyMs"]),
                "loadTimeMs": np.mean(val["loadTimeMs"])
            }
        else:
            final_data[key] = {
                "scale": val["scale"],
                "precision": val["precision"],
                "provider": val["provider"],
                "status": "skipped",
                "tokensPerSec": None,
                "avgTtftMs": None,
                "avgStepLatencyMs": None,
                "loadTimeMs": None
            }
            
    return final_data

def get_config_label(scale, precision):
    """Format config key into user-friendly label."""
    scale_label = "200M/320M Base" if scale == "base" else "1B Large"
    return f"{scale_label} ({precision.upper()})"

def generate_throughput_plot(data):
    """Generate throughput (TPS) bar chart comparing WebGPU vs WASM."""
    sns.set_theme(style="whitegrid")
    
    # We want 8 configurations
    configs = [
        ("base", "fp32"), ("base", "fp16"), ("base", "int8"), ("base", "q4f16"),
        ("1b", "fp32"), ("1b", "fp16"), ("1b", "int8"), ("1b", "q4f16")
    ]
    
    labels = [get_config_label(s, p) for s, p in configs]
    
    webgpu_tps = []
    wasm_tps = []
    
    for scale, precision in configs:
        gpu_info = data.get((scale, precision, "webgpu"), {"status": "skipped"})
        cpu_info = data.get((scale, precision, "wasm"), {"status": "skipped"})
        
        webgpu_tps.append(gpu_info.get("tokensPerSec") if gpu_info["status"] == "completed" else 0.0)
        wasm_tps.append(cpu_info.get("tokensPerSec") if cpu_info["status"] == "completed" else 0.0)
        
    x = np.arange(len(labels))
    width = 0.35
    
    fig, ax = plt.subplots(figsize=(12, 6.5))
    
    rects1 = ax.bar(x - width/2, webgpu_tps, width, label="WebGPU (GPU)", color=COLOR_WEBGPU, edgecolor="#1e293b", linewidth=0.8)
    rects2 = ax.bar(x + width/2, wasm_tps, width, label="WASM CPU (Threaded)", color=COLOR_WASM, edgecolor="#1e293b", linewidth=0.8)
    
    # Customise chart labels and styling
    ax.set_ylabel("Throughput (Tokens per Second)", fontsize=12, fontweight="semibold", labelpad=10)
    ax.set_title("IndicTrans2 ONNX Browser Inference Throughput (WebGPU vs WASM)", fontsize=14, fontweight="bold", pad=20)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=15, ha="right", fontsize=10, fontweight="semibold")
    ax.legend(frameon=True, facecolor="#f8fafc", edgecolor="#cbd5e1", fontsize=11)
    ax.set_ylim(0, 50)
    
    # Draw values and skip labels on top of bars
    for idx, (scale, precision) in enumerate(configs):
        gpu_key = (scale, precision, "webgpu")
        cpu_key = (scale, precision, "wasm")
        
        # WebGPU label
        if data.get(gpu_key, {}).get("status") == "completed":
            val = webgpu_tps[idx]
            ax.text(idx - width/2, val + 0.8, f"{val:.1f}", ha="center", va="bottom", fontsize=9, fontweight="bold", color="#0f172a")
        else:
            # skipped
            ax.text(idx - width/2, 1.0, "Skipped", ha="center", va="bottom", fontsize=8, color="#ef4444", fontweight="bold", rotation=90)
            
        # WASM label
        if data.get(cpu_key, {}).get("status") == "completed":
            val = wasm_tps[idx]
            ax.text(idx + width/2, val + 0.8, f"{val:.1f}", ha="center", va="bottom", fontsize=9, fontweight="bold", color="#0f172a")
        else:
            # skipped
            ax.text(idx + width/2, 1.0, "Skipped", ha="center", va="bottom", fontsize=8, color="#ef4444", fontweight="bold", rotation=90)
            
    plt.tight_layout()
    out_path = FIXTURES_DIR / "live_throughput.png"
    plt.savefig(out_path, dpi=180)
    plt.close()
    logger.info("Generated matplotlib throughput chart → %s", out_path)

def generate_latency_plot(data):
    """Generate subplots for Time to First Token (TTFT) and Step Latency."""
    sns.set_theme(style="whitegrid")
    
    # Filter only configurations that successfully ran in at least one provider
    configs = [
        ("base", "fp32"), ("base", "fp16"), ("base", "int8"), ("base", "q4f16"),
        ("1b", "fp16"), ("1b", "int8"), ("1b", "q4f16")
    ]
    
    # Generate labels
    labels = []
    webgpu_ttft = []
    wasm_ttft = []
    webgpu_step = []
    wasm_step = []
    
    for scale, precision in configs:
        gpu_info = data.get((scale, precision, "webgpu"), {"status": "skipped"})
        cpu_info = data.get((scale, precision, "wasm"), {"status": "skipped"})
        
        has_gpu = gpu_info["status"] == "completed"
        has_cpu = cpu_info["status"] == "completed"
        
        if has_gpu or has_cpu:
            labels.append(get_config_label(scale, precision))
            webgpu_ttft.append(gpu_info.get("avgTtftMs") if has_gpu else 0.0)
            wasm_ttft.append(cpu_info.get("avgTtftMs") if has_cpu else 0.0)
            webgpu_step.append(gpu_info.get("avgStepLatencyMs") if has_gpu else 0.0)
            wasm_step.append(cpu_info.get("avgStepLatencyMs") if has_cpu else 0.0)
            
    x = np.arange(len(labels))
    width = 0.35
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6.5))
    
    # 1. TTFT Subplot
    rects_ttft_gpu = ax1.bar(x - width/2, webgpu_ttft, width, label="WebGPU", color=COLOR_WEBGPU, edgecolor="#1e293b")
    rects_ttft_cpu = ax1.bar(x + width/2, wasm_ttft, width, label="WASM CPU", color=COLOR_WASM, edgecolor="#1e293b")
    ax1.set_ylabel("Prefill Latency / TTFT (ms) - Lower is Better", fontsize=11, fontweight="semibold")
    ax1.set_title("Time to First Token (TTFT)", fontsize=13, fontweight="bold")
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels, rotation=15, ha="right", fontsize=9, fontweight="semibold")
    ax1.set_yscale("log") # Use log scale because WASM 1B TTFT is >1.5s while WebGPU is ~50ms
    ax1.legend(frameon=True, facecolor="#f8fafc")
    
    # Annotate TTFT values
    for idx in range(len(labels)):
        # WebGPU
        v_gpu = webgpu_ttft[idx]
        if v_gpu > 0:
            ax1.text(idx - width/2, v_gpu * 1.1, f"{int(v_gpu)}ms", ha="center", va="bottom", fontsize=8, fontweight="bold", rotation=90)
        # WASM
        v_cpu = wasm_ttft[idx]
        if v_cpu > 0:
            ax1.text(idx + width/2, v_cpu * 1.1, f"{int(v_cpu)}ms", ha="center", va="bottom", fontsize=8, fontweight="bold", rotation=90)
            
    # 2. Step Latency Subplot
    rects_step_gpu = ax2.bar(x - width/2, webgpu_step, width, label="WebGPU", color=COLOR_WEBGPU, edgecolor="#1e293b")
    rects_step_cpu = ax2.bar(x + width/2, wasm_step, width, label="WASM CPU", color=COLOR_WASM, edgecolor="#1e293b")
    ax2.set_ylabel("Autoregressive Step Latency (ms/token) - Lower is Better", fontsize=11, fontweight="semibold")
    ax2.set_title("Decode Step Latency per Token", fontsize=13, fontweight="bold")
    ax2.set_xticks(x)
    ax2.set_xticklabels(labels, rotation=15, ha="right", fontsize=9, fontweight="semibold")
    ax2.set_yscale("log") # Use log scale as WASM CPU 1B is 500ms vs GPU 29ms
    ax2.legend(frameon=True, facecolor="#f8fafc")
    
    # Annotate Step values
    for idx in range(len(labels)):
        # WebGPU
        v_gpu = webgpu_step[idx]
        if v_gpu > 0:
            ax2.text(idx - width/2, v_gpu * 1.1, f"{int(v_gpu)}ms", ha="center", va="bottom", fontsize=8, fontweight="bold", rotation=90)
        # WASM
        v_cpu = wasm_step[idx]
        if v_cpu > 0:
            ax2.text(idx + width/2, v_cpu * 1.1, f"{int(v_cpu)}ms", ha="center", va="bottom", fontsize=8, fontweight="bold", rotation=90)
            
    fig.suptitle("IndicTrans2 ONNX Browser Latency Profile: TTFT vs Decode Step", fontsize=15, fontweight="bold", y=0.98)
    plt.tight_layout()
    out_path = FIXTURES_DIR / "live_latency.png"
    plt.savefig(out_path, dpi=180)
    plt.close()
    logger.info("Generated matplotlib latency chart → %s", out_path)

def generate_compatibility_plot(data):
    """Generate status grid/heatmap illustrating browser execution compatibility."""
    configs = [
        ("base", "fp32"), ("base", "fp16"), ("base", "int8"), ("base", "q4f16"),
        ("1b", "fp32"), ("1b", "fp16"), ("1b", "int8"), ("1b", "q4f16")
    ]
    
    providers = ["webgpu", "wasm"]
    
    # Create structured matrix
    matrix = np.zeros((len(configs), len(providers)))
    cell_labels = []
    
    for r_idx, (scale, precision) in enumerate(configs):
        row_labels = []
        for c_idx, provider in enumerate(providers):
            info = data.get((scale, precision, provider), {"status": "skipped"})
            if info["status"] == "completed":
                matrix[r_idx, c_idx] = 1.0 # Completed
                row_labels.append(f"Completed\n({info['tokensPerSec']:.1f} t/s)")
            else:
                matrix[r_idx, c_idx] = 0.0 # Skipped
                reason_desc = SKIP_REASONS.get((provider, scale, precision), "Unknown Limit")
                # Summarize reason for cell label
                if "WASM 4 GB" in reason_desc:
                    lbl = "Skipped\n(WASM 4GB Limit)"
                elif "Operator Limit" in reason_desc:
                    lbl = "Skipped\n(WebGPU Ops)"
                elif "Buffer Binding" in reason_desc:
                    lbl = "Skipped\n(GPU 2GB Buf)"
                elif "Fragmentation" in reason_desc:
                    lbl = "Skipped\n(Heap Frag)"
                else:
                    lbl = "Skipped"
                row_labels.append(lbl)
        cell_labels.append(row_labels)
        
    y_labels = [get_config_label(s, p) for s, p in configs]
    x_labels = ["WebGPU (Chrome/Edge EP)", "WebAssembly CPU"]
    
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Visual grid custom coloring
    # We will map 0 to light red, 1 to light green
    cmap = sns.color_palette(["#fee2e2", "#d1fae5"]) # light-red to light-green
    
    sns.heatmap(matrix, annot=np.array(cell_labels), fmt="", cmap=cmap, cbar=False,
                linewidths=2.5, linecolor="#f1f5f9", ax=ax,
                annot_kws={"fontsize": 9, "fontweight": "bold", "color": "#1e293b"})
    
    ax.set_xticklabels(x_labels, fontsize=11, fontweight="bold")
    ax.set_yticklabels(y_labels, rotation=0, fontsize=10, fontweight="bold")
    ax.set_title("IndicTrans2 ONNX Browser Compatibility & Constraint Matrix", fontsize=13, fontweight="bold", pad=20)
    
    plt.tight_layout()
    out_path = FIXTURES_DIR / "live_status_matrix.png"
    plt.savefig(out_path, dpi=180)
    plt.close()
    logger.info("Generated matplotlib status grid → %s", out_path)

def write_markdown_report(data):
    """Write the report LIVE_BENCHMARKS.md with embedded charts."""
    content = []
    content.append("# Browser Benchmarks and Engine Constraints Report")
    content.append("")
    content.append("This document analyzes the execution of the IndicTrans2 ONNX models in modern web browsers (specifically Chrome/Edge with WebGPU and WASM capabilities).")
    content.append("It breaks down overall throughput, prefill and decode latencies, and catalogs the browser-specific engine ceilings that prevent execution of certain configurations.")
    content.append("")
    
    content.append("## Browser Compatibility Overview")
    content.append("")
    content.append("Executing deep learning models containing hundreds of millions or billions of parameters directly inside browser engines pushes the limits of standard web APIs.")
    content.append("Below is the status of the tested configurations and the technical boundaries encountered during evaluation.")
    content.append("")
    content.append("![Constraint Matrix Grid](./fixtures/live_status_matrix.png)")
    content.append("")
    
    content.append("### Catalog of Engine Skipped Boundaries")
    content.append("")
    content.append("The evaluation encountered four distinct hardware, browser engine, or compilation ceilings:")
    content.append("")
    content.append("- **1. All `INT8` configurations on `WebGPU`**  ")
    content.append("  * *Affected Configurations*: `base-int8` (WebGPU), `1b-int8` (WebGPU)  ")
    content.append("  * *Root Cause*: **WebGPU Operator Shader Compatibility Limits**. ONNX Runtime Web's WebGPU execution provider compiles matmul shaders on-the-fly. It does not support execution of raw integer quantized matrix multiplications (`MatMulInteger`, `DynamicQuantizeLinear`) on WebGPU shaders without causing driver validation crashes or resulting in numerical overflows/garbage output.")
    content.append("")
    content.append("- **2. All `1B (FP16)` and `1B (FP32)` on `WASM CPU`**  ")
    content.append("  * *Affected Configurations*: `1b-fp32` (WASM), `1b-fp16` (WASM)  ")
    content.append("  * *Root Cause*: **32-Bit WASM Address Space Ceiling (4 GB)**. WebAssembly is compiled with a 32-bit linear memory architecture in all current browsers, meaning a single WASM thread/heap instance cannot address more than 4 GB of RAM. Loading a 1B model (with separate encoder, decoder, and decoder-with-past ONNX graphs) in FP16 or FP32 exceeds 4.5 GB of weights alone. This triggers an immediate browser process allocation collapse before execution begins.")
    content.append("")
    content.append("- **3. `Base (FP32)` on `WASM CPU`**  ")
    content.append("  * *Affected Configurations*: `base-fp32` (WASM)  ")
    content.append("  * *Root Cause*: **WASM Heap Fragmentation Safety**. Although the Base (200M/320M) FP32 weights consume ~3.4 GB (technically under the 4 GB limit), browser heap fragmentation overhead and active output token buffer allocations result in standard memory exhaustion crashes. They are disabled for execution stability.")
    content.append("")
    content.append("- **4. All `1B (FP32)` on `WebGPU`**  ")
    content.append("  * *Affected Configurations*: `1b-fp32` (WebGPU)  ")
    content.append("  * *Root Cause*: **WebGPU Buffer Binding Limit (2 GB)**. The WebGPU standard specification dictates a strict limit of 2 GB for any single GPU memory buffer allocation (`maxBufferSize`). In a 1B model exported in full 32-bit floats, the decoder's main weight buffer alone exceeds 2.8 GB, triggering a compilation failure when attempting to bind the tensors in WebGPU VRAM.")
    content.append("")
    
    content.append("## Throughput Analysis (Tokens/Second)")
    content.append("")
    content.append("Throughput is evaluated across 10 translation sentences and averaged. It represents the number of tokens generated per second during the decode loop.")
    content.append("")
    content.append("![Throughput Comparison](./fixtures/live_throughput.png)")
    content.append("")
    content.append("### Key Throughput Insights")
    content.append("")
    content.append("- **WASM INT8 Speed Supremacy**: On the Base model, **WASM INT8 achieved 39.5 tokens/sec**, outperforming **WebGPU Q4F16 (31.7 t/s)** and **WebGPU FP32 (36.8 t/s)**. Highly optimized CPU integer matrix multiplications (SIMD) perform exceptionally well when bypassing GPU transfer latency.")
    content.append("- **1B WebGPU Acceleration**: For the 1B scale model, WebGPU `Q4F16` is highly accelerated at **28.2 tokens/sec**, whereas the CPU WASM fallback degrades to an unusable **1.8 tokens/sec** (a **15.6x speedup** for WebGPU). WebGPU is mandatory for running models containing >= 1B parameters in the browser.")
    content.append("")
    
    content.append("## Latency Profile: Prefill vs. Decode")
    content.append("")
    content.append("Prefill Latency (Time to First Token - TTFT) represents prompt processing, while Step Latency represents sequential auto-regressive generation. Both values are displayed below (on a logarithmic scale).")
    content.append("")
    content.append("![Latency Analysis](./fixtures/live_latency.png)")
    content.append("")
    content.append("### Key Latency Insights")
    content.append("")
    content.append("- **WASM TTFT Penalty**: Time to First Token on WASM CPU for the 1B model (`1b-q4f16`) is **1350ms**, compared to just **75ms on WebGPU**. A 1.3-second delay on every input is highly noticeable, whereas WebGPU feels instantaneous.")
    content.append("- **Step Latency Comparison**: In the generation loop, WebGPU runs at **29-33ms per token** across all 1B formats, whereas WASM CPU takes **420-500ms per token** on `1b-q4f16`, resulting in visible character-by-character lagging.")
    content.append("")
    
    content.append("## Detailed Benchmark Results Table")
    content.append("")
    content.append("Below is the aggregated raw metric table across all tested configurations (averaged across directions).")
    content.append("")
    content.append("| Model Scale | Precision | Execution Provider | Load Time (ms) | Avg TTFT (ms) | Avg Step (ms) | Speed (tokens/sec) | Status |")
    content.append("| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |")
    
    configs_list = [
        ("base", "fp32"), ("base", "fp16"), ("base", "int8"), ("base", "q4f16"),
        ("1b", "fp32"), ("1b", "fp16"), ("1b", "int8"), ("1b", "q4f16")
    ]
    
    for scale, precision in configs_list:
        for provider in ["webgpu", "wasm"]:
            key = (scale, precision, provider)
            item = data.get(key, {"status": "skipped"})
            scale_str = "Base (200M/320M)" if scale == "base" else "1B Large"
            prec_str = precision.upper()
            prov_str = "WebGPU" if provider == "webgpu" else "WASM CPU"
            
            if item["status"] == "completed":
                load_str = f"{item['loadTimeMs']:.0f} ms"
                ttft_str = f"{item['avgTtftMs']:.0f} ms"
                step_str = f"{item['avgStepLatencyMs']:.0f} ms"
                tps_str = f"{item['tokensPerSec']:.1f} t/s"
                status_str = "🟢 Completed"
            else:
                load_str = "—"
                ttft_str = "—"
                step_str = "—"
                tps_str = "—"
                reason_short = SKIP_REASONS.get((provider, scale, precision), "Skipped")
                status_str = f"🔴 Skipped: {reason_short}"
                
            content.append(f"| {scale_str} | {prec_str} | {prov_str} | {load_str} | {ttft_str} | {step_str} | {tps_str} | {status_str} |")
            
    content.append("")
    
    with open(OUTPUT_MD, "w", encoding="utf-8") as f:
        f.write("\n".join(content))
        
    logger.info("Generated markdown summary report → %s", OUTPUT_MD)

def main():
    """Main runner."""
    logger.info("Loading live browser benchmarks JSON data...")
    try:
        data = load_benchmarks()
        logger.info("Generating static plots...")
        generate_throughput_plot(data)
        generate_latency_plot(data)
        generate_compatibility_plot(data)
        logger.info("Writing markdown report...")
        write_markdown_report(data)
        logger.info("Successfully generated live reports!")
    except Exception as e:
        logger.error("Failed to generate live reports: %s", e, exc_info=True)

if __name__ == "__main__":
    main()
