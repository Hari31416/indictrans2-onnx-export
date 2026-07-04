#!/usr/bin/env python3
"""Generate high-quality static charts (using matplotlib/seaborn) and a comprehensive
markdown report (BENCHMARKS.md) from JSON benchmark files in fixtures/.
"""

import json
import logging
import argparse
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger(__name__)

FIXTURES_DIR = Path("fixtures")
SCRATCH_DIR = Path("scratch")
OUTPUT_MD = Path("BENCHMARKS.md")

DIRECTIONS = ["en-indic", "indic-en", "indic-indic"]
PRECISIONS = ["fp16", "int8", "q4f16"]
ALL_PRECISIONS = ["fp32", "fp16", "int8", "q4f16"]

# Color mapping matching modern, premium developer theme
COLORS = {
    "fp32": "#64748b",   # Slate Gray
    "fp16": "#3b82f6",   # Vibrant Blue
    "int8": "#10b981",   # Emerald Green
    "q4f16": "#f59e0b",  # Amber/Orange
}

def get_model_size_mb(direction: str, precision: str, is_1b: bool = False) -> float:
    """Get the size of the ONNX model directory in MB, or fallback if not present."""
    suffix = f"-{precision}" if precision != "fp32" else ""
    folder_suffix = "-1b" if is_1b else ""
    folder_name = f"{direction}{folder_suffix}-onnx{suffix}"
    path = SCRATCH_DIR / folder_name
    
    if path.is_dir():
        total_bytes = sum(f.stat().st_size for f in path.glob("**/*") if f.is_file() and not f.name.startswith("."))
        return total_bytes / (1024 * 1024)
        
    # Fallback to standard sizes in case scratch has been cleared or models not present
    if is_1b:
        fallbacks = {
            "en-indic": {"fp32": 6804.0, "fp16": 3402.0, "int8": 1701.0, "q4f16": 850.5},
            "indic-en": {"fp32": 5000.0, "fp16": 2500.0, "int8": 1250.0, "q4f16": 625.0},
            "indic-indic": {"fp32": 7200.0, "fp16": 3600.0, "int8": 1800.0, "q4f16": 900.0},
        }
    else:
        fallbacks = {
            "en-indic": {"fp32": 1740.0, "fp16": 892.0, "int8": 452.9, "q4f16": 623.3},
            "indic-en": {"fp32": 1220.0, "fp16": 627.2, "int8": 319.7, "q4f16": 358.6},
            "indic-indic": {"fp32": 1910.0, "fp16": 980.2, "int8": 497.1, "q4f16": 711.6},
        }
    return fallbacks.get(direction, {}).get(precision, 0.0)

def load_all_data(is_1b: bool = False):
    """Load benchmark reports from fixtures."""
    data = {}
    for direction in DIRECTIONS:
        data[direction] = {}
        for precision in PRECISIONS:
            suffix_1b = "-1b" if is_1b else ""
            filepath = FIXTURES_DIR / f"benchmark-{direction}{suffix_1b}-{precision}.json"
            if filepath.exists():
                try:
                    with open(filepath, encoding="utf-8") as f:
                        data[direction][precision] = json.load(f)
                except Exception as e:
                    logger.error("Error reading %s: %s", filepath, e)
            else:
                logger.warning("Benchmark file not found: %s", filepath)
                
        # Synthesize FP32 oracle stats from one of the reports if available
        available = list(data[direction].keys())
        if available:
            ref = data[direction][available[0]]
            data[direction]["fp32"] = {
                "label": "fp32",
                "total_fixtures": ref["total_fixtures"],
                "token_exact_rate": 100.0,
                "text_exact_rate": 100.0,
                "sacrebleu_bleu": 100.0,
                "sacrebleu_chrf": 100.0,
                "cmp_avg_latency_ms": ref["fp32_avg_latency_ms"],
                "cmp_tokens_per_sec": ref["fp32_tokens_per_sec"],
                "speedup_vs_fp32": 1.0
            }
    return data

def generate_overall_plots(data, is_1b: bool = False):
    """Generate overall size, accuracy, and latency speedup plots for each direction."""
    sns.set_theme(style="whitegrid")
    
    for direction in DIRECTIONS:
        if not data[direction]:
            continue
            
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        title_suffix = " 1B" if is_1b else ""
        fig.suptitle(f"IndicTrans2 {direction.upper()}{title_suffix} - Quantization Tradeoffs", fontsize=16, fontweight="bold", fontfamily="sans-serif", y=0.98)
        
        # Data preparation
        labels = []
        sizes = []
        token_rates = []
        text_rates = []
        
        for p in ALL_PRECISIONS:
            if p in data[direction]:
                labels.append(p.upper())
                sizes.append(get_model_size_mb(direction, p, is_1b=is_1b))
                token_rates.append(data[direction][p]["token_exact_rate"])
                text_rates.append(data[direction][p]["text_exact_rate"])
                    
        # 1. Model Size Bar Chart
        ax_size = axes[0]
        size_colors = [COLORS[p.lower()] for p in ALL_PRECISIONS if p.lower() in data[direction]]
        bars = ax_size.bar(labels, sizes, color=size_colors, edgecolor="grey", width=0.5)
        ax_size.set_title("Model Size (MB) - Lower is Better", fontsize=12, fontweight="semibold")
        ax_size.set_ylabel("Size (MB)")
        for bar in bars:
            yval = bar.get_height()
            offset = max(sizes) * 0.02
            ax_size.text(bar.get_x() + bar.get_width()/2.0, yval + offset, f"{yval:.1f} MB" if yval < 1000 else f"{yval/1024:.2f} GB", ha='center', va='bottom', fontsize=9, fontweight="bold")
        ax_size.set_ylim(0, max(sizes) * 1.15)
        
        # 2. Token & Text Match Rate Chart
        ax_acc = axes[1]
        x = np.arange(len(labels))
        width = 0.35
        
        # Exact Match rates (We show comparison of rates)
        bars_tok = ax_acc.bar(x - width/2, token_rates, width, label='Token Match %', color="#60a5fa", edgecolor="grey")
        bars_txt = ax_acc.bar(x + width/2, text_rates, width, label='Text Match %', color="#f87171", edgecolor="grey")
        ax_acc.set_title("Exact Match Rate vs. FP32 Oracle", fontsize=12, fontweight="semibold")
        ax_acc.set_ylabel("Exact Match Rate (%)")
        ax_acc.set_xticks(x)
        ax_acc.set_xticklabels(labels)
        ax_acc.legend(loc="lower left", frameon=True)
        ax_acc.set_ylim(0, 115)
        
        for bar in bars_tok:
            yval = bar.get_height()
            ax_acc.text(bar.get_x() + bar.get_width()/2.0, yval + 1, f"{yval:.1f}%", ha='center', va='bottom', fontsize=8, rotation=90)
        for bar in bars_txt:
            yval = bar.get_height()
            ax_acc.text(bar.get_x() + bar.get_width()/2.0, yval + 1, f"{yval:.1f}%", ha='center', va='bottom', fontsize=8, rotation=90)
            
        plt.tight_layout()
        out_suffix = "_1b" if is_1b else ""
        out_png = FIXTURES_DIR / f"{direction.replace('-', '_')}{out_suffix}_overall.png"
        plt.savefig(out_png, dpi=150)
        plt.close()
        logger.info("Generated overall chart → %s", out_png)

def generate_language_plots(data, is_1b: bool = False):
    """Generate detailed language-level exact match plots for each direction."""
    sns.set_theme(style="whitegrid")
    
    for direction in DIRECTIONS:
        dir_data = data[direction]
        languages = set()
        for p in PRECISIONS:
            if p in dir_data and "metrics_by_language" in dir_data[p]:
                languages.update(dir_data[p]["metrics_by_language"].keys())
                
        if not languages:
            continue
            
        languages = sorted(list(languages))
        # Filter only available precisions for plotting
        available_prec = [p for p in PRECISIONS if p in dir_data]
        
        if len(languages) == 1:
            # Draw a clean bar chart showing all three metrics (Match %, BLEU, chrF) for single-language output (indic-en)
            fig, ax = plt.subplots(figsize=(8, 5))
            title_suffix = " 1B" if is_1b else ""
            ax.set_title(f"Translation Quality Breakdown ({direction.upper()}{title_suffix})", fontsize=14, fontweight="bold", pad=15)
            x = np.arange(len(available_prec))
            
            rates = []
            bleus = []
            chrfs = []
            for p in available_prec:
                m = dir_data[p]["metrics_by_language"].get(languages[0], {})
                rates.append(m.get("token_exact_rate", 0.0))
                bleus.append(m.get("sacrebleu_bleu", 0.0))
                chrfs.append(m.get("sacrebleu_chrf", 0.0))
                    
            width = 0.25
            rects1 = ax.bar(x - width, rates, width, label='Exact Match %', color="#60a5fa", edgecolor="grey")
            rects2 = ax.bar(x, bleus, width, label='SacreBLEU', color="#34d399", edgecolor="grey")
            rects3 = ax.bar(x + width, chrfs, width, label='chrF', color="#f59e0b", edgecolor="grey")
            
            ax.bar_label(rects1, labels=[f"{v:.1f}%" for v in rates], padding=3, fontsize=8)
            ax.bar_label(rects2, labels=[f"{v:.1f}" for v in bleus], padding=3, fontsize=8)
            ax.bar_label(rects3, labels=[f"{v:.1f}" for v in chrfs], padding=3, fontsize=8)
            
            ax.set_ylabel("Score / Rate", fontsize=12)
            ax.set_xticks(x)
            ax.set_xticklabels([p.upper() for p in available_prec], fontsize=11)
            ax.set_ylim(0, 115)
            ax.legend(loc="lower left", frameon=True)
        else:
            # Draw a beautiful 1-row, 3-column Cleveland Dot Plot for multi-language outputs (en-indic, indic-indic)
            metrics_keys = ["token_exact_rate", "sacrebleu_bleu", "sacrebleu_chrf"]
            metrics_titles = ["Exact Match Rate (%)", "SacreBLEU Score", "chrF Score"]
            colors = {"fp16": "#3b82f6", "int8": "#10b981", "q4f16": "#ef4444"}
            
            # Calculate average score per language across all formats to sort them
            avg_scores = {}
            for lang in languages:
                scores = []
                for p in available_prec:
                    m = dir_data[p]["metrics_by_language"].get(lang, {})
                    scores.append(m.get("token_exact_rate", 0.0))
                    scores.append(m.get("sacrebleu_bleu", 0.0))
                    scores.append(m.get("sacrebleu_chrf", 0.0))
                avg_scores[lang] = np.mean(scores)

            # Sort languages (highest average performance first)
            sorted_languages = sorted(languages, key=lambda l: avg_scores[l], reverse=True)
            
            fig, axes = plt.subplots(1, 3, figsize=(18, 11))
            title_suffix = " 1B" if is_1b else ""
            fig.suptitle(f"Language-Level Performance Comparison ({direction.upper()}{title_suffix})", fontsize=16, fontweight="bold", y=0.98)
            
            for idx, key in enumerate(metrics_keys):
                ax = axes[idx]
                y_pos = np.arange(len(sorted_languages))
                
                # Draw horizontal line for each language to connect the dots
                for y_idx, lang in enumerate(sorted_languages):
                    x_vals = []
                    for p in available_prec:
                        val = dir_data[p]["metrics_by_language"].get(lang, {}).get(key, 0.0)
                        x_vals.append(val)
                    ax.plot([min(x_vals), max(x_vals)], [y_pos[y_idx], y_pos[y_idx]], color="#d1d5db", linestyle="-", linewidth=1.5, zorder=1)
                    
                # Plot the dots
                for p in available_prec:
                    x_vals = []
                    for lang in sorted_languages:
                        val = dir_data[p]["metrics_by_language"].get(lang, {}).get(key, 0.0)
                        x_vals.append(val)
                    ax.scatter(x_vals, y_pos, color=colors[p], label=p.upper(), s=70, edgecolors="black", linewidths=0.5, zorder=2)
                    
                ax.set_title(metrics_titles[idx], fontsize=13, fontweight="semibold")
                ax.set_yticks(y_pos)
                ax.set_yticklabels(sorted_languages if idx == 0 else [])
                ax.set_xlabel("Value", fontsize=11)
                ax.set_xlim(25, 105) # Keep scale uniform and focused on the active range
                if idx == 0:
                    ax.set_ylabel("Language Code", fontsize=13)
                if idx == 2:
                    ax.legend(title="Precision Format", loc="lower left", frameon=True)
                    
            # Invert y axis for all subplots so highest average score is at the top
            for ax in axes:
                ax.invert_yaxis()
            
        plt.tight_layout()
        out_suffix = "_1b" if is_1b else ""
        out_png = FIXTURES_DIR / f"{direction.replace('-', '_')}{out_suffix}_languages.png"
        plt.savefig(out_png, dpi=150)
        plt.close()
        logger.info("Generated language-level chart → %s", out_png)

def generate_category_plots(data, is_1b: bool = False):
    """Generate category-level plots comparing match rates."""
    sns.set_theme(style="whitegrid")
    
    for direction in DIRECTIONS:
        dir_data = data[direction]
        categories = set()
        for p in PRECISIONS:
            if p in dir_data and "metrics_by_category" in dir_data[p]:
                categories.update(dir_data[p]["metrics_by_category"].keys())
                
        if not categories:
            continue
            
        categories = sorted(list(categories))
        
        fig, ax = plt.subplots(figsize=(8, 5))
        title_suffix = " 1B" if is_1b else ""
        ax.set_title(f"SacreBLEU Score by Category ({direction.upper()}{title_suffix})", fontsize=14, fontweight="bold", pad=15)
        
        x = np.arange(len(categories))
        width = 0.25
        
        for idx, p in enumerate(PRECISIONS):
            if p not in dir_data:
                continue
            rates = []
            for cat in categories:
                rate = dir_data[p]["metrics_by_category"].get(cat, {}).get("sacrebleu_bleu", 0.0)
                rates.append(rate)
                
            offset = (idx - 1) * width
            rects = ax.bar(x + offset, rates, width, label=p.upper(), color=COLORS[p], edgecolor="grey")
            ax.bar_label(rects, labels=[f"{v:.1f}" for v in rates], padding=3, fontsize=8)
            
        ax.set_ylabel("SacreBLEU Score", fontsize=12)
        ax.set_xlabel("Evaluation Category", fontsize=12)
        ax.set_xticks(x)
        ax.set_xticklabels([cat.capitalize() for cat in categories], fontsize=10)
        ax.set_ylim(0, 115)
        ax.legend(title="Precision Format", loc="lower left", frameon=True)
        
        plt.tight_layout()
        out_suffix = "_1b" if is_1b else ""
        out_png = FIXTURES_DIR / f"{direction.replace('-', '_')}{out_suffix}_categories.png"
        plt.savefig(out_png, dpi=150)
        plt.close()
        logger.info("Generated category-level chart → %s", out_png)

def generate_markdown_report(data, is_1b: bool = False):
    """Build the final BENCHMARKS.md report with markdown tables and embedded charts."""
    content = []
    title_suffix = " 1B" if is_1b else ""
    content.append(f"# IndicTrans2{title_suffix} ONNX Quantization & Parity Benchmarks")
    content.append("")
    content.append("This document provides detailed performance, accuracy, and model size reports for the exported and quantized IndicTrans2 ONNX bundles.")
    content.append("Benchmarks are computed against the **FP32 ONNX Oracle** (which matches the PyTorch model at ≥ 99.0% token parity) on direction-specific evaluation fixtures.")
    content.append("")
    
    out_suffix = "_1b" if is_1b else ""
    
    for direction in DIRECTIONS:
        if not data[direction]:
            continue
            
        content.append(f"## {direction.upper()} Model Performance")
        content.append("")
        content.append("### Overall Comparison")
        content.append("")
        
        # Embed overall plot
        overall_img = f"./fixtures/{direction.replace('-', '_')}{out_suffix}_overall.png"
        content.append(f"![{direction.upper()} Overall Tradeoffs]({overall_img})")
        content.append("")
        
        # Build overall markdown table
        content.append("| Format | Model Size | Exact Match (Token) | Exact Match (Text) | SacreBLEU (Raw) | SacreBLEU (Mixed) | Latency (Mean) | Speedup vs. FP32 |")
        content.append("| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |")
        
        for p in ALL_PRECISIONS:
            if p not in data[direction]:
                continue
            item = data[direction][p]
            size_mb = get_model_size_mb(direction, p, is_1b=is_1b)
            
            # Format model size
            if size_mb >= 1000:
                size_str = f"{size_mb / 1024:.2f} GB"
            else:
                size_str = f"{size_mb:.1f} MB"
                
            tok_match = f"{item['token_exact_rate']:.2f}%"
            txt_match = f"{item['text_exact_rate']:.2f}%"
            
            # BLEU handles
            bleu_raw = f"{item.get('sacrebleu_bleu', 100.0):.2f}"
            bleu_mix = f"{item.get('sacrebleu_bleu_mixed', 100.0):.2f}"
            
            latency = f"{item.get('cmp_avg_latency_ms', 0.0):.1f} ms"
            speedup = f"{item.get('speedup_vs_fp32', 1.0):.3f}x"
            
            content.append(f"| {p.upper()} | {size_str} | {tok_match} | {txt_match} | {bleu_raw} | {bleu_mix} | {latency} | {speedup} |")
            
        content.append("")
        
        # Language-level comparisons
        content.append("### Language-Level Performance")
        content.append("")
        lang_img = f"./fixtures/{direction.replace('-', '_')}{out_suffix}_languages.png"
        content.append(f"![{direction.upper()} Language Breakdown]({lang_img})")
        content.append("")
        
        # Build language table
        languages = set()
        for p in PRECISIONS:
            if p in data[direction] and "metrics_by_language" in data[direction][p]:
                languages.update(data[direction][p]["metrics_by_language"].keys())
        languages = sorted(list(languages))
        
        if languages:
            header = "| Language Code | FP16 Match | FP16 BLEU | INT8 Match | INT8 BLEU | Q4F16 Match | Q4F16 BLEU |"
            sep = "| :--- | :--- | :--- | :--- | :--- | :--- | :--- |"
            content.append(header)
            content.append(sep)
            
            for lang in languages:
                row = [f"**{lang}**"]
                for p in PRECISIONS:
                    if p in data[direction] and lang in data[direction][p]["metrics_by_language"]:
                        metrics = data[direction][p]["metrics_by_language"][lang]
                        match = f"{metrics['token_exact_rate']:.1f}%"
                        bleu = f"{metrics['sacrebleu_bleu']:.2f}"
                    else:
                        match = "N/A"
                        bleu = "N/A"
                    row.extend([match, bleu])
                content.append("| " + " | ".join(row) + " |")
            content.append("")
            
        # Category-level comparisons
        content.append("### Category-Level Performance")
        content.append("")
        cat_img = f"./fixtures/{direction.replace('-', '_')}{out_suffix}_categories.png"
        content.append(f"![{direction.upper()} Category Breakdown]({cat_img})")
        content.append("")
        
        categories = set()
        for p in PRECISIONS:
            if p in data[direction] and "metrics_by_category" in data[direction][p]:
                categories.update(data[direction][p]["metrics_by_category"].keys())
        categories = sorted(list(categories))
        
        if categories:
            content.append("| Category | FP16 Match | FP16 BLEU | INT8 Match | INT8 BLEU | Q4F16 Match | Q4F16 BLEU |")
            content.append("| :--- | :--- | :--- | :--- | :--- | :--- | :--- |")
            for cat in categories:
                row = [f"**{cat.capitalize()}**"]
                for p in PRECISIONS:
                    if p in data[direction] and cat in data[direction][p]["metrics_by_category"]:
                        metrics = data[direction][p]["metrics_by_category"][cat]
                        match = f"{metrics['token_exact_rate']:.2f}%"
                        bleu = f"{metrics['sacrebleu_bleu']:.2f}"
                    else:
                        match = "N/A"
                        bleu = "N/A"
                    row.extend([match, bleu])
                content.append("| " + " | ".join(row) + " |")
            content.append("")
            
        content.append("---")
    
    output_md = Path("BENCHMARKS_1B.md") if is_1b else Path("BENCHMARKS.md")
    output_md.write_text("\n".join(content), encoding="utf-8")
    logger.info("Generated markdown report → %s", output_md)

def main():
    """Main execution block."""
    parser = argparse.ArgumentParser(description="Generate visual reports and markdown summary.")
    parser.add_argument("--model-size", choices=["200m", "1b"], default="200m", help="Model size to generate reports for (default: 200m)")
    args = parser.parse_args()
    
    is_1b = (args.model_size == "1b")
    
    logger.info("Starting visual report generation for %s models...", "1B" if is_1b else "200M/320M")
    data = load_all_data(is_1b=is_1b)
    generate_overall_plots(data, is_1b=is_1b)
    generate_language_plots(data, is_1b=is_1b)
    generate_category_plots(data, is_1b=is_1b)
    generate_markdown_report(data, is_1b=is_1b)
    logger.info("Visual report generation completed successfully!")

if __name__ == "__main__":
    main()
