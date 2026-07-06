# IndicTrans2 1B ONNX Quantization & Parity Benchmarks

This document provides detailed performance, accuracy, and model size reports for the exported and quantized IndicTrans2 ONNX bundles.
Benchmarks are computed against the **FP32 ONNX Oracle** (which matches the PyTorch model at ≥ 99.0% token parity) on direction-specific evaluation fixtures.

## EN-INDIC Model Performance

### Overall Comparison

![EN-INDIC Overall Tradeoffs](./fixtures/en_indic_1b_overall.png)

| Format | Model Size | Exact Match (Token) | Exact Match (Text) | SacreBLEU (Raw) | SacreBLEU (Mixed) | Latency (Mean) | Speedup vs. FP32 |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| FP32 | 4.19 GB | 100.00% | 100.00% | 100.00 | 100.00 | 69.5 ms | 1.000x |
| FP16 | 2.11 GB | 99.73% | 99.73% | 100.00 | 99.81 | 74.3 ms | 0.935x |
| INT8 | 1.08 GB | 89.55% | 89.55% | 96.27 | 95.04 | 31.4 ms | 2.125x |
| Q4F16 | 1.01 GB | 82.45% | 82.55% | 91.99 | 92.01 | 58.4 ms | 1.186x |

### Language-Level Performance

![EN-INDIC Language Breakdown](./fixtures/en_indic_1b_languages.png)

| Language Code | FP16 Match | FP16 BLEU | INT8 Match | INT8 BLEU | Q4F16 Match | Q4F16 BLEU |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **asm_Beng** | 100.0% | 100.00 | 92.0% | 97.28 | 82.0% | 92.61 |
| **ben_Beng** | 100.0% | 100.00 | 92.0% | 94.53 | 82.0% | 90.59 |
| **brx_Deva** | 100.0% | 100.00 | 80.0% | 91.68 | 84.0% | 93.64 |
| **doi_Deva** | 100.0% | 100.00 | 86.0% | 95.99 | 78.0% | 90.60 |
| **gom_Deva** | 100.0% | 100.00 | 94.0% | 96.83 | 76.0% | 87.97 |
| **guj_Gujr** | 100.0% | 100.00 | 90.0% | 96.73 | 86.0% | 94.93 |
| **hin_Deva** | 100.0% | 100.00 | 98.0% | 98.94 | 92.0% | 97.98 |
| **kan_Knda** | 98.0% | 98.28 | 88.0% | 94.90 | 82.0% | 90.60 |
| **kas_Arab** | 100.0% | 100.00 | 92.0% | 95.75 | 86.0% | 93.50 |
| **mai_Deva** | 100.0% | 100.00 | 92.0% | 95.54 | 90.0% | 92.18 |
| **mal_Mlym** | 100.0% | 100.00 | 100.0% | 100.00 | 90.0% | 95.25 |
| **mar_Deva** | 100.0% | 100.00 | 92.0% | 96.04 | 82.0% | 91.82 |
| **mni_Beng** | 100.0% | 100.00 | 72.0% | 84.69 | 82.0% | 88.86 |
| **npi_Deva** | 100.0% | 100.00 | 96.0% | 97.29 | 82.0% | 91.65 |
| **ory_Orya** | 100.0% | 100.00 | 90.0% | 94.87 | 90.0% | 95.40 |
| **pan_Guru** | 100.0% | 100.00 | 92.0% | 96.78 | 82.0% | 91.51 |
| **san_Deva** | 100.0% | 100.00 | 76.0% | 86.50 | 70.0% | 83.93 |
| **sat_Olck** | 100.0% | 100.00 | 66.0% | 83.16 | 44.0% | 76.72 |
| **snd_Arab** | 98.0% | 98.83 | 96.0% | 97.47 | 82.0% | 92.16 |
| **tam_Taml** | 100.0% | 100.00 | 94.0% | 96.50 | 82.0% | 91.81 |
| **tel_Telu** | 100.0% | 100.00 | 94.0% | 97.87 | 92.0% | 96.44 |
| **urd_Arab** | 98.0% | 98.92 | 98.0% | 99.36 | 98.0% | 99.61 |

### Category-Level Performance

![EN-INDIC Category Breakdown](./fixtures/en_indic_1b_categories.png)

| Category | FP16 Match | FP16 BLEU | INT8 Match | INT8 BLEU | Q4F16 Match | Q4F16 BLEU |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Generic** | 99.65% | 99.74 | 87.06% | 93.24 | 81.12% | 91.48 |
| **Lexicon** | 99.62% | 99.74 | 89.39% | 95.02 | 79.55% | 91.13 |
| **Numerals** | 100.00% | 100.00 | 90.53% | 95.92 | 84.85% | 92.92 |
| **Politics** | 99.65% | 99.74 | 91.26% | 95.94 | 84.27% | 91.88 |

---
## INDIC-EN Model Performance

### Overall Comparison

![INDIC-EN Overall Tradeoffs](./fixtures/indic_en_1b_overall.png)

| Format | Model Size | Exact Match (Token) | Exact Match (Text) | SacreBLEU (Raw) | SacreBLEU (Mixed) | Latency (Mean) | Speedup vs. FP32 |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| FP32 | 3.85 GB | 100.00% | 100.00% | 100.00 | 100.00 | 49.0 ms | 1.000x |
| FP16 | 1.94 GB | 99.82% | 99.82% | 99.96 | 99.96 | 49.7 ms | 0.987x |
| INT8 | 1.00 GB | 94.45% | 94.45% | 98.00 | 98.00 | 25.2 ms | 1.900x |
| Q4F16 | 861.5 MB | 88.55% | 88.55% | 95.44 | 95.44 | 42.7 ms | 1.080x |

### Language-Level Performance

![INDIC-EN Language Breakdown](./fixtures/indic_en_1b_languages.png)

| Language Code | FP16 Match | FP16 BLEU | INT8 Match | INT8 BLEU | Q4F16 Match | Q4F16 BLEU |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **eng_Latn** | 99.8% | 99.96 | 94.5% | 98.00 | 88.5% | 95.44 |

### Category-Level Performance

![INDIC-EN Category Breakdown](./fixtures/indic_en_1b_categories.png)

| Category | FP16 Match | FP16 BLEU | INT8 Match | INT8 BLEU | Q4F16 Match | Q4F16 BLEU |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Generic** | 100.00% | 100.00 | 91.96% | 96.82 | 81.12% | 92.66 |
| **Lexicon** | 99.62% | 99.92 | 91.67% | 97.27 | 88.26% | 95.21 |
| **Numerals** | 99.62% | 99.92 | 98.48% | 99.62 | 95.08% | 98.30 |
| **Politics** | 100.00% | 100.00 | 95.80% | 98.30 | 90.21% | 95.57 |

---
## INDIC-INDIC Model Performance

### Overall Comparison

![INDIC-INDIC Overall Tradeoffs](./fixtures/indic_indic_1b_overall.png)

| Format | Model Size | Exact Match (Token) | Exact Match (Text) | SacreBLEU (Raw) | SacreBLEU (Mixed) | Latency (Mean) | Speedup vs. FP32 |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| FP32 | 4.56 GB | 100.00% | 100.00% | 100.00 | 100.00 | 94.7 ms | 1.000x |
| FP16 | 2.31 GB | 99.82% | 99.82% | 100.00 | 99.87 | 108.3 ms | 0.874x |
| INT8 | 1.19 GB | 83.64% | 83.73% | 94.22 | 91.01 | 43.7 ms | 2.240x |
| Q4F16 | 1.21 GB | 73.18% | 73.18% | 89.33 | 85.27 | 94.2 ms | 1.087x |

### Language-Level Performance

![INDIC-INDIC Language Breakdown](./fixtures/indic_indic_1b_languages.png)

| Language Code | FP16 Match | FP16 BLEU | INT8 Match | INT8 BLEU | Q4F16 Match | Q4F16 BLEU |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **asm_Beng** | 100.0% | 100.00 | 86.0% | 94.87 | 80.0% | 90.04 |
| **ben_Beng** | 100.0% | 100.00 | 96.0% | 98.16 | 84.0% | 93.66 |
| **brx_Deva** | 100.0% | 100.00 | 80.0% | 89.49 | 70.0% | 86.74 |
| **doi_Deva** | 100.0% | 100.00 | 96.0% | 99.04 | 84.0% | 91.61 |
| **gom_Deva** | 100.0% | 100.00 | 84.0% | 92.34 | 74.0% | 60.53 |
| **guj_Gujr** | 100.0% | 100.00 | 86.0% | 93.82 | 92.0% | 97.10 |
| **hin_Deva** | 100.0% | 100.00 | 88.0% | 94.85 | 88.0% | 95.56 |
| **kan_Knda** | 100.0% | 100.00 | 88.0% | 94.76 | 74.0% | 88.55 |
| **kas_Arab** | 100.0% | 100.00 | 88.0% | 94.88 | 80.0% | 91.20 |
| **mai_Deva** | 100.0% | 100.00 | 90.0% | 93.74 | 78.0% | 91.12 |
| **mal_Mlym** | 100.0% | 100.00 | 84.0% | 93.90 | 78.0% | 90.64 |
| **mar_Deva** | 98.0% | 98.85 | 86.0% | 94.54 | 70.0% | 86.08 |
| **mni_Beng** | 98.0% | 97.89 | 38.0% | 54.09 | 36.0% | 51.53 |
| **npi_Deva** | 100.0% | 100.00 | 78.0% | 87.37 | 66.0% | 79.76 |
| **ory_Orya** | 100.0% | 100.00 | 88.0% | 94.91 | 76.0% | 88.61 |
| **pan_Guru** | 100.0% | 100.00 | 94.0% | 96.84 | 80.0% | 91.95 |
| **san_Deva** | 100.0% | 100.00 | 84.0% | 90.62 | 76.0% | 87.90 |
| **sat_Olck** | 100.0% | 100.00 | 80.0% | 78.79 | 54.0% | 88.60 |
| **snd_Arab** | 100.0% | 100.00 | 66.0% | 82.16 | 46.0% | 54.02 |
| **tam_Taml** | 100.0% | 100.00 | 80.0% | 91.63 | 68.0% | 85.23 |
| **tel_Telu** | 100.0% | 100.00 | 92.0% | 95.85 | 76.0% | 90.25 |
| **urd_Arab** | 100.0% | 100.00 | 88.0% | 96.35 | 80.0% | 90.43 |

### Category-Level Performance

![INDIC-INDIC Category Breakdown](./fixtures/indic_indic_1b_categories.png)

| Category | FP16 Match | FP16 BLEU | INT8 Match | INT8 BLEU | Q4F16 Match | Q4F16 BLEU |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Generic** | 100.00% | 100.00 | 84.62% | 91.26 | 72.73% | 85.86 |
| **Lexicon** | 99.24% | 99.48 | 79.92% | 85.80 | 68.18% | 84.83 |
| **Numerals** | 100.00% | 100.00 | 84.09% | 93.63 | 73.11% | 82.06 |
| **Politics** | 100.00% | 100.00 | 85.66% | 92.32 | 78.32% | 86.29 |

---