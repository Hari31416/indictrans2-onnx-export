# IndicTrans2 ONNX Quantization & Parity Benchmarks

This document provides detailed performance, accuracy, and model size reports for the exported and quantized IndicTrans2 ONNX bundles.
Benchmarks are computed against the **FP32 ONNX Oracle** (which matches the PyTorch model at ≥ 99.0% token parity) on direction-specific evaluation fixtures.

## EN-INDIC Model Performance

### Overall Comparison

![EN-INDIC Overall Tradeoffs](./fixtures/en_indic_overall.png)

| Format | Model Size | Exact Match (Token) | Exact Match (Text) | SacreBLEU (Raw) | SacreBLEU (Mixed) | Latency (Mean) | Speedup vs. FP32 |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| FP32 | 1.77 GB | 100.00% | 100.00% | 100.00 | 100.00 | 76.1 ms | 1.000x |
| FP16 | 926.5 MB | 99.64% | 99.64% | 100.00 | 99.81 | 82.1 ms | 0.927x |
| INT8 | 487.4 MB | 73.73% | 73.82% | 91.56 | 88.74 | 45.6 ms | 1.568x |
| Q4F16 | 657.8 MB | 55.45% | 55.73% | 83.46 | 77.78 | 74.9 ms | 1.007x |

### Language-Level Performance

![EN-INDIC Language Breakdown](./fixtures/en_indic_languages.png)

| Language Code | FP16 Match | FP16 BLEU | INT8 Match | INT8 BLEU | Q4F16 Match | Q4F16 BLEU |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **asm_Beng** | 100.0% | 100.00 | 78.0% | 92.23 | 60.0% | 80.84 |
| **ben_Beng** | 100.0% | 100.00 | 88.0% | 95.20 | 70.0% | 83.59 |
| **brx_Deva** | 98.0% | 98.94 | 58.0% | 79.85 | 38.0% | 68.76 |
| **doi_Deva** | 98.0% | 99.36 | 78.0% | 91.31 | 54.0% | 78.92 |
| **gom_Deva** | 98.0% | 98.75 | 72.0% | 86.26 | 36.0% | 68.64 |
| **guj_Gujr** | 100.0% | 100.00 | 80.0% | 93.29 | 60.0% | 85.28 |
| **hin_Deva** | 100.0% | 100.00 | 88.0% | 95.59 | 80.0% | 92.55 |
| **kan_Knda** | 100.0% | 100.00 | 78.0% | 90.28 | 60.0% | 80.07 |
| **kas_Arab** | 100.0% | 100.00 | 58.0% | 80.44 | 26.0% | 61.60 |
| **mai_Deva** | 100.0% | 100.00 | 74.0% | 89.36 | 48.0% | 75.98 |
| **mal_Mlym** | 98.0% | 98.35 | 82.0% | 90.94 | 68.0% | 84.13 |
| **mar_Deva** | 100.0% | 100.00 | 70.0% | 82.82 | 62.0% | 83.33 |
| **mni_Beng** | 100.0% | 100.00 | 44.0% | 65.63 | 26.0% | 37.18 |
| **npi_Deva** | 100.0% | 100.00 | 86.0% | 93.06 | 62.0% | 84.29 |
| **ory_Orya** | 100.0% | 100.00 | 88.0% | 93.14 | 64.0% | 84.66 |
| **pan_Guru** | 100.0% | 100.00 | 82.0% | 94.07 | 72.0% | 90.35 |
| **san_Deva** | 100.0% | 100.00 | 46.0% | 68.90 | 40.0% | 64.71 |
| **sat_Olck** | 100.0% | 100.00 | 38.0% | 79.35 | 32.0% | 62.90 |
| **snd_Arab** | 100.0% | 100.00 | 74.0% | 91.82 | 62.0% | 85.83 |
| **tam_Taml** | 100.0% | 100.00 | 92.0% | 97.35 | 58.0% | 83.88 |
| **tel_Telu** | 100.0% | 100.00 | 90.0% | 94.60 | 68.0% | 84.38 |
| **urd_Arab** | 100.0% | 100.00 | 78.0% | 93.85 | 74.0% | 90.85 |

### Category-Level Performance

![EN-INDIC Category Breakdown](./fixtures/en_indic_categories.png)

| Category     | FP16 Match | FP16 BLEU | INT8 Match | INT8 BLEU | Q4F16 Match | Q4F16 BLEU |
| :----------- | :--------- | :-------- | :--------- | :-------- | :---------- | :--------- |
| **Generic**  | 98.95%     | 99.47     | 71.68%     | 87.93     | 53.15%      | 77.00      |
| **Lexicon**  | 99.62%     | 99.77     | 70.83%     | 88.45     | 49.24%      | 77.90      |
| **Numerals** | 100.00%    | 100.00    | 74.62%     | 89.34     | 61.74%      | 81.37      |
| **Politics** | 100.00%    | 100.00    | 77.62%     | 88.35     | 57.69%      | 74.46      |

---
## INDIC-EN Model Performance

### Overall Comparison

![INDIC-EN Overall Tradeoffs](./fixtures/indic_en_overall.png)

| Format | Model Size | Exact Match (Token) | Exact Match (Text) | SacreBLEU (Raw) | SacreBLEU (Mixed) | Latency (Mean) | Speedup vs. FP32 |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| FP32 | 1.26 GB | 100.00% | 100.00% | 100.00 | 100.00 | 41.4 ms | 1.000x |
| FP16 | 661.6 MB | 99.91% | 99.91% | 99.98 | 99.98 | 47.6 ms | 0.872x |
| INT8 | 354.0 MB | 86.00% | 86.00% | 93.87 | 93.87 | 38.9 ms | 1.107x |
| Q4F16 | 392.9 MB | 74.45% | 74.45% | 89.15 | 89.15 | 47.8 ms | 0.866x |

### Language-Level Performance

![INDIC-EN Language Breakdown](./fixtures/indic_en_languages.png)

| Language Code | FP16 Match | FP16 BLEU | INT8 Match | INT8 BLEU | Q4F16 Match | Q4F16 BLEU |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **eng_Latn** | 99.9% | 99.98 | 86.0% | 93.87 | 74.5% | 89.15 |

### Category-Level Performance

![INDIC-EN Category Breakdown](./fixtures/indic_en_categories.png)

| Category     | FP16 Match | FP16 BLEU | INT8 Match | INT8 BLEU | Q4F16 Match | Q4F16 BLEU |
| :----------- | :--------- | :-------- | :--------- | :-------- | :---------- | :--------- |
| **Generic**  | 100.00%    | 100.00    | 82.52%     | 91.61     | 68.18%      | 85.40      |
| **Lexicon**  | 99.62%     | 99.91     | 85.61%     | 94.43     | 65.91%      | 87.07      |
| **Numerals** | 100.00%    | 100.00    | 87.50%     | 94.49     | 84.09%      | 93.90      |
| **Politics** | 100.00%    | 100.00    | 88.46%     | 94.88     | 79.72%      | 90.21      |

---
## INDIC-INDIC Model Performance

### Overall Comparison

![INDIC-INDIC Overall Tradeoffs](./fixtures/indic_indic_overall.png)

| Format | Model Size | Exact Match (Token) | Exact Match (Text) | SacreBLEU (Raw) | SacreBLEU (Mixed) | Latency (Mean) | Speedup vs. FP32 |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| FP32 | 1.97 GB | 100.00% | 100.00% | 100.00 | 100.00 | 72.7 ms | 1.000x |
| FP16 | 1.01 GB | 99.91% | 99.91% | 100.00 | 99.99 | 82.6 ms | 0.881x |
| INT8 | 555.6 MB | 73.00% | 73.09% | 87.82 | 86.61 | 50.0 ms | 1.554x |
| Q4F16 | 770.1 MB | 44.91% | 45.36% | 72.82 | 68.27 | 77.5 ms | 0.963x |

### Language-Level Performance

![INDIC-INDIC Language Breakdown](./fixtures/indic_indic_languages.png)

| Language Code | FP16 Match | FP16 BLEU | INT8 Match | INT8 BLEU | Q4F16 Match | Q4F16 BLEU |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **asm_Beng** | 100.0% | 100.00 | 74.0% | 86.35 | 58.0% | 78.50 |
| **ben_Beng** | 100.0% | 100.00 | 88.0% | 95.81 | 58.0% | 82.00 |
| **brx_Deva** | 100.0% | 100.00 | 68.0% | 84.35 | 32.0% | 63.55 |
| **doi_Deva** | 100.0% | 100.00 | 82.0% | 91.76 | 58.0% | 81.48 |
| **gom_Deva** | 100.0% | 100.00 | 70.0% | 81.99 | 30.0% | 61.81 |
| **guj_Gujr** | 100.0% | 100.00 | 78.0% | 91.16 | 62.0% | 84.85 |
| **hin_Deva** | 100.0% | 100.00 | 84.0% | 93.39 | 56.0% | 83.91 |
| **kan_Knda** | 100.0% | 100.00 | 90.0% | 94.53 | 56.0% | 76.71 |
| **kas_Arab** | 100.0% | 100.00 | 60.0% | 81.45 | 34.0% | 69.14 |
| **mai_Deva** | 100.0% | 100.00 | 80.0% | 91.82 | 56.0% | 80.74 |
| **mal_Mlym** | 100.0% | 100.00 | 80.0% | 91.40 | 40.0% | 65.97 |
| **mar_Deva** | 100.0% | 100.00 | 74.0% | 86.40 | 44.0% | 71.13 |
| **mni_Beng** | 100.0% | 100.00 | 64.0% | 83.62 | 30.0% | 68.26 |
| **npi_Deva** | 100.0% | 100.00 | 62.0% | 73.00 | 42.0% | 64.52 |
| **ory_Orya** | 100.0% | 100.00 | 70.0% | 87.41 | 44.0% | 74.64 |
| **pan_Guru** | 100.0% | 100.00 | 80.0% | 93.29 | 56.0% | 83.44 |
| **san_Deva** | 100.0% | 100.00 | 68.0% | 78.15 | 48.0% | 64.23 |
| **sat_Olck** | 100.0% | 100.00 | 76.0% | 93.54 | 26.0% | 42.24 |
| **snd_Arab** | 98.0% | 99.73 | 34.0% | 46.87 | 6.0% | 19.98 |
| **tam_Taml** | 100.0% | 100.00 | 66.0% | 85.20 | 44.0% | 70.43 |
| **tel_Telu** | 100.0% | 100.00 | 76.0% | 88.23 | 52.0% | 74.52 |
| **urd_Arab** | 100.0% | 100.00 | 82.0% | 94.25 | 56.0% | 83.69 |

### Category-Level Performance

![INDIC-INDIC Category Breakdown](./fixtures/indic_indic_categories.png)

| Category     | FP16 Match | FP16 BLEU | INT8 Match | INT8 BLEU | Q4F16 Match | Q4F16 BLEU |
| :----------- | :--------- | :-------- | :--------- | :-------- | :---------- | :--------- |
| **Generic**  | 100.00%    | 100.00    | 68.88%     | 83.83     | 41.61%      | 62.41      |
| **Lexicon**  | 100.00%    | 100.00    | 71.59%     | 86.13     | 41.67%      | 65.16      |
| **Numerals** | 100.00%    | 100.00    | 73.11%     | 86.45     | 48.11%      | 75.42      |
| **Politics** | 99.65%     | 99.95     | 78.32%     | 89.03     | 48.25%      | 69.49      |
