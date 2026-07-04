# IndicTrans2 ONNX export pipeline — run individual steps via `make <target>`
#
# Quick start:
#   make setup              # create venv + install deps
#   make indic-en           # full P0 pipeline (export → tokenizers → validate)
#   make help               # list all targets

.DEFAULT_GOAL := help

VENV          ?= .venv
PYTHON        := $(VENV)/bin/python
UV            ?= uv
SCRATCH       ?= ./scratch
HF_ORG        ?= hari31416
ONNX_OPSET    ?= 17

EN_INDIC_MODEL    := ai4bharat/indictrans2-en-indic-dist-200M
INDIC_EN_MODEL    := ai4bharat/indictrans2-indic-en-dist-200M
INDIC_INDIC_MODEL := ai4bharat/indictrans2-indic-indic-dist-320M

EN_INDIC_OUT          := $(SCRATCH)/en-indic-onnx
EN_INDIC_INT8_OUT     := $(SCRATCH)/en-indic-onnx-int8
EN_INDIC_FP16_OUT     := $(SCRATCH)/en-indic-onnx-fp16
EN_INDIC_Q4F16_OUT    := $(SCRATCH)/en-indic-onnx-q4f16
INDIC_EN_OUT          := $(SCRATCH)/indic-en-onnx
INDIC_EN_INT8_OUT     := $(SCRATCH)/indic-en-onnx-int8
INDIC_EN_FP16_OUT     := $(SCRATCH)/indic-en-onnx-fp16
INDIC_EN_Q4F16_OUT    := $(SCRATCH)/indic-en-onnx-q4f16
INDIC_INDIC_OUT       := $(SCRATCH)/indic-indic-onnx
INDIC_INDIC_INT8_OUT  := $(SCRATCH)/indic-indic-onnx-int8
INDIC_INDIC_FP16_OUT  := $(SCRATCH)/indic-indic-onnx-fp16
INDIC_INDIC_Q4F16_OUT := $(SCRATCH)/indic-indic-onnx-q4f16

# ── 1B Models ────────────────────────────────────────────────────────────────
EN_INDIC_1B_MODEL    := ai4bharat/indictrans2-en-indic-1B
INDIC_EN_1B_MODEL    := ai4bharat/indictrans2-indic-en-1B
INDIC_INDIC_1B_MODEL := ai4bharat/indictrans2-indic-indic-1B

EN_INDIC_1B_OUT          := $(SCRATCH)/en-indic-1b-onnx
EN_INDIC_1B_INT8_OUT     := $(SCRATCH)/en-indic-1b-onnx-int8
EN_INDIC_1B_FP16_OUT     := $(SCRATCH)/en-indic-1b-onnx-fp16
EN_INDIC_1B_Q4F16_OUT    := $(SCRATCH)/en-indic-1b-onnx-q4f16
INDIC_EN_1B_OUT          := $(SCRATCH)/indic-en-1b-onnx
INDIC_EN_1B_INT8_OUT     := $(SCRATCH)/indic-en-1b-onnx-int8
INDIC_EN_1B_FP16_OUT     := $(SCRATCH)/indic-en-1b-onnx-fp16
INDIC_EN_1B_Q4F16_OUT    := $(SCRATCH)/indic-en-1b-onnx-q4f16
INDIC_INDIC_1B_OUT       := $(SCRATCH)/indic-indic-1b-onnx
INDIC_INDIC_1B_INT8_OUT  := $(SCRATCH)/indic-indic-1b-onnx-int8
INDIC_INDIC_1B_FP16_OUT  := $(SCRATCH)/indic-indic-1b-onnx-fp16
INDIC_INDIC_1B_Q4F16_OUT := $(SCRATCH)/indic-indic-1b-onnx-q4f16

EN_INDIC_1B_REPORT    := fixtures/parity-report-en-indic-1b.json
INDIC_EN_1B_REPORT    := fixtures/parity-report-indic-en-1b.json
INDIC_INDIC_1B_REPORT := fixtures/parity-report-indic-indic-1b.json

EN_INDIC_1B_BENCH_INT8    := fixtures/benchmark-en-indic-1b-int8.json
EN_INDIC_1B_BENCH_FP16    := fixtures/benchmark-en-indic-1b-fp16.json
EN_INDIC_1B_BENCH_Q4F16   := fixtures/benchmark-en-indic-1b-q4f16.json
INDIC_EN_1B_BENCH_INT8    := fixtures/benchmark-indic-en-1b-int8.json
INDIC_EN_1B_BENCH_FP16    := fixtures/benchmark-indic-en-1b-fp16.json
INDIC_EN_1B_BENCH_Q4F16   := fixtures/benchmark-indic-en-1b-q4f16.json
INDIC_INDIC_1B_BENCH_INT8  := fixtures/benchmark-indic-indic-1b-int8.json
INDIC_INDIC_1B_BENCH_FP16  := fixtures/benchmark-indic-indic-1b-fp16.json
INDIC_INDIC_1B_BENCH_Q4F16 := fixtures/benchmark-indic-indic-1b-q4f16.json

Q4F16_BLOCK_SIZE ?= 32

EN_INDIC_FIXTURES    := fixtures/en-indic-golden.jsonl
INDIC_EN_FIXTURES    := fixtures/indic-en-golden.jsonl
INDIC_INDIC_FIXTURES := fixtures/indic-indic-golden.jsonl

EN_INDIC_REPORT    := fixtures/parity-report-en-indic.json
INDIC_EN_REPORT    := fixtures/parity-report-indic-en.json
INDIC_INDIC_REPORT := fixtures/parity-report-indic-indic.json

# Benchmark reports (fp32 ONNX oracle vs quantized tiers)
EN_INDIC_BENCH_INT8    := fixtures/benchmark-en-indic-int8.json
EN_INDIC_BENCH_FP16    := fixtures/benchmark-en-indic-fp16.json
EN_INDIC_BENCH_Q4F16   := fixtures/benchmark-en-indic-q4f16.json
INDIC_EN_BENCH_INT8    := fixtures/benchmark-indic-en-int8.json
INDIC_EN_BENCH_FP16    := fixtures/benchmark-indic-en-fp16.json
INDIC_EN_BENCH_Q4F16   := fixtures/benchmark-indic-en-q4f16.json
INDIC_INDIC_BENCH_INT8  := fixtures/benchmark-indic-indic-int8.json
INDIC_INDIC_BENCH_FP16  := fixtures/benchmark-indic-indic-fp16.json
INDIC_INDIC_BENCH_Q4F16 := fixtures/benchmark-indic-indic-q4f16.json

.PHONY: help setup install clean clean-all preview reports reports-1b \
	export-en-indic tokenizers-en-indic validate-en-indic quantize-en-indic \
	convert-fp16-en-indic quantize-q4f16-en-indic \
	benchmark-int8-en-indic benchmark-fp16-en-indic benchmark-q4f16-en-indic \
	capture-fixtures-en-indic upload-en-indic en-indic \
	export-indic-en tokenizers-indic-en validate-indic-en quantize-indic-en \
	convert-fp16-indic-en quantize-q4f16-indic-en \
	benchmark-int8-indic-en benchmark-fp16-indic-en benchmark-q4f16-indic-en \
	capture-fixtures-indic-en upload-indic-en indic-en \
	export-indic-indic tokenizers-indic-indic validate-indic-indic quantize-indic-indic \
	convert-fp16-indic-indic quantize-q4f16-indic-indic \
	benchmark-int8-indic-indic benchmark-fp16-indic-indic benchmark-q4f16-indic-indic \
	capture-fixtures-indic-indic upload-indic-indic indic-indic \
	quantize-all quantize-int8-all convert-fp16-all quantize-q4f16-all \
	benchmark-all benchmark-int8-all benchmark-fp16-all benchmark-q4f16-all \
	quantize-1b-all quantize-int8-1b-all convert-fp16-1b-all quantize-q4f16-1b-all \
	benchmark-1b-all benchmark-int8-1b-all benchmark-fp16-1b-all benchmark-q4f16-1b-all


help: ## Show available targets
	@echo "IndicTrans2 ONNX export — make targets"
	@echo ""
	@echo "Setup & Batch Operations:"
	@echo "  make setup                  Create venv + install deps"
	@echo "  make clean                  Remove scratch ONNX artifacts"
	@echo "  make clean-all              Remove scratch + .venv"
	@echo "  make preview                Local preview of the ONNX components guide"
	@echo "  make reports                Generate benchmark reports and plots (overall, language, and category levels)"
	@echo "  make quantize-all           Quantize all 3 models to both INT8 & Q4F16 (incl. FP16)"
	@echo "  make benchmark-all          Evaluate/benchmark all variants (INT8, FP16, Q4F16)"
	@echo ""
	@echo "en→indic (200M):"
	@echo "  make export-en-indic"
	@echo "  make tokenizers-en-indic"
	@echo "  make validate-en-indic"
	@echo "  make quantize-en-indic"
	@echo "  make convert-fp16-en-indic   fp32 → fp16  (src/05_convert_fp16.py)"
	@echo "  make quantize-q4f16-en-indic fp16 → q4f16 (src/06_quantize_q4f16.py)"
	@echo "  make benchmark-fp16-en-indic  fp16 vs fp32 quality + speed"
	@echo "  make benchmark-q4f16-en-indic q4f16 vs fp32 quality + speed"
	@echo "  make capture-fixtures-en-indic"
	@echo "  make upload-en-indic"
	@echo "  make en-indic               Steps 1–3"
	@echo ""
	@echo "indic→en (P0, 200M):"
	@echo "  make export-indic-en        Step 1 — ONNX graphs"
	@echo "  make tokenizers-indic-en    Step 2 — fast tokenizers"
	@echo "  make validate-indic-en      Step 3 — parity vs PyTorch"
	@echo "  make quantize-indic-en      Step 4 — INT8 (after fp32 passes)"
	@echo "  make convert-fp16-indic-en   fp32 → fp16  (src/05_convert_fp16.py)"
	@echo "  make quantize-q4f16-indic-en fp16 → q4f16 (src/06_quantize_q4f16.py)"
	@echo "  make benchmark-fp16-indic-en  fp16 vs fp32 quality + speed"
	@echo "  make benchmark-q4f16-indic-en q4f16 vs fp32 quality + speed"
	@echo "  make capture-fixtures-indic-en  Generate golden fixtures"
	@echo "  make upload-indic-en        Upload bundle to HF (HF_ORG=$(HF_ORG))"
	@echo "  make indic-en               Steps 1–3"
	@echo ""
	@echo "indic→indic (P1, 320M):"
	@echo "  make export-indic-indic"
	@echo "  make tokenizers-indic-indic"
	@echo "  make validate-indic-indic"
	@echo "  make quantize-indic-indic"
	@echo "  make convert-fp16-indic-indic   fp32 → fp16  (src/05_convert_fp16.py)"
	@echo "  make quantize-q4f16-indic-indic fp16 → q4f16 (src/06_quantize_q4f16.py)"
	@echo "  make benchmark-fp16-indic-indic  fp16 vs fp32 quality + speed"
	@echo "  make benchmark-q4f16-indic-indic q4f16 vs fp32 quality + speed"
	@echo "  make capture-fixtures-indic-indic"
	@echo "  make upload-indic-indic"
	@echo "  make indic-indic            Steps 1–3"
	@echo ""
	@echo "Variables: HF_ORG=$(HF_ORG)  SCRATCH=$(SCRATCH)  ONNX_OPSET=$(ONNX_OPSET)  Q4F16_BLOCK_SIZE=$(Q4F16_BLOCK_SIZE)"

# ── Setup ────────────────────────────────────────────────────────────────────

setup: $(VENV)/bin/python ## Create venv and install Python deps

$(VENV)/bin/python: pyproject.toml
	$(UV) sync

install: setup ## Alias for setup

# ── en→indic ─────────────────────────────────────────────────────────────────

export-en-indic: setup ## Export encoder + decoder ONNX (en→indic)
	$(PYTHON) src/01_export_encoder_decoder.py \
		--model $(EN_INDIC_MODEL) \
		--output $(EN_INDIC_OUT) \
		--opset $(ONNX_OPSET)

tokenizers-en-indic: setup ## Build fast tokenizers (en→indic)
	$(PYTHON) src/02_build_fast_tokenizers.py \
		--model $(EN_INDIC_MODEL) \
		--output $(EN_INDIC_OUT)

validate-en-indic: setup ## Validate ONNX parity (en→indic)
	$(PYTHON) src/03_validate_parity.py \
		--onnx-dir $(EN_INDIC_OUT) \
		--pytorch-model $(EN_INDIC_MODEL) \
		--fixtures $(EN_INDIC_FIXTURES) \
		--report $(EN_INDIC_REPORT)

quantize-en-indic: setup ## INT8 quantize fp32 bundle (en→indic)
	$(PYTHON) src/04_quantize_int8.py \
		--input $(EN_INDIC_OUT) \
		--output $(EN_INDIC_INT8_OUT)

capture-fixtures-en-indic: setup ## Capture golden fixtures (en→indic)
	$(PYTHON) src/03_validate_parity.py \
		--pytorch-model $(EN_INDIC_MODEL) \
		--capture-fixtures $(EN_INDIC_FIXTURES)

upload-en-indic: upload-en-indic-fp32 ## Alias for upload-en-indic-fp32

upload-en-indic-fp32: setup ## Upload fp32 bundle to Hugging Face (en→indic)
	$(PYTHON) src/05_upload_hf.py \
		--model-dir $(EN_INDIC_OUT) \
		--repo-id $(HF_ORG)/indictrans2-en-indic-dist-200M-ONNX \
		--commit-message "$(COMMIT_MESSAGE)"

upload-en-indic-int8: setup ## Upload int8 bundle (en→indic)
	$(PYTHON) src/05_upload_hf.py \
		--model-dir $(EN_INDIC_INT8_OUT) \
		--repo-id $(HF_ORG)/indictrans2-en-indic-dist-200M-ONNX-int8 \
		--commit-message "$(COMMIT_MESSAGE)"

upload-en-indic-fp16: setup ## Upload fp16 bundle (en→indic)
	$(PYTHON) src/05_upload_hf.py \
		--model-dir $(EN_INDIC_FP16_OUT) \
		--repo-id $(HF_ORG)/indictrans2-en-indic-dist-200M-ONNX-fp16 \
		--commit-message "$(COMMIT_MESSAGE)"

upload-en-indic-q4f16: setup ## Upload q4f16 bundle (en→indic)
	$(PYTHON) src/05_upload_hf.py \
		--model-dir $(EN_INDIC_Q4F16_OUT) \
		--repo-id $(HF_ORG)/indictrans2-en-indic-dist-200M-ONNX-q4f16 \
		--commit-message "$(COMMIT_MESSAGE)"

upload-en-indic-all: upload-en-indic-fp32 upload-en-indic-int8 upload-en-indic-fp16 upload-en-indic-q4f16 ## Upload all precision variants (en→indic)

convert-fp16-en-indic: setup ## Convert fp32 → fp16 (en→indic)
	$(PYTHON) src/05_convert_fp16.py \
		--input $(EN_INDIC_OUT) \
		--output $(EN_INDIC_FP16_OUT)

quantize-q4f16-en-indic: setup ## q4f16 quantize fp16 bundle (en→indic)
	$(PYTHON) src/06_quantize_q4f16.py \
		--input $(EN_INDIC_FP16_OUT) \
		--output $(EN_INDIC_Q4F16_OUT) \
		--block-size $(Q4F16_BLOCK_SIZE)

benchmark-int8-en-indic: setup ## Benchmark INT8 vs fp32 oracle (en→indic)
	$(PYTHON) src/07_benchmark_precision.py \
		--fp32-dir  $(EN_INDIC_OUT) \
		--cmp-dir   $(EN_INDIC_INT8_OUT) \
		--fixtures  $(EN_INDIC_FIXTURES) \
		--pytorch-model $(EN_INDIC_MODEL) \
		--label int8 \
		--report $(EN_INDIC_BENCH_INT8)

benchmark-fp16-en-indic: setup ## Benchmark fp16 vs fp32 oracle (en→indic)
	$(PYTHON) src/07_benchmark_precision.py \
		--fp32-dir  $(EN_INDIC_OUT) \
		--cmp-dir   $(EN_INDIC_FP16_OUT) \
		--fixtures  $(EN_INDIC_FIXTURES) \
		--pytorch-model $(EN_INDIC_MODEL) \
		--label fp16 \
		--report $(EN_INDIC_BENCH_FP16)

benchmark-q4f16-en-indic: setup ## Benchmark q4f16 vs fp32 oracle (en→indic)
	$(PYTHON) src/07_benchmark_precision.py \
		--fp32-dir  $(EN_INDIC_OUT) \
		--cmp-dir   $(EN_INDIC_Q4F16_OUT) \
		--fixtures  $(EN_INDIC_FIXTURES) \
		--pytorch-model $(EN_INDIC_MODEL) \
		--label q4f16 \
		--report $(EN_INDIC_BENCH_Q4F16)

en-indic: export-en-indic tokenizers-en-indic validate-en-indic ## Full en→indic pipeline (steps 1–3)

# ── indic→en ─────────────────────────────────────────────────────────────────

export-indic-en: setup ## Export encoder + decoder ONNX (indic→en)
	$(PYTHON) src/01_export_encoder_decoder.py \
		--model $(INDIC_EN_MODEL) \
		--output $(INDIC_EN_OUT) \
		--opset $(ONNX_OPSET)

tokenizers-indic-en: setup ## Build fast tokenizers (indic→en)
	$(PYTHON) src/02_build_fast_tokenizers.py \
		--model $(INDIC_EN_MODEL) \
		--output $(INDIC_EN_OUT)

validate-indic-en: setup ## Validate ONNX parity (indic→en)
	$(PYTHON) src/03_validate_parity.py \
		--onnx-dir $(INDIC_EN_OUT) \
		--pytorch-model $(INDIC_EN_MODEL) \
		--fixtures $(INDIC_EN_FIXTURES) \
		--report $(INDIC_EN_REPORT)

quantize-indic-en: setup ## INT8 quantize fp32 bundle (indic→en)
	$(PYTHON) src/04_quantize_int8.py \
		--input $(INDIC_EN_OUT) \
		--output $(INDIC_EN_INT8_OUT)

capture-fixtures-indic-en: setup ## Capture golden fixtures (indic→en)
	$(PYTHON) src/03_validate_parity.py \
		--pytorch-model $(INDIC_EN_MODEL) \
		--capture-fixtures $(INDIC_EN_FIXTURES)

upload-indic-en: upload-indic-en-fp32 ## Alias for upload-indic-en-fp32

upload-indic-en-fp32: setup ## Upload fp32 bundle to Hugging Face (indic→en)
	$(PYTHON) src/05_upload_hf.py \
		--model-dir $(INDIC_EN_OUT) \
		--repo-id $(HF_ORG)/indictrans2-indic-en-dist-200M-ONNX \
		--commit-message "$(COMMIT_MESSAGE)"

upload-indic-en-int8: setup ## Upload int8 bundle (indic→en)
	$(PYTHON) src/05_upload_hf.py \
		--model-dir $(INDIC_EN_INT8_OUT) \
		--repo-id $(HF_ORG)/indictrans2-indic-en-dist-200M-ONNX-int8 \
		--commit-message "$(COMMIT_MESSAGE)"

upload-indic-en-fp16: setup ## Upload fp16 bundle (indic→en)
	$(PYTHON) src/05_upload_hf.py \
		--model-dir $(INDIC_EN_FP16_OUT) \
		--repo-id $(HF_ORG)/indictrans2-indic-en-dist-200M-ONNX-fp16 \
		--commit-message "$(COMMIT_MESSAGE)"

upload-indic-en-q4f16: setup ## Upload q4f16 bundle (indic→en)
	$(PYTHON) src/05_upload_hf.py \
		--model-dir $(INDIC_EN_Q4F16_OUT) \
		--repo-id $(HF_ORG)/indictrans2-indic-en-dist-200M-ONNX-q4f16 \
		--commit-message "$(COMMIT_MESSAGE)"

upload-indic-en-all: upload-indic-en-fp32 upload-indic-en-int8 upload-indic-en-fp16 upload-indic-en-q4f16 ## Upload all precision variants (indic→en)

convert-fp16-indic-en: setup ## Convert fp32 → fp16 (indic→en)
	$(PYTHON) src/05_convert_fp16.py \
		--input $(INDIC_EN_OUT) \
		--output $(INDIC_EN_FP16_OUT)

quantize-q4f16-indic-en: setup ## q4f16 quantize fp16 bundle (indic→en)
	$(PYTHON) src/06_quantize_q4f16.py \
		--input $(INDIC_EN_FP16_OUT) \
		--output $(INDIC_EN_Q4F16_OUT) \
		--block-size $(Q4F16_BLOCK_SIZE)

benchmark-int8-indic-en: setup ## Benchmark INT8 vs fp32 oracle (indic→en)
	$(PYTHON) src/07_benchmark_precision.py \
		--fp32-dir  $(INDIC_EN_OUT) \
		--cmp-dir   $(INDIC_EN_INT8_OUT) \
		--fixtures  $(INDIC_EN_FIXTURES) \
		--pytorch-model $(INDIC_EN_MODEL) \
		--label int8 \
		--report $(INDIC_EN_BENCH_INT8)

benchmark-fp16-indic-en: setup ## Benchmark fp16 vs fp32 oracle (indic→en)
	$(PYTHON) src/07_benchmark_precision.py \
		--fp32-dir  $(INDIC_EN_OUT) \
		--cmp-dir   $(INDIC_EN_FP16_OUT) \
		--fixtures  $(INDIC_EN_FIXTURES) \
		--pytorch-model $(INDIC_EN_MODEL) \
		--label fp16 \
		--report $(INDIC_EN_BENCH_FP16)

benchmark-q4f16-indic-en: setup ## Benchmark q4f16 vs fp32 oracle (indic→en)
	$(PYTHON) src/07_benchmark_precision.py \
		--fp32-dir  $(INDIC_EN_OUT) \
		--cmp-dir   $(INDIC_EN_Q4F16_OUT) \
		--fixtures  $(INDIC_EN_FIXTURES) \
		--pytorch-model $(INDIC_EN_MODEL) \
		--label q4f16 \
		--report $(INDIC_EN_BENCH_Q4F16)

indic-en: export-indic-en tokenizers-indic-en validate-indic-en ## Full indic→en pipeline (steps 1–3)

# ── indic→indic ──────────────────────────────────────────────────────────────

export-indic-indic: setup ## Export encoder + decoder ONNX (indic→indic)
	$(PYTHON) src/01_export_encoder_decoder.py \
		--model $(INDIC_INDIC_MODEL) \
		--output $(INDIC_INDIC_OUT) \
		--opset $(ONNX_OPSET)

tokenizers-indic-indic: setup ## Build fast tokenizers (indic→indic)
	$(PYTHON) src/02_build_fast_tokenizers.py \
		--model $(INDIC_INDIC_MODEL) \
		--output $(INDIC_INDIC_OUT)

validate-indic-indic: setup ## Validate ONNX parity (indic→indic)
	$(PYTHON) src/03_validate_parity.py \
		--onnx-dir $(INDIC_INDIC_OUT) \
		--pytorch-model $(INDIC_INDIC_MODEL) \
		--fixtures $(INDIC_INDIC_FIXTURES) \
		--report $(INDIC_INDIC_REPORT)

quantize-indic-indic: setup ## INT8 quantize fp32 bundle (indic→indic)
	$(PYTHON) src/04_quantize_int8.py \
		--input $(INDIC_INDIC_OUT) \
		--output $(INDIC_INDIC_INT8_OUT)

capture-fixtures-indic-indic: setup ## Capture golden fixtures (indic→indic)
	$(PYTHON) src/03_validate_parity.py \
		--pytorch-model $(INDIC_INDIC_MODEL) \
		--capture-fixtures $(INDIC_INDIC_FIXTURES)

upload-indic-indic: upload-indic-indic-fp32 ## Alias for upload-indic-indic-fp32

upload-indic-indic-fp32: setup ## Upload fp32 bundle to Hugging Face (indic→indic)
	$(PYTHON) src/05_upload_hf.py \
		--model-dir $(INDIC_INDIC_OUT) \
		--repo-id $(HF_ORG)/indictrans2-indic-indic-dist-320M-ONNX \
		--commit-message "$(COMMIT_MESSAGE)"

upload-indic-indic-int8: setup ## Upload int8 bundle (indic→indic)
	$(PYTHON) src/05_upload_hf.py \
		--model-dir $(INDIC_INDIC_INT8_OUT) \
		--repo-id $(HF_ORG)/indictrans2-indic-indic-dist-320M-ONNX-int8 \
		--commit-message "$(COMMIT_MESSAGE)"

upload-indic-indic-fp16: setup ## Upload fp16 bundle (indic→indic)
	$(PYTHON) src/05_upload_hf.py \
		--model-dir $(INDIC_INDIC_FP16_OUT) \
		--repo-id $(HF_ORG)/indictrans2-indic-indic-dist-320M-ONNX-fp16 \
		--commit-message "$(COMMIT_MESSAGE)"

upload-indic-indic-q4f16: setup ## Upload q4f16 bundle (indic→indic)
	$(PYTHON) src/05_upload_hf.py \
		--model-dir $(INDIC_INDIC_Q4F16_OUT) \
		--repo-id $(HF_ORG)/indictrans2-indic-indic-dist-320M-ONNX-q4f16 \
		--commit-message "$(COMMIT_MESSAGE)"

upload-indic-indic-all: upload-indic-indic-fp32 upload-indic-indic-int8 upload-indic-indic-fp16 upload-indic-indic-q4f16 ## Upload all precision variants (indic→indic)

convert-fp16-indic-indic: setup ## Convert fp32 → fp16 (indic→indic)
	$(PYTHON) src/05_convert_fp16.py \
		--input $(INDIC_INDIC_OUT) \
		--output $(INDIC_INDIC_FP16_OUT)

quantize-q4f16-indic-indic: setup ## q4f16 quantize fp16 bundle (indic→indic)
	$(PYTHON) src/06_quantize_q4f16.py \
		--input $(INDIC_INDIC_FP16_OUT) \
		--output $(INDIC_INDIC_Q4F16_OUT) \
		--block-size $(Q4F16_BLOCK_SIZE)

benchmark-int8-indic-indic: setup ## Benchmark INT8 vs fp32 oracle (indic→indic)
	$(PYTHON) src/07_benchmark_precision.py \
		--fp32-dir  $(INDIC_INDIC_OUT) \
		--cmp-dir   $(INDIC_INDIC_INT8_OUT) \
		--fixtures  $(INDIC_INDIC_FIXTURES) \
		--pytorch-model $(INDIC_INDIC_MODEL) \
		--label int8 \
		--report $(INDIC_INDIC_BENCH_INT8)

benchmark-fp16-indic-indic: setup ## Benchmark fp16 vs fp32 oracle (indic→indic)
	$(PYTHON) src/07_benchmark_precision.py \
		--fp32-dir  $(INDIC_INDIC_OUT) \
		--cmp-dir   $(INDIC_INDIC_FP16_OUT) \
		--fixtures  $(INDIC_INDIC_FIXTURES) \
		--pytorch-model $(INDIC_INDIC_MODEL) \
		--label fp16 \
		--report $(INDIC_INDIC_BENCH_FP16)

benchmark-q4f16-indic-indic: setup ## Benchmark q4f16 vs fp32 oracle (indic→indic)
	$(PYTHON) src/07_benchmark_precision.py \
		--fp32-dir  $(INDIC_INDIC_OUT) \
		--cmp-dir   $(INDIC_INDIC_Q4F16_OUT) \
		--fixtures  $(INDIC_INDIC_FIXTURES) \
		--pytorch-model $(INDIC_INDIC_MODEL) \
		--label q4f16 \
		--report $(INDIC_INDIC_BENCH_Q4F16)

indic-indic: export-indic-indic tokenizers-indic-indic validate-indic-indic ## Full indic→indic pipeline (steps 1–3)

# ── Batch Operations ──────────────────────────────────────────────────────────

quantize-all: quantize-int8-all convert-fp16-all quantize-q4f16-all ## Run all quantization/conversion variants for all directions

quantize-int8-all: quantize-en-indic quantize-indic-en quantize-indic-indic ## Run INT8 quantization for all directions

convert-fp16-all: convert-fp16-en-indic convert-fp16-indic-en convert-fp16-indic-indic ## Run FP16 conversion for all directions

quantize-q4f16-all: quantize-q4f16-en-indic quantize-q4f16-indic-en quantize-q4f16-indic-indic ## Run Q4F16 quantization for all directions

benchmark-all: benchmark-int8-all benchmark-fp16-all benchmark-q4f16-all ## Benchmark all directions for INT8, FP16, and Q4F16

benchmark-int8-all: benchmark-int8-en-indic benchmark-int8-indic-en benchmark-int8-indic-indic ## Run INT8 benchmarks for all directions

benchmark-fp16-all: benchmark-fp16-en-indic benchmark-fp16-indic-en benchmark-fp16-indic-indic ## Run FP16 benchmarks for all directions

benchmark-q4f16-all: benchmark-q4f16-en-indic benchmark-q4f16-indic-en benchmark-q4f16-indic-indic ## Run Q4F16 benchmarks for all directions

upload-all: upload-en-indic-all upload-indic-en-all upload-indic-indic-all ## Upload all 12 model bundles (3 directions x 4 precision variants)

# ── Batch Operations 1B ───────────────────────────────────────────────────────

quantize-1b-all: quantize-int8-1b-all convert-fp16-1b-all quantize-q4f16-1b-all ## Run all quantization/conversion variants for all 1B directions

quantize-int8-1b-all: quantize-en-indic-1b quantize-indic-en-1b quantize-indic-indic-1b ## Run INT8 quantization for all 1B directions

convert-fp16-1b-all: convert-fp16-en-indic-1b convert-fp16-indic-en-1b convert-fp16-indic-indic-1b ## Run FP16 conversion for all 1B directions

quantize-q4f16-1b-all: quantize-q4f16-en-indic-1b quantize-q4f16-indic-en-1b quantize-q4f16-indic-indic-1b ## Run Q4F16 quantization for all 1B directions

benchmark-1b-all: benchmark-int8-1b-all benchmark-fp16-1b-all benchmark-q4f16-1b-all ## Benchmark all 1B directions for INT8, FP16, and Q4F16

benchmark-int8-1b-all: benchmark-int8-en-indic-1b benchmark-int8-indic-en-1b benchmark-int8-indic-indic-1b ## Run INT8 benchmarks for all 1B directions

benchmark-fp16-1b-all: benchmark-fp16-en-indic-1b benchmark-fp16-indic-en-1b benchmark-fp16-indic-indic-1b ## Run FP16 benchmarks for all 1B directions

benchmark-q4f16-1b-all: benchmark-q4f16-en-indic-1b benchmark-q4f16-indic-en-1b benchmark-q4f16-indic-indic-1b ## Run Q4F16 benchmarks for all 1B directions

upload-1b-all: upload-en-indic-1b-all upload-indic-en-1b-all upload-indic-indic-1b-all ## Upload all 12 1B model bundles (3 directions x 4 precision variants)


# ── 1B Models Pipeline ────────────────────────────────────────────────────────

# en→indic 1B
export-en-indic-1b: setup ## Export encoder + decoder ONNX (en→indic 1B)
	$(PYTHON) src/v2/01_export_encoder_decoder.py \
		--model $(EN_INDIC_1B_MODEL) \
		--output $(EN_INDIC_1B_OUT) \
		--opset $(ONNX_OPSET)

tokenizers-en-indic-1b: setup ## Build fast tokenizers (en→indic 1B)
	$(PYTHON) src/02_build_fast_tokenizers.py \
		--model $(EN_INDIC_1B_MODEL) \
		--output $(EN_INDIC_1B_OUT)

validate-en-indic-1b: setup ## Validate ONNX parity (en→indic 1B)
	$(PYTHON) src/v2/03_validate_parity.py \
		--onnx-dir $(EN_INDIC_1B_OUT) \
		--pytorch-model $(EN_INDIC_1B_MODEL) \
		--fixtures $(EN_INDIC_FIXTURES) \
		--report $(EN_INDIC_1B_REPORT) \
		$(if $(SMOKE),--smoke)

quantize-en-indic-1b: setup ## INT8 quantize fp32 bundle (en→indic 1B)
	$(PYTHON) src/04_quantize_int8.py \
		--input $(EN_INDIC_1B_OUT) \
		--output $(EN_INDIC_1B_INT8_OUT)

convert-fp16-en-indic-1b: setup ## Convert fp32 → fp16 (en→indic 1B)
	$(PYTHON) src/v2/05_convert_fp16.py \
		--input $(EN_INDIC_1B_OUT) \
		--output $(EN_INDIC_1B_FP16_OUT)

quantize-q4f16-en-indic-1b: setup ## q4f16 quantize fp16 bundle (en→indic 1B)
	$(PYTHON) src/v2/06_quantize_q4f16.py \
		--input $(EN_INDIC_1B_FP16_OUT) \
		--output $(EN_INDIC_1B_Q4F16_OUT) \
		--block-size $(Q4F16_BLOCK_SIZE)

benchmark-int8-en-indic-1b: setup ## Benchmark INT8 vs fp32 oracle (en→indic 1B)
	$(PYTHON) src/07_benchmark_precision.py \
		--fp32-dir  $(EN_INDIC_1B_OUT) \
		--cmp-dir   $(EN_INDIC_1B_INT8_OUT) \
		--fixtures  $(EN_INDIC_FIXTURES) \
		--pytorch-model $(EN_INDIC_1B_MODEL) \
		--label int8 \
		--report $(EN_INDIC_1B_BENCH_INT8)

benchmark-fp16-en-indic-1b: setup ## Benchmark fp16 vs fp32 oracle (en→indic 1B)
	$(PYTHON) src/07_benchmark_precision.py \
		--fp32-dir  $(EN_INDIC_1B_OUT) \
		--cmp-dir   $(EN_INDIC_1B_FP16_OUT) \
		--fixtures  $(EN_INDIC_FIXTURES) \
		--pytorch-model $(EN_INDIC_1B_MODEL) \
		--label fp16 \
		--report $(EN_INDIC_1B_BENCH_FP16)

benchmark-q4f16-en-indic-1b: setup ## Benchmark q4f16 vs fp32 oracle (en→indic 1B)
	$(PYTHON) src/07_benchmark_precision.py \
		--fp32-dir  $(EN_INDIC_1B_OUT) \
		--cmp-dir   $(EN_INDIC_1B_Q4F16_OUT) \
		--fixtures  $(EN_INDIC_FIXTURES) \
		--pytorch-model $(EN_INDIC_1B_MODEL) \
		--label q4f16 \
		--report $(EN_INDIC_1B_BENCH_Q4F16)

en-indic-1b: export-en-indic-1b tokenizers-en-indic-1b validate-en-indic-1b ## Full en→indic 1B pipeline (steps 1-3)

upload-en-indic-1b: upload-en-indic-1b-fp32 ## Alias for upload-en-indic-1b-fp32

upload-en-indic-1b-fp32: setup ## Upload fp32 bundle to Hugging Face (en→indic 1B)
	$(PYTHON) src/v2/05_upload_hf.py \
		--model-dir $(EN_INDIC_1B_OUT) \
		--repo-id $(HF_ORG)/indictrans2-en-indic-1B-ONNX \
		--commit-message "$(COMMIT_MESSAGE)"

upload-en-indic-1b-int8: setup ## Upload int8 bundle (en→indic 1B)
	$(PYTHON) src/v2/05_upload_hf.py \
		--model-dir $(EN_INDIC_1B_INT8_OUT) \
		--repo-id $(HF_ORG)/indictrans2-en-indic-1B-ONNX-int8 \
		--commit-message "$(COMMIT_MESSAGE)"

upload-en-indic-1b-fp16: setup ## Upload fp16 bundle (en→indic 1B)
	$(PYTHON) src/v2/05_upload_hf.py \
		--model-dir $(EN_INDIC_1B_FP16_OUT) \
		--repo-id $(HF_ORG)/indictrans2-en-indic-1B-ONNX-fp16 \
		--commit-message "$(COMMIT_MESSAGE)"

upload-en-indic-1b-q4f16: setup ## Upload q4f16 bundle (en→indic 1B)
	$(PYTHON) src/v2/05_upload_hf.py \
		--model-dir $(EN_INDIC_1B_Q4F16_OUT) \
		--repo-id $(HF_ORG)/indictrans2-en-indic-1B-ONNX-q4f16 \
		--commit-message "$(COMMIT_MESSAGE)"

upload-en-indic-1b-all: upload-en-indic-1b-fp32 upload-en-indic-1b-int8 upload-en-indic-1b-fp16 upload-en-indic-1b-q4f16 ## Upload all precision variants (en→indic 1B)


# indic→en 1B
export-indic-en-1b: setup ## Export encoder + decoder ONNX (indic→en 1B)
	$(PYTHON) src/v2/01_export_encoder_decoder.py \
		--model $(INDIC_EN_1B_MODEL) \
		--output $(INDIC_EN_1B_OUT) \
		--opset $(ONNX_OPSET)

tokenizers-indic-en-1b: setup ## Build fast tokenizers (indic→en 1B)
	$(PYTHON) src/02_build_fast_tokenizers.py \
		--model $(INDIC_EN_1B_MODEL) \
		--output $(INDIC_EN_1B_OUT)

validate-indic-en-1b: setup ## Validate ONNX parity (indic→en 1B)
	$(PYTHON) src/v2/03_validate_parity.py \
		--onnx-dir $(INDIC_EN_1B_OUT) \
		--pytorch-model $(INDIC_EN_1B_MODEL) \
		--fixtures $(INDIC_EN_FIXTURES) \
		--report $(INDIC_EN_1B_REPORT) \
		$(if $(SMOKE),--smoke)

quantize-indic-en-1b: setup ## INT8 quantize fp32 bundle (indic→en 1B)
	$(PYTHON) src/04_quantize_int8.py \
		--input $(INDIC_EN_1B_OUT) \
		--output $(INDIC_EN_1B_INT8_OUT)

convert-fp16-indic-en-1b: setup ## Convert fp32 → fp16 (indic→en 1B)
	$(PYTHON) src/v2/05_convert_fp16.py \
		--input $(INDIC_EN_1B_OUT) \
		--output $(INDIC_EN_1B_FP16_OUT)

quantize-q4f16-indic-en-1b: setup ## q4f16 quantize fp16 bundle (indic→en 1B)
	$(PYTHON) src/v2/06_quantize_q4f16.py \
		--input $(INDIC_EN_1B_FP16_OUT) \
		--output $(INDIC_EN_1B_Q4F16_OUT) \
		--block-size $(Q4F16_BLOCK_SIZE)

benchmark-int8-indic-en-1b: setup ## Benchmark INT8 vs fp32 oracle (indic→en 1B)
	$(PYTHON) src/07_benchmark_precision.py \
		--fp32-dir  $(INDIC_EN_1B_OUT) \
		--cmp-dir   $(INDIC_EN_1B_INT8_OUT) \
		--fixtures  $(INDIC_EN_FIXTURES) \
		--pytorch-model $(INDIC_EN_1B_MODEL) \
		--label int8 \
		--report $(INDIC_EN_1B_BENCH_INT8)

benchmark-fp16-indic-en-1b: setup ## Benchmark fp16 vs fp32 oracle (indic→en 1B)
	$(PYTHON) src/07_benchmark_precision.py \
		--fp32-dir  $(INDIC_EN_1B_OUT) \
		--cmp-dir   $(INDIC_EN_1B_FP16_OUT) \
		--fixtures  $(INDIC_EN_FIXTURES) \
		--pytorch-model $(INDIC_EN_1B_MODEL) \
		--label fp16 \
		--report $(INDIC_EN_1B_BENCH_FP16)

benchmark-q4f16-indic-en-1b: setup ## Benchmark q4f16 vs fp32 oracle (indic→en 1B)
	$(PYTHON) src/07_benchmark_precision.py \
		--fp32-dir  $(INDIC_EN_1B_OUT) \
		--cmp-dir   $(INDIC_EN_1B_Q4F16_OUT) \
		--fixtures  $(INDIC_EN_FIXTURES) \
		--pytorch-model $(INDIC_EN_1B_MODEL) \
		--label q4f16 \
		--report $(INDIC_EN_1B_BENCH_Q4F16)

indic-en-1b: export-indic-en-1b tokenizers-indic-en-1b validate-indic-en-1b ## Full indic→en 1B pipeline (steps 1-3)

upload-indic-en-1b: upload-indic-en-1b-fp32 ## Alias for upload-indic-en-1b-fp32

upload-indic-en-1b-fp32: setup ## Upload fp32 bundle to Hugging Face (indic→en 1B)
	$(PYTHON) src/v2/05_upload_hf.py \
		--model-dir $(INDIC_EN_1B_OUT) \
		--repo-id $(HF_ORG)/indictrans2-indic-en-1B-ONNX \
		--commit-message "$(COMMIT_MESSAGE)"

upload-indic-en-1b-int8: setup ## Upload int8 bundle (indic→en 1B)
	$(PYTHON) src/v2/05_upload_hf.py \
		--model-dir $(INDIC_EN_1B_INT8_OUT) \
		--repo-id $(HF_ORG)/indictrans2-indic-en-1B-ONNX-int8 \
		--commit-message "$(COMMIT_MESSAGE)"

upload-indic-en-1b-fp16: setup ## Upload fp16 bundle (indic→en 1B)
	$(PYTHON) src/v2/05_upload_hf.py \
		--model-dir $(INDIC_EN_1B_FP16_OUT) \
		--repo-id $(HF_ORG)/indictrans2-indic-en-1B-ONNX-fp16 \
		--commit-message "$(COMMIT_MESSAGE)"

upload-indic-en-1b-q4f16: setup ## Upload q4f16 bundle (indic→en 1B)
	$(PYTHON) src/v2/05_upload_hf.py \
		--model-dir $(INDIC_EN_1B_Q4F16_OUT) \
		--repo-id $(HF_ORG)/indictrans2-indic-en-1B-ONNX-q4f16 \
		--commit-message "$(COMMIT_MESSAGE)"

upload-indic-en-1b-all: upload-indic-en-1b-fp32 upload-indic-en-1b-int8 upload-indic-en-1b-fp16 upload-indic-en-1b-q4f16 ## Upload all precision variants (indic→en 1B)


# indic→indic 1B
export-indic-indic-1b: setup ## Export encoder + decoder ONNX (indic→indic 1B)
	$(PYTHON) src/v2/01_export_encoder_decoder.py \
		--model $(INDIC_INDIC_1B_MODEL) \
		--output $(INDIC_INDIC_1B_OUT) \
		--opset $(ONNX_OPSET)

tokenizers-indic-indic-1b: setup ## Build fast tokenizers (indic→indic 1B)
	$(PYTHON) src/02_build_fast_tokenizers.py \
		--model $(INDIC_INDIC_1B_MODEL) \
		--output $(INDIC_INDIC_1B_OUT)

validate-indic-indic-1b: setup ## Validate ONNX parity (indic→indic 1B)
	$(PYTHON) src/v2/03_validate_parity.py \
		--onnx-dir $(INDIC_INDIC_1B_OUT) \
		--pytorch-model $(INDIC_INDIC_1B_MODEL) \
		--fixtures $(INDIC_INDIC_FIXTURES) \
		--report $(INDIC_INDIC_1B_REPORT) \
		$(if $(SMOKE),--smoke)

quantize-indic-indic-1b: setup ## INT8 quantize fp32 bundle (indic→indic 1B)
	$(PYTHON) src/04_quantize_int8.py \
		--input $(INDIC_INDIC_1B_OUT) \
		--output $(INDIC_INDIC_1B_INT8_OUT)

convert-fp16-indic-indic-1b: setup ## Convert fp32 → fp16 (indic→indic 1B)
	$(PYTHON) src/v2/05_convert_fp16.py \
		--input $(INDIC_INDIC_1B_OUT) \
		--output $(INDIC_INDIC_1B_FP16_OUT)

quantize-q4f16-indic-indic-1b: setup ## q4f16 quantize fp16 bundle (indic→indic 1B)
	$(PYTHON) src/v2/06_quantize_q4f16.py \
		--input $(INDIC_INDIC_1B_FP16_OUT) \
		--output $(INDIC_INDIC_1B_Q4F16_OUT) \
		--block-size $(Q4F16_BLOCK_SIZE)

benchmark-int8-indic-indic-1b: setup ## Benchmark INT8 vs fp32 oracle (indic→indic 1B)
	$(PYTHON) src/07_benchmark_precision.py \
		--fp32-dir  $(INDIC_INDIC_1B_OUT) \
		--cmp-dir   $(INDIC_INDIC_1B_INT8_OUT) \
		--fixtures  $(INDIC_INDIC_FIXTURES) \
		--pytorch-model $(INDIC_INDIC_1B_MODEL) \
		--label int8 \
		--report $(INDIC_INDIC_1B_BENCH_INT8)

benchmark-fp16-indic-indic-1b: setup ## Benchmark fp16 vs fp32 oracle (indic→indic 1B)
	$(PYTHON) src/07_benchmark_precision.py \
		--fp32-dir  $(INDIC_INDIC_1B_OUT) \
		--cmp-dir   $(INDIC_INDIC_1B_FP16_OUT) \
		--fixtures  $(INDIC_INDIC_FIXTURES) \
		--pytorch-model $(INDIC_INDIC_1B_MODEL) \
		--label fp16 \
		--report $(INDIC_INDIC_1B_BENCH_FP16)

benchmark-q4f16-indic-indic-1b: setup ## Benchmark q4f16 vs fp32 oracle (indic→indic 1B)
	$(PYTHON) src/07_benchmark_precision.py \
		--fp32-dir  $(INDIC_INDIC_1B_OUT) \
		--cmp-dir   $(INDIC_INDIC_1B_Q4F16_OUT) \
		--fixtures  $(INDIC_INDIC_FIXTURES) \
		--pytorch-model $(INDIC_INDIC_1B_MODEL) \
		--label q4f16 \
		--report $(INDIC_INDIC_1B_BENCH_Q4F16)

indic-indic-1b: export-indic-indic-1b tokenizers-indic-indic-1b validate-indic-indic-1b ## Full indic→indic 1B pipeline (steps 1-3)

upload-indic-indic-1b: upload-indic-indic-1b-fp32 ## Alias for upload-indic-indic-1b-fp32

upload-indic-indic-1b-fp32: setup ## Upload fp32 bundle to Hugging Face (indic→indic 1B)
	$(PYTHON) src/v2/05_upload_hf.py \
		--model-dir $(INDIC_INDIC_1B_OUT) \
		--repo-id $(HF_ORG)/indictrans2-indic-indic-1B-ONNX \
		--commit-message "$(COMMIT_MESSAGE)"

upload-indic-indic-1b-int8: setup ## Upload int8 bundle (indic→indic 1B)
	$(PYTHON) src/v2/05_upload_hf.py \
		--model-dir $(INDIC_INDIC_1B_INT8_OUT) \
		--repo-id $(HF_ORG)/indictrans2-indic-indic-1B-ONNX-int8 \
		--commit-message "$(COMMIT_MESSAGE)"

upload-indic-indic-1b-fp16: setup ## Upload fp16 bundle (indic→indic 1B)
	$(PYTHON) src/v2/05_upload_hf.py \
		--model-dir $(INDIC_INDIC_1B_FP16_OUT) \
		--repo-id $(HF_ORG)/indictrans2-indic-indic-1B-ONNX-fp16 \
		--commit-message "$(COMMIT_MESSAGE)"

upload-indic-indic-1b-q4f16: setup ## Upload q4f16 bundle (indic→indic 1B)
	$(PYTHON) src/v2/05_upload_hf.py \
		--model-dir $(INDIC_INDIC_1B_Q4F16_OUT) \
		--repo-id $(HF_ORG)/indictrans2-indic-indic-1B-ONNX-q4f16 \
		--commit-message "$(COMMIT_MESSAGE)"

upload-indic-indic-1b-all: upload-indic-indic-1b-fp32 upload-indic-indic-1b-int8 upload-indic-indic-1b-fp16 upload-indic-indic-1b-q4f16 ## Upload all precision variants (indic→indic 1B)

readmes-1b: setup ## Generate READMEs for all 1B models (dry-run mode)
	$(PYTHON) src/v2/dry_run_generate_readmes.py

# ── Cleanup ──────────────────────────────────────────────────────────────────

clean: ## Remove scratch ONNX artifacts
	rm -rf \
		$(EN_INDIC_OUT) $(EN_INDIC_INT8_OUT) $(EN_INDIC_FP16_OUT) $(EN_INDIC_Q4F16_OUT) \
		$(INDIC_EN_OUT) $(INDIC_EN_INT8_OUT) $(INDIC_EN_FP16_OUT) $(INDIC_EN_Q4F16_OUT) \
		$(INDIC_INDIC_OUT) $(INDIC_INDIC_INT8_OUT) $(INDIC_INDIC_FP16_OUT) $(INDIC_INDIC_Q4F16_OUT) \
		$(EN_INDIC_1B_OUT) $(EN_INDIC_1B_INT8_OUT) $(EN_INDIC_1B_FP16_OUT) $(EN_INDIC_1B_Q4F16_OUT) \
		$(INDIC_EN_1B_OUT) $(INDIC_EN_1B_INT8_OUT) $(INDIC_EN_1B_FP16_OUT) $(INDIC_EN_1B_Q4F16_OUT) \
		$(INDIC_INDIC_1B_OUT) $(INDIC_INDIC_1B_INT8_OUT) $(INDIC_INDIC_1B_FP16_OUT) $(INDIC_INDIC_1B_Q4F16_OUT)

clean-all: clean ## Remove scratch artifacts and Python venv
	rm -rf $(VENV)

preview: ## Serve onnx-components.html locally
	@echo "Starting local preview server on http://localhost:8000/onnx-components.html..."
	python3 -m http.server 8000

reports: setup ## Generate benchmark reports and plots (overall, language, and category levels)
	$(PYTHON) src/generate_visual_reports.py

reports-1b: setup ## Generate benchmark reports and plots for 1B models (overall, language, and category levels)
	$(PYTHON) src/generate_visual_reports.py --model-size 1b
