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

EN_INDIC_OUT      := $(SCRATCH)/en-indic-onnx
EN_INDIC_INT8_OUT := $(SCRATCH)/en-indic-onnx-int8
INDIC_EN_OUT      := $(SCRATCH)/indic-en-onnx
INDIC_EN_INT8_OUT := $(SCRATCH)/indic-en-onnx-int8
INDIC_INDIC_OUT   := $(SCRATCH)/indic-indic-onnx
INDIC_INDIC_INT8_OUT := $(SCRATCH)/indic-indic-onnx-int8

EN_INDIC_FIXTURES    := fixtures/en-indic-golden.jsonl
INDIC_EN_FIXTURES    := fixtures/indic-en-golden.jsonl
INDIC_INDIC_FIXTURES := fixtures/indic-indic-golden.jsonl

EN_INDIC_REPORT    := fixtures/parity-report-en-indic.json
INDIC_EN_REPORT    := fixtures/parity-report-indic-en.json
INDIC_INDIC_REPORT := fixtures/parity-report-indic-indic.json

.PHONY: help setup install clean clean-all preview \
	export-en-indic tokenizers-en-indic validate-en-indic quantize-en-indic \
	capture-fixtures-en-indic upload-en-indic en-indic \
	export-indic-en tokenizers-indic-en validate-indic-en quantize-indic-en \
	capture-fixtures-indic-en upload-indic-en indic-en \
	export-indic-indic tokenizers-indic-indic validate-indic-indic quantize-indic-indic \
	capture-fixtures-indic-indic upload-indic-indic indic-indic

help: ## Show available targets
	@echo "IndicTrans2 ONNX export — make targets"
	@echo ""
	@echo "Setup:"
	@echo "  make setup                  Create .venv and install requirements"
	@echo "  make clean                  Remove scratch ONNX artifacts"
	@echo "  make clean-all              Remove scratch + .venv"
	@echo "  make preview                Local preview of the ONNX components guide"
	@echo ""
	@echo "en→indic (200M):"
	@echo "  make export-en-indic"
	@echo "  make tokenizers-en-indic"
	@echo "  make validate-en-indic"
	@echo "  make quantize-en-indic"
	@echo "  make capture-fixtures-en-indic"
	@echo "  make upload-en-indic"
	@echo "  make en-indic               Steps 1–3"
	@echo ""
	@echo "indic→en (P0, 200M):"
	@echo "  make export-indic-en        Step 1 — ONNX graphs"
	@echo "  make tokenizers-indic-en    Step 2 — fast tokenizers"
	@echo "  make validate-indic-en      Step 3 — parity vs PyTorch"
	@echo "  make quantize-indic-en      Step 4 — INT8 (after fp32 passes)"
	@echo "  make capture-fixtures-indic-en  Generate golden fixtures"
	@echo "  make upload-indic-en        Upload bundle to HF (HF_ORG=$(HF_ORG))"
	@echo "  make indic-en               Steps 1–3"
	@echo ""
	@echo "indic→indic (P1, 320M):"
	@echo "  make export-indic-indic"
	@echo "  make tokenizers-indic-indic"
	@echo "  make validate-indic-indic"
	@echo "  make quantize-indic-indic"
	@echo "  make capture-fixtures-indic-indic"
	@echo "  make upload-indic-indic"
	@echo "  make indic-indic            Steps 1–3"
	@echo ""
	@echo "Variables: HF_ORG=$(HF_ORG)  SCRATCH=$(SCRATCH)  ONNX_OPSET=$(ONNX_OPSET)"

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

upload-en-indic: setup ## Upload fp32 bundle to Hugging Face
	$(PYTHON) src/05_upload_hf.py \
		--model-dir $(EN_INDIC_OUT) \
		--repo-id $(HF_ORG)/indictrans2-en-indic-dist-200M-ONNX \
		--commit-message "$(COMMIT_MESSAGE)"

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

upload-indic-en: setup ## Upload fp32 bundle to Hugging Face
	$(PYTHON) src/05_upload_hf.py \
		--model-dir $(INDIC_EN_OUT) \
		--repo-id $(HF_ORG)/indictrans2-indic-en-dist-200M-ONNX \
		--commit-message "$(COMMIT_MESSAGE)"

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

upload-indic-indic: setup ## Upload fp32 bundle to Hugging Face
	$(PYTHON) src/05_upload_hf.py \
		--model-dir $(INDIC_INDIC_OUT) \
		--repo-id $(HF_ORG)/indictrans2-indic-indic-dist-320M-ONNX \
		--commit-message "$(COMMIT_MESSAGE)"

indic-indic: export-indic-indic tokenizers-indic-indic validate-indic-indic ## Full indic→indic pipeline (steps 1–3)

# ── Cleanup ──────────────────────────────────────────────────────────────────

clean: ## Remove scratch ONNX artifacts
	rm -rf $(EN_INDIC_OUT) $(EN_INDIC_INT8_OUT) $(INDIC_EN_OUT) $(INDIC_EN_INT8_OUT) $(INDIC_INDIC_OUT) $(INDIC_INDIC_INT8_OUT)

clean-all: clean ## Remove scratch artifacts and Python venv
	rm -rf $(VENV)

preview: ## Serve onnx-components.html locally
	@echo "Starting local preview server on http://localhost:8000/onnx-components.html..."
	python3 -m http.server 8000
