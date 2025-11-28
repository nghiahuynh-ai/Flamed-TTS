# Common settings -----------------------------------------------------------
PYTHON ?= python
.DEFAULT_GOAL := help

# WandB / training defaults
WANDB_PROJECT ?= flamed-tts-v4
WANDB_RUN ?= v4
WANDB_VERSION ?= local
WANDB_MODE ?= online
EXP_ROOT ?= experiments
EXP_NAME ?= $(WANDB_RUN)
# Comma-separated modules to train: PriorGenerator, ProbGenerator, or both
PIPELINE ?= PriorGenerator,ProbGenerator
# Pretrained checkpoint for PriorGenerator when training ProbGenerator alone
PRIOR_CKPT ?=
# Comma-separated CUDA indices understood by train.py (e.g., "0" or "0,1")
DEVICES ?= 0,1,2,3
BATCH_SIZE ?= 16
EPOCHS ?= 100
CKPT ?=

# Synthesis defaults
SYNTH_CKPT ?= ckpts/averaged.ckpt
SYNTH_CFG ?= ckpts/cfg.yaml
SYNTH_TEXT ?= 
PROMPT_DIR ?= ../librispeech/LibriSpeech-Clipped-5s
PROMPT_LIST ?= 
METADATA_FILE ?= ../librispeech/test-clean-clipped-5s.txt
OUTPUT_DIR ?= ../outputs-ckpt30-34/5s
SYNTH_DEVICE ?= cuda:0
NSTEPS_DURGEN ?= 8
NSTEPS_DENOISER ?= 128
TEMP_DURGEN ?= 0.3
TEMP_DENOISER ?= 0.3
GUIDANCE_SCALE ?= 3.5
SKIP_EXISTING ?= true
SYNTH_BATCH_SIZE ?= 1
WEIGHTS_ONLY ?= true

# Avg Checkpoints
AVG_CKPTS ?= ckpts/ckpt-epoch=40-total_loss_val_epoch=7.85.ckpt ckpts/ckpt-epoch=41-total_loss_val_epoch=7.84.ckpt ckpts/ckpt-epoch=42-total_loss_val_epoch=7.84.ckpt ckpts/ckpt-epoch=43-total_loss_val_epoch=7.83.ckpt ckpts/ckpt-epoch=44-total_loss_val_epoch=7.83.ckpt
AVG_OUTPUT ?= ckpts/averaged1.ckpt

# Evaluation defaults
EVAL_MANIFEST ?=
EVAL_SYNTH ?=
EVAL_PROMPT ?=
EVAL_TGT ?=
EVAL_OUTPUT_DIR ?=
EVAL_NAME ?= flamedv3
EVAL_METRICS ?= utmos sim wer prosody
EVAL_SR ?= 16000
EVAL_DEVICE ?= cuda:0
EVAL_CODEC ?= facodec
EVAL_SIM_MODEL ?=
EVAL_SIM_CKPT ?=
EVAL_UTMOS_CKPT ?=
EVAL_CACHE_DIR ?= $(HOME)/.cache/flamed/eval
EVAL_VENV ?= .venv-eval
EVAL_VENV_PYTHON ?= python3.10
EVAL_PYTHON ?= $(EVAL_VENV)/bin/python
EVAL_REQUIREMENTS ?= evaluate/requirements.txt

export WANDB_API_KEY
export WANDB_MODE

.PHONY: help train synth eval avg

help:
	@echo "Targets:"
	@echo "  make train  - Launch Lightning training with WandB logging."
	@echo "  make synth  - Run synthesis via synthesize.py (prompt-list or metadata mode)."
	@echo "  make eval   - Run evaluation metrics (set EVAL_* vars accordingly)."
	@echo "  make avg    - Average checkpoint weights (set AVG_CKPTS and AVG_OUTPUT)."
	@echo ""
	@echo "Override variables on the command line, e.g.:"
	@echo "  make train WANDB_RUN=my_run DEVICES=0,1 EXP_ROOT=/tmp/runs"
	@echo "  make synth SYNTH_CKPT=ckpt.pt SYNTH_CFG=config.yaml PROMPT_DIR=./prompts \\"
	@echo "      PROMPT_LIST=\"p1.wav p2.wav\" SYNTH_TEXT=\"Hello world\""

train:
	@test -n "$(strip $(WANDB_PROJECT))" || (echo "Set WANDB_PROJECT to your project name." && exit 1)
	@test -n "$(strip $(EXP_NAME))" || (echo "Set EXP_NAME for this experiment run." && exit 1)
	@if [ -z "$(strip $(WANDB_API_KEY))" ]; then \
		echo "Warning: WANDB_API_KEY not exported; ensure you are logged in via 'wandb login'."; \
	fi
	@mkdir -p "$(EXP_ROOT)"
	$(PYTHON) train.py \
		--proj_name "$(WANDB_PROJECT)" \
		--ver "$(WANDB_VERSION)" \
		--exp_root "$(EXP_ROOT)" \
		--exp_name "$(EXP_NAME)" \
		--devices "$(DEVICES)" \
		--pipeline "$(PIPELINE)" \
		$(if $(strip $(PRIOR_CKPT)),--prior_ckpt "$(PRIOR_CKPT)",) \
		--batch_size $(BATCH_SIZE) \
		--epochs $(EPOCHS) \
		$(if $(strip $(CKPT)),--ckpt "$(CKPT)",)

synth:
	@test -n "$(strip $(SYNTH_CKPT))" || (echo "SYNTH_CKPT is required." && exit 1)
	@test -n "$(strip $(SYNTH_CFG))" || (echo "SYNTH_CFG is required." && exit 1)
	@test -n "$(strip $(PROMPT_DIR))" || (echo "PROMPT_DIR is required." && exit 1)
	@if [ -n "$(strip $(PROMPT_LIST))" ] && [ -n "$(strip $(METADATA_FILE))" ]; then \
		echo "Use either PROMPT_LIST or METADATA_FILE, not both."; \
		exit 2; \
	fi
	@if [ -z "$(strip $(PROMPT_LIST))" ] && [ -z "$(strip $(METADATA_FILE))" ]; then \
		echo "Provide PROMPT_LIST (direct mode) or METADATA_FILE (batch mode)."; \
		exit 2; \
	fi
	@if [ -n "$(strip $(PROMPT_LIST))" ] && [ -z "$(strip $(SYNTH_TEXT))" ]; then \
		echo "SYNTH_TEXT is required when PROMPT_LIST is used."; \
		exit 2; \
	fi
	$(PYTHON) synthesize.py \
		--ckpt-path "$(SYNTH_CKPT)" \
		--cfg-path "$(SYNTH_CFG)" \
		--prompt-dir "$(PROMPT_DIR)" \
		--output-dir "$(OUTPUT_DIR)" \
		--device "$(SYNTH_DEVICE)" \
		--nsteps-durgen $(NSTEPS_DURGEN) \
		--nsteps-denoiser $(NSTEPS_DENOISER) \
		--temp-durgen $(TEMP_DURGEN) \
		--temp-denoiser $(TEMP_DENOISER) \
		$(if $(strip $(GUIDANCE_SCALE)),--guidance-scale $(GUIDANCE_SCALE),) \
		--weights-only $(WEIGHTS_ONLY) \
		--skip-existing $(SKIP_EXISTING) \
		--batch-size $(SYNTH_BATCH_SIZE) \
		$(if $(strip $(PROMPT_LIST)),--text "$(SYNTH_TEXT)" --prompt-list $(PROMPT_LIST),) \
		$(if $(strip $(METADATA_FILE)),--metadata-file "$(METADATA_FILE)",)

eval:
	@test -n "$(strip $(EVAL_SYNTH))" || (echo "Set EVAL_SYNTH to the synthesized wav directory." && exit 1)
	@test -n "$(strip $(EVAL_OUTPUT_DIR))" || (echo "Set EVAL_OUTPUT_DIR to the output directory for metrics." && exit 1)
	@test -n "$(strip $(EVAL_MANIFEST))" || (echo "Set EVAL_MANIFEST for prompt/transcript mappings." && exit 1)
	@test -n "$(strip $(EVAL_PROMPT))" || (echo "Set EVAL_PROMPT to the prompt/reference wav directory." && exit 1)
	@if [ ! -x "$(EVAL_PYTHON)" ]; then $(EVAL_VENV_PYTHON) -m venv "$(EVAL_VENV)"; fi
	@$(EVAL_PYTHON) -m pip install -r "$(EVAL_REQUIREMENTS)"
	@$(EVAL_PYTHON) -m pip install -e "evaluate/s3prl"
	"$(EVAL_PYTHON)" evaluate/evaluate.py \
		--manifest "$(EVAL_MANIFEST)" \
		--synth_path "$(EVAL_SYNTH)" \
		--prompt_path "$(EVAL_PROMPT)" \
		--output_path "$(EVAL_OUTPUT_DIR)" \
		--name "$(EVAL_NAME)" \
		--metrics $(EVAL_METRICS) \
		--sr $(EVAL_SR) \
		--device "$(EVAL_DEVICE)" \
		--codec "$(EVAL_CODEC)" \
		$(if $(strip $(EVAL_TGT)),--tgt_path "$(EVAL_TGT)",) \
		$(if $(strip $(EVAL_SIM_MODEL)),--sim_model "$(EVAL_SIM_MODEL)",) \
		$(if $(strip $(EVAL_SIM_CKPT)),--sim_ckpt "$(EVAL_SIM_CKPT)",) \
		$(if $(strip $(EVAL_UTMOS_CKPT)),--utmos_ckpt "$(EVAL_UTMOS_CKPT)",) \
		--cache_dir "$(EVAL_CACHE_DIR)"

avg:
	@test -n "$(strip $(AVG_CKPTS))" || (echo "Provide AVG_CKPTS with space-separated checkpoint paths." && exit 1)
	$(PYTHON) avg_weights.py --output "$(AVG_OUTPUT)" --ckpts $(AVG_CKPTS)
