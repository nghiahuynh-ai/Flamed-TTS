# Common settings -----------------------------------------------------------
PYTHON ?= python
.DEFAULT_GOAL := help

# WandB / training defaults
WANDB_PROJECT ?= flamed-tts-v3
WANDB_RUN ?= forcing-cfg-distill
WANDB_VERSION ?= local
WANDB_MODE ?= online
EXP_ROOT ?= experiments
EXP_NAME ?= $(WANDB_RUN)
# Comma-separated CUDA indices understood by train.py (e.g., "0" or "0,1")
DEVICES ?= 0,1,2,3
BATCH_SIZE ?= 16
EPOCHS ?= 100
CKPT ?=

# Synthesis defaults
SYNTH_CKPT ?=
SYNTH_CFG ?=
SYNTH_TEXT ?=
PROMPT_DIR ?=
PROMPT_LIST ?=
METADATA_FILE ?=
OUTPUT_DIR ?= outputs
SYNTH_DEVICE ?= cuda:0
NSTEPS_DURGEN ?= 64
NSTEPS_DENOISER ?= 64
TEMP_DURGEN ?= 0.3
TEMP_DENOISER ?= 0.3
SKIP_EXISTING ?= true
SYNTH_BATCH_SIZE ?= 4
WEIGHTS_ONLY ?= true
AVG_CKPTS ?=
AVG_OUTPUT ?= averaged.ckpt

export WANDB_API_KEY
export WANDB_MODE

.PHONY: help train synth eval avg

help:
	@echo "Targets:"
	@echo "  make train  - Launch Lightning training with WandB logging."
	@echo "  make synth  - Run synthesis via synthesize.py (prompt-list or metadata mode)."
	@echo "  make eval   - Placeholder until the evaluation script is published."
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
		--weights-only $(WEIGHTS_ONLY) \
		--skip-existing $(SKIP_EXISTING) \
		--batch-size $(SYNTH_BATCH_SIZE) \
		$(if $(strip $(PROMPT_LIST)),--text "$(SYNTH_TEXT)" --prompt-list $(PROMPT_LIST),) \
		$(if $(strip $(METADATA_FILE)),--metadata-file "$(METADATA_FILE)",)

eval:
	@echo "Evaluation workflow is TBD. Wire this target up once the eval script is available."

avg:
	@test -n "$(strip $(AVG_CKPTS))" || (echo "Provide AVG_CKPTS with space-separated checkpoint paths." && exit 1)
	$(PYTHON) avg_weights.py --output "$(AVG_OUTPUT)" --ckpts $(AVG_CKPTS)
