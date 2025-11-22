# Flamed-TTS: Flow Matching Attention-Free Models for Efficient Generating and Dynamic Pacing Zero-shot Text-to-Speech

[![python](https://img.shields.io/badge/Python-3.10-brightgreen)]([https://github.com/SWivid/F5-TTS](https://github.com/flamedtts/Flamed-TTS))
[![arXiv](https://img.shields.io/badge/arXiv-2410.06885-b31b1b.svg?logo=arXiv)]()
[![demo](https://img.shields.io/badge/GitHub-Demo%20page-orange.svg)](https://flamed-tts.github.io/)

![Overall Architecture](https://github.com/flamedtts/Flamed-TTS/blob/main/figs/Flamed-TTS.png)
<div align="center">
	<img src="https://github.com/flamedtts/Flamed-TTS/blob/main/figs/CodeDecoder_Denoiser.png" width="640" style="display: block; margin: auto;"/>
</div>

## 🔥 News
- [Coming soon] Release training instructions
- [2025.08] Release checkpoint
- [2025.08] Release inference code
- [2025.08] Init Repo
- [2025.08] Submitted to `AAAI 2026`

## 🎯 Overview

This repo implements a novel zero-shot TTS framework, named Flamed-TTS, focusing on the low-latency generation and dynamic pacing in speech synthesis.

## 🛠️ Installation Dependencies

Prepare your environment by creating a conda setup, preferably on Linux. Then, install the necessary requirements using pip:
```bash
# Create a python 3.10 conda env (you could also use virtualenv)
conda create -n flamed-tts python=3.10
conda activate flamed-tts
cd Flamed-TTS
pip install -r requirements.txt
```

To train the model yourself, a GPU is recommended for optimal performance. However, you can generate samples using our pretrained models without requiring a GPU.

## 🚀 Inference

### Download pretrained weights

To perform inference with pretrained weights, you must download the pretrained weights for both FaCodec and OZSpeech.

* With FaCodec, you can download the FaCodec Encoder and FaCodec Decoder directly from Hugging Face: [FaCodec Encoder](https://huggingface.co/amphion/naturalspeech3_facodec/blob/main/ns3_facodec_encoder.bin), [FaCodec Decoder](https://huggingface.co/amphion/naturalspeech3_facodec/blob/main/ns3_facodec_decoder.bin). After downloading process done, please move those checkpoint files to the directory:
```
flamed/
└── models/
    └── facodec/
        └── checkpoints/
            ├── ns3_facodec_encoder.bin
            └── ns3_facodec_decoder.bin
```
* With Flamed-TTS, please refer [this link](https://drive.google.com/drive/folders/17A5OJoF6yUqiy62n1ghEGJ6EwHUexUEs?usp=sharing). You need to download both pretrained weights and config file for initializing model.

### Inference sample(s) with given content and speech prompt(s)

Use the `make synth` target, which wraps `synthesize.py` and validates all arguments for you. Supply the mandatory variables directly on the command line:

```bash
make synth \
	SYNTH_CKPT=path/to/ckpt.pt \
	SYNTH_CFG=configs/prob.yaml \
	PROMPT_DIR=assets/prompts \
	PROMPT_LIST="prompt_1.wav prompt_2.wav prompt_3.wav" \
	SYNTH_TEXT="content to be synthesized" \
	OUTPUT_DIR=outputs \
	SYNTH_DEVICE=cuda:0 \
	NSTEPS_DURGEN=16 \
	NSTEPS_DENOISER=128 \
	TEMP_DURGEN=1.0 \
	TEMP_DENOISER=0.3 \
	GUIDANCE_SCALE=3.5
```

- `PROMPT_LIST` is space-separated within quotes and every entry must exist under `PROMPT_DIR` (absolute paths also work).
- Set `WEIGHTS_ONLY=false` if your checkpoint is a full Lightning snapshot instead of the default `weights_only` format.
- Leave `DENOISER_METHOD=euler` for the original flow-matching sampler, or set `DENOISER_METHOD=forcing` with matching `FORCING_STEPS_MIN`/`FORCING_STEPS_MAX` to enable cascaded updates for later latents.

### Inference using a metadata file

Batch synthesis also goes through `make synth`, but you pass `METADATA_FILE` instead of `PROMPT_LIST`/`SYNTH_TEXT`. Each metadata line follows `target_filename.wav|prompt_filename.wav|content_to_be_synthesized`.

```bash
make synth \
	SYNTH_CKPT=path/to/ckpt.pt \
	SYNTH_CFG=configs/prob.yaml \
	PROMPT_DIR=assets/prompts \
	METADATA_FILE=lists/metadata.txt \
	OUTPUT_DIR=outputs \
	SYNTH_DEVICE=cuda:0 \
	NSTEPS_DURGEN=64 \
	NSTEPS_DENOISER=64 \
	TEMP_DURGEN=0.3 \
	TEMP_DENOISER=0.3 \
	GUIDANCE_SCALE=3.5 \
	SKIP_EXISTING=true \
	SYNTH_BATCH_SIZE=4
```

Optional knobs from the Makefile include `SKIP_EXISTING` to avoid re-synthesizing files that are already present, `SYNTH_BATCH_SIZE` to control how many metadata entries are processed per forward pass, and the same `DENOISER_METHOD`/forcing step overrides described above.

## 🔄 Training Flamed-TTS from scratch

TBD.

## ⚠️ Disclaimer

No individual or organization may use any technology described in this paper to generate, edit, or manipulate the speech of any person, including but not limited to government officials, political figures, or celebrities, without their explicit consent. Unauthorized use may violate applicable copyright, intellectual property, or privacy laws and could result in legal consequences.


























