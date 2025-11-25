#!/usr/bin/env python3
"""
Smoke tests for Flamed covering both inference sampling and compute_loss passes.
It fabricates simple deterministic tensors, runs sampling (without FaCodec) and
verifies that the loss pipeline executes end-to-end on dummy batches.
"""

import argparse
import os
import tempfile
from typing import Dict

import torch
from omegaconf import OmegaConf

# Disable numba's on-disk cache early to avoid issues inside sandboxed envs.
os.environ.setdefault("NUMBA_DISABLE_JIT", "1")
mpl_cache = os.path.join(tempfile.gettempdir(), "matplotlib-cache")
os.makedirs(mpl_cache, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", mpl_cache)

from flamed import Flamed  # noqa: E402
from flamed.text.symbols import symbols  # noqa: E402


def build_cfg(device: torch.device):
    """Load prior/prob configs and attach a dummy codec stub."""
    prior_cfg = OmegaConf.load("configs/prior.yaml")
    prob_cfg = OmegaConf.load("configs/prob.yaml")

    prior_cfg.device = str(device)
    prob_cfg.device = str(device)

    codec_cfg = OmegaConf.create(
        {
            "sr": 16000,
            "device": str(device),
            "encoder": {"checkpoint": None, "device": str(device)},
            "decoder": {"checkpoint": None, "device": str(device)},
        }
    )

    return OmegaConf.create(
        {
            "prior_generator": prior_cfg,
            "prob_generator": prob_cfg,
            "codec_cfg": codec_cfg,
        }
    )


def fabricate_dummy_inputs(
    cfg,
    batch_size: int,
    src_len: int,
    prompt_len: int,
    device: torch.device,
) -> Dict[str, torch.Tensor]:
    """Create simple deterministic tensors for inference smoke testing."""
    phoneme_vocab = len(symbols)
    n_quantizers = cfg.prior_generator.codec.n_quantizers
    spk_dim = cfg.prob_generator.spk_dim

    actual_src_len = max(2, min(src_len, phoneme_vocab - 1))
    phonemes = torch.zeros((batch_size, src_len), dtype=torch.long, device=device)
    base_seq = torch.arange(actual_src_len, device=device) % phoneme_vocab
    phonemes[:, :actual_src_len] = base_seq

    x_len = torch.full((batch_size,), actual_src_len, dtype=torch.long, device=device)

    prompts = torch.zeros(
        (batch_size, n_quantizers, prompt_len),
        dtype=torch.long,
        device=device,
    )

    spks = torch.zeros(batch_size, spk_dim, device=device)

    return {
        "phonemes": phonemes,
        "x_len": x_len,
        "prompts": prompts,
        "spks": spks,
    }


def fabricate_dummy_training_batch(
    cfg,
    batch_size: int,
    src_len: int,
    tgt_len: int,
    prompt_len: int,
    device: torch.device,
) -> Dict[str, torch.Tensor]:
    """Create tensors that mimic a collated training batch for compute_loss."""
    if tgt_len < src_len:
        raise ValueError("tgt_len must be >= src_len for positive durations.")

    base_inputs = fabricate_dummy_inputs(cfg, batch_size, src_len, prompt_len, device)
    n_quantizers = cfg.prior_generator.codec.n_quantizers
    vocab_size = cfg.prior_generator.codec.vocab_size
    emb_dim = cfg.prob_generator.target_dim
    spk_dim = cfg.prob_generator.spk_dim

    codes = torch.arange(tgt_len, device=device) % vocab_size
    codes = codes.unsqueeze(0).unsqueeze(0).expand(batch_size, n_quantizers, tgt_len).clone()

    y_len = torch.full((batch_size,), tgt_len, dtype=torch.long, device=device)

    phone_durations = torch.full((batch_size, src_len), tgt_len // src_len, dtype=torch.long, device=device)
    phone_durations = torch.clamp(phone_durations, min=1)
    used = phone_durations[0, 0].item() * src_len
    remainder = tgt_len - used
    if remainder < 0:
        remainder = 0
    if phone_durations.min() == phone_durations.max():
        # distribute any remainder across the leading phonemes
        if remainder:
            phone_durations[:, :remainder] += 1
    sil_durations = torch.zeros((batch_size, src_len), dtype=torch.long, device=device)

    prompt_tokens = []
    base_prompt = torch.arange(prompt_len, device=device) % vocab_size
    for idx in range(n_quantizers):
        prompt_tokens.append((base_prompt + idx) % vocab_size)
    prompt_tokens = torch.stack(prompt_tokens)
    prompts = prompt_tokens.unsqueeze(0).expand(batch_size, -1, -1).clone()

    embs = torch.linspace(0.0, 1.0, steps=tgt_len * emb_dim, device=device).view(1, tgt_len, emb_dim)
    embs = embs.expand(batch_size, -1, -1).clone()

    spks = torch.zeros((batch_size, spk_dim), dtype=torch.float32, device=device)
    for idx in range(batch_size):
        spks[idx, idx % spk_dim] = 1.0

    return {
        "phonemes": base_inputs["phonemes"],
        "x_len": base_inputs["x_len"],
        "codes": codes,
        "y_len": y_len,
        "phone_durations": phone_durations,
        "sil_durations": sil_durations,
        "embs": embs,
        "prompts": prompts,
        "spks": spks,
    }


@torch.inference_mode()
def run_inference(
    model: Flamed,
    inputs: Dict[str, torch.Tensor],
    dur_steps: int,
    denoiser_steps: int,
    temperature: float,
):
    phonemes = inputs["phonemes"]
    x_len = inputs["x_len"]
    prompts = inputs["prompts"]
    spks = inputs["spks"]

    prior_embs, logits, tgt_mask = model.prior_generator.sample(
        texts=phonemes,
        src_lens=x_len,
        max_src_len=phonemes.size(-1),
        prompts=prompts,
        prompts_len=prompts.size(-1),
        nfe=dur_steps,
        temperature=temperature,
    )

    latents = model.prob_generator.sample(
        cond=prior_embs,
        spk=spks,
        mask=~tgt_mask.unsqueeze(-1),
        nfe=denoiser_steps,
        temperature=temperature,
    )

    return {
        "phonemes": tuple(phonemes.shape),
        "prior_embs": tuple(prior_embs.shape),
        "prior_logits": tuple(logits.shape),
        "tgt_mask": tuple(tgt_mask.shape),
        "latents": tuple(latents.shape),
    }


@torch.inference_mode()
def run_compute_loss(model: Flamed, batch: Dict[str, torch.Tensor]):
    losses = model.forward(
        phonemes=batch["phonemes"],
        x_len=batch["x_len"],
        codes=batch["codes"],
        y_len=batch["y_len"],
        phone_durations=batch["phone_durations"],
        sil_durations=batch["sil_durations"],
        embs=batch["embs"],
        prompts=batch["prompts"],
        spks=batch["spks"],
    )
    return {name: float(value.detach().cpu().item()) for name, value in losses.items()}


def resolve_device(device_arg: str) -> torch.device:
    try:
        device = torch.device(device_arg)
    except RuntimeError as exc:
        raise RuntimeError(f"Invalid device string '{device_arg}'") from exc
    if device.type.startswith("cuda") and not torch.cuda.is_available():
        print("CUDA not available. Falling back to CPU.")
        return torch.device("cpu")
    return device


def parse_args():
    parser = argparse.ArgumentParser(description="Flamed smoke test exercising sampling and compute_loss without FaCodec.")
    parser.add_argument("--device", default="cpu", help="Device to run the test on (cpu or cuda:0).")
    parser.add_argument("--batch-size", type=int, default=2, help="Dummy batch size.")
    parser.add_argument("--src-len", type=int, default=32, help="Dummy phoneme sequence length.")
    parser.add_argument("--tgt-len", type=int, default=64, help="Dummy acoustic code length for loss testing.")
    parser.add_argument("--prompt-len", type=int, default=16, help="Dummy prompt length.")
    parser.add_argument("--dur-steps", type=int, default=8, help="Euler steps for duration generator sampling.")
    parser.add_argument("--denoiser-steps", type=int, default=16, help="Euler steps for denoiser sampling.")
    parser.add_argument("--temperature", type=float, default=0.7, help="Noise scale for sampling routines.")
    parser.add_argument("--seed", type=int, default=0, help="Random seed for reproducibility.")
    return parser.parse_args()


def main():
    args = parse_args()
    torch.manual_seed(args.seed)

    device = resolve_device(args.device)
    cfg = build_cfg(device)
    model = Flamed(cfg).to(device)

    inputs = fabricate_dummy_inputs(
        cfg=cfg,
        batch_size=args.batch_size,
        src_len=args.src_len,
        prompt_len=args.prompt_len,
        device=device,
    )

    train_batch = fabricate_dummy_training_batch(
        cfg=cfg,
        batch_size=args.batch_size,
        src_len=args.src_len,
        tgt_len=args.tgt_len,
        prompt_len=args.prompt_len,
        device=device,
    )
    loss_stats = run_compute_loss(model, train_batch)
    
    print("\n=== Total Parameters ===")
    print(f"{sum(p.numel() for p in model.parameters()):,}")

    print("\n=== Compute Loss ===")
    for key, value in loss_stats.items():
        print(f"{key:>12}: {value:.6f}")

    sample_stats = run_inference(
        model,
        inputs,
        dur_steps=args.dur_steps,
        denoiser_steps=args.denoiser_steps,
        temperature=args.temperature,
    )

    print("\n=== Inference Shapes ===")
    for key, shape in sample_stats.items():
        print(f"{key:>12}: {shape}")
        
if __name__ == "__main__":
    main()
