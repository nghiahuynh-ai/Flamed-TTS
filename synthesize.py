#!/usr/bin/env python3
import argparse
import math
import os
from typing import Dict, List, Optional, Tuple

import librosa
import soundfile as sf
import torch
from omegaconf import OmegaConf
from torch.nn.utils.rnn import pad_sequence
from tqdm import tqdm

from flamed import Flamed
from flamed.models.facodec import FACodecEncoder, FACodecDecoder

SR = 16000
CURDIR = os.path.dirname(__file__)


def str2bool(value):
    if isinstance(value, bool):
        return value
    value = str(value).strip().lower()
    if value in {"true", "1", "yes", "y"}:
        return True
    if value in {"false", "0", "no", "n"}:
        return False
    raise argparse.ArgumentTypeError(f"Cannot interpret '{value}' as boolean.")


def resolve_device(device_str: str) -> torch.device:
    device = torch.device(device_str)
    if device.type.startswith("cuda") and not torch.cuda.is_available():
        print("CUDA not available. Falling back to CPU.")
        return torch.device("cpu")
    return device


def load_audio(wav_path: str) -> torch.Tensor:
    wav = librosa.load(wav_path, sr=SR)[0]
    wav = torch.from_numpy(wav).float().unsqueeze(0).unsqueeze(0)
    return wav


def get_codec(device: torch.device):
    fa_encoder = FACodecEncoder(
        ngf=32,
        up_ratios=[2, 4, 5, 5],
        out_channels=256,
    )

    fa_decoder = FACodecDecoder(
        in_channels=256,
        upsample_initial_channel=1024,
        ngf=32,
        up_ratios=[5, 5, 4, 2],
        vq_num_q_c=2,
        vq_num_q_p=1,
        vq_num_q_r=3,
        vq_dim=256,
        codebook_dim=8,
        codebook_size_prosody=10,
        codebook_size_content=10,
        codebook_size_residual=10,
        use_gr_x_timbre=True,
        use_gr_residual_f0=True,
        use_gr_residual_phone=True,
    )

    encoder_ckpt = os.path.join(CURDIR, "flamed", "models", "facodec", "checkpoints", "ns3_facodec_encoder.bin")
    decoder_ckpt = os.path.join(CURDIR, "flamed", "models", "facodec", "checkpoints", "ns3_facodec_decoder.bin")
    fa_encoder.load_state_dict(torch.load(encoder_ckpt, map_location=device))
    fa_decoder.load_state_dict(torch.load(decoder_ckpt, map_location=device))
    fa_encoder.to(device).eval()
    fa_decoder.to(device).eval()

    return fa_encoder, fa_decoder


def prepare_model(cfg_path: str, ckpt_path: str, device: torch.device, weights_only: bool, scheduler: str | None = None):
    cfg = OmegaConf.load(cfg_path)
    cfg["prob_generator"]["device"] = str(device)
    cfg["prior_generator"]["device"] = str(device)
    if scheduler is not None:
        cfg["prob_generator"]["scheduler"] = scheduler

    model = Flamed.from_pretrained(
        cfg=cfg,
        ckpt_path=ckpt_path,
        device=device,
        weights_only=weights_only,
        training_mode=False,
    )
    model.to(device)
    return model


def _resolve_prompt_path(prompt_dir: str, prompt_name: str) -> str:
    if os.path.isabs(prompt_name):
        return prompt_name
    return os.path.join(prompt_dir, prompt_name)


def chunked(seq, size):
    for idx in range(0, len(seq), size):
        yield seq[idx : idx + size]


def encode_prompt_features(
    model: Flamed,
    codec_encoder,
    codec_decoder,
    prompt_path: str,
    cache: Dict[str, Tuple[torch.Tensor, torch.Tensor]],
):
    if prompt_path in cache:
        return cache[prompt_path]

    with torch.inference_mode():
        acoustic_prompt = model._preprocess_acoustic_prompt(prompt_path, sr=SR)
        enc_out = codec_encoder(acoustic_prompt)
        _, prompts, _, _, timbre = codec_decoder(enc_out, eval_vq=False, vq=True)
    prompts = prompts.permute(1, 0, 2).contiguous().squeeze(0).detach().cpu()
    timbre = timbre.squeeze(0).detach().cpu()
    cache[prompt_path] = (prompts, timbre)
    return cache[prompt_path]


def pad_prompts(prompt_tensors: List[torch.Tensor], pad_value: int, device: torch.device):
    if not prompt_tensors:
        raise ValueError("pad_prompts received an empty list.")
    n_quantizers = prompt_tensors[0].size(0)
    max_len = max(tensor.size(-1) for tensor in prompt_tensors)
    padded = torch.full(
        (len(prompt_tensors), n_quantizers, max_len),
        fill_value=pad_value,
        dtype=prompt_tensors[0].dtype,
        device=device,
    )
    for idx, tensor in enumerate(prompt_tensors):
        length = tensor.size(-1)
        padded[idx, :, :length] = tensor.to(device)
    return padded, max_len


def build_metadata_batch(
    model: Flamed,
    codec_encoder,
    codec_decoder,
    batch_items: List[Dict[str, str]],
    prompt_cache: Dict[str, Tuple[torch.Tensor, torch.Tensor]],
):
    phoneme_tensors, src_lens = [], []
    prompt_tensors, timbres = [], []

    for item in batch_items:
        seq, _, _ = model._preprocess_english(item["text"])
        seq = seq.squeeze(0)
        phoneme_tensors.append(seq)
        src_lens.append(seq.size(0))

        prompt_codes, timbre = encode_prompt_features(
            model, codec_encoder, codec_decoder, item["prompt_path"], prompt_cache
        )
        prompt_tensors.append(prompt_codes)
        timbres.append(timbre)

    phonemes = pad_sequence(phoneme_tensors, batch_first=True, padding_value=0)
    src_lens_tensor = torch.tensor(src_lens, dtype=torch.long)

    pad_value = model.prior_generator.config["codec"]["vocab_size"]
    prompts, _ = pad_prompts(prompt_tensors, pad_value=pad_value, device=torch.device("cpu"))
    timbre_tensor = torch.stack(timbres, dim=0)

    return phonemes, src_lens_tensor, prompts, timbre_tensor


def synthesize_with_prompts(
    model: Flamed,
    codec_encoder,
    codec_decoder,
    text: str,
    prompt_dir: str,
    prompt_list: List[str],
    output_dir: str,
    nsteps_durgen: int,
    nsteps_denoiser: int,
    temp_durgen: float,
    temp_denoiser: float,
    guidance_scale: Optional[float],
    denoiser_method: str,
    forcing_steps_min: Optional[int],
    forcing_steps_max: Optional[int],
):
    os.makedirs(output_dir, exist_ok=True)
    infer_times, output_durations = [], []

    for prompt_name in tqdm(prompt_list, desc="Synthesizing prompts"):
        prompt_path = _resolve_prompt_path(prompt_dir, prompt_name)
        audio_prompt = load_audio(prompt_path)

        results = model.sample(
            text=text,
            prompt_raw=audio_prompt,
            sr=SR,
            codec_encoder=codec_encoder,
            codec_decoder=codec_decoder,
            nsteps_durgen=nsteps_durgen,
            nsteps_denoiser=nsteps_denoiser,
            temp_durgen=temp_durgen,
            temp_denoiser=temp_denoiser,
            guidance_scale=guidance_scale,
            denoiser_method=denoiser_method,
            forcing_steps_min=forcing_steps_min,
            forcing_steps_max=forcing_steps_max,
        )

        infer_times.append(results["time"])
        output_durations.append(len(results["wav"]) / SR)
        out_name = f"{os.path.splitext(os.path.basename(prompt_name))[0]}-{nsteps_durgen}-{nsteps_denoiser}-{temp_durgen}-{temp_denoiser}.wav"
        sf.write(os.path.join(output_dir, out_name), results["wav"], SR)

    if not infer_times:
        return None
    rtf = [t / d for t, d in zip(infer_times, output_durations)]
    return sum(rtf) / len(rtf)


def synthesize_with_metadata(
    model: Flamed,
    codec_encoder,
    codec_decoder,
    metadata_file: str,
    prompt_dir: str,
    output_dir: str,
    nsteps_durgen: int,
    nsteps_denoiser: int,
    temp_durgen: float,
    temp_denoiser: float,
    skip_existing: bool,
    batch_size: int,
    guidance_scale: Optional[float],
    denoiser_method: str,
    forcing_steps_min: Optional[int],
    forcing_steps_max: Optional[int],
):
    with open(metadata_file, "r", encoding="utf-8") as fin:
        entries = [line.strip() for line in fin if line.strip()]

    target_dir = os.path.join(output_dir, f"nfe{nsteps_denoiser}-temp{temp_denoiser}-cfg{guidance_scale}-{denoiser_method}")
    os.makedirs(target_dir, exist_ok=True)

    prompt_cache: Dict[str, Tuple[torch.Tensor, torch.Tensor]] = {}
    infer_times, output_durations = [], []

    pending: List[Dict[str, str]] = []
    for entry in entries:
        try:
            filename, prompt_filename, transcript, _, _, _ = entry.split("|")
        except ValueError:
            print(f"[WARN] Malformed line skipped: {entry}")
            continue

        out_path = os.path.join(target_dir, filename)
        if skip_existing and os.path.exists(out_path):
            continue

        prompt_path = _resolve_prompt_path(prompt_dir, prompt_filename)
        pending.append(
            {
                "filename": filename,
                "prompt_path": prompt_path,
                "text": transcript,
                "out_path": out_path,
            }
        )

    if not pending:
        return None

    num_batches = math.ceil(len(pending) / batch_size)
    for batch in tqdm(
        chunked(pending, batch_size),
        total=num_batches,
        desc="Synthesizing metadata entries",
    ):
        phonemes, src_lens, prompts, timbres = build_metadata_batch(
            model=model,
            codec_encoder=codec_encoder,
            codec_decoder=codec_decoder,
            batch_items=batch,
            prompt_cache=prompt_cache,
        )
        batch_outputs = model.sample_batch(
            phonemes=phonemes,
            src_lens=src_lens,
            prompts=prompts,
            timbres=timbres,
            codec_decoder=codec_decoder,
            temp_durgen=temp_durgen,
            temp_denoiser=temp_denoiser,
            nsteps_durgen=nsteps_durgen,
            nsteps_denoiser=nsteps_denoiser,
            guidance_scale=guidance_scale,
            denoiser_method=denoiser_method,
            forcing_steps_min=forcing_steps_min,
            forcing_steps_max=forcing_steps_max,
        )
        wav_batch = batch_outputs["wav"]
        per_sample_time = batch_outputs["time"] / len(batch)
        for item, wav_tensor in zip(batch, wav_batch):
            wav = wav_tensor[0].detach().cpu().numpy()
            sf.write(item["out_path"], wav, SR)
            infer_times.append(per_sample_time)
            output_durations.append(len(wav) / SR)

    if not infer_times:
        return None
    rtf = [t / d for t, d in zip(infer_times, output_durations)]
    return sum(rtf) / len(rtf)


def _normalize_args(args: argparse.Namespace) -> argparse.Namespace:
    if getattr(args, "prompt_dir", None) is None and hasattr(args, "input_dir"):
        args.prompt_dir = args.input_dir
    return args


def _validate_args(args: argparse.Namespace):
    metadata_mode = args.metadata_file is not None
    prompt_mode = args.prompt_list is not None
    if metadata_mode == prompt_mode:
        raise ValueError("Specify either --prompt-list (direct mode) or --metadata-file (batch mode), but not both.")
    if args.prompt_dir is None:
        raise ValueError("--prompt-dir/--input-dir is required.")
    if prompt_mode and not args.text:
        raise ValueError("--text is required when using --prompt-list.")
    if metadata_mode:
        if not os.path.isfile(args.metadata_file):
            raise ValueError(f"Metadata file not found: {args.metadata_file}")
        if args.batch_size < 1:
            raise ValueError("--batch-size must be >= 1.")
    if args.denoiser_method == "forcing":
        if args.forcing_steps_min is None or args.forcing_steps_max is None:
            raise ValueError("--forcing-steps-min and --forcing-steps-max are required when --denoiser-method forcing is selected.")
        if args.forcing_steps_min <= 0 or args.forcing_steps_max <= 0:
            raise ValueError("--forcing-steps-min and --forcing-steps-max must be positive integers.")
        if args.forcing_steps_min > args.forcing_steps_max:
            raise ValueError("--forcing-steps-min cannot exceed --forcing-steps-max.")
        

def build_arg_parser():
    parser = argparse.ArgumentParser(description="Unified Flamed-TTS synthesis script.")
    parser.add_argument("--ckpt-path", type=str, required=True, help="Path to Flamed checkpoint.")
    parser.add_argument("--cfg-path", type=str, required=True, help="Path to model config yaml.")
    parser.add_argument("--text", type=str, default=None, help="Text content (prompt-list mode).")
    parser.add_argument("--prompt-list", nargs="+", default=None, help="Prompt filenames for direct synthesis.")
    parser.add_argument("--prompt-dir", "--input-dir", dest="prompt_dir", type=str, default=None, help="Directory containing prompt WAV files.")
    parser.add_argument("--metadata-file", "--text-file", dest="metadata_file", type=str, default=None, help="Metadata file with lines formatted as target|prompt|text.")
    parser.add_argument("--output-dir", type=str, default=".", help="Directory to store outputs.")
    parser.add_argument("--weights-only", type=str2bool, default=True, help="Load checkpoint weights_only flag (default: True).")
    parser.add_argument("--nsteps-durgen", type=int, default=64, help="Duration generator sampling steps.")
    parser.add_argument("--nsteps-denoiser", type=int, default=64, help="Denoiser sampling steps.")
    parser.add_argument("--temp-durgen", type=float, default=0.3, help="Duration generator temperature.")
    parser.add_argument("--temp-denoiser", type=float, default=0.3, help="Denoiser temperature.")
    parser.add_argument("--guidance-scale", type=float, default=None, help="Classifier-free guidance scale for the denoiser (default: use config value).")
    parser.add_argument("--device", type=str, default="cuda:0", help="Device to run inference on.")
    parser.add_argument("--skip-existing", type=str2bool, default=True, help="Skip samples whose output files already exist (metadata mode).")
    parser.add_argument("--batch-size", type=int, default=4, help="Number of metadata samples to synthesize per batch.")
    parser.add_argument("--denoiser-method", choices=["euler", "forcing"], default="euler", help="Integrator to sample latents (euler or forcing).")
    parser.add_argument("--forcing-steps-min", type=int, default=None, help="Minimum steps for the earliest latent when using forcing.")
    parser.add_argument("--forcing-steps-max", type=int, default=None, help="Maximum steps for the latest latent when using forcing.")
    parser.add_argument("--scheduler", type=str, default=None, help="Scheduler choice forwarded to prob_generator (e.g., partial).")
    return parser


def main(args: Optional[argparse.Namespace] = None):
    parser = build_arg_parser()
    cli_invocation = args is None
    if cli_invocation:
        args = parser.parse_args()

    args = _normalize_args(args)
    try:
        _validate_args(args)
    except ValueError as exc:
        if cli_invocation:
            parser.error(str(exc))
        else:
            raise

    device = resolve_device(args.device)
    codec_encoder, codec_decoder = get_codec(device)
    model = prepare_model(args.cfg_path, args.ckpt_path, device, args.weights_only, scheduler=args.scheduler)

    rtf = None
    if args.metadata_file:
        rtf = synthesize_with_metadata(
            model=model,
            codec_encoder=codec_encoder,
            codec_decoder=codec_decoder,
            metadata_file=args.metadata_file,
            prompt_dir=args.prompt_dir,
            output_dir=args.output_dir,
            nsteps_durgen=args.nsteps_durgen,
            nsteps_denoiser=args.nsteps_denoiser,
            temp_durgen=args.temp_durgen,
            temp_denoiser=args.temp_denoiser,
            skip_existing=args.skip_existing,
            batch_size=args.batch_size,
            guidance_scale=args.guidance_scale,
            denoiser_method=args.denoiser_method,
            forcing_steps_min=args.forcing_steps_min,
            forcing_steps_max=args.forcing_steps_max,
        )
    else:
        rtf = synthesize_with_prompts(
            model=model,
            codec_encoder=codec_encoder,
            codec_decoder=codec_decoder,
            text=args.text,
            prompt_dir=args.prompt_dir,
            prompt_list=args.prompt_list,
            output_dir=args.output_dir,
            nsteps_durgen=args.nsteps_durgen,
            nsteps_denoiser=args.nsteps_denoiser,
            temp_durgen=args.temp_durgen,
            temp_denoiser=args.temp_denoiser,
            guidance_scale=args.guidance_scale,
            denoiser_method=args.denoiser_method,
            forcing_steps_min=args.forcing_steps_min,
            forcing_steps_max=args.forcing_steps_max,
        )

    if rtf is not None:
        print("=" * 20, "Avg RTF", "=" * 20)
        print(">" * 5, "RTF:", round(rtf, 3))
    else:
        print("No samples were generated.")

    return rtf


if __name__ == "__main__":
    main()
