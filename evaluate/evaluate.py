from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, Mapping

from utils import cal_phone, cal_prosody, cal_speechrate, cal_sim, cal_utmos, cal_wer


DEFAULT_METRICS = ("utmos", "sim", "wer", "prosody")


@dataclass
class EvalContext:
    synth_path: Path
    tgt_path: Path | None
    prompt_path: Path | None
    output_dir: Path
    model_name: str
    codec: str
    sample_rate: int
    device: str
    sim_model: str | None
    sim_ckpt: str | None
    utmos_ckpt: str | None
    cache_dir: Path
    target2prompt: Dict[str, str]
    target2transcript: Dict[str, str]


def load_manifest(manifest: Path | None) -> tuple[Dict[str, str], Dict[str, str]]:
    """Parse manifest file into prompt/transcript lookups."""
    target2prompt: Dict[str, str] = {}
    target2transcript: Dict[str, str] = {}

    if manifest is None:
        return target2prompt, target2transcript

    with manifest.open() as fin:
        for line_no, line in enumerate(fin, start=1):
            parts = line.strip().split("|")
            if len(parts) < 3:
                print(f"[WARN] Skipping malformed manifest line {line_no}: {line.strip()}")
                continue
            target, prompt, transcript = parts[:3]
            target2prompt[target] = prompt
            target2transcript[target] = transcript
    return target2prompt, target2transcript


def require_path(path: Path | None, description: str) -> Path:
    if path is None:
        raise ValueError(f"{description} is required for the selected metrics.")
    return path


def require_mapping(mapping: Mapping[str, str], description: str) -> Mapping[str, str]:
    if not mapping:
        raise ValueError(f"{description} is required for the selected metrics.")
    return mapping


def build_metric_calculators(
    requested_metrics: Iterable[str],
    ctx: EvalContext,
) -> Dict[str, Callable[[str], Any]]:
    """Create callables for each requested metric with all shared params bound."""
    calculators: Dict[str, Callable[[str], Any]] = {}

    for metric in requested_metrics:
        name = metric.lower()
        if name == "sim":
            calculators[name] = lambda output_file, c=ctx: cal_sim(
                synth_path=str(c.synth_path),
                prompt_path=str(require_path(c.prompt_path, "--prompt_path")),
                output_file=output_file,
                target2prompt=require_mapping(c.target2prompt, "target->prompt mapping (manifest)"),
                codec=c.codec,
                sr=c.sample_rate,
                device=c.device,
                model_name=c.sim_model or "wavlm_large",
                ckpt=c.sim_ckpt,
                cache_dir=c.cache_dir / "sim",
            )
        elif name == "utmos":
            calculators[name] = lambda output_file, c=ctx: cal_utmos(
                audio_path=str(c.synth_path),
                output_file=output_file,
                sr=c.sample_rate,
                device=c.device,
                ckpt=c.utmos_ckpt,
                cache_dir=c.cache_dir / "utmos",
            )
        elif name == "wer":
            calculators[name] = lambda output_file, c=ctx: cal_wer(
                audio_path=str(c.synth_path),
                output_file=output_file,
                target2transcript=require_mapping(c.target2transcript, "target->transcript mapping (manifest)"),
                sr=c.sample_rate,
                device=c.device,
            )
        elif name == "prosody":
            calculators[name] = lambda output_file, c=ctx: cal_prosody(
                synth_path=str(c.synth_path),
                prompt_path=str(require_path(c.prompt_path, "--prompt_path")),
                output_file=output_file,
                target2prompt=require_mapping(c.target2prompt, "target->prompt mapping (manifest)"),
            )
        elif name == "speechrate":
            calculators[name] = lambda output_file, c=ctx: cal_speechrate(
                audio_path=str(c.synth_path),
                output_file=output_file,
                target2transcript=require_mapping(c.target2transcript, "target->transcript mapping (manifest)"),
            )
        elif name == "phone":
            calculators[name] = lambda output_file, c=ctx: cal_phone(
                tgt_path=str(require_path(c.tgt_path, "--tgt_path")),
                output_file=output_file,
                target2transcript=require_mapping(c.target2transcript, "target->transcript mapping (manifest)"),
            )
        else:
            print(f"[WARN] Unsupported metric '{metric}', skipping.")
    return calculators


def run_metrics(
    calculators: Mapping[str, Callable[[str], Any]],
    output_dir: Path,
    model_name: str,
) -> Dict[str, Any]:
    results: Dict[str, Any] = {}
    for metric, func in calculators.items():
        output_file = output_dir / f"{metric}_{model_name}.csv"
        print(f"Evaluating metric '{metric}' -> {output_file} ...")
        results[metric] = func(output_file=str(output_file))
    return results


def safe_extract(results: Mapping[str, Any], key: str, extractor: Callable[[Any], float], default: float = -1.0) -> float:
    try:
        return extractor(results[key])
    except Exception:
        return default


def summarize(results: Mapping[str, Any]) -> Dict[str, float]:
    return {
        "UTMOS": safe_extract(results, "utmos", lambda v: round(float(v), 2)),
        "WER": safe_extract(results, "wer", lambda v: round(float(v), 2)),
        "SIM-O": safe_extract(results, "sim", lambda v: round(float(v["SIM-O"]), 2)),
        "SIM-R": safe_extract(results, "sim", lambda v: round(float(v["SIM-R"]), 2)),
        "F0 ACC": safe_extract(results, "prosody", lambda v: round(float(v["F0 ACC"]), 2)),
        "F0 RMSE": safe_extract(results, "prosody", lambda v: round(float(v["F0 RMSE"]), 2)),
        "ENERGY ACC": safe_extract(results, "prosody", lambda v: round(float(v["ENERGY ACC"]), 2)),
        "ENERGY RMSE": safe_extract(results, "prosody", lambda v: round(float(v["ENERGY RMSE"]), 3)),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run evaluation metrics for synthesized audio.")
    parser.add_argument("--manifest", type=Path, default=None, help="Manifest file mapping targets to prompts/transcripts.")
    parser.add_argument("--synth_path", type=Path, required=True, help="Directory with synthesized .wav files.")
    parser.add_argument("--tgt_path", type=Path, default=None, help="Directory with target/reference .wav files (for phone metrics).")
    parser.add_argument("--prompt_path", type=Path, default=None, help="Directory with prompt/reference .wav files.")
    parser.add_argument("--output_path", type=Path, required=True, help="Directory to write metric CSVs.")
    parser.add_argument("--name", required=True, help="Model/run name used in output filenames.")
    parser.add_argument("--sr", type=int, default=16000, help="Audio sampling rate.")
    parser.add_argument("--metrics", nargs="+", default=list(DEFAULT_METRICS), help="Metrics to compute.")
    parser.add_argument("--device", default="cuda:0", help="Device string passed to torch/transformers.")
    parser.add_argument("--codec", default="facodec", help="Codec used to reconstruct prompts for similarity.")
    parser.add_argument("--sim_model", default='wavlm_large', help="Speaker model name for SIM computation.")
    parser.add_argument("--sim_ckpt", default=None, help="Checkpoint path or URL for the SIM model.")
    parser.add_argument("--utmos_ckpt", default=None, help="Checkpoint path or URL for UTMOS.")
    parser.add_argument("--cache_dir", type=Path, default=Path.home() / ".cache" / "flamed" / "eval", help="Where to cache downloaded checkpoints.")
    return parser.parse_args()


def main():
    args = parse_args()

    output_dir = args.output_path
    output_dir.mkdir(parents=True, exist_ok=True)
    cache_dir = Path(args.cache_dir).expanduser()

    target2prompt, target2transcript = load_manifest(args.manifest)
    ctx = EvalContext(
        synth_path=args.synth_path,
        tgt_path=args.tgt_path,
        prompt_path=args.prompt_path,
        output_dir=output_dir,
        model_name=args.name,
        codec=args.codec,
        sample_rate=args.sr,
        device=args.device,
        sim_model=args.sim_model,
        sim_ckpt=args.sim_ckpt,
        utmos_ckpt=args.utmos_ckpt,
        cache_dir=cache_dir,
        target2prompt=target2prompt,
        target2transcript=target2transcript,
    )

    calculators = build_metric_calculators(args.metrics, ctx)
    if not calculators:
        raise ValueError("No valid metrics requested.")

    results = run_metrics(calculators, output_dir, args.name)
    summary = summarize(results)

    summary_line = "\t".join(f"{k}: {v}" for k, v in summary.items())
    print(summary_line)


if __name__ == "__main__":
    main()
