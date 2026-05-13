from __future__ import annotations

import argparse
import os
import platform
from pathlib import Path

import lightning.pytorch as pl
import torch
from lightning.pytorch.callbacks import LearningRateMonitor, ModelCheckpoint
from lightning.pytorch.loggers import WandbLogger
from omegaconf import OmegaConf

from flamed import Flamed


def _configure_start_method(preferred_method=None):
    """Ensure multiprocessing start method matches the requested or default policy."""
    import multiprocessing as mp

    env_requested = os.environ.get("FLAMED_MP_START_METHOD")
    preferred = preferred_method or env_requested
    default = "fork" if platform.system() == "Linux" else "spawn"
    target_method = preferred or default

    available = mp.get_all_start_methods()
    if target_method not in available:
        fallback = "spawn" if "spawn" in available else default
        target_method = fallback

    current = mp.get_start_method(allow_none=True)
    if current == target_method:
        return

    try:
        mp.set_start_method(target_method, force=True)
    except RuntimeError:
        pass


def _parse_bool(value: str) -> bool:
    normalized = str(value).strip().lower()
    if normalized in {"1", "true", "t", "yes", "y", "on"}:
        return True
    if normalized in {"0", "false", "f", "no", "n", "off"}:
        return False
    raise ValueError(f"Invalid boolean value: {value!r}")


def _parse_pipeline(pipeline_str):
    allowed = {"PriorGenerator", "ProbGenerator"}
    modules = [item.strip() for item in pipeline_str.split(",") if item.strip()]
    if not modules:
        raise ValueError("PIPELINE must include at least one of PriorGenerator or ProbGenerator.")
    invalid = [module for module in modules if module not in allowed]
    if invalid:
        raise ValueError(
            f"Unsupported pipeline entries: {', '.join(invalid)}. "
            f"Valid options: {', '.join(sorted(allowed))}."
        )

    seen = set()
    deduped = []
    for module in modules:
        if module in seen:
            continue
        seen.add(module)
        deduped.append(module)
    return deduped


def _parse_devices(devices_raw: str, *, accelerator: str):
    devices_raw = devices_raw.strip()
    if accelerator == "cpu":
        if devices_raw.lower() == "auto":
            return 1
        if "," in devices_raw:
            return max(1, len([device for device in devices_raw.split(",") if device.strip()]))
        return max(1, int(devices_raw))

    if devices_raw.lower() == "auto":
        return "auto"
    if "," not in devices_raw:
        return [int(devices_raw)]
    return [int(device.strip()) for device in devices_raw.split(",") if device.strip()]


def _resolve_precision(precision_arg: str | None, *, accelerator: str) -> str:
    if precision_arg is not None:
        return precision_arg
    if accelerator != "gpu":
        return "32-true"

    bf16_supported_fn = getattr(torch.cuda, "is_bf16_supported", None)
    if callable(bf16_supported_fn) and bf16_supported_fn():
        return "bf16-mixed"
    return "16-mixed"


def _set_torch_runtime_flags(*, accelerator: str, matmul_precision: str, disable_tf32: bool):
    if accelerator != "gpu":
        return
    torch.set_float32_matmul_precision(matmul_precision)
    allow_tf32 = not disable_tf32
    if hasattr(torch.backends, "cuda") and hasattr(torch.backends.cuda, "matmul"):
        torch.backends.cuda.matmul.allow_tf32 = allow_tf32
    if hasattr(torch.backends, "cudnn") and torch.backends.cudnn.is_available():
        torch.backends.cudnn.allow_tf32 = allow_tf32
        torch.backends.cudnn.benchmark = True


def _load_base_configs(config_dir: Path):
    prob_cfg = OmegaConf.load(config_dir / "prob.yaml")
    prior_cfg = OmegaConf.load(config_dir / "prior.yaml")
    codec_cfg = OmegaConf.load(config_dir / "codec.yaml")
    optimizer_cfg = OmegaConf.load(config_dir / "optimizer.yaml")
    data_cfg = OmegaConf.load(config_dir / "data.yaml")
    return prob_cfg, prior_cfg, codec_cfg, optimizer_cfg, data_cfg


def _apply_runtime_overrides(
    *,
    prob_cfg,
    prior_cfg,
    codec_cfg,
    optimizer_cfg,
    data_cfg,
    accelerator: str,
    args: argparse.Namespace,
):
    runtime_device = "cuda" if accelerator == "gpu" else "cpu"
    prob_cfg["device"] = runtime_device
    prior_cfg["device"] = runtime_device
    codec_cfg["device"] = runtime_device
    codec_cfg["encoder"]["device"] = runtime_device
    codec_cfg["decoder"]["device"] = runtime_device
    optimizer_cfg["device"] = runtime_device

    optimizer_cfg["epochs"] = int(args.epochs)
    optimizer_cfg["batch_size"] = int(args.batch_size)
    if args.max_steps is not None:
        optimizer_cfg["max_steps"] = int(args.max_steps)
    if args.warmup_steps is not None:
        optimizer_cfg["warmup_steps"] = int(args.warmup_steps)

    data_cfg["batch_size"] = int(args.batch_size)
    if args.num_workers is not None:
        data_cfg["num_workers"] = int(args.num_workers)
    if args.pin_memory is not None:
        data_cfg["pin_memory"] = bool(args.pin_memory)
    if args.prefetch_factor is not None:
        data_cfg["prefetch_factor"] = int(args.prefetch_factor)
    if args.adaptive_batching:
        adaptive_cfg = data_cfg.get("adaptive_batching")
        if adaptive_cfg is None:
            adaptive_cfg = {}
            data_cfg["adaptive_batching"] = adaptive_cfg
        adaptive_cfg["enabled"] = True
        if args.adaptive_target_utilization is not None:
            adaptive_cfg["target_memory_utilization"] = float(args.adaptive_target_utilization)
        if args.adaptive_max_batch_size is not None:
            adaptive_cfg["max_batch_size"] = int(args.adaptive_max_batch_size)
        if args.adaptive_memory_budget is not None:
            adaptive_cfg["memory_budget"] = int(args.adaptive_memory_budget)
        if args.adaptive_target_batch_cost is not None:
            adaptive_cfg["target_batch_cost"] = int(args.adaptive_target_batch_cost)
        if args.adaptive_seed is not None:
            adaptive_cfg["seed"] = int(args.adaptive_seed)


def train(args: argparse.Namespace):
    pipeline = _parse_pipeline(args.pipeline)
    accelerator = "gpu" if torch.cuda.is_available() else "cpu"
    devices = _parse_devices(args.devices, accelerator=accelerator)
    precision = _resolve_precision(args.precision, accelerator=accelerator)

    _set_torch_runtime_flags(
        accelerator=accelerator,
        matmul_precision=args.matmul_precision,
        disable_tf32=args.disable_tf32,
    )

    project_root = Path(__file__).resolve().parent
    config_dir = project_root / "configs"
    exp_root = Path(args.exp_root).expanduser().resolve()
    exp_dir = exp_root / args.exp_name
    exp_dir.mkdir(parents=True, exist_ok=True)

    prob_cfg, prior_cfg, codec_cfg, optimizer_cfg, data_cfg = _load_base_configs(config_dir)
    _apply_runtime_overrides(
        prob_cfg=prob_cfg,
        prior_cfg=prior_cfg,
        codec_cfg=codec_cfg,
        optimizer_cfg=optimizer_cfg,
        data_cfg=data_cfg,
        accelerator=accelerator,
        args=args,
    )

    cfg = OmegaConf.create(
        {
            "prior_generator": prior_cfg,
            "prob_generator": prob_cfg,
            "codec_cfg": codec_cfg,
            "pipeline": pipeline,
        }
    )
    OmegaConf.save(cfg, exp_dir / "config.yaml")

    requires_prior_ckpt = ("ProbGenerator" in pipeline) and ("PriorGenerator" not in pipeline)
    if requires_prior_ckpt and not args.prior_ckpt:
        raise ValueError("prior_ckpt is required when training ProbGenerator without PriorGenerator.")

    if requires_prior_ckpt:
        model = Flamed.from_pretrained(
            cfg,
            args.prior_ckpt,
            device="cpu",
            weights_only=False,
            training_mode=True,
            modules=["prior_generator"],
        )
    else:
        model = Flamed(cfg)

    model.setup_dataset_optimizer(data_cfg, optimizer_cfg)
    train_data, val_data = model.get_dataset()

    checkpoint_callback = ModelCheckpoint(
        monitor="total_loss_val_epoch",
        filename="ckpt-{epoch:02d}-{total_loss_val_epoch:.2f}",
        save_top_k=10,
        mode="min",
        save_last=True,
    )
    lr_monitor = LearningRateMonitor(logging_interval="step")

    logger = WandbLogger(
        project=args.proj_name,
        name=args.exp_name,
        save_dir=str(exp_dir),
        version=args.ver,
        resume="allow",
    )

    train_uses_custom_batch_sampler = bool(
        getattr(model.dataset, "train_loader_uses_custom_batch_sampler", False)
    )
    if train_uses_custom_batch_sampler:
        print(
            "[train.py] Adaptive/custom batch sampler detected; "
            "setting Trainer(use_distributed_sampler=False)."
        )

    trainer = pl.Trainer(
        devices=devices,
        accelerator=accelerator,
        precision=precision,
        max_epochs=int(args.epochs),
        accumulate_grad_batches=max(1, int(args.accumulate_grad_batches)),
        gradient_clip_val=float(args.gradient_clip_val),
        log_every_n_steps=int(args.log_every_n_steps),
        check_val_every_n_epoch=int(args.check_val_every_n_epoch),
        num_sanity_val_steps=int(args.num_sanity_val_steps),
        enable_checkpointing=True,
        logger=logger,
        default_root_dir=str(exp_dir),
        callbacks=[checkpoint_callback, lr_monitor],
        use_distributed_sampler=not train_uses_custom_batch_sampler,
    )

    trainer.fit(
        model=model,
        ckpt_path=args.ckpt,
        train_dataloaders=train_data,
        val_dataloaders=val_data,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--proj_name", type=str, required=True)
    parser.add_argument("--ver", type=str, required=True)
    parser.add_argument("--exp_root", type=str, default="experiments")
    parser.add_argument("--exp_name", type=str, required=True)
    parser.add_argument("--devices", type=str, default="0")
    parser.add_argument(
        "--pipeline",
        type=str,
        default="PriorGenerator,ProbGenerator",
        help="Comma-separated modules to train: PriorGenerator, ProbGenerator, or both.",
    )
    parser.add_argument(
        "--prior_ckpt",
        type=str,
        default=None,
        help=(
            "Checkpoint containing PriorGenerator weights "
            "(required when training ProbGenerator alone)."
        ),
    )
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--ckpt", type=str, default=None)

    parser.add_argument(
        "--precision",
        type=str,
        default=None,
        help="Lightning precision mode (for example: 16-mixed, bf16-mixed, 32-true).",
    )
    parser.add_argument("--accumulate_grad_batches", type=int, default=1)
    parser.add_argument("--gradient_clip_val", type=float, default=0.0)
    parser.add_argument("--log_every_n_steps", type=int, default=1)
    parser.add_argument("--check_val_every_n_epoch", type=int, default=1)
    parser.add_argument("--num_sanity_val_steps", type=int, default=2)
    parser.add_argument("--max_steps", type=int, default=None)
    parser.add_argument("--warmup_steps", type=int, default=None)

    parser.add_argument("--num_workers", type=int, default=None)
    parser.add_argument(
        "--pin_memory",
        type=_parse_bool,
        default=None,
        help="Override data.pin_memory with true/false.",
    )
    parser.add_argument("--prefetch_factor", type=int, default=None)

    parser.add_argument("--adaptive_batching", action="store_true")
    parser.add_argument("--adaptive_target_utilization", type=float, default=None)
    parser.add_argument("--adaptive_max_batch_size", type=int, default=None)
    parser.add_argument("--adaptive_memory_budget", type=int, default=None)
    parser.add_argument("--adaptive_target_batch_cost", type=int, default=None)
    parser.add_argument("--adaptive_seed", type=int, default=None)

    parser.add_argument(
        "--matmul_precision",
        type=str,
        choices=["highest", "high", "medium"],
        default="high",
    )
    parser.add_argument("--disable_tf32", action="store_true")
    parser.add_argument(
        "--mp_start_method",
        type=str,
        choices=["fork", "spawn", "forkserver"],
        default="spawn",
        help=(
            "Override multiprocessing start method "
            "(defaults to fork on Linux, spawn elsewhere, or FLAMED_MP_START_METHOD)."
        ),
    )
    cli_args = parser.parse_args()

    _configure_start_method(cli_args.mp_start_method)
    train(cli_args)
