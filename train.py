import os
import platform
import torch
import argparse
from flamed import Flamed
import lightning.pytorch as pl
from omegaconf import OmegaConf
from lightning.pytorch.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger


def _configure_start_method(preferred_method=None):
    """Ensure multiprocessing start method matches the requested or default policy."""
    import multiprocessing as mp

    env_requested = os.environ.get('FLAMED_MP_START_METHOD')
    preferred = preferred_method or env_requested
    default = 'fork' if platform.system() == 'Linux' else 'spawn'
    target_method = preferred or default

    available = mp.get_all_start_methods()
    if target_method not in available:
        fallback = 'spawn' if 'spawn' in available else default
        target_method = fallback

    current = mp.get_start_method(allow_none=True)
    if current == target_method:
        return

    try:
        mp.set_start_method(target_method, force=True)
    except RuntimeError:
        # Happens when the method was already set elsewhere; safe to ignore.
        pass


def _parse_pipeline(pipeline_str):
    allowed = {'PriorGenerator', 'ProbGenerator'}
    modules = [item.strip() for item in pipeline_str.split(',') if item.strip()]
    if not modules:
        raise ValueError("PIPELINE must include at least one of PriorGenerator or ProbGenerator.")
    invalid = [m for m in modules if m not in allowed]
    if invalid:
        raise ValueError(f"Unsupported pipeline entries: {', '.join(invalid)}. Valid options: {', '.join(sorted(allowed))}.")
    # Preserve order while removing duplicates
    seen = set()
    deduped = []
    for module in modules:
        if module in seen:
            continue
        seen.add(module)
        deduped.append(module)
    return deduped


def train(proj_name, version, exp_root, exp_name, devices, batch_size, epochs, ckpt, pipeline, prior_ckpt):
    
    if not os.path.exists(os.path.join(exp_root, exp_name)):
        os.mkdir(os.path.join(exp_root, exp_name))

    prob_cfg = OmegaConf.load('configs/prob.yaml')
    prior_cfg = OmegaConf.load('configs/prior.yaml')
    codec_cfg = OmegaConf.load('configs/codec.yaml')
    optimizer_cfg = OmegaConf.load('configs/optimizer.yaml')
    data_config = OmegaConf.load('configs/data.yaml')
    accelerator = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    prob_cfg['device'] = accelerator
    prior_cfg['device'] = accelerator
    codec_cfg['device'] = accelerator
    codec_cfg['encoder']['device'] = accelerator
    codec_cfg['decoder']['device'] = accelerator
    optimizer_cfg['device'] = accelerator
    
    optimizer_cfg['epochs'] = epochs
    optimizer_cfg['batch_size'] = batch_size
    data_config['batch_size'] = batch_size

    cfg = OmegaConf.create({
        'prior_generator': prior_cfg,
        'prob_generator': prob_cfg,
        'codec_cfg': codec_cfg,
        'pipeline': pipeline,
    })
    OmegaConf.save(cfg, os.path.join(os.path.join(exp_root, exp_name), 'config.yaml'))

    requires_prior_ckpt = ('ProbGenerator' in pipeline) and ('PriorGenerator' not in pipeline)
    if requires_prior_ckpt and not prior_ckpt:
        raise ValueError("prior_ckpt is required when training ProbGenerator without PriorGenerator.")

    if requires_prior_ckpt:
        model = Flamed.from_pretrained(
            cfg,
            prior_ckpt,
            device=accelerator,
            weights_only=False,
            training_mode=True,
            modules=['prior_generator'],
        )
    else:
        model = Flamed(cfg)

    model.setup_dataset_optimizer(data_config, optimizer_cfg)
    train_data, val_data = model.get_dataset()

    checkpoint_callback = ModelCheckpoint(
        monitor='total_loss_val_epoch',
        filename='ckpt-{epoch:02d}-{total_loss_val_epoch:.2f}',
        save_top_k=10,
        mode='min',
        save_last=True,
    )

    logger = WandbLogger(
        project=proj_name,
        name=exp_name, 
        save_dir=os.path.join(exp_root, exp_name),
        version=version,
        resume="allow"
    )

    trainer = pl.Trainer(
        devices=devices, 
        accelerator=accelerator, 
        max_epochs=epochs,
        enable_checkpointing=True, 
        logger=logger,
        log_every_n_steps=1, 
        check_val_every_n_epoch=1,
        default_root_dir=os.path.join(exp_root, exp_name),
        callbacks=[checkpoint_callback],
    )
    
    trainer.fit(
        model=model,
        ckpt_path=ckpt,
        train_dataloaders=train_data,
        val_dataloaders=val_data,
    )


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--proj_name', type=str, required=True)
    parser.add_argument('--ver', type=str, required=True)
    parser.add_argument('--exp_root', type=str, default=None)
    parser.add_argument('--exp_name', type=str, required=True)
    parser.add_argument('--devices', type=str, default='0')
    parser.add_argument(
        '--pipeline',
        type=str,
        default='PriorGenerator,ProbGenerator',
        help='Comma-separated modules to train: PriorGenerator, ProbGenerator, or both.',
    )
    parser.add_argument(
        '--prior_ckpt',
        type=str,
        default=None,
        help='Checkpoint containing PriorGenerator weights (required when training ProbGenerator alone).',
    )
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--ckpt', type=str, default=None)
    parser.add_argument(
        '--mp_start_method',
        type=str,
        choices=['fork', 'spawn', 'forkserver'],
        default='spawn',
        help="Override the multiprocessing start method (defaults to fork on Linux, spawn elsewhere, or set FLAMED_MP_START_METHOD).",
    )
    args = parser.parse_args()

    mp_start_method = args.mp_start_method
    _configure_start_method(mp_start_method)
    
    proj_name = args.proj_name
    version = args.ver
    exp_root = args.exp_root
    exp_name = args.exp_name
    devices = [int(device) for device in args.devices.split(',')]
    pipeline = _parse_pipeline(args.pipeline)
    prior_ckpt = args.prior_ckpt
    batch_size = args.batch_size
    epochs = args.epochs
    ckpt = args.ckpt
    
    train(proj_name, version, exp_root, exp_name, devices, batch_size, epochs, ckpt, pipeline, prior_ckpt)
