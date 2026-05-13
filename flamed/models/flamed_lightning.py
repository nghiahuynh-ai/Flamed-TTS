from __future__ import annotations

from abc import ABC
from typing import Any

import torch
import wandb
from lightning import LightningModule
from lightning.pytorch.utilities import rank_zero_only
from omegaconf import DictConfig
from transformers import get_cosine_schedule_with_warmup

from flamed.data import FlamedDataset
from flamed.models.facodec import FACodecDecoder, FACodecEncoder


class FlamedLightning(LightningModule, ABC):
    def __init__(self):
        super().__init__()
        self._last_logged_val_epoch = -1
        self._val_codec_encoder: FACodecEncoder | None = None
        self._val_codec_decoder: FACodecDecoder | None = None

    def setup_dataset_optimizer(
        self,
        dataset_cfg: DictConfig,
        optimizer_cfg: DictConfig,
    ):
        self.dataset_cfg = dataset_cfg
        self.optimizer_cfg = optimizer_cfg
        self.dataset = FlamedDataset(dataset_cfg)

        params = [parameter for parameter in self.parameters() if parameter.requires_grad]
        if not params:
            raise ValueError(
                f"No trainable parameters found for pipeline {getattr(self, 'pipeline', None)}."
            )

        self.optimizer = torch.optim.AdamW(
            params,
            lr=optimizer_cfg["lr"],
            betas=optimizer_cfg["betas"],
            eps=optimizer_cfg["eps"],
            weight_decay=optimizer_cfg["weight_decay"],
        )

        max_steps = int(optimizer_cfg["max_steps"])
        if max_steps < 1:
            raise ValueError("optimizer.max_steps must be >= 1.")
        self.scheduler = get_cosine_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps=int(optimizer_cfg["warmup_steps"]),
            num_training_steps=max_steps,
        )

    def configure_optimizers(self):
        return {
            "optimizer": self.optimizer,
            "lr_scheduler": {
                "scheduler": self.scheduler,
                "interval": "step",
                "frequency": 1,
                "monitor": "total_loss_val",
            },
        }

    def get_dataset(self):
        self.dataset.setup()
        train_data = self.dataset.train_dataloader()
        val_data = self.dataset.val_dataloader()
        return train_data, val_data

    def training_step(self, batch, batch_idx):  # pylint: disable=unused-argument
        (
            phonemes,
            x_len,
            codes,
            y_len,
            phone_durations,
            sil_durations,
            embs,
            prompts,
            spks,
        ) = batch
        losses = self(
            phonemes,
            x_len,
            codes,
            y_len,
            phone_durations,
            sil_durations,
            embs,
            prompts,
            spks,
            training=True,
        )

        total_loss = None
        logging_data = {}
        for key, value in losses.items():
            if "_loss" in key:
                total_loss = value if total_loss is None else (total_loss + value)
                logging_data[f"{key}_train"] = value
            else:
                logging_data[key] = value
        if total_loss is None:
            raise RuntimeError("No *_loss entries were returned from forward().")

        logging_data["total_loss_train"] = total_loss
        logging_data["lr"] = float(self.scheduler.optimizer.param_groups[0]["lr"])
        logging_data["step"] = float(self.global_step)
        self._logging(logging_data, batch_size=int(phonemes.size(0)))
        return total_loss

    def validation_step(self, batch, batch_idx):  # pylint: disable=unused-argument
        (
            phonemes,
            x_len,
            codes,
            y_len,
            phone_durations,
            sil_durations,
            embs,
            prompts,
            spks,
        ) = batch
        losses = self(
            phonemes,
            x_len,
            codes,
            y_len,
            phone_durations,
            sil_durations,
            embs,
            prompts,
            spks,
            training=False,
        )

        total_loss = None
        logging_data = {}
        for key, value in losses.items():
            if "_loss" in key:
                total_loss = value if total_loss is None else (total_loss + value)
                logging_data[f"{key}_val"] = value
        if total_loss is None:
            raise RuntimeError("No *_loss entries were returned from forward().")

        logging_data["total_loss_val"] = total_loss
        logging_data["step"] = float(self.global_step)
        self._logging(logging_data, batch_size=int(phonemes.size(0)))
        return total_loss

    def test_step(self, batch, batch_idx):  # pylint: disable=unused-argument
        return None

    def _logging(self, logs: dict[str, Any], *, batch_size: int):
        sync_dist = self._should_sync_dist()
        for key, value in logs.items():
            self.log(
                name=key,
                value=value,
                on_step=True,
                on_epoch=True,
                logger=True,
                batch_size=batch_size,
                sync_dist=sync_dist,
            )

    def _should_sync_dist(self) -> bool:
        trainer = getattr(self, "trainer", None)
        if trainer is None:
            return False
        return bool(getattr(trainer, "num_devices", 1) > 1)

    @rank_zero_only
    def on_validation_batch_end(self, outputs, batch, batch_idx):  # pylint: disable=unused-argument
        if self.trainer and self.trainer.sanity_checking:
            return
        if self._last_logged_val_epoch == self.current_epoch:
            return
        if not getattr(self, "prob_generator", None):
            return
        if not bool(self.optimizer_cfg.get("log_val_audio", True)):
            return

        self._last_logged_val_epoch = self.current_epoch
        phonemes, x_len, _, y_len, _, _, embs, prompts, spks = batch

        with torch.inference_mode():
            codec_encoder, codec_decoder = self._get_codec_models_for_val()
            results = self.sample(
                prompt_processed=prompts[0],
                phonemes=phonemes[0, : x_len[0].item()],
                timbre=spks[0],
                codec_encoder=codec_encoder,
                codec_decoder=codec_decoder,
            )
            wav = results["wav"]
            gt_wav = codec_decoder.inference(
                embs[0, : y_len[0].item(), :].unsqueeze(0).permute(0, 2, 1),
                spks[0].unsqueeze(0),
            )

        logger_experiment = getattr(self.logger, "experiment", None)
        if logger_experiment is None or not hasattr(logger_experiment, "log"):
            return

        logger_experiment.log(
            {
                "synthesize/val_synth": wandb.Audio(
                    wav,
                    sample_rate=self.cfg["codec_cfg"]["sr"],
                )
            },
            step=int(self.global_step),
        )
        logger_experiment.log(
            {
                "synthesize/val_gt": wandb.Audio(
                    gt_wav[0][0].detach().cpu().numpy(),
                    sample_rate=self.cfg["codec_cfg"]["sr"],
                )
            },
            step=int(self.global_step),
        )

    def _get_codec_models_for_val(self) -> tuple[FACodecEncoder, FACodecDecoder]:
        if self._val_codec_encoder is None or self._val_codec_decoder is None:
            self._val_codec_encoder = FACodecEncoder.from_pretrained(
                self.cfg["codec_cfg"]["encoder"]
            ).eval()
            self._val_codec_decoder = FACodecDecoder.from_pretrained(
                self.cfg["codec_cfg"]["decoder"]
            ).eval()
            self._val_codec_encoder.requires_grad_(False)
            self._val_codec_decoder.requires_grad_(False)

        device = self.device
        self._val_codec_encoder.to(device)
        self._val_codec_decoder.to(device)
        return self._val_codec_encoder, self._val_codec_decoder

    def on_fit_end(self):
        self._release_codec_models()

    def _release_codec_models(self):
        if self._val_codec_encoder is not None:
            self._val_codec_encoder.cpu()
            self._val_codec_encoder = None
        if self._val_codec_decoder is not None:
            self._val_codec_decoder.cpu()
            self._val_codec_decoder = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
