from __future__ import annotations

import multiprocessing as mp
import os
import random
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterator, Optional

import numpy as np
import tgt
import torch
from lightning import LightningDataModule
from torch.utils.data import DataLoader

from flamed.data.adaptive_batching import (
    AdaptiveMemoryBatchSampler,
    estimate_flamed_sample_lengths,
)
from flamed.text import text_to_sequence


@dataclass(frozen=True)
class ManifestEntry:
    filename: str
    duration: float
    transcript: str
    style_prompt: str
    textgrid_path: str
    tgt_codes_path: str
    cond_codes_path: str


def _length_quantile(lengths: list[int], quantile: float) -> int:
    sorted_lengths = sorted(max(1, int(length)) for length in lengths)
    index = int(round((len(sorted_lengths) - 1) * quantile))
    index = max(0, min(len(sorted_lengths) - 1, index))
    return sorted_lengths[index]


class FlamedDataset(LightningDataModule):
    def __init__(self, config):
        super().__init__()
        self.save_hyperparameters(logger=False)

        self.name = config["name"]
        self.data_root = config["data_root"]
        self.train_manifest = config["train_manifest"]
        self.valid_manifest = config["valid_manifest"]
        self.sampling_rate = int(config["sampling_rate"])
        self.dur_min = float(config["dur_min"])
        self.dur_max = float(config["dur_max"])
        self.n_words_min = int(config["n_words_min"])
        self.prompt_dur_max = float(config["prompt_dur_max"])
        self.prompt_reduced_factor = float(config["prompt_reduced_factor"])
        self.down_factors = list(config["down_factors"])
        self.vocab_size = int(config["vocab_size"])
        self.batch_size = int(config["batch_size"])
        self.val_batch_size = int(config.get("val_batch_size", self.batch_size))
        self.train_shuffle = bool(config.get("shuffle_train", True))
        self.val_shuffle = bool(config.get("shuffle_val", False))
        self.drop_last_train = bool(config.get("drop_last_train", False))

        self.num_workers = self._resolve_num_workers(config.get("num_workers", "auto"))
        self.pin_memory = self._resolve_pin_memory(config.get("pin_memory", "auto"))
        self.persistent_workers = self._resolve_persistent_workers(
            config.get("persistent_workers", "auto"),
            self.num_workers,
        )
        self.prefetch_factor = self._resolve_prefetch_factor(
            config.get("prefetch_factor"),
            self.num_workers,
        )
        self.mp_context = self._resolve_mp_context(config.get("multiprocessing_context", "auto"))

        self.cleaners = config["cleaners"]
        self.add_blank = bool(config["add_blank"])
        self.seed = config["seed"]
        self.sil_phones = config["sil_phones"]

        adaptive_cfg_raw = config.get("adaptive_batching")
        if adaptive_cfg_raw is None:
            adaptive_cfg_raw = {}
        if not isinstance(adaptive_cfg_raw, Mapping):
            raise ValueError("data.adaptive_batching must be a mapping when provided.")
        self.adaptive_cfg = dict(adaptive_cfg_raw)
        self.adaptive_batching_enabled = bool(self.adaptive_cfg.get("enabled", False))
        self.train_loader_uses_custom_batch_sampler = False

        self.trainset: TextCodesDataset | None = None
        self.validset: TextCodesDataset | None = None

    def setup(self, stage: Optional[str] = None):  # pylint: disable=unused-argument
        if self.trainset is not None and self.validset is not None:
            return

        self.trainset = TextCodesDataset(
            data_root=self.data_root,
            manifest=self.train_manifest,
            cleaners=self.cleaners,
            dur_min=self.dur_min,
            dur_max=self.dur_max,
            n_words_min=self.n_words_min,
            prompt_dur_max=self.prompt_dur_max,
            sampling_rate=self.sampling_rate,
            down_factors=self.down_factors,
            sil_phones=self.sil_phones,
            add_blank=self.add_blank,
            seed=self.seed,
        )
        self.validset = TextCodesDataset(
            data_root=self.data_root,
            manifest=self.valid_manifest,
            cleaners=self.cleaners,
            dur_min=self.dur_min,
            dur_max=self.dur_max,
            n_words_min=self.n_words_min,
            prompt_dur_max=self.prompt_dur_max,
            sampling_rate=self.sampling_rate,
            down_factors=self.down_factors,
            sil_phones=self.sil_phones,
            add_blank=self.add_blank,
            seed=self.seed,
        )

    def train_dataloader(self):
        if self.trainset is None:
            self.setup()
        if self.trainset is None:
            raise RuntimeError("trainset is not initialized.")

        collate = self._build_collate()
        common_kwargs = self._build_common_loader_kwargs()

        if self.adaptive_batching_enabled:
            target_memory_utilization = float(
                self.adaptive_cfg.get("target_memory_utilization", 0.8)
            )
            if target_memory_utilization <= 0.0 or target_memory_utilization > 1.0:
                raise ValueError(
                    "data.adaptive_batching.target_memory_utilization must be in (0, 1]."
                )

            max_batch_size = int(
                self.adaptive_cfg.get("max_batch_size", max(1, self.batch_size * 2))
            )
            if max_batch_size < 1:
                raise ValueError("data.adaptive_batching.max_batch_size must be >= 1.")

            reference_quantile = float(
                self.adaptive_cfg.get("reference_length_quantile", 0.95)
            )
            if reference_quantile <= 0.0 or reference_quantile > 1.0:
                raise ValueError(
                    "data.adaptive_batching.reference_length_quantile must be in (0, 1]."
                )

            text_length_weight = float(self.adaptive_cfg.get("text_length_weight", 0.0))
            sample_lengths = estimate_flamed_sample_lengths(
                self.trainset,
                sampling_rate=float(self.sampling_rate),
                down_factor=float(self.trainset.down_factor),
                text_length_weight=text_length_weight,
            )

            reference_length = _length_quantile(sample_lengths, reference_quantile)
            memory_budget_raw = self.adaptive_cfg.get("memory_budget")
            if memory_budget_raw is None:
                memory_budget = self.batch_size * (reference_length**2)
            else:
                memory_budget = int(memory_budget_raw)
            if memory_budget < 1:
                raise ValueError("data.adaptive_batching.memory_budget must be >= 1.")

            target_batch_cost_raw = self.adaptive_cfg.get("target_batch_cost")
            if target_batch_cost_raw is None:
                target_batch_cost = int(max(1, round(memory_budget * target_memory_utilization)))
            else:
                target_batch_cost = int(target_batch_cost_raw)
            if target_batch_cost < 1:
                raise ValueError("data.adaptive_batching.target_batch_cost must be >= 1.")

            seed_raw = self.adaptive_cfg.get("seed", self.seed)
            if seed_raw is None:
                sampler_seed = int.from_bytes(os.urandom(8), byteorder="big") & 0x7FFF_FFFF
                print(
                    "[dataset.py] data.adaptive_batching.seed is null; "
                    f"generated random sampler seed={sampler_seed}."
                )
            else:
                sampler_seed = int(seed_raw)

            if self.drop_last_train:
                print(
                    "[dataset.py] drop_last_train=true is not supported with adaptive batching; "
                    "overriding to false."
                )
            train_batch_sampler = AdaptiveMemoryBatchSampler(
                sample_lengths=sample_lengths,
                target_batch_cost=target_batch_cost,
                max_batch_size=max_batch_size,
                shuffle=bool(self.adaptive_cfg.get("shuffle", self.train_shuffle)),
                drop_last=False,
                seed=sampler_seed,
            )
            self.train_loader_uses_custom_batch_sampler = True
            print(
                "[dataset.py] Adaptive train batching enabled: "
                f"target_memory_utilization={target_memory_utilization:.2f}, "
                f"target_batch_cost={target_batch_cost}, "
                f"memory_budget={memory_budget}, "
                f"reference_length(q={reference_quantile:.2f})={reference_length}, "
                f"text_length_weight={text_length_weight:.2f}, "
                f"max_batch_size={max_batch_size}, "
                f"seed={sampler_seed}."
            )
            return DataLoader(
                dataset=self.trainset,
                batch_sampler=train_batch_sampler,
                num_workers=self.num_workers,
                pin_memory=self.pin_memory,
                persistent_workers=self.persistent_workers if self.num_workers > 0 else False,
                collate_fn=collate,
                **common_kwargs,
            )

        self.train_loader_uses_custom_batch_sampler = False
        return DataLoader(
            dataset=self.trainset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            shuffle=self.train_shuffle,
            drop_last=self.drop_last_train,
            persistent_workers=self.persistent_workers if self.num_workers > 0 else False,
            collate_fn=collate,
            **common_kwargs,
        )

    def val_dataloader(self):
        if self.validset is None:
            self.setup()
        if self.validset is None:
            raise RuntimeError("validset is not initialized.")

        collate = self._build_collate()
        common_kwargs = self._build_common_loader_kwargs()
        return DataLoader(
            dataset=self.validset,
            batch_size=self.val_batch_size,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            shuffle=self.val_shuffle,
            drop_last=False,
            persistent_workers=self.persistent_workers if self.num_workers > 0 else False,
            collate_fn=collate,
            **common_kwargs,
        )

    def _build_collate(self) -> "TextCodesBatchCollate":
        prompt_max_len = int(self.prompt_dur_max * self.sampling_rate // np.prod(self.down_factors))
        return TextCodesBatchCollate(
            prompt_max_len=prompt_max_len,
            prompt_reduced_factor=self.prompt_reduced_factor,
            vocab_size=self.vocab_size,
        )

    def _build_common_loader_kwargs(self) -> dict[str, Any]:
        loader_kwargs: dict[str, Any] = {}
        if self.num_workers > 0 and self.prefetch_factor is not None:
            loader_kwargs["prefetch_factor"] = self.prefetch_factor
        if self.num_workers > 0 and self.mp_context is not None:
            loader_kwargs["multiprocessing_context"] = self.mp_context
        return loader_kwargs

    def teardown(self, stage: Optional[str] = None):  # pylint: disable=unused-argument
        return None

    def state_dict(self):
        return {}

    def load_state_dict(self, state_dict: Dict[str, Any]):  # pylint: disable=unused-argument
        return None

    @staticmethod
    def _resolve_num_workers(value):
        if value is None:
            value = "auto"
        if isinstance(value, str):
            if value.lower() == "auto":
                cpu_count = os.cpu_count() or 1
                if cpu_count <= 2:
                    return 1
                return max(2, min(8, cpu_count // 2))
            raise ValueError(f"Unsupported num_workers value: {value}")
        return max(0, int(value))

    @staticmethod
    def _resolve_pin_memory(value):
        if value is None:
            value = "auto"
        if isinstance(value, str):
            if value.lower() == "auto":
                return torch.cuda.is_available()
            raise ValueError(f"Unsupported pin_memory value: {value}")
        return bool(value)

    @staticmethod
    def _resolve_persistent_workers(value, num_workers: int):
        if num_workers <= 0:
            return False
        if value is None:
            value = "auto"
        if isinstance(value, str):
            if value.lower() == "auto":
                return True
            raise ValueError(f"Unsupported persistent_workers value: {value}")
        return bool(value)

    @staticmethod
    def _resolve_prefetch_factor(value, num_workers: int):
        if num_workers <= 0:
            return None
        if value is None:
            return None
        return max(1, int(value))

    @staticmethod
    def _resolve_mp_context(value):
        if value is None:
            value = "auto"
        ctx_name = value
        if isinstance(value, str):
            if value.lower() == "auto":
                ctx_name = mp.get_start_method(allow_none=True)
            else:
                ctx_name = value.lower()
        if ctx_name in (None, "none"):
            return None
        try:
            return mp.get_context(ctx_name)
        except ValueError as err:
            raise ValueError(f"Unsupported multiprocessing context: {ctx_name}") from err


class TextCodesDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        data_root,
        manifest,
        cleaners,
        dur_min=0.3,
        dur_max=15.0,
        n_words_min=3,
        prompt_dur_max=3.0,
        sampling_rate=16000,
        down_factors=None,
        sil_phones=None,
        add_blank=True,
        seed=None,
    ):
        self.data_root = str(data_root)
        self.manifest = str(manifest)
        self.cleaners = cleaners
        self.dur_min = float(dur_min)
        self.dur_max = float(dur_max)
        self.n_words_min = int(n_words_min)
        self.prompt_dur_max = float(prompt_dur_max)
        self.sampling_rate = int(sampling_rate)
        self.add_blank = bool(add_blank)

        if down_factors is None:
            self.down_factors = [2, 4, 5, 5]
        else:
            self.down_factors = list(down_factors)
        self.down_factor = int(np.prod(self.down_factors))

        if sil_phones is None:
            self.sil_phones = ["sil", "sp", "spn", ""]
        else:
            self.sil_phones = list(sil_phones)

        manifest_path = Path(self.manifest)
        if not manifest_path.is_absolute():
            manifest_path = Path(self.data_root) / self.manifest
        if not manifest_path.exists():
            raise FileNotFoundError(f"Manifest file not found: {manifest_path}")

        entries: list[ManifestEntry] = []
        filtered_count = 0
        duration_total_seconds = 0.0

        with manifest_path.open("r", encoding="utf-8") as manifest_file:
            for line_number, line in enumerate(manifest_file, start=1):
                raw_line = line.rstrip("\n")
                if raw_line == "":
                    continue
                entry = self._parse_manifest_entry(
                    raw_line,
                    line_number=line_number,
                    manifest_path=manifest_path,
                )
                if not self._keep_entry(entry):
                    filtered_count += 1
                    continue
                entries.append(entry)
                duration_total_seconds += entry.duration

        rng = random.Random(seed)
        rng.shuffle(entries)
        self.entries = entries
        self.sample_lengths = [
            max(1, int(round(entry.duration * self.sampling_rate / self.down_factor)))
            for entry in self.entries
        ]

        duration_hours = round(duration_total_seconds / 3600.0, 3)
        print("+-" * 50)
        print(f">>>\t {self.manifest}: {duration_hours} hours")
        print(f">>>\t Valid utterances: {len(self.entries)}")
        print(f">>>\t Filtered utterances: {filtered_count}")
        print("+-" * 50)

    def _parse_manifest_entry(
        self,
        raw_line: str,
        *,
        line_number: int,
        manifest_path: Path,
    ) -> ManifestEntry:
        parts = raw_line.split("|")
        if len(parts) != 7:
            raise ValueError(
                f"Invalid manifest format at {manifest_path}:{line_number}. "
                f"Expected 7 fields, got {len(parts)}."
            )
        (
            filename,
            duration_raw,
            transcript,
            style_prompt,
            textgrid_path,
            tgt_codes_path,
            cond_codes_path,
        ) = parts
        try:
            duration = float(duration_raw)
        except ValueError as err:
            raise ValueError(
                f"Invalid duration {duration_raw!r} at {manifest_path}:{line_number}."
            ) from err
        return ManifestEntry(
            filename=filename,
            duration=duration,
            transcript=transcript,
            style_prompt=style_prompt,
            textgrid_path=self._resolve_data_path(textgrid_path),
            tgt_codes_path=self._resolve_data_path(tgt_codes_path),
            cond_codes_path=self._resolve_data_path(cond_codes_path),
        )

    def _resolve_data_path(self, path_value: str) -> str:
        path = Path(path_value)
        if path.is_absolute():
            return str(path)
        return str(Path(self.data_root) / path)

    def _keep_entry(self, entry: ManifestEntry) -> bool:
        if entry.duration < self.dur_min or entry.duration > self.dur_max:
            return False
        word_count = len([token for token in entry.transcript.split(" ") if token])
        if word_count < self.n_words_min:
            return False
        return True

    def iter_manifest_entries(self) -> Iterator[ManifestEntry]:
        return iter(self.entries)

    def get_datapoint(self, entry: ManifestEntry):
        textgrid = tgt.io.read_textgrid(entry.textgrid_path, include_empty_intervals=True)
        phones_tier = textgrid.get_tier_by_name("phones")

        spk, codes, embs = self._load_target_arrays(entry)

        phones, phone_durations, sil_durations = self.get_alignment(phones_tier)
        phone_durations_tensor = torch.as_tensor(phone_durations, dtype=torch.long)
        sil_durations_tensor = torch.as_tensor(sil_durations, dtype=torch.long)
        phonemes_tensor = torch.as_tensor(
            text_to_sequence("{" + " ".join(phones) + "}", self.cleaners),
            dtype=torch.long,
        )

        return {
            "phoneme": phonemes_tensor,
            "code": codes,
            "emb": embs,
            "spk": spk,
            "phone_dur": phone_durations_tensor,
            "sil_dur": sil_durations_tensor,
        }

    def _load_target_arrays(
        self,
        entry: ManifestEntry,
    ) -> tuple[torch.FloatTensor, torch.LongTensor, torch.FloatTensor]:
        target_path = Path(entry.tgt_codes_path)
        payload = self._load_npy_payload(target_path)
        payload_map = self._extract_mapping_from_payload(payload, target_path)

        raw_codes = self._extract_codes_from_payload(payload_map, target_path)
        raw_embs = self._extract_embeddings_from_payload(payload_map, target_path)
        raw_spk = self._extract_speaker_from_payload(payload_map)
        if raw_spk is None:
            raw_spk = self._load_speaker_from_condition_path(Path(entry.cond_codes_path))
        if raw_spk is None:
            raise ValueError(
                f"Missing speaker embedding in {target_path}. "
                "Expected one of: spk, spkemb, speaker, timbre."
            )

        codes = self._to_codes_tensor(raw_codes, target_path)
        embs = self._to_embeddings_tensor(raw_embs, target_path, target_length=int(codes.shape[1]))
        spk = self._to_speaker_tensor(raw_spk, target_path)
        return spk, codes, embs

    @staticmethod
    def _load_npy_payload(npy_path: Path) -> Any:
        if npy_path.suffix.lower() != ".npy":
            raise ValueError(f"Expected a .npy file, got: {npy_path}")
        try:
            try:
                return np.load(npy_path, allow_pickle=False)
            except ValueError as err:
                if "allow_pickle=False" not in str(err):
                    raise
                return np.load(npy_path, allow_pickle=True)
        except Exception as err:  # noqa: BLE001
            raise RuntimeError(f"Failed to load npy payload from {npy_path}: {err}") from err

    @staticmethod
    def _unwrap_object_scalar(value: Any) -> Any:
        if isinstance(value, np.ndarray) and value.dtype == object and value.shape == ():
            return value.item()
        return value

    @classmethod
    def _extract_mapping_from_payload(cls, payload: Any, npy_path: Path) -> Mapping[str, Any]:
        if isinstance(payload, np.ndarray) and payload.dtype.names is not None:
            field_names = payload.dtype.names
            if field_names is None:
                raise ValueError(f"Structured payload has no fields in {npy_path}.")
            if payload.shape == ():
                return {name: cls._unwrap_object_scalar(payload[name]) for name in field_names}
            if payload.shape[0] == 1:
                return {
                    name: cls._unwrap_object_scalar(payload[name][0])
                    for name in field_names
                }
            return {name: cls._unwrap_object_scalar(payload[name]) for name in field_names}

        if isinstance(payload, np.ndarray) and payload.dtype == object and payload.shape == ():
            payload = payload.item()

        if isinstance(payload, Mapping):
            return payload

        if isinstance(payload, tuple) and len(payload) == 3:
            return {
                "spk": payload[0],
                "discrete": payload[1],
                "continuous": payload[2],
            }

        raise ValueError(
            f"Unsupported npy payload format in {npy_path}. "
            "Expected a mapping-like object with modal arrays."
        )

    @staticmethod
    def _extract_codes_from_payload(payload_map: Mapping[str, Any], npy_path: Path) -> Any:
        quantizers = payload_map.get("quantizers")
        if quantizers is not None:
            try:
                quantizer_count = len(quantizers)
            except TypeError as err:
                raise ValueError(f"`quantizers` must be indexable in {npy_path}.") from err
            if quantizer_count <= 1:
                raise ValueError(
                    f"Expected at least two quantizer sequences, got {quantizer_count} in {npy_path}."
                )
            return quantizers[1]

        for key in ("codes", "code", "discrete"):
            value = payload_map.get(key)
            if value is not None:
                return value
        raise ValueError(
            f"Missing codes in {npy_path}. Expected one of keys: quantizers/codes/code/discrete."
        )

    @staticmethod
    def _extract_embeddings_from_payload(payload_map: Mapping[str, Any], npy_path: Path) -> Any:
        for key in ("vqemb", "emb", "embs", "continuous"):
            value = payload_map.get(key)
            if value is not None:
                return value
        raise ValueError(
            f"Missing embedding matrix in {npy_path}. Expected one of keys: vqemb/emb/embs/continuous."
        )

    @staticmethod
    def _extract_speaker_from_payload(payload_map: Mapping[str, Any]) -> Any | None:
        for key in ("spk", "spkemb", "speaker", "timbre"):
            value = payload_map.get(key)
            if value is not None:
                return value
        return None

    def _load_speaker_from_condition_path(self, cond_path: Path) -> Any | None:
        if not cond_path.is_file() or cond_path.suffix.lower() != ".npy":
            return None
        payload = self._load_npy_payload(cond_path)
        payload_map = self._extract_mapping_from_payload(payload, cond_path)
        return self._extract_speaker_from_payload(payload_map)

    @staticmethod
    def _to_codes_tensor(raw_codes: Any, npy_path: Path) -> torch.LongTensor:
        codes = torch.as_tensor(np.asarray(raw_codes), dtype=torch.long)
        if codes.dim() == 1:
            return codes.unsqueeze(0)
        if codes.dim() != 2:
            raise ValueError(
                f"Codes tensor must be 1D or 2D in {npy_path}, got shape {tuple(codes.shape)}."
            )
        # Heuristic: keep [Q, T]; transpose when payload is [T, Q].
        if codes.shape[0] > codes.shape[1] and codes.shape[1] <= 8:
            return codes.transpose(0, 1).contiguous()
        return codes

    @staticmethod
    def _to_embeddings_tensor(
        raw_embs: Any,
        npy_path: Path,
        *,
        target_length: int,
    ) -> torch.FloatTensor:
        embs = torch.as_tensor(np.asarray(raw_embs), dtype=torch.float32)
        if embs.dim() == 1:
            embs = embs.unsqueeze(1)
        if embs.dim() == 3 and embs.shape[0] == 1:
            embs = embs.squeeze(0)
        if embs.dim() != 2:
            raise ValueError(
                f"Embeddings tensor must be 2D in {npy_path}, got shape {tuple(embs.shape)}."
            )

        if embs.shape[0] != target_length and embs.shape[1] == target_length:
            embs = embs.transpose(0, 1).contiguous()
        if embs.shape[0] != target_length:
            raise ValueError(
                f"Temporal length mismatch in {npy_path}: "
                f"codes length={target_length}, embs shape={tuple(embs.shape)}."
            )
        return embs

    @staticmethod
    def _to_speaker_tensor(raw_speaker: Any, npy_path: Path) -> torch.FloatTensor:
        speaker = torch.as_tensor(np.asarray(raw_speaker), dtype=torch.float32).reshape(-1)
        if speaker.numel() == 0:
            raise ValueError(f"Speaker embedding is empty in {npy_path}.")
        return speaker

    def get_alignment(self, textgrid_tier):
        pre_phones = ["bos"]
        pre_durations = [0]
        for interval in textgrid_tier._objects:
            start_time = interval.start_time
            end_time = interval.end_time
            phone = interval.text or "sp"
            start_code = start_time * self.sampling_rate // self.down_factor
            end_code = end_time * self.sampling_rate // self.down_factor
            pre_phones.append(phone)
            pre_durations.append(end_code - start_code)

        phones: list[str] = []
        phone_durations: list[int] = []
        sil_durations: list[int] = []
        for idx, phone in enumerate(pre_phones):
            if phone in self.sil_phones:
                continue

            phones.append(phone)
            phone_durations.append(int(pre_durations[idx]))
            if idx == len(pre_phones) - 1:
                sil_durations.append(0)
            elif pre_phones[idx + 1] in self.sil_phones:
                sil_durations.append(int(pre_durations[idx + 1]))
            else:
                sil_durations.append(0)

        if not phones:
            raise ValueError("No non-silence phones were extracted from textgrid tier.")
        phones[0] = "sp"
        return phones, phone_durations, sil_durations

    def __getitem__(self, index):
        return self.get_datapoint(self.entries[index])

    def __len__(self):
        return len(self.entries)


class TextCodesBatchCollate:
    def __init__(
        self,
        prompt_max_len=800,
        prompt_reduced_factor=0.8,
        vocab_size=1024,
    ):
        self.vocab_size = int(vocab_size)
        self.prompt_max_len = int(prompt_max_len)
        self.prompt_reduced_factor = float(prompt_reduced_factor)

    def _process_acoustic_prompt(self, prompts):
        max_len = max(1, min([prompt.size(1) for prompt in prompts] + [self.prompt_max_len]))
        max_len_reduced = max(1, int(round(self.prompt_reduced_factor * max_len)))
        max_len_reduced = min(max_len_reduced, max_len)

        prompt_segments = []
        for prompt in prompts:
            start_max = max(0, prompt.size(1) - max_len_reduced)
            start_idx = random.randint(0, start_max)
            end_idx = start_idx + max_len_reduced
            prompt_segments.append(prompt[:, start_idx:end_idx])

        prompts_tensor = torch.stack(prompt_segments, dim=0)
        prompts_tensor[:, 1:3, :] = self.vocab_size
        return prompts_tensor

    def __call__(self, batch):
        if len(batch) == 0:
            raise ValueError("Empty batch is not supported.")

        batch_size = len(batch)
        x_max_len = max(item["phoneme"].shape[-1] for item in batch)
        y_max_len = max(item["code"].shape[-1] for item in batch)
        n_codes = batch[0]["code"].shape[-2]
        emb_dim = batch[0]["emb"].shape[-1]

        phonemes = torch.zeros((batch_size, x_max_len), dtype=torch.long)
        codes = torch.full(
            (batch_size, n_codes, y_max_len),
            fill_value=self.vocab_size,
            dtype=torch.long,
        )
        embs = torch.zeros((batch_size, y_max_len, emb_dim), dtype=torch.float)
        phone_durations = torch.zeros((batch_size, x_max_len), dtype=torch.long)
        sil_durations = torch.zeros((batch_size, x_max_len), dtype=torch.long)

        prompts = []
        spks = []
        x_lens = []
        y_lens = []
        for index, item in enumerate(batch):
            p_i = item["phoneme"]
            c_i = item["code"]
            e_i = item["emb"]
            s_i = item["spk"]
            pd_i = item["phone_dur"]
            sd_i = item["sil_dur"]

            phonemes[index, : p_i.shape[-1]] = p_i
            codes[index, :, : c_i.shape[-1]] = c_i
            embs[index, : e_i.shape[0], :] = e_i
            phone_durations[index, : pd_i.shape[-1]] = pd_i
            sil_durations[index, : sd_i.shape[-1]] = sd_i

            prompts.append(c_i)
            spks.append(s_i)
            x_lens.append(int(p_i.shape[-1]))
            y_lens.append(int(c_i.shape[-1]))

        spks_tensor = torch.stack(spks, dim=0)
        x_len = torch.tensor(x_lens, dtype=torch.long)
        y_len = torch.tensor(y_lens, dtype=torch.long)
        prompts_tensor = self._process_acoustic_prompt(prompts)

        return (
            phonemes,
            x_len,
            codes,
            y_len,
            phone_durations,
            sil_durations,
            embs,
            prompts_tensor,
            spks_tensor,
        )
