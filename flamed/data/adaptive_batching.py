from __future__ import annotations

import math
import os
import random
from collections.abc import Iterable, Sequence
from typing import Any

import torch.distributed as dist
from torch.utils.data import Sampler

try:
    from tqdm.auto import tqdm
except ModuleNotFoundError:
    tqdm = None


def _estimate_code_length(
    duration_seconds: float,
    *,
    sampling_rate: float,
    down_factor: float,
) -> int:
    if (
        not math.isfinite(duration_seconds)
        or not math.isfinite(sampling_rate)
        or not math.isfinite(down_factor)
        or duration_seconds <= 0.0
        or sampling_rate <= 0.0
        or down_factor <= 0.0
    ):
        return 1
    return max(1, int(round(duration_seconds * sampling_rate / down_factor)))


def _estimate_text_length(transcript: str) -> int:
    tokens = [token for token in transcript.strip().split() if token]
    return max(1, len(tokens))


def _iter_manifest_entries(dataset: Any):
    iterator_fn = getattr(dataset, "iter_manifest_entries", None)
    if callable(iterator_fn):
        entries_iter = iterator_fn()
        if entries_iter is not None:
            return entries_iter

    entries = getattr(dataset, "entries", None)
    if isinstance(entries, Sequence) and len(entries) > 0:
        return iter(entries)

    return None


def _progress(
    iterable: Iterable[Any],
    *,
    total: int | None,
    desc: str,
    unit: str,
):
    if tqdm is None:
        return iterable
    return tqdm(
        iterable,
        total=total,
        desc=desc,
        unit=unit,
        dynamic_ncols=True,
        leave=False,
    )


def _resolve_distributed_state() -> tuple[int, int]:
    if dist.is_available() and dist.is_initialized():
        rank = int(dist.get_rank())
        world_size = int(dist.get_world_size())
        return rank, max(1, world_size)

    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    world_size = max(1, world_size)
    if rank < 0 or rank >= world_size:
        rank = 0
    return rank, world_size


def estimate_flamed_sample_lengths(
    dataset: Any,
    *,
    sampling_rate: float,
    down_factor: float,
    text_length_weight: float = 0.0,
) -> list[int]:
    """Estimate sequence lengths used by the Flamed adaptive batch sampler."""
    if sampling_rate <= 0.0:
        raise ValueError("sampling_rate must be > 0.")
    if down_factor <= 0.0:
        raise ValueError("down_factor must be > 0.")
    if text_length_weight < 0.0:
        raise ValueError("text_length_weight must be >= 0.")

    entries = _iter_manifest_entries(dataset)
    if entries is not None:
        total = None
        if hasattr(dataset, "__len__"):
            try:
                total = len(dataset)
            except TypeError:
                total = None
        lengths: list[int] = []
        for entry in _progress(
            entries,
            total=total,
            desc="adaptive_batching: estimating lengths",
            unit="sample",
        ):
            duration = float(getattr(entry, "duration", 0.0))
            transcript = str(getattr(entry, "transcript", ""))
            code_length = _estimate_code_length(
                duration_seconds=duration,
                sampling_rate=sampling_rate,
                down_factor=down_factor,
            )
            text_length = _estimate_text_length(transcript)
            combined_length = max(1, int(round(code_length + (text_length_weight * text_length))))
            lengths.append(combined_length)
        if lengths:
            return lengths

    sample_lengths = getattr(dataset, "sample_lengths", None)
    if isinstance(sample_lengths, Sequence) and len(sample_lengths) > 0:
        return [max(1, int(length)) for length in sample_lengths]

    raise ValueError(
        "Adaptive batching requires manifest-backed TextCodesDataset entries "
        "or a non-empty sample_lengths attribute."
    )


class AdaptiveMemoryBatchSampler(Sampler[list[int]]):
    """
    Variable-size batch sampler targeting a memory budget proxy:
        estimated_cost = batch_size * (max_sequence_length_in_batch ** 2)
    """

    def __init__(
        self,
        sample_lengths: Sequence[int],
        *,
        target_batch_cost: int,
        max_batch_size: int,
        shuffle: bool = True,
        drop_last: bool = False,
        seed: int = 0,
    ) -> None:
        if not sample_lengths:
            raise ValueError("sample_lengths must not be empty.")
        if target_batch_cost < 1:
            raise ValueError("target_batch_cost must be >= 1.")
        if max_batch_size < 1:
            raise ValueError("max_batch_size must be >= 1.")
        if drop_last:
            raise ValueError(
                "drop_last=True is not allowed for AdaptiveMemoryBatchSampler because "
                "all training samples must be used."
            )

        self.sample_lengths = [max(1, int(length)) for length in sample_lengths]
        self.target_batch_cost = int(target_batch_cost)
        self.max_batch_size = int(max_batch_size)
        self.shuffle = bool(shuffle)
        self.drop_last = False
        self.seed = int(seed)
        self._epoch = 0

    def _ordered_indices(self) -> list[int]:
        indices = list(range(len(self.sample_lengths)))
        if self.shuffle:
            rng = random.Random(self.seed + self._epoch)
            rng.shuffle(indices)
        self._epoch += 1
        return indices

    def _build_batches(self, ordered_indices: Sequence[int]) -> list[list[int]]:
        batches: list[list[int]] = []
        cursor = 0
        total = len(ordered_indices)
        while cursor < total:
            batch: list[int] = []
            batch_max_len = 0
            while cursor < total and len(batch) < self.max_batch_size:
                sample_idx = int(ordered_indices[cursor])
                sample_len = self.sample_lengths[sample_idx]

                next_batch_size = len(batch) + 1
                next_max_len = max(batch_max_len, sample_len)
                next_cost = next_batch_size * (next_max_len**2)
                if batch and next_cost > self.target_batch_cost:
                    break

                batch.append(sample_idx)
                batch_max_len = next_max_len
                cursor += 1

            if not batch:
                batch = [int(ordered_indices[cursor])]
                cursor += 1

            batches.append(batch)

        return batches

    def __iter__(self):
        ordered = self._ordered_indices()
        batches = self._build_batches(ordered)
        for batch in self._shard_batches_for_current_rank(batches):
            yield batch

    def __len__(self) -> int:
        ordered = list(range(len(self.sample_lengths)))
        total_batches = len(self._build_batches(ordered))
        if total_batches == 0:
            return 0
        rank, world_size = _resolve_distributed_state()
        if world_size == 1:
            return total_batches
        return int(math.ceil(total_batches / float(world_size)))

    def _shard_batches_for_current_rank(self, batches: list[list[int]]) -> list[list[int]]:
        rank, world_size = _resolve_distributed_state()
        if world_size == 1:
            return batches
        if len(batches) == 0:
            return []

        per_rank_batch_count = int(math.ceil(len(batches) / float(world_size)))
        rank_batches = list(batches[rank::world_size])

        if len(rank_batches) < per_rank_batch_count:
            filler_source = rank_batches if len(rank_batches) > 0 else batches
            filler_index = 0
            while len(rank_batches) < per_rank_batch_count:
                rank_batches.append(filler_source[filler_index % len(filler_source)])
                filler_index += 1

        return rank_batches
