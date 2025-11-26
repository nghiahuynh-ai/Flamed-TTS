#!/usr/bin/env python3
import argparse
from collections import OrderedDict
from collections.abc import Mapping
from typing import List

import torch


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Average a list of PyTorch checkpoint weight files into a single checkpoint."
    )
    parser.add_argument(
        "--ckpts",
        nargs="+",
        required=True,
        help="Whitespace separated list of checkpoint paths to average.",
    )
    parser.add_argument(
        "--output",
        "-o",
        required=True,
        help="Path to write the averaged checkpoint.",
    )
    return parser.parse_args()


def extract_state_dict(obj: Mapping, ckpt_path: str) -> OrderedDict:
    """Return a mutable copy of the checkpoint's state dict."""
    if "state_dict" in obj and isinstance(obj["state_dict"], Mapping):
        return OrderedDict(obj["state_dict"])

    if all(isinstance(v, torch.Tensor) for v in obj.values()):
        return OrderedDict(obj)

    raise ValueError(
        f"Checkpoint '{ckpt_path}' is not a pure weight dictionary or Lightning-style checkpoint."
    )


def load_state_dict(path: str) -> OrderedDict:
    data = torch.load(path, weights_only=False, map_location="cpu")
    if not isinstance(data, Mapping):
        raise ValueError(f"Checkpoint '{path}' must be a mapping of tensors.")
    return extract_state_dict(data, path)


def ensure_compatible(state_dicts: List[OrderedDict]):
    base_keys = list(state_dicts[0].keys())
    base_key_set = set(base_keys)
    for idx, sd in enumerate(state_dicts[1:], start=2):
        sd_keys = set(sd.keys())
        missing = base_key_set - sd_keys
        extra = sd_keys - base_key_set
        if missing or extra:
            raise ValueError(
                f"Checkpoint #{idx} has mismatched parameters. Missing: {sorted(missing)}; Extra: {sorted(extra)}"
            )
        for key in base_keys:
            ref = state_dicts[0][key]
            cur = sd[key]
            if ref.shape != cur.shape:
                raise ValueError(f"Parameter '{key}' has mismatched shapes: {ref.shape} vs {cur.shape} (ckpt #{idx}).")
            if ref.dtype != cur.dtype:
                raise ValueError(f"Parameter '{key}' has mismatched dtypes: {ref.dtype} vs {cur.dtype} (ckpt #{idx}).")


def average_state_dicts(state_dicts: List[OrderedDict]) -> OrderedDict:
    ensure_compatible(state_dicts)
    num = len(state_dicts)
    averaged = OrderedDict()

    for key in state_dicts[0].keys():
        tensors = [sd[key] for sd in state_dicts]
        first = tensors[0]

        if torch.is_floating_point(first):
            acc = first.to(torch.float64).clone()
            for tensor in tensors[1:]:
                acc += tensor.to(torch.float64)
            acc /= num
            averaged[key] = acc.to(first.dtype)
        else:
            if not all(torch.equal(first, tensor) for tensor in tensors[1:]):
                raise ValueError(
                    f"Non-floating parameter '{key}' differs across checkpoints; cannot average safely."
                )
            averaged[key] = first.clone()

    return averaged


def main():
    args = parse_args()
    state_dicts = [load_state_dict(path) for path in args.ckpts]
    averaged = average_state_dicts(state_dicts)
    torch.save(averaged, args.output)
    print(f"Averaged {len(state_dicts)} checkpoints into '{args.output}'.")


if __name__ == "__main__":
    main()
