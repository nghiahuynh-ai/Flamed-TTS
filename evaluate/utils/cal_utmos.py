import glob
import os
from pathlib import Path
from typing import Iterable

import fire
import librosa
import pandas as pd
import torch
from tqdm import tqdm

from models.utmos.lightning_module import BaselineLightningModule
from utils.checkpoints import resolve_checkpoint


DEFAULT_UTMOS_CKPT = os.environ.get("UTMOS_CKPT", "ckpts/epoch=3-step=7459.ckpt")
DEFAULT_CACHE_DIR = Path(os.environ.get("UTMOS_CACHE", Path.home() / ".cache" / "flamed" / "utmos")).expanduser()


def calc_mos(audio_path, model, sr=16000, device="cuda:0"):
    wav, _ = librosa.load(path=audio_path, sr=sr)
    wav = torch.from_numpy(wav).unsqueeze(0).float()
    batch = {
        "wav": wav.to(device),
        "domains": torch.tensor([0]).to(device),
        "judge_id": torch.tensor([288]).to(device),
    }
    with torch.no_grad():
        output = model(batch)
    return output.mean(dim=1).squeeze().detach().cpu().numpy() * 2 + 3


def cal_utmos(
    audio_path,
    output_file,
    sr=16000,
    device="cuda:0",
    ckpt: str | None = None,
    cache_dir: str | Path | None = None,
):
    ckpt_path = resolve_checkpoint(ckpt or DEFAULT_UTMOS_CKPT, cache_dir or DEFAULT_CACHE_DIR)

    results = []
    wav_paths: Iterable[Path] = sorted(Path(audio_path).glob("*.wav"))
    model = BaselineLightningModule.load_from_checkpoint(str(ckpt_path), map_location=device).eval()
    model.to(device)

    for path in tqdm(wav_paths):
        utmos = calc_mos(str(path), model, sr, device)
        results.append([path.name, utmos])

    df = pd.DataFrame(results, columns=["File", "UTMOS"])
    df.to_csv(output_file, index=False)

    utmos_mean = df["UTMOS"].mean()
    print("=" * 100, "\nUTMOS :", utmos_mean)

    return utmos_mean


if __name__ == "__main__":
    fire.Fire(cal_utmos)

    
