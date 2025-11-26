from __future__ import annotations

from pathlib import Path
from urllib.parse import urlparse
from urllib.request import urlretrieve


def resolve_checkpoint(path_or_url: str, cache_dir: str | Path | None = None) -> Path:
    """
    Resolve a checkpoint path. Supports local paths and http/https URLs.
    When a URL is provided, the file is downloaded once into cache_dir.
    """
    if not path_or_url:
        raise ValueError("Checkpoint path or URL must be provided.")

    parsed = urlparse(path_or_url)
    if parsed.scheme in {"http", "https"}:
        cache_root = Path(cache_dir or Path.home() / ".cache" / "flamed" / "checkpoints")
        cache_root.mkdir(parents=True, exist_ok=True)
        filename = Path(parsed.path).name or "checkpoint.ckpt"
        target_path = cache_root / filename
        if not target_path.exists():
            urlretrieve(path_or_url, target_path)
        return target_path

    return Path(path_or_url).expanduser().resolve(strict=False)
