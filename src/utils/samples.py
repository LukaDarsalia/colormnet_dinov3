from __future__ import annotations

from pathlib import Path
from typing import List, Tuple, Union

from PIL import Image


IMAGE_EXTS = {".jpg", ".jpeg", ".png"}


def _list_video_dirs(root: Union[str, Path]) -> List[Path]:
    root = Path(root)
    if not root.exists():
        return []
    return [p for p in sorted(root.iterdir()) if p.is_dir() and not p.name.startswith('.')]


def _list_frames(video_dir: Path) -> List[Path]:
    frames = []
    for p in sorted(video_dir.iterdir()):
        if p.is_file() and p.suffix.lower() in IMAGE_EXTS and not p.name.startswith('.'):
            frames.append(p)
    return frames


def sample_images(root: Union[str, Path], max_videos: int = 2, max_frames: int = 3) -> List[Tuple[Image.Image, str]]:
    samples: List[Tuple[Image.Image, str]] = []
    for video_dir in _list_video_dirs(root)[:max_videos]:
        for frame_path in _list_frames(video_dir)[:max_frames]:
            try:
                img = Image.open(frame_path).convert("RGB")
            except Exception:
                continue
            caption = f"{video_dir.name}/{frame_path.name}"
            samples.append((img, caption))
    return samples
