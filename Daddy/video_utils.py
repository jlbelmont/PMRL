"""
Headless video utilities for the slim agent.

Implements MP4 + GIF export plus PNG frame dumps per SLIM_AGENT_VIDEO_SPEC.md.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Iterable, List

import imageio
import numpy as np
import math


def _ensure_dir(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def _to_game_size(frame: np.ndarray, target_hw: tuple[int, int] = (144, 160)) -> np.ndarray:
    """
    Upscale frames with nearest-neighbor to match original Game Boy resolution (144x160).
    Only scales up; keeps aspect ratio by using the max scale factor and crops if slightly over.
    """
    if frame.ndim < 2:
        return frame
    target_h, target_w = target_hw
    h, w = frame.shape[:2]
    if h == target_h and w == target_w:
        return frame
    scale = max(math.ceil(target_h / h), math.ceil(target_w / w), 1)
    up = np.repeat(np.repeat(frame, scale, axis=0), scale, axis=1)
    return up[:target_h, :target_w]  # crop if we overshoot


def save_mp4(frames: Iterable[np.ndarray], path: str | Path, fps: int = 30) -> dict:
    start = time.time()
    path = Path(path)
    _ensure_dir(path)
    frames_list = [_to_game_size(f) for f in frames]
    imageio.mimsave(path, frames_list, fps=fps, codec="libx264", macro_block_size=1)
    duration = time.time() - start
    return {
        "path": str(path),
        "frames": len(frames_list),
        "fps": fps,
        "seconds": duration,
        "type": "mp4",
        "bytes": Path(path).stat().st_size,
    }


def save_gif(frames: Iterable[np.ndarray], path: str | Path, fps: int = 15) -> dict:
    start = time.time()
    path = Path(path)
    _ensure_dir(path)
    frames_list = [_to_game_size(f) for f in frames]
    imageio.mimsave(path, frames_list, fps=fps)
    duration = time.time() - start
    return {
        "path": str(path),
        "frames": len(frames_list),
        "fps": fps,
        "seconds": duration,
        "type": "gif",
        "bytes": Path(path).stat().st_size,
    }


def save_frames_as_png(frames: Iterable[np.ndarray], output_dir: str | Path) -> List[str]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    written = []
    for idx, frame in enumerate(frames):
        fname = output_dir / f"frame_{idx:05d}.png"
        imageio.imwrite(fname, frame)
        written.append(str(fname))
    return written


def maybe_save_video(
    frames: List[np.ndarray],
    video_type: str,
    output_path: str | Path,
    fps: int = 30,
) -> list[dict]:
    """
    Helper to save MP4/GIF/Both depending on CLI flag.
    """
    artifacts: list[dict] = []
    if video_type in {"mp4", "both"}:
        artifacts.append(save_mp4(frames, output_path, fps=fps))
    if video_type in {"gif", "both"}:
        gif_path = Path(output_path).with_suffix(".gif")
        artifacts.append(save_gif(frames, gif_path, fps=max(1, fps // 2)))
    return artifacts
