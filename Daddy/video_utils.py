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


def _pad_to_block(frame: np.ndarray, block: int = 16) -> np.ndarray:
    """
    Pad a frame so H and W are divisible by block (macroblock-friendly for ffmpeg).
    Uses edge padding to avoid introducing new colors.
    """
    if frame.ndim < 2:
        return frame
    h, w = frame.shape[:2]
    pad_h = (block - h % block) % block
    pad_w = (block - w % block) % block
    if pad_h == 0 and pad_w == 0:
        return frame
    pad_spec = ((0, pad_h), (0, pad_w))
    if frame.ndim == 3:
        pad_spec += ((0, 0),)
    return np.pad(frame, pad_spec, mode="edge")


def _ensure_dir(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def save_mp4(frames: Iterable[np.ndarray], path: str | Path, fps: int = 30) -> dict:
    start = time.time()
    path = Path(path)
    _ensure_dir(path)
    frames_list = [_pad_to_block(f) for f in frames]
    imageio.mimsave(path, frames_list, fps=fps, codec="libx264")
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
    frames_list = [_pad_to_block(f, block=16) for f in frames]
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
