
# Slim Agent Video Specification (MP4 + GIF Support)

## 🎥 Video Output Requirements (MP4 + GIF)

The slim agent **must** support **both MP4 and GIF** export formats to allow for flexible visualization and compatibility across tools, including SLURM cluster environments, academic compute clusters, and local development.

---

## ✔ Supported Formats
- **MP4** (`.mp4`, H.264 codec, cluster-friendly, high quality)
- **GIF** (`.gif`, lightweight, useful for quick previews)
- **PNG Frame Dumps** (optional for debugging)

---

## 🧩 Implementation Requirements

### 1. Implement `Daddy/video_utils.py` with:

#### MP4 Saving

**Function:**
```
save_mp4(frames, path, fps=30)
```

**Requirements:**
- Accepts list of numpy `uint8` arrays shaped `(H, W, 3)`
- Uses `imageio` or `imageio-ffmpeg`
- Codec must be `libx264`
- Must be headless (no GUI)
- Works under SLURM, qlogin, and qlogin-gpu

**Example Implementation Detail:**
```python
import imageio

imageio.mimsave(path, frames, fps=fps, codec='libx264')
```

---

#### GIF Saving

**Function:**
```
save_gif(frames, path, fps=15)
```

**Requirements:**
- Uses `imageio`
- Lightweight and cluster-safe
- No GUI or interactive components

---

#### PNG Frame Dump

**Function:**
```
save_frames_as_png(frames, output_dir)
```

**Requirements:**
- Saves each frame as a numbered PNG
- Useful in cluster jobs without video toolchain

---

## ☁ Cluster Requirements

All video generation must:

- Run fully **headless**
- Work inside:
  - `qlogin-gpu`
  - `qlogin`
  - `sbatch` SLURM jobs
- Save files locally
- Encode on CPU only
- Avoid allocating GPU RAM for encoding
- Produce deterministic output

---

## 🔧 CLI Support (`debug_rollout.py`)

Add command-line options:

```
--video mp4
--video gif
--video both
--video-fps 30
```

**Examples:**
```
python Daddy/debug_rollout.py --video mp4
python Daddy/debug_rollout.py --video gif --video-fps 12
python Daddy/debug_rollout.py --video both
```

---

## 📊 Video Logging Requirements

When a video is saved, log:

- Output file path
- Number of frames
- FPS used
- Total encoding time
- Resulting file size
- Video type (MP4/GIF/Both)

---

## 🧠 Copilot Prompt (Paste Into VS Code)

```
Implement Daddy/video_utils.py with full MP4 + GIF support:

- save_mp4(frames, path, fps=30)
- save_gif(frames, path, fps=15)
- save_frames_as_png(frames, output_dir)

Rules:
- Must work headless on the WashU Academic Compute Cluster
- Must use imageio and libx264 for MP4
- Must accept numpy frames (H,W,3)
- No GUI or display required
- Must integrate with debug_rollout.py via --video options
```

---

## 📦 Download-Ready File

This document defines the complete MP4/GIF requirement for the slim agent and should be saved as:

```
Daddy/SLIM_AGENT_VIDEO_SPEC.md
```
