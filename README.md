# Video Processing — Canine Nose-Print Recognition

A high-performance Python video processing module designed to extract the sharpest, highest-quality frame(s) from close-up videos of canine nose prints for biometric identification.

---

## Features

- **Streaming Two-Pass Extraction ($O(1)$ RAM Footprint)**: Pass 1 streams and evaluates candidate metrics frame-by-frame without buffering entire uncompressed videos in memory. Pass 2 selectively seeks and extracts winning candidates at original full resolution.
- **Center-Weighted ROI Scoring**: Evaluates sharpness within a central Region of Interest (`--crop-fraction`, default: `0.55`) to prevent high-contrast backgrounds (tiles, rugs, clothing) from skewing selection away from the nose print, while preserving the complete uncropped image on disk.
- **Fast Downscaled Evaluation**: Computes edge variance on a downscaled ROI (default: 640px width) for real-time throughput on 1080p/4K captures.
- **Exposure & Glare Telemetry**: Computes and logs `overexposure_pct` (glare / highlight blowout), `underexposure_mean` (luminance), and `contrast_std` (dynamic range) per candidate for diagnostic review.
- **Top-$K$ Diverse Frame Extraction**: Employs Temporal Non-Maximum Suppression (NMS) with configurable frame gaps (`--min-gap`) to ensure candidate frames are distinct rather than adjacent duplicates.
- **Frame Subsampling**: Configurable step intervals (`--frame-step`) to accelerate processing on high-framerate feeds.
- **Safe Quality Gating**: Returns `UNVALIDATED` by default when no threshold is set, preventing premature `"PASS"` assumptions during Phase 0 prototyping. Evaluates `PASS` / `FAIL_BLURRY` once calibrated with `--min-sharpness`.

---

## Installation

### Prerequisites
- Python 3.8+
- [OpenCV](https://pypi.org/project/opencv-python/)
- [NumPy](https://numpy.org/)

### Setup

```bash
# Clone the repository
git clone https://github.com/sntharun/video-processing.git
cd video-processing

# (Optional) Create and activate a virtual environment
python3 -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
pip install opencv-python numpy
```

---

## Quick Start & CLI Usage

### 1. Standard Single-Frame Extraction
Extract the single sharpest frame from a video:

```bash
python3 video_processor.py input_video.mp4
```

### 2. Extract Top-$K$ Diverse Candidate Frames
Extract the top 3 sharpest distinct frames (separated by at least 10 frames):

```bash
python3 video_processor.py input_video.mp4 --top-k 3 --min-gap 10
```
*Outputs:* `best_frame_1.jpg`, `best_frame_2.jpg`, `best_frame_3.jpg`

### 3. Custom ROI Crop & Frame Subsampling
Evaluate a wider central region (65% width/height) and sample every 2nd frame:

```bash
python3 video_processor.py input_video.mp4 --crop-fraction 0.65 --frame-step 2
```

### 4. Quality Threshold Evaluation
Flag frames that do not meet a calibrated minimum sharpness threshold of `100.0`:

```bash
python3 video_processor.py input_video.mp4 --min-sharpness 100.0
```

### 5. Custom Output Path & Scan Duration
Process the first 2 seconds and output to a custom filename:

```bash
python3 video_processor.py input_video.mp4 -o output/sharp_frame.jpg --max-duration 2.0
```

---

## CLI Options Reference

| Option | Short | Type | Default | Description |
| :--- | :--- | :--- | :--- | :--- |
| `video_path` | — | `str` | *(Required)* | Path to the input video file (`.mp4`, `.mov`, `.avi`, etc.) |
| `--output` | `-o` | `str` | `best_frame.jpg` | Base path/filename for the output saved image(s) |
| `--top-k` | `-k` | `int` | `1` | Number of candidate frames to select and save |
| `--min-gap` | — | `int` | `10` | Minimum temporal separation (frames) between selected top-k picks |
| `--max-duration` | — | `float` | `3.0` | Maximum video duration (in seconds) to scan |
| `--frame-step` | — | `int` | `1` | Subsample step (`1` = every frame, `2` = every 2nd frame) |
| `--crop-fraction` | — | `float` | `0.55` | Central ROI fraction (0.0–1.0) to crop before scoring |
| `--eval-width` | — | `int` | `640` | Resize width for sharpness scoring (`0` to disable resizing) |
| `--min-sharpness`| — | `float` | `0.0` | Minimum sharpness threshold for `PASS` (`0.0` reports `UNVALIDATED`) |

---

## Python API Usage

You can also import and integrate the module directly into your Python workflows:

```python
from video_processor import process_video

result = process_video(
    video_path="IMG_4987.mp4",
    output_path="best_frame.jpg",
    top_k=3,
    min_frame_gap=10,
    max_duration_seconds=3.0,
    frame_step=1,
    eval_width=640,
    crop_fraction=0.55,
    min_sharpness_threshold=100.0,
)

print(f"Quality Status: {result['quality_status']}")
print(f"Best Frame: {result['best_frame']} (Score: {result['sharpness_score']})")
print(f"Exposure Stats — Glare: {result['overexposure_pct']}%, Mean: {result['underexposure_mean']}, Contrast: {result['contrast_std']}")
```

---

## Output JSON Schema

The tool outputs structured JSON to standard output:

### Example: Top-1 Output (`--top-k 1`)
```json
{
  "best_frame": "best_frame.jpg",
  "frame_number": 14,
  "sharpness_score": 123.65,
  "overexposure_pct": 0.45,
  "underexposure_mean": 118.4,
  "contrast_std": 45.2,
  "total_frames": 90,
  "quality_status": "UNVALIDATED",
  "diagnostics": {
    "fps": 30.0,
    "resolution": [1920, 1080],
    "processed_frames": 90,
    "processing_time_ms": 78.42,
    "eval_width": 640,
    "crop_fraction": 0.55
  }
}
```

### Example: Top-$K$ Output (`--top-k 3`)
```json
{
  "best_frame": "best_frame_1.jpg",
  "frame_number": 14,
  "sharpness_score": 123.65,
  "overexposure_pct": 0.45,
  "underexposure_mean": 118.4,
  "contrast_std": 45.2,
  "total_frames": 90,
  "quality_status": "UNVALIDATED",
  "diagnostics": {
    "fps": 30.0,
    "resolution": [1920, 1080],
    "processed_frames": 90,
    "processing_time_ms": 82.15,
    "eval_width": 640,
    "crop_fraction": 0.55
  },
  "top_frames": [
    {
      "rank": 1,
      "file": "best_frame_1.jpg",
      "frame_number": 14,
      "sharpness_score": 123.65,
      "overexposure_pct": 0.45,
      "underexposure_mean": 118.4,
      "contrast_std": 45.2
    },
    {
      "rank": 2,
      "file": "best_frame_2.jpg",
      "frame_number": 28,
      "sharpness_score": 115.30,
      "overexposure_pct": 0.12,
      "underexposure_mean": 112.1,
      "contrast_std": 42.8
    },
    {
      "rank": 3,
      "file": "best_frame_3.jpg",
      "frame_number": 45,
      "sharpness_score": 109.82,
      "overexposure_pct": 0.85,
      "underexposure_mean": 121.0,
      "contrast_std": 46.1
    }
  ]
}
```

---

## License

Private repository. All rights reserved.
