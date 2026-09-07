# Video Processing — Canine Nose-Print Recognition

A high-performance Python video processing module designed to extract the sharpest, highest-quality frame(s) from close-up videos of canine nose prints for biometric identification.

---

## Features

- **Streaming Evaluation ($O(K)$ Memory footprint)**: Evaluates sharpness frame-by-frame without buffering uncompressed video streams in RAM.
- **Fast Downscaled Scoring**: Computes Laplacian variance on a downscaled representation (default: 640px width) for maximum throughput, while extracting and saving winning frames at original full resolution.
- **Top-$K$ Diverse Frame Extraction**: Employs Temporal Non-Maximum Suppression (NMS) with configurable frame gaps (`--min-gap`) to ensure candidate frames are distinct rather than adjacent duplicates.
- **Frame Subsampling**: Configurable step intervals (`--frame-step`) to accelerate processing for high-framerate video feeds.
- **Quality Gating & Diagnostics**: Automatically evaluates quality (`PASS` / `FAIL_BLURRY`) against a sharpness threshold and returns comprehensive execution telemetry (FPS, resolution, duration, elapsed time).

---

## Installation

### Prerequisites
- Python 3.8+
- [OpenCV](https://pypi.org/project/opencv-python/)

### Setup

```bash
# Clone the repository
git clone https://github.com/sntharun/video-processing.git
cd video-processing

# (Optional) Create and activate a virtual environment
python3 -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
pip install opencv-python
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

### 3. Fast Mode with Quality Threshold
Subsample every 2nd frame and flag frames that don't meet a minimum sharpness threshold of `100.0`:

```bash
python3 video_processor.py input_video.mp4 --frame-step 2 --min-sharpness 100.0
```

### 4. Custom Output Path & Scan Duration
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
| `--eval-width` | — | `int` | `640` | Resize width for sharpness scoring (`0` to disable resizing) |
| `--min-sharpness`| — | `float` | `0.0` | Minimum sharpness threshold for `PASS` quality status |

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
    min_sharpness_threshold=100.0,
)

print(f"Quality Status: {result['quality_status']}")
print(f"Best Frame: {result['best_frame']} (Score: {result['sharpness_score']})")
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
  "total_frames": 90,
  "quality_status": "PASS",
  "diagnostics": {
    "fps": 30.0,
    "resolution": [1920, 1080],
    "processed_frames": 90,
    "processing_time_ms": 78.42,
    "eval_width": 640
  }
}
```

### Example: Top-$K$ Output (`--top-k 3`)
```json
{
  "best_frame": "best_frame_1.jpg",
  "frame_number": 14,
  "sharpness_score": 123.65,
  "total_frames": 90,
  "quality_status": "PASS",
  "diagnostics": {
    "fps": 30.0,
    "resolution": [1920, 1080],
    "processed_frames": 90,
    "processing_time_ms": 82.15,
    "eval_width": 640
  },
  "top_frames": [
    {
      "rank": 1,
      "file": "best_frame_1.jpg",
      "frame_number": 14,
      "sharpness_score": 123.65
    },
    {
      "rank": 2,
      "file": "best_frame_2.jpg",
      "frame_number": 28,
      "sharpness_score": 115.30
    },
    {
      "rank": 3,
      "file": "best_frame_3.jpg",
      "frame_number": 45,
      "sharpness_score": 109.82
    }
  ]
}
```

---

## License

Private repository. All rights reserved.
