# Nose-aware Video Processing — Phase 0

## What changed
The original module selected the sharpest **whole frame** using Laplacian variance.
This version selects the best **nose crop** after a nose/snout region is provided.

## Two modes

### 1. Manual ROI — recommended for Phase 0 validation
This lets you test whether nose-specific quality scoring works before training a detector.

Example:
python video_processor_v2.py dog.mp4 --nose-box 780,500,1150,850

Replace the coordinates with the nose bounding box in your frame.

### 2. Automatic detector — target architecture
Use a custom YOLO/Ultralytics model trained to detect `nose` or `snout`.

Example:
python video_processor_v2.py dog.mp4 --detector dog_nose_detector.pt

The detector model is NOT included. It must be trained/obtained separately.

## Output
- best_nose_frame.jpg = full frame selected
- best_nose.jpg = cropped nose region
- JSON diagnostics = frame number, nose box, sharpness, quality score, etc.

## Important
The weights in the combined score are starting values for experimentation, not validated thresholds.
Do not claim production-level nose recognition from this module until it is tested on real stray-dog videos.
