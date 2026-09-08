"""
video_processor_v2.py
Canine Nose-Print Recognition — nose-aware video frame selection.

IMPORTANT:
- This version does NOT pretend that whole-frame sharpness = nose quality.
- For TRUE automatic nose-aware selection, provide a custom YOLO/Ultralytics
  model trained to detect the dog's nose/snout.
- Without a detector, use --nose-box for Phase-0/manual validation.

Pipeline:
Video -> frames -> nose detection -> nose crop -> nose quality score
      -> best nose-print frame + metadata

Expected custom detector:
  A YOLO model whose class names include "nose" or "snout".
"""

import argparse
import json
import os
import time
import cv2

MAX_DURATION_SECONDS = 3.0
DEFAULT_EVAL_WIDTH = 640
DEFAULT_CONFIDENCE = 0.35

# Quality-score weights. These are starting values for experimentation,
# NOT validated production thresholds.
W_SHARPNESS = 0.70
W_BRIGHTNESS = 0.15
W_SIZE = 0.15


def laplacian_sharpness(image, eval_width=DEFAULT_EVAL_WIDTH):
    """Return Laplacian-variance sharpness score for an image."""
    if image is None or image.size == 0:
        return 0.0

    h, w = image.shape[:2]
    if eval_width and w > eval_width:
        scale = eval_width / float(w)
        image = cv2.resize(
            image,
            (eval_width, max(1, int(h * scale))),
            interpolation=cv2.INTER_AREA,
        )

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    lap = cv2.Laplacian(gray, cv2.CV_64F)
    return float(lap.var())


def brightness_quality(image):
    """
    Simple exposure-quality score in [0, 1].
    Penalizes very dark or very bright nose crops.
    """
    if image is None or image.size == 0:
        return 0.0

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    mean = float(gray.mean())

    # Broad usable range for prototype testing.
    if 70 <= mean <= 190:
        return 1.0

    if mean < 70:
        return max(0.0, mean / 70.0)

    return max(0.0, (255.0 - mean) / 65.0)


def size_quality(box, frame_shape):
    """
    Prefer a reasonably large nose crop.
    This is intentionally a soft score, not a hard rule.
    """
    x1, y1, x2, y2 = box
    fh, fw = frame_shape[:2]

    area_ratio = max(0.0, (x2 - x1) * (y2 - y1)) / float(max(1, fw * fh))

    # Prototype preference: around 2% or more of frame area is useful.
    # Cap at 1.0 so large detections don't dominate.
    return min(1.0, area_ratio / 0.02)


def crop_box(frame, box, padding=0.10):
    """Crop a bounding box with a small relative padding."""
    h, w = frame.shape[:2]
    x1, y1, x2, y2 = map(int, box)

    bw = max(1, x2 - x1)
    bh = max(1, y2 - y1)

    px = int(bw * padding)
    py = int(bh * padding)

    x1 = max(0, x1 - px)
    y1 = max(0, y1 - py)
    x2 = min(w, x2 + px)
    y2 = min(h, y2 + py)

    if x2 <= x1 or y2 <= y1:
        return None

    return frame[y1:y2, x1:x2]


def normalize_sharpness(score, reference=250.0):
    """
    Convert raw Laplacian variance to [0,1].
    Reference is only a starting point and must be calibrated experimentally.
    """
    return min(1.0, max(0.0, score / reference))


def combined_quality(nose_crop, box, frame_shape):
    """Calculate a prototype nose-quality score in [0,1]."""
    sharpness = laplacian_sharpness(nose_crop)
    sharpness_q = normalize_sharpness(sharpness)
    brightness_q = brightness_quality(nose_crop)
    size_q = size_quality(box, frame_shape)

    total = (
        W_SHARPNESS * sharpness_q
        + W_BRIGHTNESS * brightness_q
        + W_SIZE * size_q
    )

    return {
        "score": float(total),
        "sharpness": float(sharpness),
        "sharpness_quality": float(sharpness_q),
        "brightness_quality": float(brightness_q),
        "size_quality": float(size_q),
    }


def load_detector(model_path):
    """
    Load a custom Ultralytics YOLO detector.

    The model should be trained to detect dog nose/snout.
    """
    try:
        from ultralytics import YOLO
    except ImportError as exc:
        raise RuntimeError(
            "Ultralytics is not installed. Run: pip install ultralytics"
        ) from exc

    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"Detector model not found: {model_path}"
        )

    return YOLO(model_path)


def detect_noses(model, frame, confidence=DEFAULT_CONFIDENCE):
    """
    Return candidate nose boxes from a custom YOLO model.

    The detector should have a class named 'nose' or 'snout'.
    If it has one class only, that class is accepted.
    """
    results = model.predict(
        source=frame,
        conf=confidence,
        verbose=False,
    )

    candidates = []

    if not results:
        return candidates

    result = results[0]
    boxes = getattr(result, "boxes", None)

    if boxes is None:
        return candidates

    names = getattr(result, "names", {}) or {}

    for i in range(len(boxes)):
        xyxy = boxes.xyxy[i].tolist()
        conf = float(boxes.conf[i].item())
        cls_id = int(boxes.cls[i].item())

        class_name = str(names.get(cls_id, cls_id)).lower()

        # Prefer explicit nose/snout labels.
        if class_name not in {"nose", "snout"} and len(names) > 1:
            continue

        candidates.append({
            "box": xyxy,
            "detector_confidence": conf,
            "class_name": class_name,
        })

    return candidates


def parse_manual_box(text):
    """Parse x1,y1,x2,y2 from command line."""
    values = [float(v.strip()) for v in text.split(",")]
    if len(values) != 4:
        raise ValueError("--nose-box must be x1,y1,x2,y2")
    return values


def process_video(
    video_path,
    output_frame="best_nose_frame.jpg",
    start_time=0.0,
    output_nose="best_nose.jpg",
    detector_path=None,
    manual_box=None,
    max_duration=MAX_DURATION_SECONDS,
    frame_step=1,
    confidence=DEFAULT_CONFIDENCE,
):
    start = time.perf_counter()

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise IOError(f"Could not open video: {video_path}")
    
    if start_time > 0:
        cap.set(cv2.CAP_PROP_POS_MSEC, start_time * 1000)

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    max_frames = int(fps * max_duration)

    detector = load_detector(detector_path) if detector_path else None

    best = None
    frame_number = 0
    processed = 0

    while frame_number < max_frames:
        ok, frame = cap.read()
        if not ok:
            break

        if frame_number % frame_step != 0:
            frame_number += 1
            continue

        processed += 1

        if detector is not None:
            candidates = detect_noses(detector, frame, confidence)
        elif manual_box is not None:
            candidates = [{
                "box": manual_box,
                "detector_confidence": 1.0,
                "class_name": "manual_roi",
            }]
        else:
            cap.release()
            raise ValueError(
                "For nose-aware mode provide either --detector MODEL.pt "
                "or --nose-box x1,y1,x2,y2."
            )

        for candidate in candidates:
            box = candidate["box"]
            nose = crop_box(frame, box)

            if nose is None or nose.size == 0:
                continue

            quality = combined_quality(nose, box, frame.shape)

            # Detection confidence is not included in the quality score
            # because confidence calibration depends on the detector.
            # It is stored as diagnostic information.
            record = {
                "frame_number": frame_number,
                "box": [round(float(v), 2) for v in box],
                "detector_confidence": round(
                    float(candidate["detector_confidence"]), 4
                ),
                "class_name": candidate["class_name"],
                **quality,
            }

            if best is None or record["score"] > best["record"]["score"]:
                best = {
                    "record": record,
                    "frame": frame.copy(),
                    "nose": nose.copy(),
                }

        frame_number += 1

    cap.release()

    if best is None:
        raise ValueError(
            "No nose detection/ROI was usable in the scanned video."
        )

    cv2.imwrite(output_frame, best["frame"])
    cv2.imwrite(output_nose, best["nose"])

    elapsed_ms = round((time.perf_counter() - start) * 1000, 2)

    result = {
        "best_frame": output_frame,
        "best_nose": output_nose,
        "frame_number": best["record"]["frame_number"],
        "nose_quality_score": round(best["record"]["score"], 4),
        "sharpness_score": round(best["record"]["sharpness"], 2),
        "detector_confidence": best["record"]["detector_confidence"],
        "nose_box": best["record"]["box"],
        "total_frames_read": min(frame_number, max_frames),
        "processed_frames": processed,
        "fps": round(float(fps), 2),
        "resolution": [width, height],
        "processing_time_ms": elapsed_ms,
        "note": (
            "Quality weights and normalization are prototype starting points; "
            "validate them experimentally."
        ),
    }

    return result


def main():
    parser = argparse.ArgumentParser(
        description="Select the best nose-print frame from a short dog video."
    )

    parser.add_argument("video_path")
    parser.add_argument(
        "-o", "--output-frame",
        default="best_nose_frame.jpg",
    )
    parser.add_argument(
        "--output-nose",
        default="best_nose.jpg",
    )
    parser.add_argument(
        "--detector",
        help="Path to custom YOLO nose/snout detector (.pt).",
    )
    parser.add_argument(
        "--nose-box",
        help="Manual ROI for Phase 0 testing: x1,y1,x2,y2",
    )
    parser.add_argument(
        "--start-time",
        type=float,
        default=0.0,
        help="Start processing at this many seconds into the video.",
    )
    parser.add_argument(
        "--max-duration",
        type=float,
        default=MAX_DURATION_SECONDS,
    )
    parser.add_argument(
        "--frame-step",
        type=int,
        default=1,
    )
    parser.add_argument(
        "--confidence",
        type=float,
        default=DEFAULT_CONFIDENCE,
    )

    args = parser.parse_args()

    manual_box = parse_manual_box(args.nose_box) if args.nose_box else None

    if not args.detector and not manual_box:
        parser.error(
            "Provide --detector MODEL.pt for automatic nose detection, "
            "or --nose-box x1,y1,x2,y2 for manual Phase-0 testing."
        )

    try:
        result = process_video(
            video_path=args.video_path,
            start_time=args.start_time,
            output_frame=args.output_frame,
            output_nose=args.output_nose,
            detector_path=args.detector,
            manual_box=manual_box,
            max_duration=args.max_duration,
            frame_step=args.frame_step,
            confidence=args.confidence,
        )
        print(json.dumps(result, indent=2))
    except Exception as exc:
        print(f"ERROR: {exc}")
        raise SystemExit(1)


if __name__ == "__main__":
    main()
