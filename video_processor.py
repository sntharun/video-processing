"""
Video Processing module — Canine Nose-Print Recognition (Phase 0 prototype)

Reads a short video stream, computes sharpness scores efficiently, selects
the top-K sharpest distinct frames using temporal non-maximum suppression (NMS),
and saves the result with diagnostic metadata.

Key Enhancements:
- Streaming single-pass evaluation (O(K) memory footprint, avoids loading all frames to RAM)
- Downscaled sharpness evaluation for fast processing on 4K/1080p clips
- Configurable frame subsampling (frame_step)
- Top-K diverse frame extraction with temporal separation (min_frame_gap)
- Actionable quality status and diagnostic metrics (FPS, resolution, processing time)
"""

import argparse
import json
import os
import sys
import time
import cv2


MAX_DURATION_SECONDS = 3.0  # Spec default cap (~2-3s clips)
DEFAULT_EVAL_WIDTH = 640    # Resolution width for fast scoring
DEFAULT_MIN_FRAME_GAP = 10  # Minimum frame distance between top-k picks


def calculate_sharpness(frame, eval_width=DEFAULT_EVAL_WIDTH):
    """Compute Laplacian variance on a grayscale (optionally downscaled) frame.
    Higher score indicates sharper edge definition and fine pattern detail.
    """
    if eval_width and eval_width > 0 and frame.shape[1] > eval_width:
        scale = eval_width / float(frame.shape[1])
        new_height = int(frame.shape[0] * scale)
        eval_frame = cv2.resize(frame, (eval_width, new_height), interpolation=cv2.INTER_AREA)
    else:
        eval_frame = frame

    gray = cv2.cvtColor(eval_frame, cv2.COLOR_BGR2GRAY)
    laplacian = cv2.Laplacian(gray, cv2.CV_64F)
    return float(laplacian.var())


def select_top_k_indices(scores, top_k=1, min_frame_gap=DEFAULT_MIN_FRAME_GAP):
    """Select top-k frame indices using temporal non-maximum suppression (NMS)
    so candidate frames are separated by at least `min_frame_gap` frames.

    Args:
        scores: list of (frame_number, score)
        top_k: number of candidates to select
        min_frame_gap: minimum temporal distance between selected frames

    Returns:
        List of (frame_number, score) sorted by score descending.
    """
    sorted_scores = sorted(scores, key=lambda x: x[1], reverse=True)
    selected = []

    for fn, score in sorted_scores:
        if len(selected) >= top_k:
            break
        # Check temporal separation against already selected frames
        if all(abs(fn - sel_fn) >= min_frame_gap for sel_fn, _ in selected):
            selected.append((fn, score))

    # If temporal constraint was too strict to fill top_k, fill with remaining highest
    if len(selected) < top_k and len(sorted_scores) > len(selected):
        selected_fns = {fn for fn, _ in selected}
        for fn, score in sorted_scores:
            if len(selected) >= top_k:
                break
            if fn not in selected_fns:
                selected.append((fn, score))
                selected_fns.add(fn)

    return selected


def process_video(
    video_path,
    output_path="best_frame.jpg",
    top_k=1,
    min_frame_gap=DEFAULT_MIN_FRAME_GAP,
    max_duration_seconds=MAX_DURATION_SECONDS,
    frame_step=1,
    eval_width=DEFAULT_EVAL_WIDTH,
    min_sharpness_threshold=0.0,
):
    """Processes a video file in a memory-efficient streaming manner, scores
    sharpness, and saves the top candidate frame(s).

    Args:
        video_path (str): Path to input video file (.mp4, .mov, etc.)
        output_path (str): Output filename or pattern for saved frames.
        top_k (int): Number of top distinct frames to extract.
        min_frame_gap (int): Minimum frame separation for top-k candidates.
        max_duration_seconds (float): Max video duration to scan.
        frame_step (int): Sample every Nth frame (1 = all frames, 2 = every 2nd).
        eval_width (int): Downscaled width for sharpness evaluation.
        min_sharpness_threshold (float): Threshold for PASS/FAIL_BLURRY quality.

    Returns:
        dict: Metadata matching the Phase 0 output contract with diagnostics.
    """
    t_start = time.perf_counter()

    video = cv2.VideoCapture(video_path)
    if not video.isOpened():
        raise IOError(f"Could not open video: {video_path}")

    fps = video.get(cv2.CAP_PROP_FPS) or 30.0
    width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    max_frames = int(fps * max_duration_seconds)

    # Pass 1: Stream & score without keeping full uncompressed frames in RAM (O(1) memory)
    scores = []
    frame_number = 0
    total_frames_read = 0

    while frame_number < max_frames:
        success, frame = video.read()
        if not success:
            break

        total_frames_read += 1

        if frame_number % frame_step == 0:
            score = calculate_sharpness(frame, eval_width=eval_width)
            scores.append((frame_number, score))

        frame_number += 1

    if not scores:
        video.release()
        raise ValueError("No valid frames could be read from the video.")

    # Select top-k distinct winning frame indices
    top_candidates = select_top_k_indices(scores, top_k=top_k, min_frame_gap=min_frame_gap)
    winning_indices = {fn: score for fn, score in top_candidates}

    # Pass 2: Extract and save winning full-resolution frames
    saved_frames = {}
    for target_fn in winning_indices.keys():
        video.set(cv2.CAP_PROP_POS_FRAMES, target_fn)
        success, target_frame = video.read()
        if success:
            saved_frames[target_fn] = target_frame

    video.release()

    # Determine file paths and save frames
    top_frames_metadata = []
    base_name, ext = os.path.splitext(output_path)
    if not ext:
        ext = ".jpg"

    for rank, (fn, score) in enumerate(top_candidates, start=1):
        if fn in saved_frames:
            if top_k == 1:
                frame_filename = output_path
            else:
                frame_filename = f"{base_name}_{rank}{ext}"

            cv2.imwrite(frame_filename, saved_frames[fn])
            top_frames_metadata.append({
                "rank": rank,
                "file": frame_filename,
                "frame_number": fn,
                "sharpness_score": round(float(score), 2),
            })

    t_end = time.perf_counter()
    processing_time_ms = round((t_end - t_start) * 1000, 2)

    best_cand = top_frames_metadata[0] if top_frames_metadata else None
    best_score = best_cand["sharpness_score"] if best_cand else 0.0
    best_fn = best_cand["frame_number"] if best_cand else 0
    best_file = best_cand["file"] if best_cand else output_path

    # Quality status evaluation
    if min_sharpness_threshold > 0.0 and best_score < min_sharpness_threshold:
        quality_status = "FAIL_BLURRY"
    else:
        quality_status = "PASS"

    result = {
        "best_frame": best_file,
        "frame_number": best_fn,
        "sharpness_score": best_score,
        "total_frames": total_frames_read,
        "quality_status": quality_status,
        "diagnostics": {
            "fps": round(float(fps), 2),
            "resolution": [width, height],
            "processed_frames": len(scores),
            "processing_time_ms": processing_time_ms,
            "eval_width": eval_width,
        }
    }

    if top_k > 1:
        result["top_frames"] = top_frames_metadata

    return result


def main():
    parser = argparse.ArgumentParser(
        description="Extract the highest quality frame(s) from canine nose videos."
    )
    parser.add_argument("video_path", help="Path to input video file")
    parser.add_argument(
        "-o", "--output", default="best_frame.jpg", help="Output file path (default: best_frame.jpg)"
    )
    parser.add_argument(
        "-k", "--top-k", type=int, default=1, help="Number of candidate frames to save (default: 1)"
    )
    parser.add_argument(
        "--min-gap",
        type=int,
        default=DEFAULT_MIN_FRAME_GAP,
        help="Minimum frame gap between top-k picks (default: 10)",
    )
    parser.add_argument(
        "--max-duration",
        type=float,
        default=MAX_DURATION_SECONDS,
        help="Max video duration in seconds to scan (default: 3.0)",
    )
    parser.add_argument(
        "--frame-step",
        type=int,
        default=1,
        help="Subsample step (1 = all frames, 2 = every 2nd frame) (default: 1)",
    )
    parser.add_argument(
        "--eval-width",
        type=int,
        default=DEFAULT_EVAL_WIDTH,
        help="Resize width for sharpness scoring, 0 to disable (default: 640)",
    )
    parser.add_argument(
        "--min-sharpness",
        type=float,
        default=0.0,
        help="Minimum sharpness score threshold for PASS quality (default: 0.0)",
    )

    args = parser.parse_args()

    try:
        result = process_video(
            video_path=args.video_path,
            output_path=args.output,
            top_k=args.top_k,
            min_frame_gap=args.min_gap,
            max_duration_seconds=args.max_duration,
            frame_step=args.frame_step,
            eval_width=args.eval_width if args.eval_width > 0 else None,
            min_sharpness_threshold=args.min_sharpness,
        )
        print(json.dumps(result, indent=2))
    except Exception as e:
        sys.stderr.write(f"Error: {e}\n")
        sys.exit(1)


if __name__ == "__main__":
    main()