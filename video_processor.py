"""
Video Processing module — Canine Nose-Print Recognition (Phase 0 prototype)

Reads a short video stream, computes sharpness scores efficiently on a center ROI,
computes exposure/glare and motion diagnostic telemetry, selects the top-K sharpest
distinct frames using temporal non-maximum suppression (NMS), and extracts full-resolution
frames with diagnostic metadata.

Key Enhancements (Step 1):
- Glare-masked Laplacian scoring (dilates and ignores specular reflections on wet noses)
- Multi-metric support: glare_masked_laplacian (default), raw laplacian, tenengrad
- Inter-frame motion estimation to flag frames affected by camera/subject movement
- Exposure-quality scaling factor (smooth penalty for severe under/over-exposure)
- Streaming two-pass evaluation (O(1) frame memory footprint during Pass 1)
- Center-weighted ROI cropping before scoring to prevent background bias
- Fast downscaled sharpness evaluation for high-resolution video streams
- Top-K diverse frame extraction with temporal NMS (min_frame_gap)
- Safe quality status handling: UNVALIDATED when threshold is unset, PASS / FAIL_BLURRY when set
"""

import argparse
import json
import os
import sys
import time
import cv2
import numpy as np


MAX_DURATION_SECONDS = 3.0    # Spec default cap (~2-3s clips)
DEFAULT_EVAL_WIDTH = 640      # Downscaled width for fast sharpness scoring
DEFAULT_MIN_FRAME_GAP = 10    # Minimum frame distance between top-k picks
DEFAULT_CROP_FRACTION = 0.55  # Heuristic central ROI fraction
DEFAULT_METRIC = "glare_masked_laplacian"


def crop_center(frame, fraction=DEFAULT_CROP_FRACTION):
    """Crops the central bounding box of the frame based on `fraction`.

    Canine nose prints are typically framed near the center of the video.
    Evaluating only the central region prevents high-contrast background elements
    (e.g., floor tiles, patterned fabrics, hands, collar edges) from inflating
    the sharpness score over the actual nose texture.

    Args:
        frame (np.ndarray): Full frame array (H, W, C) or (H, W).
        fraction (float): Fraction of width and height to retain (0.0 < fraction <= 1.0).

    Returns:
        np.ndarray: Cropped central image region.
    """
    if fraction is None or fraction <= 0.0 or fraction >= 1.0:
        return frame

    h, w = frame.shape[:2]
    crop_h = int(h * fraction)
    crop_w = int(w * fraction)

    if crop_h <= 0 or crop_w <= 0:
        return frame

    start_y = max(0, (h - crop_h) // 2)
    start_x = max(0, (w - crop_w) // 2)

    return frame[start_y : start_y + crop_h, start_x : start_x + crop_w]


def calculate_exposure_stats(gray_frame):
    """Computes exposure, glare, and contrast metrics from a grayscale image.

    Args:
        gray_frame (np.ndarray): Grayscale image array (values in 0..255).

    Returns:
        dict:
            - overexposure_pct (float): Percentage of pixels with intensity > 245 (glare / clipped highlights).
            - underexposure_mean (float): Mean luminance across the region (0.0 - 255.0).
            - contrast_std (float): Standard deviation of pixel intensities (dynamic range).
    """
    if gray_frame is None or gray_frame.size == 0:
        return {
            "overexposure_pct": 0.0,
            "underexposure_mean": 0.0,
            "contrast_std": 0.0,
        }

    overexposure_pct = round(float(np.mean(gray_frame > 245) * 100.0), 2)
    underexposure_mean = round(float(np.mean(gray_frame)), 2)
    contrast_std = round(float(np.std(gray_frame)), 2)

    return {
        "overexposure_pct": overexposure_pct,
        "underexposure_mean": underexposure_mean,
        "contrast_std": contrast_std,
    }


def calculate_masked_laplacian(gray_frame, glare_threshold=245):
    """Computes Laplacian variance while masking out saturated specular highlights.

    Wet dog noses create specular glints that produce artificial high-frequency edges.
    This function creates a dilated mask over pixels >= glare_threshold so the variance
    is computed purely on true nose leather / ridge texture.

    Args:
        gray_frame (np.ndarray): Grayscale image (2D).
        glare_threshold (int): Pixel intensity considered saturated highlight (default: 245).

    Returns:
        tuple (float, float):
            - masked_sharpness (float): Laplacian variance on non-glare pixels.
            - raw_sharpness (float): Standard Laplacian variance across all pixels.
    """
    if gray_frame is None or gray_frame.size == 0:
        return 0.0, 0.0

    lap = cv2.Laplacian(gray_frame, cv2.CV_64F)
    raw_sharpness = float(lap.var())

    glare_mask = (gray_frame >= glare_threshold).astype(np.uint8)
    if np.sum(glare_mask) > 0:
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        dilated_mask = cv2.dilate(glare_mask, kernel)
        valid_mask = dilated_mask == 0

        valid_pixels = np.sum(valid_mask)
        if valid_pixels >= int(0.20 * gray_frame.size):
            valid_lap = lap[valid_mask]
            mean_lap = np.mean(valid_lap)
            masked_sharpness = float(np.mean((valid_lap - mean_lap) ** 2))
        else:
            masked_sharpness = raw_sharpness * 0.1
    else:
        masked_sharpness = raw_sharpness

    return float(masked_sharpness), float(raw_sharpness)


def calculate_tenengrad(gray_frame, ksize=3):
    """Computes Tenengrad gradient energy (Sobel gradient magnitude sum).

    Args:
        gray_frame (np.ndarray): Grayscale image (2D).
        ksize (int): Sobel kernel size (default: 3).

    Returns:
        float: Mean gradient magnitude squared.
    """
    if gray_frame is None or gray_frame.size == 0:
        return 0.0

    gx = cv2.Sobel(gray_frame, cv2.CV_64F, 1, 0, ksize=ksize)
    gy = cv2.Sobel(gray_frame, cv2.CV_64F, 0, 1, ksize=ksize)
    grad_mag_sq = gx ** 2 + gy ** 2
    return float(np.mean(grad_mag_sq))


def calculate_exposure_quality_multiplier(mean_luminance):
    """Returns an exposure quality scaling factor in [0.1, 1.0].

    Penalizes extreme underexposure (<70) or extreme overexposure (>185).
    """
    if 70.0 <= mean_luminance <= 185.0:
        return 1.0
    if mean_luminance < 70.0:
        return max(0.1, mean_luminance / 70.0)
    return max(0.1, (255.0 - mean_luminance) / 70.0)


def calculate_sharpness(
    frame,
    eval_width=DEFAULT_EVAL_WIDTH,
    crop_fraction=DEFAULT_CROP_FRACTION,
    metric=DEFAULT_METRIC,
):
    """Computes sharpness score on the center-cropped, downscaled grayscale frame.

    Args:
        frame (np.ndarray): Input BGR image.
        eval_width (int or None): Width to resize ROI to before scoring. None/0 disables resizing.
        crop_fraction (float): Central ROI fraction to evaluate.
        metric (str): Sharpness algorithm ('glare_masked_laplacian', 'laplacian', 'tenengrad').

    Returns:
        float: Sharpness score.
    """
    score, _, _ = calculate_frame_metrics(
        frame, eval_width=eval_width, crop_fraction=crop_fraction, metric=metric
    )
    return score


def calculate_frame_metrics(
    frame,
    eval_width=DEFAULT_EVAL_WIDTH,
    crop_fraction=DEFAULT_CROP_FRACTION,
    metric=DEFAULT_METRIC,
    prev_gray_roi=None,
):
    """Extracts central ROI, optionally downscales, and computes sharpness, motion & exposure telemetry.

    Args:
        frame (np.ndarray): Input BGR image.
        eval_width (int or None): Width to resize ROI to before scoring. None/0 disables resizing.
        crop_fraction (float): Central ROI fraction to evaluate.
        metric (str): Sharpness algorithm ('glare_masked_laplacian', 'laplacian', 'tenengrad').
        prev_gray_roi (np.ndarray or None): Previous frame's grayscale ROI for motion estimation.

    Returns:
        tuple (float, dict, np.ndarray):
            - effective_score (float): Quality-weighted sharpness score for frame ranking.
            - exposure_stats (dict): overexposure_pct, underexposure_mean, contrast_std,
                                    motion_score, raw_sharpness, metric_used.
            - gray (np.ndarray): Grayscale ROI array (used for subsequent motion diff).
    """
    roi = crop_center(frame, fraction=crop_fraction)

    if eval_width and eval_width > 0 and roi.shape[1] > eval_width:
        scale = eval_width / float(roi.shape[1])
        new_height = max(1, int(roi.shape[0] * scale))
        eval_frame = cv2.resize(roi, (eval_width, new_height), interpolation=cv2.INTER_AREA)
    else:
        eval_frame = roi

    if len(eval_frame.shape) == 3 and eval_frame.shape[2] == 3:
        gray = cv2.cvtColor(eval_frame, cv2.COLOR_BGR2GRAY)
    else:
        gray = eval_frame

    exposure_stats = calculate_exposure_stats(gray)

    if prev_gray_roi is not None and prev_gray_roi.shape == gray.shape:
        motion_score = round(float(np.mean(np.abs(gray.astype(np.float32) - prev_gray_roi.astype(np.float32)))), 2)
    else:
        motion_score = 0.0

    if metric == "laplacian":
        lap = cv2.Laplacian(gray, cv2.CV_64F)
        raw_sharpness = float(lap.var())
        sharpness_score = raw_sharpness
    elif metric == "tenengrad":
        sharpness_score = calculate_tenengrad(gray)
        raw_sharpness = sharpness_score
    else:  # default: glare_masked_laplacian
        sharpness_score, raw_sharpness = calculate_masked_laplacian(gray)

    exposure_mult = calculate_exposure_quality_multiplier(exposure_stats["underexposure_mean"])
    effective_score = float(sharpness_score * exposure_mult)

    exposure_stats.update({
        "raw_sharpness": round(raw_sharpness, 2),
        "sharpness_score": round(sharpness_score, 2),
        "motion_score": motion_score,
        "metric_used": metric,
    })

    return effective_score, exposure_stats, gray


def select_top_k_indices(scored_items, top_k=1, min_frame_gap=DEFAULT_MIN_FRAME_GAP):
    """Selects top-k frame candidates using temporal Non-Maximum Suppression (NMS).

    Args:
        scored_items: List of tuples (frame_number, score, exposure_stats)
        top_k: Number of candidates to select
        min_frame_gap: Minimum temporal frame distance between selected candidates

    Returns:
        List of (frame_number, score, exposure_stats) sorted by score descending.
    """
    sorted_items = sorted(scored_items, key=lambda x: x[1], reverse=True)
    selected = []

    for item in sorted_items:
        if len(selected) >= top_k:
            break
        fn = item[0]
        if all(abs(fn - sel[0]) >= min_frame_gap for sel in selected):
            selected.append(item)

    if len(selected) < top_k and len(sorted_items) > len(selected):
        selected_fns = {sel[0] for sel in selected}
        for item in sorted_items:
            if len(selected) >= top_k:
                break
            if item[0] not in selected_fns:
                selected.append(item)
                selected_fns.add(item[0])

    return selected


def process_video(
    video_path,
    output_path="best_frame.jpg",
    top_k=1,
    min_frame_gap=DEFAULT_MIN_FRAME_GAP,
    max_duration_seconds=MAX_DURATION_SECONDS,
    frame_step=1,
    eval_width=DEFAULT_EVAL_WIDTH,
    crop_fraction=DEFAULT_CROP_FRACTION,
    min_sharpness_threshold=0.0,
    metric=DEFAULT_METRIC,
):
    """Processes a video stream in a memory-efficient two-pass manner.

    - Pass 1 (O(1) memory): Streams frames, crops central ROI, computes sharpness, glare mask, motion & exposure.
    - Candidate Selection: Applies temporal NMS to find the top_k diverse candidate frame indices.
    - Pass 2: Seeks directly to the winning frame(s) to read and save the full-resolution uncropped image(s).

    Args:
        video_path (str): Path to input video file (.mp4, .mov, etc.)
        output_path (str): Output filename or base path for saved frame(s).
        top_k (int): Number of distinct candidate frames to extract.
        min_frame_gap (int): Minimum frame separation between top-k picks.
        max_duration_seconds (float): Max video duration in seconds to scan.
        frame_step (int): Subsample step (1 = all frames, 2 = every 2nd frame).
        eval_width (int): Downscaled width for ROI sharpness evaluation (0 to disable).
        crop_fraction (float): Central ROI crop fraction for scoring (0.0 to 1.0).
        min_sharpness_threshold (float or None): Minimum score for PASS status.
        metric (str): Sharpness metric ('glare_masked_laplacian', 'laplacian', 'tenengrad').

    Returns:
        dict: Metadata matching the Phase 0 contract with diagnostics and telemetry.
    """
    t_start = time.perf_counter()

    video = cv2.VideoCapture(video_path)
    if not video.isOpened():
        raise IOError(f"Could not open video: {video_path}")

    fps = video.get(cv2.CAP_PROP_FPS) or 30.0
    width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    max_frames = int(fps * max_duration_seconds)

    scored_frames = []
    frame_number = 0
    total_frames_read = 0
    prev_gray_roi = None

    while frame_number < max_frames:
        success, frame = video.read()
        if not success:
            break

        total_frames_read += 1

        if frame_number % frame_step == 0:
            score, exposure_stats, prev_gray_roi = calculate_frame_metrics(
                frame,
                eval_width=eval_width,
                crop_fraction=crop_fraction,
                metric=metric,
                prev_gray_roi=prev_gray_roi,
            )
            scored_frames.append((frame_number, score, exposure_stats))

        frame_number += 1

    if not scored_frames:
        video.release()
        raise ValueError("No valid frames could be read from the video.")

    top_candidates = select_top_k_indices(
        scored_frames, top_k=top_k, min_frame_gap=min_frame_gap
    )

    saved_frames = {}
    target_fns = sorted([item[0] for item in top_candidates])

    for target_fn in target_fns:
        video.set(cv2.CAP_PROP_POS_FRAMES, target_fn)
        success, target_frame = video.read()
        if success and target_frame is not None:
            saved_frames[target_fn] = target_frame
        else:
            video.set(cv2.CAP_PROP_POS_FRAMES, 0)
            current_fn = 0
            while current_fn <= target_fn:
                ok, f = video.read()
                if not ok:
                    break
                if current_fn == target_fn:
                    saved_frames[target_fn] = f
                    break
                current_fn += 1

    video.release()

    top_frames_metadata = []
    base_name, ext = os.path.splitext(output_path)
    if not ext:
        ext = ".jpg"

    for rank, (fn, score, exp_stats) in enumerate(top_candidates, start=1):
        if fn in saved_frames:
            if top_k == 1:
                frame_filename = output_path
            else:
                frame_filename = f"{base_name}_{rank}{ext}"

            out_dir = os.path.dirname(frame_filename)
            if out_dir:
                os.makedirs(out_dir, exist_ok=True)

            cv2.imwrite(frame_filename, saved_frames[fn])

            top_frames_metadata.append({
                "rank": rank,
                "file": frame_filename,
                "frame_number": fn,
                "effective_score": round(float(score), 2),
                "sharpness_score": exp_stats["sharpness_score"],
                "raw_sharpness": exp_stats["raw_sharpness"],
                "motion_score": exp_stats["motion_score"],
                "overexposure_pct": exp_stats["overexposure_pct"],
                "underexposure_mean": exp_stats["underexposure_mean"],
                "contrast_std": exp_stats["contrast_std"],
            })

    t_end = time.perf_counter()
    processing_time_ms = round((t_end - t_start) * 1000, 2)

    best_cand = top_frames_metadata[0] if top_frames_metadata else None
    best_score = best_cand["effective_score"] if best_cand else 0.0
    best_fn = best_cand["frame_number"] if best_cand else 0
    best_file = best_cand["file"] if best_cand else output_path
    best_overexposure_pct = best_cand["overexposure_pct"] if best_cand else 0.0
    best_underexposure_mean = best_cand["underexposure_mean"] if best_cand else 0.0
    best_contrast_std = best_cand["contrast_std"] if best_cand else 0.0
    best_raw_sharpness = best_cand["raw_sharpness"] if best_cand else 0.0
    best_motion_score = best_cand["motion_score"] if best_cand else 0.0

    if min_sharpness_threshold is not None and min_sharpness_threshold > 0.0:
        if best_score >= min_sharpness_threshold:
            quality_status = "PASS"
        else:
            quality_status = "FAIL_BLURRY"
    else:
        quality_status = "UNVALIDATED"

    result = {
        "best_frame": best_file,
        "frame_number": best_fn,
        "effective_score": best_score,
        "sharpness_score": best_cand["sharpness_score"] if best_cand else 0.0,
        "raw_sharpness": best_raw_sharpness,
        "motion_score": best_motion_score,
        "overexposure_pct": best_overexposure_pct,
        "underexposure_mean": best_underexposure_mean,
        "contrast_std": best_contrast_std,
        "total_frames": total_frames_read,
        "quality_status": quality_status,
        "diagnostics": {
            "fps": round(float(fps), 2),
            "resolution": [width, height],
            "processed_frames": len(scored_frames),
            "processing_time_ms": processing_time_ms,
            "eval_width": eval_width,
            "crop_fraction": crop_fraction,
            "metric": metric,
        },
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
        "--crop-fraction",
        type=float,
        default=DEFAULT_CROP_FRACTION,
        help="Central ROI fraction (0.0-1.0) to evaluate (default: 0.55)",
    )
    parser.add_argument(
        "--metric",
        choices=["glare_masked_laplacian", "laplacian", "tenengrad"],
        default=DEFAULT_METRIC,
        help="Sharpness metric to use (default: glare_masked_laplacian)",
    )
    parser.add_argument(
        "--min-sharpness",
        type=float,
        default=0.0,
        help="Minimum sharpness threshold for PASS. 0.0 leaves quality status as UNVALIDATED (default: 0.0)",
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
            crop_fraction=args.crop_fraction,
            min_sharpness_threshold=args.min_sharpness,
            metric=args.metric,
        )
        print(json.dumps(result, indent=2))
    except Exception as e:
        sys.stderr.write(f"Error: {e}\n")
        sys.exit(1)


if __name__ == "__main__":
    main()