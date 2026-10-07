"""BasicVSR++ on the mosaic, then MiniMax H3 on one locked crop."""

from __future__ import annotations

from fractions import Fraction
from pathlib import Path

import cv2
import numpy as np

from mosaicdiff.geometry import (
    H3_FPS,
    edge_fade,
    frames_at_h3_fps,
    letterbox,
)
from mosaicdiff.videoio import VideoWriter, open_capture
from mosaicdiff.vsr import restore_squares

CHUNK = 90
OVERLAP = 4


class Cancelled(Exception):
    pass


def unique_path(path: Path) -> Path:
    if not path.exists():
        return path
    stem, suffix = path.stem, path.suffix
    number = 1
    while True:
        candidate = path.with_name(f"{stem} ({number}){suffix}")
        if not candidate.exists():
            return candidate
        number += 1


def _check(cancel) -> None:
    if cancel.is_set():
        raise Cancelled()


def _detect(source, detector, log, progress, cancel):
    capture, width, height, fps, count = open_capture(source)
    boxes = []
    batch = []
    seen = 0
    try:
        while True:
            _check(cancel)
            ok, frame = capture.read()
            if not ok:
                break
            batch.append(frame)
            seen += 1
            if len(batch) == detector.batch:
                boxes.extend(detector.best_boxes(batch))
                batch = []
                if count:
                    progress(0.25 * min(1.0, seen / count), "Detecting")
        if batch:
            boxes.extend(detector.best_boxes(batch))
    finally:
        capture.release()
    log(f"{sum(box is not None for box in boxes)} of {len(boxes)} frames have a mosaic")
    return boxes, width, height, fps, len(boxes)


def _restore_video(source, vsr_path, boxes, model, device, fps, progress, cancel) -> None:
    capture, width, height, _fps, count = open_capture(source)
    writer = VideoWriter(vsr_path, width, height, Fraction(fps).limit_denominator(1000))
    hold: list[tuple[np.ndarray, tuple[int, int, int, int] | None]] = []
    frame_index = 0
    skip_head = 0
    try:
        while True:
            _check(cancel)
            ok, frame = capture.read()
            if not ok:
                break
            hold.append((frame, boxes[frame_index] if frame_index < len(boxes) else None))
            frame_index += 1
            if len(hold) >= CHUNK:
                _flush_chunk(hold, model, device, writer, keep_tail=OVERLAP, skip_head=skip_head)
                skip_head = OVERLAP
                if count:
                    progress(0.25 + 0.30 * min(1.0, frame_index / count), "BasicVSR++")
        if hold:
            _check(cancel)
            _flush_chunk(hold, model, device, writer, keep_tail=0, skip_head=skip_head)
    finally:
        capture.release()
        writer.close()


def _flush_chunk(hold, model, device, writer, keep_tail: int, skip_head: int) -> None:
    if not hold:
        return
    indexed = [(index, frame, box) for index, (frame, box) in enumerate(hold) if box is not None]
    restored: dict[int, np.ndarray] = {}
    if indexed:
        squares = []
        metas = []
        for _index, frame, box in indexed:
            x1, y1, x2, y2 = box
            crop = frame[y1:y2, x1:x2]
            if crop.size == 0:
                continue
            square, meta = letterbox(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))
            squares.append(square)
            metas.append((_index, box, meta))
        if squares:
            output = restore_squares(model, squares, device)
            for (index, box, meta), square in zip(metas, output):
                restored[index] = _unletterbox(square, box, meta)
    end = len(hold) if keep_tail <= 0 else max(skip_head, len(hold) - keep_tail)
    for index in range(skip_head, end):
        frame, box = hold[index]
        patch = restored.get(index)
        if patch is not None and box is not None:
            frame = _paste(frame, patch, box)
        writer.write(frame)
    del hold[:end]


def _unletterbox(square_rgb: np.ndarray, box, meta) -> np.ndarray:
    x0, y0, fitted_w, fitted_h = meta
    content = square_rgb[y0 : y0 + fitted_h, x0 : x0 + fitted_w]
    bgr = cv2.cvtColor(content, cv2.COLOR_RGB2BGR)
    x1, y1, x2, y2 = box
    return cv2.resize(bgr, (max(1, x2 - x1), max(1, y2 - y1)), interpolation=cv2.INTER_LANCZOS4)


def _paste(frame: np.ndarray, patch: np.ndarray, box) -> np.ndarray:
    x1, y1, x2, y2 = box
    x1 = max(0, x1)
    y1 = max(0, y1)
    x2 = min(frame.shape[1], x2)
    y2 = min(frame.shape[0], y2)
    if x2 <= x1 or y2 <= y1:
        return frame
    fitted = patch
    if fitted.shape[1] != x2 - x1 or fitted.shape[0] != y2 - y1:
        fitted = cv2.resize(fitted, (x2 - x1, y2 - y1), interpolation=cv2.INTER_LANCZOS4)
    short = min(y2 - y1, x2 - x1)
    falloff = max(24, min(round(short * 0.06), short // 6))
    fade = edge_fade(y2 - y1, x2 - x1, falloff)
    base = frame.copy()
    region = base[y1:y2, x1:x2].astype(np.float32)
    mixed = region * (1.0 - fade) + fitted.astype(np.float32) * fade
    base[y1:y2, x1:x2] = np.clip(np.round(mixed), 0, 255).astype(np.uint8)
    return base


def _write_output(vsr_path, destination, pastes, crop, output_fps, source_fps, frame_count: int) -> None:
    capture, width, height, _fps, _count = open_capture(vsr_path)
    keep = set(frames_at_h3_fps(frame_count, source_fps)) if output_fps == H3_FPS else None
    writer = VideoWriter(destination, width, height, Fraction(output_fps).limit_denominator(1000))
    index = 0
    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            if keep is not None and index not in keep:
                index += 1
                continue
            image_path = pastes.get(index)
            if image_path is not None:
                patch = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
                if patch is not None:
                    frame = _paste(frame, patch, crop)
            writer.write(frame)
            index += 1
    finally:
        capture.release()
        writer.close()


