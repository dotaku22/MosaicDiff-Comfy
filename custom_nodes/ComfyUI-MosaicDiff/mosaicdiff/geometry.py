"""Frame lengths and crop boxes for the BasicVSR++ and MiniMax H3 passes."""

from __future__ import annotations

import math

H3_FPS = 24.0
MIN_FRAMES = 5
MAX_SAMPLE_FRAMES = 362
FRAME_MOD = 17
FRAME_REM = 5
RESTORATION_SIZE = 256


def align_down(count: int) -> int:
    """Largest count <= ``count`` with ``count % 17 == 5``, or 0 when that is under 5."""
    if count < MIN_FRAMES:
        return 0
    drop = (count % FRAME_MOD - FRAME_REM) % FRAME_MOD
    aligned = count - drop
    return aligned if aligned >= MIN_FRAMES else 0


def align_up(count: int) -> int:
    count = max(MIN_FRAMES, int(count))
    while count % FRAME_MOD != FRAME_REM:
        count += 1
    return count


def context_frames(seconds: float) -> int:
    seconds = min(15.0, max(1.0, float(seconds)))
    return min(align_up(round(seconds * H3_FPS)), MAX_SAMPLE_FRAMES)


def fit_context(seconds: float, sample_frames: int) -> tuple[int, int]:
    requested = context_frames(seconds)
    available = max(0, int(sample_frames))
    fitted = align_down(min(available, requested, MAX_SAMPLE_FRAMES))
    if fitted < MIN_FRAMES:
        fitted = align_down(available)
    overlap = 22 if fitted > 22 else 0
    if overlap >= fitted:
        overlap = 0
    return fitted, overlap


def frames_at_h3_fps(frame_count: int, source_fps: float) -> list[int]:
    """Source indices for a constant 24 fps timeline."""
    frame_count = int(frame_count)
    if frame_count <= 0:
        return []
    if source_fps <= H3_FPS + 0.05:
        return list(range(frame_count))
    out_count = max(1, int(round(frame_count * H3_FPS / float(source_fps))))
    picked: list[int] = []
    last = -1
    for slot in range(out_count):
        index = min(frame_count - 1, (slot * frame_count) // out_count)
        if index != last:
            picked.append(index)
            last = index
    return picked


def split_samples(frame_indices: list[int]) -> list[list[int]]:
    windows: list[list[int]] = []
    start = 0
    total = len(frame_indices)
    while start < total:
        if total - start <= MAX_SAMPLE_FRAMES:
            count = align_down(total - start)
            if count >= MIN_FRAMES:
                windows.append(frame_indices[start : start + count])
            break
        count = align_down(MAX_SAMPLE_FRAMES)
        windows.append(frame_indices[start : start + count])
        start += count
    return windows


def context_piece_spans(total: int, length: int, overlap: int = 22) -> list[tuple[int, int]]:
    """Pieces no longer than the H3 length, sharing about one second.

    The last piece is the tail plus that overlap, so a 15 second region is not
    sampled as one block when the H3 length is shorter.
    """
    total = int(total)
    length = int(length)
    if length % FRAME_MOD != FRAME_REM:
        length = align_down(length)
    if length < MIN_FRAMES:
        length = align_down(total)
    overlap = 22 if length > 22 else 0
    if total <= length:
        count = total if total % FRAME_MOD == FRAME_REM else align_down(total)
        return [(0, count)] if count >= MIN_FRAMES else []
    stride = length - overlap
    spans: list[tuple[int, int]] = []
    start = 0
    while start < total:
        remaining = total - start
        if remaining <= length:
            count = remaining if remaining % FRAME_MOD == FRAME_REM else align_down(remaining)
            if count >= MIN_FRAMES:
                spans.append((start, count))
            break
        tail_room = total - (start + stride)
        if tail_room <= length:
            tail = tail_room if tail_room % FRAME_MOD == FRAME_REM else align_up(tail_room)
            if tail > length:
                tail = align_down(min(length, max(tail_room, MIN_FRAMES)))
            if tail < MIN_FRAMES:
                spans.append((start, length))
                start += stride
                continue
            spans.append((start, length))
            spans.append((total - tail, tail))
            break
        spans.append((start, length))
        start += stride
    return spans


def split_context_pieces(frame_indices: list[int], length: int, overlap: int = 22) -> list[list[int]]:
    """Cut a 24 fps timeline into H3-length pieces that share ``overlap`` frames."""
    return [
        frame_indices[start : start + count]
        for start, count in context_piece_spans(len(frame_indices), length, overlap)
    ]


def generation_size(width: int, height: int, side: int) -> tuple[int, int]:
    """Keep the crop's shape at the pixel count of a ``side`` square."""
    side = max(256, min(1280, int(side)))
    megapixels = (side / 1024) ** 2
    total = megapixels * 1024 * 1024
    scale = math.sqrt(total / (width * height))
    out_w = max(32, round(width * scale / 32) * 32)
    out_h = max(32, round(height * scale / 32) * 32)
    return out_w, out_h


def expand_box(x1: int, y1: int, x2: int, y2: int, frame_h: int, frame_w: int) -> tuple[int, int, int, int]:
    """Grow a detection toward the 256 restoration square, staying inside the frame."""
    width = x2 - x1
    height = y2 - y1
    border = max(20, int(max(width, height) * 0.06))
    x1 = max(0, x1 - border)
    y1 = max(0, y1 - border)
    x2 = min(frame_w, x2 + border)
    y2 = min(frame_h, y2 + border)
    width = max(1, x2 - x1)
    height = max(1, y2 - y1)
    scale = min(1.0, RESTORATION_SIZE / width, RESTORATION_SIZE / height)
    missing_w = int((RESTORATION_SIZE - width * scale) / scale) if scale > 0 else 0
    missing_h = int((RESTORATION_SIZE - height * scale) / scale) if scale > 0 else 0
    budget_w = width
    budget_h = height
    left_room = x1
    right_room = frame_w - x2
    top_room = y1
    bottom_room = frame_h - y2
    grow_x = min(left_room, right_room, missing_w // 2, budget_w)
    extra_left = min(left_room - grow_x, missing_w - grow_x * 2, budget_w - grow_x)
    extra_right = min(right_room - grow_x, missing_w - grow_x * 2 - extra_left, budget_w - grow_x - extra_left)
    grow_y = min(top_room, bottom_room, missing_h // 2, budget_h)
    extra_top = min(top_room - grow_y, missing_h - grow_y * 2, budget_h - grow_y)
    extra_bottom = min(bottom_room - grow_y, missing_h - grow_y * 2 - extra_top, budget_h - grow_y - extra_top)
    x1 = max(0, x1 - math.floor(grow_x / 2) - extra_left)
    x2 = min(frame_w, x2 + math.ceil(grow_x / 2) + extra_right)
    y1 = max(0, y1 - math.floor(grow_y / 2) - extra_top)
    y2 = min(frame_h, y2 + math.ceil(grow_y / 2) + extra_bottom)
    return int(x1), int(y1), int(x2), int(y2)


def _place_square(
    origin_x: int,
    origin_y: int,
    side: int,
    frame_w: int,
    frame_h: int,
    box: tuple[int, int, int, int] | None,
) -> tuple[int, int, int, int]:
    """Keep an even square inside the frame and, when it fits, over this box."""
    origin_x = int(origin_x)
    origin_y = int(origin_y)
    if box is not None:
        x1, y1, x2, y2 = box
        if x2 - side <= x1:
            origin_x = min(max(origin_x, x2 - side), x1)
        if y2 - side <= y1:
            origin_y = min(max(origin_y, y2 - side), y1)
    origin_x = max(0, min(origin_x, frame_w - side))
    origin_y = max(0, min(origin_y, frame_h - side))
    origin_x -= origin_x % 2
    origin_y -= origin_y % 2
    if origin_x + side > frame_w:
        origin_x = max(0, frame_w - side)
    if origin_y + side > frame_h:
        origin_y = max(0, frame_h - side)
    return origin_x, origin_y, origin_x + side, origin_y + side


def _smooth_centers(values: list[float | None], radius: int) -> list[float | None]:
    smoothed: list[float | None] = []
    for index, value in enumerate(values):
        if value is None:
            smoothed.append(None)
            continue
        total = 0.0
        count = 0
        start = max(0, index - radius)
        end = min(len(values), index + radius + 1)
        for other in values[start:end]:
            if other is None:
                continue
            total += other
            count += 1
        smoothed.append(total / count)
    return smoothed


def tracking_squares(
    boxes: list[tuple[int, int, int, int] | None],
    frame_w: int,
    frame_h: int,
    radius: int = 15,
) -> list[tuple[int, int, int, int] | None]:
    """One square size, moved so each frame's square stays on that frame's mosaic.

    The size is the largest single BasicVSR++ box, not the path the mosaic
    travels. A square of the whole path gets cut to the short side of a tall
    frame and the mosaic walks out of it.
    """
    present = [box for box in boxes if box is not None]
    if not present:
        raise ValueError("tracking squares need at least one box")
    side = max(max(box[2] - box[0], box[3] - box[1]) for box in present)
    side = max(2, side)
    if side % 2:
        side += 1
    limit = min(frame_w - frame_w % 2, frame_h - frame_h % 2)
    side = min(side, max(2, limit))
    if side % 2:
        side -= 1
    centers_x: list[float | None] = []
    centers_y: list[float | None] = []
    for box in boxes:
        if box is None:
            centers_x.append(None)
            centers_y.append(None)
            continue
        held = _place_square(
            int(round((box[0] + box[2]) / 2 - side / 2)),
            int(round((box[1] + box[3]) / 2 - side / 2)),
            side,
            frame_w,
            frame_h,
            box,
        )
        centers_x.append(held[0] + side / 2)
        centers_y.append(held[1] + side / 2)
    smooth_x = _smooth_centers(centers_x, radius)
    smooth_y = _smooth_centers(centers_y, radius)
    tracked: list[tuple[int, int, int, int] | None] = []
    for box, center_x, center_y in zip(boxes, smooth_x, smooth_y):
        if center_x is None or center_y is None:
            tracked.append(None)
            continue
        tracked.append(
            _place_square(
                int(round(center_x - side / 2)),
                int(round(center_y - side / 2)),
                side,
                frame_w,
                frame_h,
                box,
            )
        )
    return tracked


def squares_for_frames(
    boxes: list[tuple[int, int, int, int] | None],
    frame_indices: list[int],
    frame_w: int,
    frame_h: int,
) -> list[tuple[int, int, int, int]]:
    """The tracking square for each sampled frame. Gaps reuse the nearest box."""
    tracked = tracking_squares(boxes, frame_w, frame_h)
    crops: list[tuple[int, int, int, int]] = []
    for index in frame_indices:
        crop = tracked[index] if 0 <= index < len(tracked) else None
        if crop is None:
            crop = _nearest_square(tracked, index)
        crops.append(crop)
    return crops


def _nearest_square(
    tracked: list[tuple[int, int, int, int] | None],
    index: int,
) -> tuple[int, int, int, int]:
    for distance in range(len(tracked)):
        for other in (index - distance, index + distance):
            if 0 <= other < len(tracked) and tracked[other] is not None:
                return tracked[other]
    raise ValueError("tracking squares need at least one box")


def square_crop(
    boxes: list[tuple[int, int, int, int]],
    frame_w: int,
    frame_h: int,
) -> tuple[int, int, int, int]:
    """One even square around the BasicVSR++ boxes, the same cut as the training clips."""
    x1 = min(box[0] for box in boxes)
    y1 = min(box[1] for box in boxes)
    x2 = max(box[2] for box in boxes)
    y2 = max(box[3] for box in boxes)
    side = max(x2 - x1, y2 - y1, 2)
    if side % 2:
        side += 1
    side = min(side, frame_w - frame_w % 2, frame_h - frame_h % 2)
    center_x = (x1 + x2) / 2
    center_y = (y1 + y2) / 2
    origin_x = int(round(center_x - side / 2))
    origin_y = int(round(center_y - side / 2))
    origin_x = max(0, min(origin_x, frame_w - side))
    origin_y = max(0, min(origin_y, frame_h - side))
    return origin_x, origin_y, origin_x + side, origin_y + side


def stable_crop(
    boxes: list[tuple[int, int, int, int]],
    frame_w: int,
    frame_h: int,
) -> tuple[int, int, int, int]:
    """One even rectangle covering every restored box, plus a small margin."""
    x1 = min(box[0] for box in boxes)
    y1 = min(box[1] for box in boxes)
    x2 = max(box[2] for box in boxes)
    y2 = max(box[3] for box in boxes)
    pad = max(32, int(round(0.06 * max(x2 - x1, y2 - y1))))
    x1 = max(0, x1 - pad)
    y1 = max(0, y1 - pad)
    x2 = min(frame_w, x2 + pad)
    y2 = min(frame_h, y2 + pad)
    x1 -= x1 % 2
    y1 -= y1 % 2
    if x2 % 2:
        x2 = x2 + 1 if x2 + 1 <= frame_w else x2 - 1
    if y2 % 2:
        y2 = y2 + 1 if y2 + 1 <= frame_h else y2 - 1
    if x2 <= x1 or y2 <= y1:
        return 0, 0, frame_w - frame_w % 2, frame_h - frame_h % 2
    return x1, y1, x2, y2


def window_crops(
    boxes: list[tuple[int, int, int, int] | None],
    windows: list[list[int]],
    frame_w: int,
    frame_h: int,
) -> list[tuple[int, int, int, int]]:
    """One crop per sample, covering the boxes in that sample only.

    A close-up would otherwise stretch the crop for the whole file, and a wider
    shot would then be a small patch inside it. A sample with no detection keeps
    the file crop so a gap between mosaics still has a rectangle to paste.
    """
    present = [box for box in boxes if box is not None]
    if not present:
        raise ValueError("window_crops needs at least one box")
    fallback = stable_crop(present, frame_w, frame_h)
    crops: list[tuple[int, int, int, int]] = []
    for indices in windows:
        chosen = [
            boxes[index]
            for index in indices
            if 0 <= index < len(boxes) and boxes[index] is not None
        ]
        crops.append(stable_crop(chosen, frame_w, frame_h) if chosen else fallback)
    return crops


def letterbox(image, size: int = RESTORATION_SIZE):
    """Fit ``image`` inside a square. Returns the square and the content rectangle."""
    import cv2

    height, width = image.shape[:2]
    scale = size / max(height, width)
    fitted_w = max(1, int(round(width * scale)))
    fitted_h = max(1, int(round(height * scale)))
    fitted = cv2.resize(image, (fitted_w, fitted_h), interpolation=cv2.INTER_AREA)
    import numpy as np

    canvas = np.zeros((size, size, 3), dtype=np.uint8)
    x0 = (size - fitted_w) // 2
    y0 = (size - fitted_h) // 2
    canvas[y0 : y0 + fitted_h, x0 : x0 + fitted_w] = fitted
    return canvas, (x0, y0, fitted_w, fitted_h)


def edge_fade(height: int, width: int, falloff: int):
    import numpy as np

    falloff = max(1, min(falloff, height // 2, width // 2))
    y = np.minimum(np.arange(height), np.arange(height)[::-1]).clip(max=falloff).astype(np.float32) / falloff
    x = np.minimum(np.arange(width), np.arange(width)[::-1]).clip(max=falloff).astype(np.float32) / falloff
    y = y * y * (3 - 2 * y)
    x = x * x * (3 - 2 * x)
    return (y[:, None] * x[None, :])[:, :, None]
