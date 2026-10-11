"""ComfyUI nodes for the parts the stock graph cannot do.

Load Video, the model loaders, and SamplerCustomAdvanced stay the normal
Comfy nodes. These nodes find the mosaic, restore it with BasicVSR++, encode
that square in short pieces, and paste the sampled crop back.
"""

from __future__ import annotations

import math
import shutil
import subprocess
import sys
import threading
import time
from pathlib import Path

import numpy as np
import torch

_PACK = Path(__file__).resolve().parent
if str(_PACK) not in sys.path:
    sys.path.insert(0, str(_PACK))


def _log(message: str) -> None:
    print(message, flush=True)


def _noop(*_args, **_kwargs) -> None:
    return None


def _names(folder: str, suffix: str) -> list[str]:
    import folder_paths

    found = folder_paths.get_filename_list(folder)
    names = [name for name in found if name.lower().endswith(suffix)]
    return names or [f"missing{suffix}"]


def _model_file(folder: str, name: str) -> Path:
    import folder_paths

    found = folder_paths.get_full_path(folder, name)
    if not found:
        raise FileNotFoundError(f"Model file not found: {name}")
    return Path(found)


def _video_file(video) -> Path:
    source = video.get_stream_source()
    if isinstance(source, str):
        path = Path(source)
        if path.is_file():
            return path
    raise RuntimeError("Load Video did not provide a video file")


def _preview_frames(path: Path, indexes: list[int], box=None, limit: int = 8) -> torch.Tensor:
    import cv2

    chosen = [int(index) for index in indexes[:limit]] or [0]
    wanted = set(chosen)
    found: dict[int, np.ndarray] = {}
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise RuntimeError(f"Could not open {path}")
    index = 0
    try:
        while len(found) < len(wanted):
            ok, frame = capture.read()
            if not ok:
                break
            if index in wanted:
                if box is not None:
                    x1, y1, x2, y2 = (int(value) for value in box)
                    frame = frame[y1:y2, x1:x2]
                if frame.size == 0:
                    index += 1
                    continue
                height, width = frame.shape[:2]
                scale = 480 / max(height, width)
                if scale < 1:
                    frame = cv2.resize(
                        frame,
                        (max(1, int(width * scale)), max(1, int(height * scale))),
                        interpolation=cv2.INTER_AREA,
                    )
                found[index] = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            index += 1
    finally:
        capture.release()
    frames = [found[index] for index in chosen if index in found]
    if not frames:
        raise RuntimeError(f"No preview frames in {path}")
    return torch.from_numpy(np.stack(frames).astype(np.float32) / 255.0)


def _read_square(
    video: Path,
    indexes: list[int],
    crops: list[tuple[int, int, int, int]],
    width: int,
    height: int,
) -> torch.Tensor:
    import cv2

    need = {int(index): crop for index, crop in zip(indexes, crops)}
    found: dict[int, np.ndarray] = {}
    capture = cv2.VideoCapture(str(video))
    if not capture.isOpened():
        raise RuntimeError(f"Could not open {video}")
    index = 0
    try:
        while len(found) < len(need):
            ok, frame = capture.read()
            if not ok:
                break
            crop = need.get(index)
            if crop is not None:
                x1, y1, x2, y2 = crop
                image = frame[y1:y2, x1:x2]
                if image.size == 0:
                    raise RuntimeError(f"Empty H3 crop {x1},{y1}-{x2},{y2}")
                fitted = cv2.resize(image, (width, height), interpolation=cv2.INTER_AREA)
                found[index] = fitted
            index += 1
    finally:
        capture.release()
    missing = sorted(set(need) - set(found))
    if missing:
        raise RuntimeError(f"VSR video ended before H3 frame(s) {missing[:8]}")
    frames = []
    for index in indexes:
        bgr = found[int(index)]
        rgb = torch.from_numpy(np.ascontiguousarray(bgr[:, :, ::-1])).float().div_(255.0)
        frames.append(rgb)
    return torch.stack(frames, dim=0)


def _job() -> dict:
    import folder_paths

    root = Path(folder_paths.base_path)
    return {"comfy_root": str(root), "nodes_dir": str(root / "custom_nodes")}


class MosaicDiffPrepare:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "video": ("VIDEO",),
                "detector": (_names("mosaicdiff", ".onnx"),),
                "basicvsr": (_names("mosaicdiff", ".pth"),),
                "resolution": ("INT", {"default": 512, "min": 256, "max": 1280, "step": 32}),
                "context_seconds": ("INT", {"default": 5, "min": 1, "max": 15}),
            }
        }

    RETURN_TYPES = ("MOSAICDIFF_PREP", "IMAGE", "INT", "INT", "INT", "INT", "INT", "IMAGE")
    RETURN_NAMES = (
        "prep",
        "reference",
        "width",
        "height",
        "length",
        "context_length",
        "context_overlap",
        "preview",
    )
    FUNCTION = "prepare"
    CATEGORY = "MosaicDiff"
    DESCRIPTION = "Detect the mosaic, restore it with BasicVSR++, and pass a square that stays on the mosaic."

    def prepare(self, video, detector, basicvsr, resolution, context_seconds):
        import comfy.model_management
        import folder_paths

        from mosaicdiff.detect import Detector
        from mosaicdiff.geometry import (
            align_down,
            expand_box,
            fit_context,
            frames_at_h3_fps,
            generation_size,
            squares_for_frames,
        )
        from mosaicdiff.pipeline import _detect, _restore_video
        from mosaicdiff.vsr import load_restorer

        source = _video_file(video)
        if not torch.cuda.is_available():
            raise RuntimeError("MosaicDiff needs an NVIDIA GPU")
        comfy.model_management.unload_all_models()
        comfy.model_management.soft_empty_cache()

        device = torch.device("cuda:0")
        _log(f"Detecting mosaic in {source.name}")
        detector_model = Detector(_model_file("mosaicdiff", detector), device)
        _log(f"Detector model on {detector_model.provider}")
        cancel = threading.Event()
        started = time.perf_counter()
        try:
            boxes, width, height, fps, _count = _detect(source, detector_model, _log, _noop, cancel)
        finally:
            detector_model.close()
            torch.cuda.empty_cache()
        _log(f"Detection finished in {time.perf_counter() - started:.1f}s")
        if not any(box is not None for box in boxes):
            raise RuntimeError(f"No mosaic found in {source.name}")

        grown = [None if box is None else expand_box(*box, height, width) for box in boxes]
        _log(f"BasicVSR++ on {sum(box is not None for box in grown)} frames")
        restorer = load_restorer(_model_file("mosaicdiff", basicvsr), device)
        work = Path(folder_paths.get_temp_directory()) / "mosaicdiff"
        work.mkdir(parents=True, exist_ok=True)
        vsr_video = work / f"{source.stem}.vsr.mp4"
        started = time.perf_counter()
        try:
            _restore_video(source, vsr_video, grown, restorer, device, fps, _noop, cancel)
        finally:
            del restorer
            torch.cuda.empty_cache()
        _log(f"BasicVSR++ finished in {time.perf_counter() - started:.1f}s")

        present = [index for index, box in enumerate(grown) if box is not None]
        span = frames_at_h3_fps(present[-1] - present[0] + 1, fps)
        sampled = [present[0] + index for index in span]
        kept = align_down(len(sampled))
        if kept < 5:
            raise RuntimeError("The restored section is shorter than 5 frames at 24 fps")
        if kept != len(sampled):
            _log(f"H3 keeps {kept} of {len(sampled)} frames so the length matches the model")
        sampled = sampled[:kept]
        crops = squares_for_frames(grown, sampled, width, height)
        side = crops[0][2] - crops[0][0]
        gen_w, gen_h = generation_size(side, side, int(resolution))
        context, overlap = fit_context(float(context_seconds), len(sampled))
        later = crops[min(100, len(crops) - 1)]
        _log(
            f"Square {side}x{side} follows the mosaic, {len(sampled)} frames, "
            f"generation {gen_w}x{gen_h}, first at {crops[0][0]},{crops[0][1]}, "
            f"frame {min(100, len(crops) - 1)} at {later[0]},{later[1]}"
        )
        preview = _read_square(vsr_video, sampled[:8], crops[:8], gen_w, gen_h)
        _log(
            f"Reference stays on disk, {len(sampled)} frames. "
            "It is encoded in pieces of 17 frames, not loaded as one image."
        )
        prep = {
            "source": str(source),
            "vsr": str(vsr_video),
            "fps": float(fps),
            "frame_count": len(boxes),
            "frame_indices": [int(index) for index in sampled],
            "crop": [int(value) for value in crops[0]],
            "crops": [[int(value) for value in crop] for crop in crops],
            "width": gen_w,
            "height": gen_h,
        }
        return (prep, preview, gen_w, gen_h, len(sampled), context, overlap, preview)


def _qwen_orders(frame_count: int, limit: int = 32) -> list[int]:
    """2 fps samples, capped so the text encoder never sees the whole clip."""
    orders = list(range(0, max(1, int(frame_count)), 12)) or [0]
    if len(orders) <= limit:
        return orders
    if limit <= 1:
        return [orders[0]]
    last = len(orders) - 1
    picked = []
    seen = set()
    for step in range(limit):
        order = orders[round(step * last / (limit - 1))]
        if order in seen:
            continue
        seen.add(order)
        picked.append(order)
    return picked


def _load_vae(vae, shape, encode: bool) -> None:
    import comfy.model_management

    vae.throw_exception_if_invalid()
    memory = vae.memory_used_encode(shape, vae.vae_dtype) if encode else vae.memory_used_decode(shape, vae.vae_dtype)
    comfy.model_management.load_models_gpu(
        [vae.patcher],
        memory_required=max(1, int(memory)),
        force_full_load=vae.disable_offload,
    )


def _encode_reference_latent(vae, video: Path, indexes: list[int], crops: list, width: int, height: int):
    """Encode the square the way the H3 VAE does: 17-frame clips, then drop the last 3 tokens.

    One clip is about 50 MB at 512. The full float video is not built.
    """
    import cv2
    import comfy.model_management

    model = vae.first_stage_model
    if not hasattr(model, "clip_length"):
        raise RuntimeError("MosaicDiff Reference needs the MiniMax H3 video VAE")
    clip_length = int(model.clip_length)
    token_drop = int(model.token_drop)
    frame_count = len(indexes)
    qwen_at = set(_qwen_orders(frame_count))
    qwen: dict[int, torch.Tensor] = {}
    need = {int(index): (order, tuple(int(value) for value in crop)) for order, (index, crop) in enumerate(zip(indexes, crops))}
    _load_vae(vae, (1, 3, clip_length, height, width), encode=True)
    total_clips = math.ceil(frame_count / clip_length)
    canvas = None
    slot = 0
    per_clip = 0
    found = 0
    holding: list[torch.Tensor] = []
    capture = cv2.VideoCapture(str(video))
    if not capture.isOpened():
        raise RuntimeError(f"Could not open {video}")
    started = time.perf_counter()
    _log(f"Encoding reference in pieces of {clip_length} frames, {frame_count} frames, {total_clips} pieces")

    def _take(clip: torch.Tensor) -> None:
        nonlocal canvas, slot, per_clip
        if clip.shape[0] < clip_length:
            pad = clip[-1:].repeat(clip_length - clip.shape[0], 1, 1, 1)
            clip = torch.cat((clip, pad), dim=0)
        pixels = vae.process_input(clip.movedim(-1, 1).movedim(1, 0).unsqueeze(0)).to(vae.vae_dtype)
        with comfy.model_management.cuda_device_context(vae.device):
            moment = model._adaptive_encode(model._normalize_pixels(pixels.to(vae.device)))
        mean = torch.chunk(moment.float(), 2, dim=1)[0].cpu()
        del moment, pixels
        if canvas is None:
            per_clip = int(mean.shape[2])
            canvas = torch.empty((1, mean.shape[1], total_clips * per_clip, mean.shape[3], mean.shape[4]), dtype=mean.dtype)
        elif int(mean.shape[2]) != per_clip:
            raise RuntimeError(f"VAE piece has {int(mean.shape[2])} tokens, the first piece has {per_clip}")
        canvas[:, :, slot:slot + per_clip] = mean
        slot += per_clip
        if slot // per_clip == 1 or (slot // per_clip) % 100 == 0 or slot // per_clip == total_clips:
            _log(f"Encoded {slot // per_clip} of {total_clips} pieces")

    index = 0
    try:
        while found < len(need):
            comfy.model_management.throw_exception_if_processing_interrupted()
            ok, frame = capture.read()
            if not ok:
                break
            item = need.get(index)
            if item is not None:
                order, crop = item
                x1, y1, x2, y2 = crop
                image = frame[y1:y2, x1:x2]
                if image.size == 0:
                    raise RuntimeError(f"Empty H3 crop {x1},{y1}-{x2},{y2}")
                fitted = cv2.resize(image, (width, height), interpolation=cv2.INTER_AREA)
                rgb = torch.from_numpy(np.ascontiguousarray(fitted[:, :, ::-1])).float().div_(255.0)
                if order in qwen_at:
                    qwen[order] = rgb
                holding.append(rgb)
                found += 1
                if len(holding) == clip_length:
                    _take(torch.stack(holding, dim=0))
                    holding.clear()
            index += 1
    finally:
        capture.release()
    missing = len(need) - found
    if missing:
        raise RuntimeError(f"VSR video ended {missing} frame(s) before the reference was encoded")
    if holding:
        _take(torch.stack(holding, dim=0))
        holding.clear()
    if canvas is None or slot != total_clips * per_clip:
        raise RuntimeError("Reference encode produced no latent")
    latent = canvas[:, :, : slot - token_drop] if token_drop else canvas[:, :, :slot]
    del canvas
    mean_b = model.latents_mean.view(1, -1, 1, 1, 1).to(latent)
    std_b = model.latents_std.view(1, -1, 1, 1, 1).to(latent)
    encoded = (latent - mean_b) / std_b
    orders = [order for order in _qwen_orders(frame_count) if order in qwen]
    if not orders:
        raise RuntimeError("No frames were kept for the text encoder")
    pictures = torch.stack([qwen[order] for order in orders], dim=0)
    timestamps = [order / 24.0 for order in orders]
    _log(
        f"Reference latent {tuple(encoded.shape)} from {frame_count} frames in {time.perf_counter() - started:.1f}s. "
        f"Text encoder sees {len(orders)} of {frame_count} frames."
    )
    return encoded, pictures, timestamps


class _FrameSink:
    """Stand-in for the VAE decode buffer so frames are written instead of stored."""

    def __init__(self, shape, write_frame) -> None:
        self.shape = shape
        self._write_frame = write_frame

    def __getitem__(self, item):
        return _FrameSlice(self)


class _FrameSlice:
    def __init__(self, sink: _FrameSink) -> None:
        self._sink = sink

    def copy_(self, src) -> None:
        import comfy.model_management

        comfy.model_management.throw_exception_if_processing_interrupted()
        frames = src.detach()
        rgb = frames[0].permute(1, 2, 3, 0).clamp(0, 1).mul(255).round().to(dtype=torch.uint8).cpu().numpy()
        for index in range(rgb.shape[0]):
            self._sink._write_frame(rgb[index])


def _open_intermediate(path: Path, width: int, height: int, fps: float):
    from fractions import Fraction

    from mosaicdiff.videoio import VideoWriter, _rate

    exe = shutil.which("ffmpeg")
    if exe and width >= 256 and height >= 144:
        print("Encoding with h264_nvenc", flush=True)
        process = subprocess.Popen(
            [
                exe,
                "-hide_banner",
                "-loglevel",
                "error",
                "-y",
                "-f",
                "rawvideo",
                "-pix_fmt",
                "rgb24",
                "-s",
                f"{width}x{height}",
                "-r",
                _rate(Fraction(fps).limit_denominator(1000)),
                "-i",
                "-",
                "-c:v",
                "h264_nvenc",
                "-preset",
                "p4",
                "-rc",
                "vbr",
                "-cq",
                "1",
                "-pix_fmt",
                "yuv420p",
                str(path),
            ],
            stdin=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        return process, None
    writer = VideoWriter(path, width, height, Fraction(fps).limit_denominator(1000))
    return None, writer


class MosaicDiffReference:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "clip": ("CLIP",),
                "vae": ("VAE",),
                "prompt": ("STRING", {"multiline": True, "default": "Restore the vulva or penis the reference video"}),
                "prep": ("MOSAICDIFF_PREP",),
                "width": ("INT", {"default": 512, "min": 32, "max": 8192, "step": 32}),
                "height": ("INT", {"default": 512, "min": 32, "max": 8192, "step": 32}),
                "length": ("INT", {"default": 124, "min": 5, "max": 100000}),
            }
        }

    RETURN_TYPES = ("CONDITIONING", "LATENT")
    RETURN_NAMES = ("positive", "latent")
    FUNCTION = "encode"
    CATEGORY = "MosaicDiff"
    DESCRIPTION = "Encode the restored square from disk in 17-frame pieces and pin its border. The full clip is never one image."

    def encode(self, clip, vae, prompt, prep, width, height, length):
        import comfy.nested_tensor
        import node_helpers
        from comfy_extras.nodes_minimax_h3 import _empty_av_latent

        from mosaicdiff.h3_worker import _denoise_mask

        indexes = [int(index) for index in prep["frame_indices"]]
        crops = prep.get("crops") or [prep["crop"]] * len(indexes)
        if len(crops) != len(indexes):
            raise RuntimeError(f"Prepare stored {len(crops)} crops for {len(indexes)} frames")
        if int(width) != int(prep["width"]) or int(height) != int(prep["height"]) or int(length) != len(indexes):
            raise RuntimeError(
                f"Prepare has {len(indexes)} frames at {prep['width']}x{prep['height']}, "
                f"the graph asked for {int(length)} frames at {int(width)}x{int(height)}. Run Prepare again."
            )
        encoded, pictures, timestamps = _encode_reference_latent(
            vae,
            Path(prep["vsr"]),
            indexes,
            crops,
            int(width),
            int(height),
        )
        latent, _frame_count = _empty_av_latent(int(width), int(height), len(indexes))
        video, audio = latent["samples"].unbind()
        if tuple(encoded.shape) != tuple(video.shape):
            raise RuntimeError(
                f"Reference latent is {tuple(encoded.shape)}, generation latent is {tuple(video.shape)}"
            )
        ref = {
            "kind": "video",
            "latent_t": int(encoded.shape[2]),
            "latent_h": int(encoded.shape[3]),
            "latent_w": int(encoded.shape[4]),
            "ref_audio_t": 0,
            "latent": encoded,
            "audio_latent": None,
        }
        tokens = clip.tokenize(
            prompt,
            minimax_ref_items=[{"type": "video", "data": pictures, "timestamps": timestamps}],
        )
        cond = clip.encode_from_tokens_scheduled(tokens)
        cond = node_helpers.conditioning_set_values(cond, {"minimax_refs": [ref]})
        pinned = encoded.detach().to(device=video.device, dtype=video.dtype).clone()
        mask = _denoise_mask(torch, None, int(pinned.shape[2]), int(pinned.shape[-2]), int(pinned.shape[-1]))
        audio_mask = torch.zeros((1, 1, audio.shape[-2], audio.shape[-1]), dtype=torch.float32)
        latent["samples"] = comfy.nested_tensor.NestedTensor((pinned, audio))
        latent["noise_mask"] = comfy.nested_tensor.NestedTensor((mask, audio_mask))
        held = float((mask[0, 0, 0] <= 0).float().mean())
        _log(f"Noise mask holds {held:.0%} of each frame, including the border")
        return (cond, latent)


class MosaicDiffPinReference:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "latent": ("LATENT",),
                "images": ("IMAGE",),
                "vae": ("VAE",),
                "width": ("INT", {"default": 512, "min": 32, "max": 8192, "step": 32}),
                "height": ("INT", {"default": 512, "min": 32, "max": 8192, "step": 32}),
            }
        }

    RETURN_TYPES = ("LATENT",)
    FUNCTION = "pin"
    CATEGORY = "MosaicDiff"
    DESCRIPTION = "Keep the border of the reference crop while MiniMax H3 repaints the middle."

    def pin(self, latent, images, vae, width, height):
        from mosaicdiff.h3_worker import _pin_reference_border

        video, _audio = latent["samples"].unbind()
        if int(images.shape[0]) <= 8 and int(video.shape[2]) > 20:
            raise RuntimeError(
                "MosaicDiff Pin Reference got the short preview, not the full clip. "
                "Reload the MosaicDiff workflow. MosaicDiff Reference encodes the clip from disk."
            )
        _pin_reference_border({"torch": torch, "vae": vae}, latent, images, int(width), int(height), None)
        return (latent,)


class MosaicDiffPlace:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "latent": ("LATENT",),
                "vae": ("VAE",),
                "prep": ("MOSAICDIFF_PREP",),
            }
        }

    RETURN_TYPES = ("VIDEO", "IMAGE")
    RETURN_NAMES = ("video", "preview")
    FUNCTION = "place"
    CATEGORY = "MosaicDiff"
    OUTPUT_NODE = True
    DESCRIPTION = "Decode the sampled square in pieces, upscale it, and paste it back into the full video."

    def place(self, latent, vae, prep):
        import cv2
        import folder_paths
        from comfy_api.latest._input_impl.video_types import VideoFromFile
        from fractions import Fraction

        from mosaicdiff.geometry import H3_FPS, frames_at_h3_fps
        from mosaicdiff.h3_worker import _rtx_fit
        from mosaicdiff.pipeline import _paste, unique_path
        from mosaicdiff.videoio import VideoWriter, copy_audio, open_capture

        indices = [int(index) for index in prep["frame_indices"]]
        crops = prep.get("crops") or [prep["crop"]] * len(indices)
        if len(crops) != len(indices):
            raise RuntimeError(f"Prepare stored {len(crops)} crops for {len(indices)} frames")
        video_latent, _audio = latent["samples"].unbind()
        source = Path(prep["source"])
        vsr_video = Path(prep["vsr"])
        work = Path(folder_paths.get_temp_directory()) / "mosaicdiff"
        work.mkdir(parents=True, exist_ok=True)
        decoded_path = work / f"{source.stem}.h3.mp4"
        destination = unique_path(Path(folder_paths.get_output_directory()) / f"{source.stem}_mosaicdiff.mp4")
        model = vae.first_stage_model
        if not hasattr(model, "decode_output_shape"):
            raise RuntimeError("MosaicDiff Place needs the MiniMax H3 video VAE")
        out_shape = model.decode_output_shape(tuple(video_latent.shape))
        if int(out_shape[2]) != len(indices):
            raise RuntimeError(f"Latent decodes to {int(out_shape[2])} frames, Prepare has {len(indices)}")
        _load_vae(vae, tuple(video_latent.shape), encode=False)
        written = 0
        started = time.perf_counter()
        _log(f"Decoding {len(indices)} frames to disk")
        process, fallback = _open_intermediate(decoded_path, int(out_shape[4]), int(out_shape[3]), H3_FPS)

        def _write_frame(rgb: np.ndarray) -> None:
            nonlocal written
            if written >= len(indices):
                return
            rgb = np.ascontiguousarray(rgb)
            if process is not None:
                process.stdin.write(rgb.tobytes())
            else:
                fallback.write(np.ascontiguousarray(rgb[:, :, ::-1]))
            written += 1

        sink = _FrameSink(out_shape, _write_frame)
        try:
            model.decode(video_latent.to(vae.device), output_buffer=sink)
            if process is not None:
                process.stdin.close()
                error = process.stderr.read().decode("utf-8", errors="replace").strip()
                code = process.wait()
                if code != 0:
                    raise RuntimeError(error or f"NVENC encoder exited with status {code}")
            elif fallback is not None:
                fallback.close()
        except Exception:
            if process is not None and process.poll() is None:
                process.kill()
            decoded_path.unlink(missing_ok=True)
            raise
        if written != len(indices):
            decoded_path.unlink(missing_ok=True)
            raise RuntimeError(f"Decoded {written} frames, expected {len(indices)}")
        _log(f"Decode finished in {time.perf_counter() - started:.1f}s")

        output_fps = H3_FPS if float(prep["fps"]) > H3_FPS + 0.05 else float(prep["fps"])
        keep = set(frames_at_h3_fps(int(prep["frame_count"]), float(prep["fps"]))) if output_fps == H3_FPS else None
        square = crops[0]
        side_w = int(square[2]) - int(square[0])
        side_h = int(square[3]) - int(square[1])
        job = _job()
        capture, _width, _height, _fps, _count = open_capture(vsr_video)
        h3 = cv2.VideoCapture(str(decoded_path))
        if not h3.isOpened():
            capture.release()
            decoded_path.unlink(missing_ok=True)
            raise RuntimeError(f"Could not open {decoded_path}")
        writer = VideoWriter(destination, int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)), int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)), Fraction(output_fps).limit_denominator(1000))
        queue: list[np.ndarray] = []
        next_i = 0
        started = time.perf_counter()
        try:
            index = 0
            while True:
                ok, frame = capture.read()
                if not ok:
                    break
                if keep is not None and index not in keep:
                    index += 1
                    continue
                if next_i < len(indices) and index == indices[next_i]:
                    if not queue:
                        batch = []
                        for _step in range(16):
                            got, patch = h3.read()
                            if not got:
                                break
                            batch.append(cv2.cvtColor(patch, cv2.COLOR_BGR2RGB))
                        if not batch:
                            raise RuntimeError("Decoded video ended before every square was pasted")
                        stacked = torch.from_numpy(np.stack(batch)).float().div_(255.0)
                        fitted = _rtx_fit(
                            job,
                            stacked,
                            int(prep["width"]),
                            int(prep["height"]),
                            side_w,
                            side_h,
                            announce=next_i == 0,
                            release=next_i == 0,
                        )
                        rgb = fitted.detach().clamp(0, 1).mul(255).round().to(dtype=torch.uint8).cpu().numpy()
                        for order in range(rgb.shape[0]):
                            queue.append(np.ascontiguousarray(rgb[order][:, :, ::-1]))
                    frame = _paste(frame, queue.pop(0), tuple(int(value) for value in crops[next_i]))
                    next_i += 1
                writer.write(frame)
                index += 1
            if next_i != len(indices):
                raise RuntimeError(f"Placed {next_i} of {len(indices)} squares")
        except Exception:
            writer.close()
            destination.unlink(missing_ok=True)
            raise
        finally:
            capture.release()
            h3.release()
            decoded_path.unlink(missing_ok=True)
        writer.close()
        _log(f"RTX upscale and paste finished in {time.perf_counter() - started:.1f}s")
        with_audio = destination.with_name(destination.stem + ".audio" + destination.suffix)
        try:
            if copy_audio(destination, source, with_audio):
                with_audio.replace(destination)
        except Exception as exc:
            with_audio.unlink(missing_ok=True)
            _log(f"Audio was not copied: {exc}")
        vsr_video.unlink(missing_ok=True)
        _log(f"Wrote {destination}")
        capture = cv2.VideoCapture(str(destination))
        count = max(1, int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 1))
        capture.release()
        preview = _preview_frames(destination, sorted({0, count // 3, (2 * count) // 3, count - 1}))
        return (VideoFromFile(str(destination)), preview)


NODE_CLASS_MAPPINGS = {
    "MosaicDiffPrepare": MosaicDiffPrepare,
    "MosaicDiffReference": MosaicDiffReference,
    "MosaicDiffPinReference": MosaicDiffPinReference,
    "MosaicDiffPlace": MosaicDiffPlace,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "MosaicDiffPrepare": "MosaicDiff Prepare",
    "MosaicDiffReference": "MosaicDiff Reference",
    "MosaicDiffPinReference": "MosaicDiff Pin Reference",
    "MosaicDiffPlace": "MosaicDiff Place",
}
