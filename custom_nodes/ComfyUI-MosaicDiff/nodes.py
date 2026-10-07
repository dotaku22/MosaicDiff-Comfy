"""ComfyUI nodes for the parts the stock graph cannot do.

Load Video, the model loaders, and SamplerCustomAdvanced stay the normal
Comfy nodes. These nodes find the mosaic, restore it with BasicVSR++, pin the
reference border, and paste the sampled crop back.
"""

from __future__ import annotations

import sys
import tempfile
import threading
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


def _names(suffix: str) -> list[str]:
    import folder_paths

    found = folder_paths.get_filename_list("mosaicdiff")
    names = [name for name in found if name.lower().endswith(suffix)]
    return names or [f"missing{suffix}"]


def _model_file(name: str) -> Path:
    import folder_paths

    found = folder_paths.get_full_path("mosaicdiff", name)
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


def _read_crops(video: Path, indexes: list[int], crop: list[int]) -> dict[int, np.ndarray]:
    import cv2

    need = {int(index) for index in indexes}
    found: dict[int, np.ndarray] = {}
    capture = cv2.VideoCapture(str(video))
    if not capture.isOpened():
        raise RuntimeError(f"Could not open {video}")
    x1, y1, x2, y2 = (int(value) for value in crop)
    index = 0
    try:
        while len(found) < len(need):
            ok, frame = capture.read()
            if not ok:
                break
            if index in need:
                image = frame[y1:y2, x1:x2]
                if image.size == 0:
                    raise RuntimeError(f"Empty H3 crop {x1},{y1}-{x2},{y2}")
                found[index] = image.copy()
            index += 1
    finally:
        capture.release()
    missing = sorted(need - set(found))
    if missing:
        raise RuntimeError(f"VSR video ended before H3 frame(s) {missing[:8]}")
    return found


def _frames_from_crops(crops: dict[int, np.ndarray], indexes: list[int]) -> torch.Tensor:
    frames = []
    for index in indexes:
        bgr = crops[int(index)]
        rgb = torch.from_numpy(np.ascontiguousarray(bgr[:, :, ::-1])).float().div_(255.0)
        frames.append(rgb)
    return torch.stack(frames, dim=0)


def _save_pngs(images: torch.Tensor, folder: Path) -> None:
    from PIL import Image

    folder.mkdir(parents=True, exist_ok=True)
    for index in range(int(images.shape[0])):
        array = images[index].clamp(0, 1).mul(255).round().to(dtype=torch.uint8).cpu().numpy()
        Image.fromarray(array).save(folder / f"{index:06d}.png")


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
                "detector": (_names(".onnx"),),
                "basicvsr": (_names(".pth"),),
                "resolution": ("INT", {"default": 512, "min": 512, "max": 1280, "step": 32}),
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
    DESCRIPTION = "Find the mosaic and restore it with BasicVSR++. Reference frames are the crop at 24 fps."

    def prepare(self, video, detector, basicvsr, resolution, context_seconds):
        import comfy.model_management
        import folder_paths

        from mosaicdiff.detect import Detector
        from mosaicdiff.geometry import expand_box, fit_context, frames_at_h3_fps, generation_size, split_samples, stable_crop
        from mosaicdiff.pipeline import _detect, _restore_video
        from mosaicdiff.vsr import load_restorer

        source = _video_file(video)
        if not torch.cuda.is_available():
            raise RuntimeError("MosaicDiff needs an NVIDIA GPU")
        comfy.model_management.unload_all_models()
        comfy.model_management.soft_empty_cache()

        device = torch.device("cuda:0")
        _log(f"Detecting mosaic in {source.name}")
        detector_model = Detector(_model_file(detector), device)
        _log(f"Detector model on {detector_model.provider}")
        cancel = threading.Event()
        try:
            boxes, width, height, fps, _count = _detect(source, detector_model, _log, _noop, cancel)
        finally:
            detector_model.close()
            torch.cuda.empty_cache()
        if not any(box is not None for box in boxes):
            raise RuntimeError(f"No mosaic found in {source.name}")

        grown = [None if box is None else expand_box(*box, height, width) for box in boxes]
        _log(f"BasicVSR++ on {sum(box is not None for box in grown)} frames")
        restorer = load_restorer(_model_file(basicvsr), device)
        work = Path(folder_paths.get_temp_directory()) / "mosaicdiff"
        work.mkdir(parents=True, exist_ok=True)
        vsr_video = work / f"{source.stem}.vsr.mp4"
        try:
            _restore_video(source, vsr_video, grown, restorer, device, fps, _noop, cancel)
        finally:
            del restorer
            torch.cuda.empty_cache()

        crop = [int(value) for value in stable_crop([box for box in grown if box is not None], width, height)]
        present = [index for index, box in enumerate(grown) if box is not None]
        span = frames_at_h3_fps(present[-1] - present[0] + 1, fps)
        sampled = [present[0] + index for index in span]
        windows = split_samples(sampled)
        if not windows:
            raise RuntimeError("The restored section is shorter than 5 frames at 24 fps")
        gen_w, gen_h = generation_size(crop[2] - crop[0], crop[3] - crop[1], int(resolution))
        context, overlap = fit_context(float(context_seconds), max(len(window) for window in windows))
        _log(
            f"H3 crop {crop[2] - crop[0]}x{crop[3] - crop[1]} at {crop[0]},{crop[1]}, "
            f"{len(windows)} sample(s), {gen_w}x{gen_h}"
        )
        first = [int(index) for index in windows[0]]
        reference = _frames_from_crops(_read_crops(vsr_video, first, crop), first)
        prep = {
            "source": str(source),
            "vsr": str(vsr_video),
            "crop": crop,
            "fps": float(fps),
            "frame_count": len(boxes),
            "windows": windows,
            "width": gen_w,
            "height": gen_h,
            "prompt_frames": first,
        }
        preview = _preview_frames(vsr_video, first, crop)
        return (prep, reference, gen_w, gen_h, len(first), context, overlap, preview)


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

        _pin_reference_border({"torch": torch, "vae": vae}, latent, images, int(width), int(height), None)
        return (latent,)


class MosaicDiffPlace:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE",),
                "prep": ("MOSAICDIFF_PREP",),
                "model": ("MODEL",),
                "clip": ("CLIP",),
                "vae": ("VAE",),
                "noise": ("NOISE",),
                "sampler": ("SAMPLER",),
                "sigmas": ("SIGMAS",),
                "prompt": ("STRING", {"multiline": True, "default": "Restore the vulva or penis the reference video"}),
            }
        }

    RETURN_TYPES = ("VIDEO", "IMAGE")
    RETURN_NAMES = ("video", "preview")
    FUNCTION = "place"
    CATEGORY = "MosaicDiff"
    OUTPUT_NODE = True
    DESCRIPTION = "Fit the sampled frames onto the crop and paste them back. Further samples use the same sampler, sigmas, and noise."

    def place(self, images, prep, model, clip, vae, noise, sampler, sigmas, prompt):
        import comfy.model_management
        import folder_paths
        from comfy_api.latest._input_impl.video_types import VideoFromFile
        from comfy_extras.nodes_custom_sampler import BasicGuider, SamplerCustomAdvanced
        from comfy_extras.nodes_minimax_h3 import MiniMaxH3ReferenceToVideo
        from nodes import VAEDecode

        from mosaicdiff.geometry import H3_FPS
        from mosaicdiff.h3_worker import _pin_reference_border, _rtx_fit
        from mosaicdiff.pipeline import _write_output, unique_path
        from mosaicdiff.videoio import copy_audio

        windows = prep["windows"]
        crop = [int(value) for value in prep["crop"]]
        crop_w = crop[2] - crop[0]
        crop_h = crop[3] - crop[1]
        gen_w = int(prep["width"])
        gen_h = int(prep["height"])
        job = _job()
        job["prompt"] = prompt
        source = Path(prep["source"])
        vsr_video = Path(prep["vsr"])
        destination = unique_path(Path(folder_paths.get_output_directory()) / f"{source.stem}_mosaicdiff.mp4")
        first = [int(index) for index in windows[0]]
        if int(images.shape[0]) != len(first):
            raise RuntimeError(f"Sampler returned {int(images.shape[0])} frames, expected {len(first)}")

        with tempfile.TemporaryDirectory(prefix="mosaicdiff-", dir=str(destination.parent)) as temp_name:
            temp = Path(temp_name)
            fitted = _rtx_fit(job, images, gen_w, gen_h, crop_w, crop_h)
            specs = [{"frame_indices": first, "out_dir": str(temp / "window_000")}]
            _save_pngs(fitted, Path(specs[0]["out_dir"]))
            del fitted
            loaded = {"torch": torch, "vae": vae}
            decode = VAEDecode()
            for number, indices in enumerate(windows[1:], start=1):
                comfy.model_management.throw_exception_if_processing_interrupted()
                indices = [int(index) for index in indices]
                _log(f"Sample {number + 1}/{len(windows)}")
                frames = _frames_from_crops(_read_crops(vsr_video, indices, crop), indices)
                conditioned = MiniMaxH3ReferenceToVideo.execute(
                    clip,
                    prompt,
                    gen_w,
                    gen_h,
                    len(indices),
                    "match",
                    vae=vae,
                    audio_vae=None,
                    ref_videos={"ref_video_0": frames},
                )
                positive, latent = conditioned[0], conditioned[1]
                _pin_reference_border(loaded, latent, frames, gen_w, gen_h, None)
                guider = BasicGuider.execute(model, positive)[0]
                with torch.no_grad():
                    sampled = SamplerCustomAdvanced.execute(noise, guider, sampler, sigmas, latent)[0]
                    decoded = decode.decode(vae, sampled)[0]
                decoded = _rtx_fit(job, decoded, gen_w, gen_h, crop_w, crop_h)
                out_dir = temp / f"window_{number:03d}"
                _save_pngs(decoded, out_dir)
                specs.append({"frame_indices": indices, "out_dir": str(out_dir)})
                del frames, positive, latent, sampled, decoded
            pastes = {}
            for spec in specs:
                out_dir = Path(spec["out_dir"])
                for order, frame_idx in enumerate(spec["frame_indices"]):
                    image = out_dir / f"{order:06d}.png"
                    if image.is_file():
                        pastes[int(frame_idx)] = image
            output_fps = H3_FPS if float(prep["fps"]) > H3_FPS + 0.05 else float(prep["fps"])
            _write_output(
                vsr_video,
                destination,
                pastes,
                tuple(crop),
                output_fps,
                float(prep["fps"]),
                int(prep["frame_count"]),
            )
        with_audio = destination.with_name(destination.stem + ".audio" + destination.suffix)
        try:
            if copy_audio(destination, source, with_audio):
                with_audio.replace(destination)
        except Exception as exc:
            with_audio.unlink(missing_ok=True)
            _log(f"Audio was not copied: {exc}")
        vsr_video.unlink(missing_ok=True)
        _log(f"Wrote {destination}")
        import cv2

        capture = cv2.VideoCapture(str(destination))
        count = max(1, int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 1))
        capture.release()
        preview = _preview_frames(destination, sorted({0, count // 3, (2 * count) // 3, count - 1}))
        return (VideoFromFile(str(destination)), preview)


NODE_CLASS_MAPPINGS = {
    "MosaicDiffPrepare": MosaicDiffPrepare,
    "MosaicDiffPinReference": MosaicDiffPinReference,
    "MosaicDiffPlace": MosaicDiffPlace,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "MosaicDiffPrepare": "MosaicDiff Prepare",
    "MosaicDiffPinReference": "MosaicDiff Pin Reference",
    "MosaicDiffPlace": "MosaicDiff Place",
}
