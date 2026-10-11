"""Read frames and write an H.264 file."""

from __future__ import annotations

import shutil
import subprocess
from fractions import Fraction
from pathlib import Path

import av
import cv2
import numpy as np


def _rate(fps: Fraction) -> str:
    fps = Fraction(fps)
    if fps.denominator == 1:
        return str(fps.numerator)
    return f"{fps.numerator}/{fps.denominator}"


def _nvenc_process(path: Path, width: int, height: int, fps: Fraction):
    """Jasna encodes with NVENC. PyAV's bundled copy cannot open that encoder."""
    if width < 256 or height < 144:
        return None
    exe = shutil.which("ffmpeg")
    if not exe:
        return None
    return subprocess.Popen(
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
            _rate(fps),
            "-i",
            "-",
            "-c:v",
            "h264_nvenc",
            "-preset",
            "p4",
            "-rc",
            "vbr",
            "-cq",
            "19",
            "-pix_fmt",
            "yuv420p",
            str(path),
        ],
        stdin=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )


def open_capture(path: Path):
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise RuntimeError(f"Could not open {path}")
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = float(capture.get(cv2.CAP_PROP_FPS) or 0) or 30.0
    count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    return capture, width, height, fps, count


class VideoWriter:
    def __init__(self, path: Path, width: int, height: int, fps: Fraction) -> None:
        self.path = path
        self.width = width - width % 2
        self.height = height - height % 2
        self._process = _nvenc_process(path, self.width, self.height, fps)
        self.container = None
        self.stream = None
        if self._process is not None:
            self.codec = "h264_nvenc"
            print("Encoding with h264_nvenc", flush=True)
            return
        self.codec = "libx264"
        self.container = av.open(str(path), mode="w")
        self.stream = self.container.add_stream("libx264", rate=fps)
        self.stream.width = self.width
        self.stream.height = self.height
        self.stream.pix_fmt = "yuv420p"
        self.stream.options = {"crf": "18", "preset": "veryfast"}
        print("Encoding with libx264", flush=True)

    def write(self, bgr: np.ndarray) -> None:
        frame = bgr[: self.height, : self.width]
        if frame.shape[0] != self.height or frame.shape[1] != self.width:
            frame = cv2.resize(frame, (self.width, self.height), interpolation=cv2.INTER_AREA)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        if self._process is not None:
            self._process.stdin.write(np.ascontiguousarray(rgb).tobytes())
            return
        video = av.VideoFrame.from_ndarray(rgb, format="rgb24")
        for packet in self.stream.encode(video):
            self.container.mux(packet)

    def close(self) -> None:
        if self._process is not None:
            self._process.stdin.close()
            error = self._process.stderr.read().decode("utf-8", errors="replace").strip()
            code = self._process.wait()
            if code != 0:
                raise RuntimeError(error or f"NVENC encoder exited with status {code}")
            return
        for packet in self.stream.encode():
            self.container.mux(packet)
        self.container.close()


def copy_audio(silent_video: Path, source: Path, destination: Path) -> bool:
    """Copy the source audio onto ``silent_video``. Returns False when there is no audio."""
    video_in = av.open(str(silent_video))
    src = av.open(str(source))
    try:
        if not src.streams.audio:
            return False
        destination.parent.mkdir(parents=True, exist_ok=True)
        out = av.open(str(destination), mode="w")
        try:
            v_in = video_in.streams.video[0]
            v_out = out.add_stream_from_template(v_in)
            a_out = out.add_stream_from_template(src.streams.audio[0])
            for packet in video_in.demux(v_in):
                if packet.dts is None:
                    continue
                packet.stream = v_out
                out.mux(packet)
            for packet in src.demux(src.streams.audio[0]):
                if packet.dts is None:
                    continue
                packet.stream = a_out
                out.mux(packet)
        finally:
            out.close()
    finally:
        video_in.close()
        src.close()
    return True
