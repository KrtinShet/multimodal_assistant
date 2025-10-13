from __future__ import annotations

import ctypes
import os
import platform
from typing import Optional

import numpy as np

from multimodal_assistant.processors.aec_core import IAEC
from multimodal_assistant.utils.logger import setup_logger


class WebRtcApmAEC(IAEC):
    """ctypes wrapper around a small C++ shim for WebRTC AudioProcessing.

    This class expects a shared library built from native/webrtc_apm/apm_shim.cc
    and available as libapm_shim.(dylib|so|dll) under native/webrtc_apm/dist.
    """

    def __init__(
        self,
        sample_rate_hz: int = 16000,
        frames_per_buffer: int = 160,  # 10 ms at 16 kHz
        enable_aec3: bool = True,
        enable_hpf: bool = True,
        enable_ns: bool = True,
        enable_agc: bool = True,
        stream_delay_ms: int = 60,
        lib_path: Optional[str] = None,
    ) -> None:
        self.logger = setup_logger("multimodal_assistant.aec.webrtc_apm")
        self.rate = sample_rate_hz
        self.fp = frames_per_buffer
        self.stream_delay_ms = stream_delay_ms
        self._handle = None
        self._lib = None

        lib_name = {
            "Darwin": "libapm_shim.dylib",
            "Linux": "libapm_shim.so",
            "Windows": "apm_shim.dll",
        }.get(platform.system(), "libapm_shim.so")
        # Allow env override first
        env_path = os.environ.get("WEBRTC_APM_LIB") or os.environ.get("APM_SHIM_PATH")
        candidates = []
        if lib_path:
            candidates.append(lib_path)
        if env_path:
            candidates.append(env_path)
        # Try repo root/native path
        repo_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
        candidates.append(os.path.join(repo_root, "native", "webrtc_apm", "dist", lib_name))
        # Try src/native path if present
        src_root = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
        candidates.append(os.path.join(src_root, "native", "webrtc_apm", "dist", lib_name))

        lib_file = None
        for c in candidates:
            if c and os.path.exists(c):
                lib_file = c
                break
        if not lib_file:
            raise RuntimeError(
                "WebRTC APM shim not found. Set WEBRTC_APM_LIB or build native/webrtc_apm (see BUILD.md)."
            )

        self._lib = ctypes.CDLL(lib_file)
        # Signatures
        self._lib.apm_create.restype = ctypes.c_void_p
        self._lib.apm_create.argtypes = [
            ctypes.c_int,  # sample_rate_hz
            ctypes.c_int,  # num_channels
            ctypes.c_int,  # enable_aec3
            ctypes.c_int,  # enable_hpf
            ctypes.c_int,  # enable_ns
            ctypes.c_int,  # enable_agc
        ]
        self._lib.apm_destroy.argtypes = [ctypes.c_void_p]
        self._lib.apm_set_stream_delay_ms.argtypes = [ctypes.c_void_p, ctypes.c_int]
        self._lib.apm_process_reverse.argtypes = [
            ctypes.c_void_p,
            ctypes.POINTER(ctypes.c_float),
            ctypes.c_int,
        ]
        self._lib.apm_process_near.argtypes = [
            ctypes.c_void_p,
            ctypes.POINTER(ctypes.c_float),
            ctypes.POINTER(ctypes.c_float),
            ctypes.c_int,
        ]

        self._handle = self._lib.apm_create(
            int(self.rate), 1, int(enable_aec3), int(enable_hpf), int(enable_ns), int(enable_agc)
        )
        if not self._handle:
            raise RuntimeError("Failed to create WebRTC APM instance")
        self._lib.apm_set_stream_delay_ms(self._handle, int(self.stream_delay_ms))

        self._in_buf = np.zeros(self.fp, dtype=np.float32)
        self._out_buf = np.zeros(self.fp, dtype=np.float32)

        self.logger.info(
            f"WebRTC APM ready (rate={self.rate}, fp={self.fp}, delay={self.stream_delay_ms}ms)"
        )

    def process_near(self, frame: np.ndarray) -> np.ndarray:
        if frame.dtype != np.float32:
            frame = frame.astype(np.float32)

        if frame.size != self.fp:
            # pad/trim to frame size
            tmp = np.zeros(self.fp, dtype=np.float32)
            n = min(self.fp, frame.size)
            tmp[:n] = frame[:n]
            frame = tmp

        inp = frame.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
        outp = self._out_buf.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
        self._lib.apm_process_near(self._handle, inp, outp, int(self.fp))
        return self._out_buf.copy()

    def push_reverse(self, frame: np.ndarray) -> None:
        if frame.dtype != np.float32:
            frame = frame.astype(np.float32)
        if frame.size != self.fp:
            tmp = np.zeros(self.fp, dtype=np.float32)
            n = min(self.fp, frame.size)
            tmp[:n] = frame[:n]
            frame = tmp
        ptr = frame.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
        self._lib.apm_process_reverse(self._handle, ptr, int(self.fp))

    def shutdown(self) -> None:
        if self._handle and self._lib:
            self._lib.apm_destroy(self._handle)
            self._handle = None
