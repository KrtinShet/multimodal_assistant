from __future__ import annotations

import numpy as np


def resample_linear(x: np.ndarray, src_rate: int, dst_rate: int) -> np.ndarray:
    """Simple linear resampler for mono PCM float32.

    Not intended as final production SRC; replace with `soxr` for highest quality.
    """
    if src_rate == dst_rate:
        return x.astype(np.float32, copy=True)
    if x.size == 0:
        return np.zeros(0, dtype=np.float32)

    ratio = dst_rate / float(src_rate)
    n_out = int(np.round(x.size * ratio))
    if n_out <= 1:
        return np.zeros(n_out, dtype=np.float32)

    # Create positions in input space
    src_pos = np.linspace(0, x.size - 1, num=x.size, dtype=np.float64)
    dst_pos = np.linspace(0, x.size - 1, num=n_out, dtype=np.float64)
    return np.interp(dst_pos, src_pos, x).astype(np.float32)


def ensure_10ms_frames(x: np.ndarray, rate_hz: int) -> list[np.ndarray]:
    """Slice audio into 10 ms frames, padding last frame with zeros if needed."""
    fp = rate_hz // 100
    frames = []
    i = 0
    while i < x.size:
        end = i + fp
        frame = np.zeros(fp, dtype=np.float32)
        chunk = x[i:end]
        frame[: chunk.size] = chunk
        frames.append(frame)
        i = end
    return frames

