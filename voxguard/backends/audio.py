"""Audio helpers: WAV I/O, resampling, G.711 mu-law, and a telephony channel simulator."""
from __future__ import annotations

import io
import wave
from math import gcd

import numpy as np


def read_wav(path_or_bytes) -> tuple[np.ndarray, int]:
    src = io.BytesIO(path_or_bytes) if isinstance(path_or_bytes, (bytes, bytearray)) else path_or_bytes
    with wave.open(src, "rb") as w:
        sr = w.getframerate()
        n = w.getnframes()
        ch = w.getnchannels()
        sw = w.getsampwidth()
        raw = w.readframes(n)
    if sw != 2:
        raise ValueError("only 16-bit PCM WAV is supported")
    x = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
    if ch > 1:
        x = x.reshape(-1, ch).mean(axis=1)
    return x, sr


def wav_bytes(x: np.ndarray, sr: int) -> bytes:
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sr)
        w.writeframes(to_pcm16(x))
    return buf.getvalue()


def write_wav(path, x: np.ndarray, sr: int) -> None:
    with open(path, "wb") as f:
        f.write(wav_bytes(x, sr))


def to_pcm16(x: np.ndarray) -> bytes:
    return (np.clip(x, -1, 1) * 32767).astype("<i2").tobytes()


def resample(x: np.ndarray, sr_in: int, sr_out: int) -> np.ndarray:
    if sr_in == sr_out:
        return x
    from scipy.signal import resample_poly
    g = gcd(sr_in, sr_out)
    return resample_poly(x, sr_out // g, sr_in // g).astype(np.float32)


def mulaw_encode(x: np.ndarray, mu: int = 255) -> np.ndarray:
    x = np.clip(x, -1, 1)
    y = np.sign(x) * np.log1p(mu * np.abs(x)) / np.log1p(mu)
    return ((y + 1) / 2 * mu + 0.5).astype(np.uint8)


def mulaw_decode(q: np.ndarray, mu: int = 255) -> np.ndarray:
    y = 2 * (q.astype(np.float32) / mu) - 1
    return (np.sign(y) * ((1 + mu) ** np.abs(y) - 1) / mu).astype(np.float32)


def telephony_channel(
    x: np.ndarray,
    sr: int,
    snr_db: float | None = 20.0,
    packet_loss: float = 0.0,
    seed: int = 0,
) -> tuple[np.ndarray, int]:
    """Simulate a PSTN call leg: 300-3400 Hz band, 8 kHz, mu-law, noise, packet loss.

    Returns audio at 8 kHz. Agents that need 16 or 24 kHz get it upsampled, which
    is what happens in production: the lost bandwidth does not come back.
    """
    from scipy.signal import butter, sosfilt
    rng = np.random.default_rng(seed)
    y = resample(x, sr, 8000)
    sos = butter(4, [300, 3400], btype="bandpass", fs=8000, output="sos")
    y = sosfilt(sos, y).astype(np.float32)
    if snr_db is not None:
        p = np.mean(y ** 2) + 1e-12
        noise = rng.normal(0, np.sqrt(p / 10 ** (snr_db / 10)), size=y.shape)
        y = y + noise.astype(np.float32)
    peak = np.abs(y).max() + 1e-9
    y = y / peak * 0.9
    y = mulaw_decode(mulaw_encode(y))
    if packet_loss > 0:
        frame = 160  # 20 ms at 8 kHz
        for i in range(0, len(y), frame):
            if rng.random() < packet_loss:
                y[i:i + frame] = 0
    return y, 8000
