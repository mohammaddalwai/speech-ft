"""Uploaded bytes -> 16 kHz mono float32, decoded in memory (nothing is written to disk).

PyAV (bundled with faster-whisper) opens any common format, including what browsers record (webm/opus, mp4/aac,
ogg). The audio is kept at its own sample rate, averaged to mono and resampled with soxr: exactly what src/data.py
does for every evaluation clip, so serving does not add a second resampler.
"""
import io

import av
import numpy as np
import soxr

SR = 16000


class AudioError(ValueError):
    """The bytes are not decodable audio."""


class TooLong(ValueError):
    """The audio is longer than allowed."""


def decode(data: bytes, max_seconds: float) -> np.ndarray:
    if not data:
        raise AudioError("empty upload")
    chunks, rate = [], None
    try:
        with av.open(io.BytesIO(data), mode="r") as container:
            stream = next((s for s in container.streams if s.type == "audio"), None)
            if stream is None:
                raise AudioError("no audio stream in the file")
            to_float = av.AudioResampler(format="fltp")  # sample format only: keeps channels and sample rate
            n = 0
            for frame in container.decode(stream):
                for f in to_float.resample(frame):
                    rate = rate or f.sample_rate
                    chunks.append(f.to_ndarray())
                    n += f.samples
                if rate and n > (max_seconds + 1) * rate:  # stop early: a small file can hold hours of audio
                    raise TooLong(f"audio is longer than {max_seconds:g} s")
            for f in to_float.resample(None):
                chunks.append(f.to_ndarray())
    except (TooLong, AudioError):
        raise
    except Exception as e:  # PyAV raises many error types for broken or non-audio input
        raise AudioError(f"could not decode audio ({type(e).__name__})") from None
    if not chunks or rate is None:
        raise AudioError("no audio samples in the file")
    wav = np.concatenate(chunks, axis=1).mean(axis=0)  # (channels, samples) -> mono, as src/data.py
    if rate != SR:
        wav = soxr.resample(wav, rate, SR)
    wav = np.ascontiguousarray(wav, dtype=np.float32)
    if len(wav) > max_seconds * SR:
        raise TooLong(f"audio is {len(wav) / SR:.1f} s; the limit is {max_seconds:g} s")
    if len(wav) == 0:
        raise AudioError("no audio samples in the file")
    return wav
