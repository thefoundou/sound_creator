"""
Audio post-processing: weirdify, resample chop, live effects, pleasantize.
"""

import numpy as np
import random

from generators import SAMPLE_RATE


def apply_weirdify(audio_int16, crush=0, ring=0, warp=0, stutter=0, glitch=0, sr=SAMPLE_RATE, bpm=None):
    """
    Chain of warp effects applied to a 16-bit PCM audio array.
    Each parameter is 0–100 (0 = off, 100 = full effect).
    """
    signal = audio_int16.astype(np.float32) / 32767.0

    # ── Bit Crush ─────────────────────────────────────────────────────────────
    if crush > 0:
        bits = max(1, 16 - int(crush / 100 * 14))
        steps = float(2 ** bits)
        signal = np.round(signal * steps) / steps

    # ── Ring Modulation ───────────────────────────────────────────────────────
    if ring > 0:
        mod_freq = 1.0 + (ring / 100.0) ** 2 * 999.0
        t = np.linspace(0, len(signal) / sr, len(signal), endpoint=False)
        signal = signal * np.sin(2 * np.pi * mod_freq * t)

    # ── Warp / Waveshaper ─────────────────────────────────────────────────────
    if warp > 0:
        drive = 1.0 + (warp / 100.0) ** 2 * 29.0
        signal = np.tanh(signal * drive) / np.tanh(np.array(drive))

    # ── Stutter ───────────────────────────────────────────────────────────────
    if stutter > 0:
        if bpm:
            beat_samples = int(sr * 60.0 / bpm)
            subdivision  = 4 * (2 ** int(stutter / 100 * 3))
            seg_len = max(1, beat_samples // subdivision)
        else:
            seg_ms  = max(5, int(300 - (stutter / 100.0) * 290))
            seg_len = max(1, int(sr * seg_ms / 1000))
        out = np.zeros_like(signal)
        i = 0
        while i < len(signal):
            end   = min(i + seg_len, len(signal))
            chunk = signal[i:end].copy()
            if random.random() < stutter / 150.0:
                chunk = chunk[::-1]
            out[i:end] = chunk
            i += seg_len
        signal = out

    # ── Glitch ────────────────────────────────────────────────────────────────
    if glitch > 0:
        num_ops  = max(1, int((glitch / 100.0) ** 1.5 * 40))
        max_size = max(2, int(sr * 0.06))
        for _ in range(num_ops):
            size = random.randint(64, max_size)
            src  = random.randint(0, max(0, len(signal) - size))
            dst  = random.randint(0, max(0, len(signal) - size))
            signal[dst:dst + size] = signal[src:src + size]

    peak = np.max(np.abs(signal))
    if peak > 0:
        signal /= peak

    return (signal * 32767).astype(np.int16)


def resample_chop(audio_int16, chop_length=0.5, chop_reuse=0.0, sr=SAMPLE_RATE):
    """
    Break audio into chops and randomly reassemble them.

    chop_length  seconds per chop (0.02 – 2.0)
    chop_reuse   0.0–1.0 probability that any slot re-uses a previous chop
    """
    signal = audio_int16.astype(np.float32) / 32767.0
    n      = len(signal)
    chop_n = max(64, int(chop_length * sr))

    chops = []
    for i in range(0, n, chop_n):
        chunk = signal[i:i + chop_n]
        if len(chunk) > 0:
            chops.append(chunk)
    if not chops:
        return audio_int16

    n_slots = len(chops)
    indices = list(range(n_slots))
    random.shuffle(indices)

    assembled = []
    fade = min(int(0.005 * sr), chop_n // 4)
    for slot_i in range(n_slots):
        if random.random() < chop_reuse and len(chops) > 1:
            idx = random.randint(0, len(chops) - 1)
        else:
            idx = indices[slot_i % len(indices)]
        chunk = chops[idx].copy()
        fn = min(fade, len(chunk) // 4)
        if fn > 0:
            ramp = np.linspace(0.0, 1.0, fn, dtype=np.float32)
            chunk[:fn]  *= ramp
            chunk[-fn:] *= ramp[::-1]
        assembled.append(chunk)

    result = np.concatenate(assembled)
    peak   = np.max(np.abs(result))
    if peak > 0:
        result = result / peak * 0.95
    return (result * 32767).astype(np.int16)


def apply_live_effects(audio_int16, reverb=0, delay=0, filter_amt=0, sr=SAMPLE_RATE):
    """
    Live preview effects: reverb, delay, low-pass filter.
    Each parameter is 0–100.
    """
    signal = audio_int16.astype(np.float32) / 32767.0
    n      = len(signal)

    # ── Low-pass filter (via FFT) ─────────────────────────────────────────────
    if filter_amt > 0:
        spectrum = np.fft.rfft(signal)
        freqs    = np.fft.rfftfreq(n, d=1.0 / sr)
        cutoff   = 18000.0 * (1.0 - filter_amt / 100.0) ** 2 + 200.0
        gain = 1.0 / (1.0 + (freqs / max(cutoff, 1.0)) ** 4)
        spectrum *= gain
        signal = np.fft.irfft(spectrum, n=n).astype(np.float32)

    # ── Delay ─────────────────────────────────────────────────────────────────
    if delay > 0:
        delay_time = 0.08 + (delay / 100.0) * 0.45
        feedback   = 0.15 + (delay / 100.0) * 0.55
        wet_mix    = 0.15 + (delay / 100.0) * 0.45
        delay_samp = int(delay_time * sr)
        wet        = np.zeros(n + delay_samp * 4, dtype=np.float32)
        wet[:n]    = signal
        for tap in range(1, 5):
            offset = delay_samp * tap
            gain   = feedback ** tap
            if offset < len(wet):
                end = min(offset + n, len(wet))
                wet[offset:end] += signal[:end - offset] * gain
        wet = wet[:n]
        signal = signal * (1.0 - wet_mix) + wet * wet_mix

    # ── Reverb (FFT convolution with exponential impulse) ─────────────────────
    if reverb > 0:
        decay_time = 0.3 + (reverb / 100.0) * 2.5
        wet_mix    = 0.10 + (reverb / 100.0) * 0.55
        ir_len     = int(decay_time * sr)
        ir         = np.random.randn(ir_len).astype(np.float32)
        ir        *= np.exp(-np.linspace(0, 6.0, ir_len))
        ir        /= np.sqrt(np.sum(ir ** 2) + 1e-9)
        fft_n   = 1
        while fft_n < n + ir_len:
            fft_n <<= 1
        wet     = np.fft.irfft(np.fft.rfft(signal, fft_n) *
                               np.fft.rfft(ir, fft_n))[:n].astype(np.float32)
        signal  = signal * (1.0 - wet_mix) + wet * wet_mix

    peak = np.max(np.abs(signal))
    if peak > 0:
        signal = signal / peak * 0.92

    return (signal * 32767).astype(np.int16)


def pleasantize(audio_int16, sr=SAMPLE_RATE):
    """
    Make a sound more pleasant without changing its character:
      1. Gentle fade-in / fade-out
      2. Low-cut below 60 Hz
      3. High-shelf rolloff 9–16 kHz
      4. Very light soft saturation
      5. Normalise to 90% peak
    """
    signal = audio_int16.astype(np.float32) / 32767.0
    n      = len(signal)

    fade = min(int(sr * 0.02), n // 4)
    t    = np.linspace(0.0, 1.0, fade)
    signal[:fade]  *= t ** 0.5
    signal[-fade:] *= t[::-1] ** 0.5

    spectrum = np.fft.rfft(signal)
    freqs    = np.fft.rfftfreq(n, d=1.0 / sr)

    gain = np.ones(len(freqs), dtype=np.float32)

    gain[freqs < 60.0] = np.interp(
        freqs[freqs < 60.0], [0.0, 60.0], [0.05, 1.0])

    shelf_lo, shelf_hi = 9000.0, 16000.0
    hi_mask = (freqs >= shelf_lo) & (freqs <= shelf_hi)
    gain[hi_mask] = np.interp(freqs[hi_mask],
                               [shelf_lo, shelf_hi], [1.0, 0.35])
    gain[freqs > shelf_hi] = 0.35

    spectrum *= gain
    signal    = np.fft.irfft(spectrum, n=n).astype(np.float32)

    drive  = 1.4
    signal = np.tanh(signal * drive) / float(np.tanh(drive))

    peak = np.max(np.abs(signal))
    if peak > 0:
        signal = signal / peak * 0.90

    return (signal * 32767).astype(np.int16)
