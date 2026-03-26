"""
Core synthesis: constants, waveform primitives, sound type generators,
generate_sound / mix_sounds, and write_wav.
"""

import numpy as np
import wave
import random

SAMPLE_RATE = 44100

# --- Musical tuning helpers ---

NOTE_SEMITONES = {
    'C': 0, 'C#': 1, 'D': 2, 'D#': 3, 'E': 4,
    'F': 5, 'F#': 6, 'G': 7, 'G#': 8, 'A': 9, 'A#': 10, 'B': 11,
}

def note_to_freq(note, octave):
    """Return Hz for a note name ('C', 'C#', …) and octave (2–5)."""
    semitones = (octave - 4) * 12 + NOTE_SEMITONES[note]
    return 261.63 * (2 ** (semitones / 12))


# --- Envelope Helpers ---

def apply_adsr(signal, sr, attack=0.01, decay=0.05, sustain=0.7, release=0.1):
    """Apply an ADSR amplitude envelope to a signal."""
    n = len(signal)
    envelope = np.ones(n)

    a = int(attack * sr)
    d = int(decay * sr)
    r = int(release * sr)
    s_end = n - r

    if a > 0:
        envelope[:a] = np.linspace(0, 1, a)
    if d > 0 and a + d < n:
        envelope[a:a + d] = np.linspace(1, sustain, d)
    if a + d < s_end:
        envelope[a + d:s_end] = sustain
    if r > 0:
        envelope[s_end:] = np.linspace(sustain, 0, n - s_end)

    return signal * envelope


# --- Synthesis Functions ---

def sine_tone(freq, duration, sr=SAMPLE_RATE):
    t = np.linspace(0, duration, int(sr * duration), endpoint=False)
    return np.sin(2 * np.pi * freq * t)


def square_wave(freq, duration, sr=SAMPLE_RATE):
    t = np.linspace(0, duration, int(sr * duration), endpoint=False)
    return np.sign(np.sin(2 * np.pi * freq * t))


def sawtooth_wave(freq, duration, sr=SAMPLE_RATE):
    t = np.linspace(0, duration, int(sr * duration), endpoint=False)
    return 2 * (t * freq - np.floor(0.5 + t * freq))


def white_noise(duration, sr=SAMPLE_RATE):
    return np.random.uniform(-1, 1, int(sr * duration))


def fm_synthesis(carrier_freq, mod_freq, mod_index, duration, sr=SAMPLE_RATE):
    """Frequency Modulation synthesis."""
    t = np.linspace(0, duration, int(sr * duration), endpoint=False)
    modulator = mod_index * np.sin(2 * np.pi * mod_freq * t)
    return np.sin(2 * np.pi * carrier_freq * t + modulator)


def additive_harmonics(base_freq, duration, num_harmonics=6, sr=SAMPLE_RATE):
    """Additive synthesis with random harmonic weights."""
    t = np.linspace(0, duration, int(sr * duration), endpoint=False)
    signal = np.zeros(len(t))
    for h in range(1, num_harmonics + 1):
        weight = random.uniform(0.1, 1.0) / h
        signal += weight * np.sin(2 * np.pi * base_freq * h * t)
    return signal


def resonant_noise(cutoff_freq, duration, sr=SAMPLE_RATE):
    """Bandpass noise approximated by summing closely-spaced sine waves with random phases."""
    t = np.linspace(0, duration, int(sr * duration), endpoint=False)
    signal = np.zeros(len(t))
    bandwidth = min(300, cutoff_freq * 0.3)
    num_partials = 24
    for _ in range(num_partials):
        freq = cutoff_freq + random.uniform(-bandwidth / 2, bandwidth / 2)
        phase = random.uniform(0, 2 * np.pi)
        signal += np.sin(2 * np.pi * freq * t + phase)
    return signal / num_partials


def plucked_string(freq, duration, sr=SAMPLE_RATE):
    """Karplus-Strong string synthesis."""
    delay = int(sr / freq)
    noise_len = delay
    buf = np.random.uniform(-1, 1, noise_len)
    output = np.zeros(int(sr * duration))
    for i in range(len(output)):
        output[i] = buf[i % delay]
        buf[i % delay] = 0.996 * 0.5 * (buf[i % delay] + buf[(i + 1) % delay])
    return output


def pitched_percussion(freq, duration, sr=SAMPLE_RATE):
    """Sine + noise blend for drum-like tones."""
    t = np.linspace(0, duration, int(sr * duration), endpoint=False)
    pitch_env = np.exp(-t * random.uniform(3, 15))
    tone = np.sin(2 * np.pi * freq * t * pitch_env)
    noise = white_noise(duration, sr) * 0.3
    return tone + noise


# --- Sound Type Generators ---

SOUND_TYPES = [
    "pitched_pluck",
    "sine",
    "square",
    "saw",
    "triangle",
]


def gen_pitched_pluck(sr, root_freq=None, bpm=None):
    """Mallet/kalimba-style pluck — short impact, exponential harmonic decay."""
    freq = root_freq if root_freq else random.uniform(300, 1800)
    duration = random.uniform(0.4, 1.4)
    n = int(sr * duration)
    t = np.linspace(0, duration, n, endpoint=False)

    sig = np.zeros(n)
    decay_base = random.uniform(6.0, 12.0)
    harmonic_weights = [1.0, 0.55, 0.28, 0.14, 0.07]
    for i, weight in enumerate(harmonic_weights):
        h = i + 1
        hfreq = freq * h
        if hfreq >= sr / 2:
            break
        env = np.exp(-decay_base * h * t)
        sig += weight * np.sin(2 * np.pi * hfreq * t) * env

    attack = min(int(0.002 * sr), n)
    sig[:attack] *= np.linspace(0, 1, attack)
    return sig


def gen_sine(sr, root_freq=None, bpm=None):
    """Pure sine tone with ADSR envelope."""
    freq = root_freq if root_freq else random.uniform(100, 800)
    duration = random.uniform(0.5, 2.0)
    n = int(sr * duration)
    t = np.linspace(0, duration, n, endpoint=False)
    sig = np.sin(2 * np.pi * freq * t)
    return apply_adsr(sig, sr, attack=0.01, decay=0.05, sustain=0.8, release=0.15)


def gen_square(sr, root_freq=None, bpm=None):
    """Square wave with ADSR envelope."""
    freq = root_freq if root_freq else random.uniform(100, 800)
    duration = random.uniform(0.5, 2.0)
    n = int(sr * duration)
    t = np.linspace(0, duration, n, endpoint=False)
    sig = np.sign(np.sin(2 * np.pi * freq * t))
    return apply_adsr(sig, sr, attack=0.01, decay=0.05, sustain=0.8, release=0.15)


def gen_saw(sr, root_freq=None, bpm=None):
    """Sawtooth wave with ADSR envelope."""
    freq = root_freq if root_freq else random.uniform(100, 800)
    duration = random.uniform(0.5, 2.0)
    n = int(sr * duration)
    t = np.linspace(0, duration, n, endpoint=False)
    sig = 2 * (t * freq - np.floor(0.5 + t * freq))
    return apply_adsr(sig, sr, attack=0.01, decay=0.05, sustain=0.8, release=0.15)


def gen_triangle(sr, root_freq=None, bpm=None):
    """Triangle wave with ADSR envelope."""
    freq = root_freq if root_freq else random.uniform(100, 800)
    duration = random.uniform(0.5, 2.0)
    n = int(sr * duration)
    t = np.linspace(0, duration, n, endpoint=False)
    sig = 2 * np.abs(2 * (t * freq - np.floor(0.5 + t * freq))) - 1
    return apply_adsr(sig, sr, attack=0.01, decay=0.05, sustain=0.8, release=0.15)


GENERATORS = {
    "pitched_pluck": gen_pitched_pluck,
    "sine":          gen_sine,
    "square":        gen_square,
    "saw":           gen_saw,
    "triangle":      gen_triangle,
}


# --- Main generation helpers ---

def _fit_to_duration(signal, duration, sr):
    """Trim or loop a signal to exactly `duration` seconds."""
    target = int(duration * sr)
    if len(signal) >= target:
        return signal[:target]
    repeats = -(-target // len(signal))
    return np.tile(signal, repeats)[:target]


def generate_sound(sound_type=None, sr=SAMPLE_RATE, duration=None,
                   root_freq=None, bpm=None):
    if sound_type is None:
        sound_type = random.choice(list(GENERATORS.keys()))
    signal = GENERATORS[sound_type](sr, root_freq=root_freq, bpm=bpm)

    if duration is not None:
        signal = _fit_to_duration(signal, duration, sr)

    peak = np.max(np.abs(signal))
    if peak > 0:
        signal = signal / peak

    audio = (signal * 32767).astype(np.int16)
    return audio, sound_type


def mix_sounds(type_a, type_b, blend=0.5, duration=None, root_freq=None, bpm=None):
    """Blend two sound types together. blend=1.0 is 100% type_a, 0.0 is 100% type_b."""
    audio_a, _ = generate_sound(type_a, duration=duration, root_freq=root_freq, bpm=bpm)
    audio_b, _ = generate_sound(type_b, duration=duration, root_freq=root_freq, bpm=bpm)

    a = audio_a.astype(np.float32) / 32767.0
    b = audio_b.astype(np.float32) / 32767.0
    length = min(len(a), len(b))
    mixed = blend * a[:length] + (1.0 - blend) * b[:length]

    peak = np.max(np.abs(mixed))
    if peak > 0:
        mixed /= peak

    audio = (mixed * 32767).astype(np.int16)
    label = f"mix_{type_a}_x_{type_b}"
    return audio, label


def write_wav(filepath, audio, sr=SAMPLE_RATE):
    with wave.open(filepath, "w") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sr)
        wf.writeframes(audio.tobytes())
