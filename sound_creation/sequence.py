"""
Melodic sequence generation: generate_sequence and generate_random_sequence.
"""

import math
import numpy as np
import random

from generators import GENERATORS, generate_sound, SAMPLE_RATE, note_to_freq


def generate_sequence(notes_list, sound_type=None, sr=SAMPLE_RATE, bpm=None,
                      note_duration=None, swing=0.0, velocity=0.8,
                      velocity_variance=0.15, note_variance=0.0,
                      phrase_contour=0.0, accent_strength=0.0):
    """
    Generate a sequence of notes with swing, velocity, and duration variance.

    notes_list        List of (freq_hz, duration_multiplier) tuples.
                      freq_hz=None means a rest (silence for that slot).
                      duration_multiplier scales the base note_duration
                      (1.0=normal, 1.5=dotted, 2.0=double).
                      Plain float/int entries are treated as (freq, 1.0).
    swing             0.0 = straight, 1.0 = heavy swing (odd notes pushed late)
    velocity          0.0–1.0 base amplitude
    velocity_variance 0.0–1.0 per-note velocity randomness (0=uniform, 1=±40 %)
    note_variance     0.0–1.0 per-note duration spread (0=exact, 1=±50 %)
    phrase_contour    0.0–1.0 phrase-aware velocity shaping + phrase-final
                      lengthening (0=flat/off, 1=full arc + stretch)
    accent_strength   0.0–1.0 metric accent on strong beats (assumes 4/4)
    """
    if not notes_list:
        return np.array([], dtype=np.int16), "empty_sequence"

    if sound_type is None:
        sound_type = random.choice(list(GENERATORS.keys()))

    def _base_dur():
        if note_duration is not None:
            return note_duration
        if bpm is not None:
            beat = 60.0 / bpm
            candidates = [beat / 4, beat / 3, beat / 2, beat]
            candidates = [d for d in candidates if 0.04 <= d <= 1.0]
            return random.choice(candidates) if candidates else 0.25
        return random.uniform(0.15, 0.35)

    base_dur    = _base_dur()
    swing_delay = swing * 0.35 * base_dur
    fade_n      = min(int(0.008 * sr), 256)

    # ── Phrase boundaries ─────────────────────────────────────────────────
    # Split the note list into phrases of 4–8 notes for contour/lengthening.
    total = len(notes_list)
    phrase_breaks = set()       # indices that START a new phrase
    if phrase_contour > 0 and total > 0:
        pos = 0
        while pos < total:
            phrase_breaks.add(pos)
            pos += random.randint(4, 8)
    # Build a list of (phrase_start, phrase_len) spans
    _breaks = sorted(phrase_breaks) if phrase_breaks else [0]
    phrase_spans = []
    for bi, start in enumerate(_breaks):
        end = _breaks[bi + 1] if bi + 1 < len(_breaks) else total
        phrase_spans.append((start, end - start))

    def _phrase_of(idx):
        """Return (position_in_phrase, phrase_length) for note *idx*."""
        for start, length in phrase_spans:
            if start <= idx < start + length:
                return idx - start, length
        return 0, 1

    # ── Metric accent weights (4/4 assumed) ───────────────────────────────
    # Beat 1 strongest, beat 3 secondary, beats 2 & 4 weakest.
    _accent_pattern = [1.0, 0.7, 0.85, 0.7]
    # Cumulative duration tracks beat position within a bar.
    cum_dur = 0.0
    bar_len = (4 * 60.0 / bpm) if bpm else None

    segments = []
    for i, entry in enumerate(notes_list):
        if isinstance(entry, (tuple, list)):
            freq, dur_mult = entry[0], entry[1]
        else:
            freq, dur_mult = entry, 1.0

        dur = max(0.04, base_dur * dur_mult
                  * (1.0 + note_variance * random.uniform(-0.5, 0.5)))

        # ── Phrase-final lengthening ──────────────────────────────────────
        if phrase_contour > 0:
            pos_in_phrase, phr_len = _phrase_of(i)
            if phr_len > 1 and pos_in_phrase == phr_len - 1:
                # Last note of phrase: stretch up to +25 %
                dur *= 1.0 + 0.25 * phrase_contour
            elif phr_len > 2 and pos_in_phrase == phr_len - 2:
                # Second-to-last: mild stretch up to +10 %
                dur *= 1.0 + 0.10 * phrase_contour

        if swing_delay > 0.001 and (i % 2 == 1):
            segments.append(np.zeros(int(swing_delay * sr), dtype=np.int16))

        if freq is None:
            cum_dur += dur
            segments.append(np.zeros(int(dur * sr), dtype=np.int16))
            continue

        note_audio, _ = generate_sound(sound_type, sr=sr, duration=dur,
                                       root_freq=freq, bpm=bpm)

        # ── Velocity: random spread ───────────────────────────────────────
        vel_spread = velocity_variance * 0.40
        vel = velocity * (1.0 + random.uniform(-vel_spread, vel_spread))

        # ── Velocity: phrase contour (sine arc) ──────────────────────────
        if phrase_contour > 0:
            pos_in_phrase, phr_len = _phrase_of(i)
            if phr_len > 1:
                # Sine arc: 0 at edges → 1 at centre
                arc = math.sin(math.pi * pos_in_phrase / (phr_len - 1))
                # Blend between flat (1.0) and shaped arc
                contour_scale = 1.0 - phrase_contour * 0.35 * (1.0 - arc)
                vel *= contour_scale

        # ── Velocity: metric accent ──────────────────────────────────────
        if accent_strength > 0 and bar_len and bar_len > 0:
            beat_pos = (cum_dur % bar_len) / (bar_len / 4.0)
            beat_idx = int(beat_pos) % 4
            accent_w = _accent_pattern[beat_idx]
            # Blend between flat (1.0) and accented
            vel *= 1.0 - accent_strength * (1.0 - accent_w)

        vel = np.clip(vel, 0.02, 1.0)
        sig = note_audio.astype(np.float32) * vel

        n = len(sig)
        fn = min(fade_n, n // 4)
        if fn > 0:
            ramp = np.linspace(0.0, 1.0, fn, dtype=np.float32)
            sig[:fn]  *= ramp
            sig[-fn:] *= ramp[::-1]

        segments.append(np.clip(sig, -32767, 32767).astype(np.int16))
        cum_dur += dur

    audio = np.concatenate(segments)
    label = f"sequence_{sound_type}_{len(notes_list)}"
    return audio, label


def _pick_chord(scale, num_degrees=4):
    """Pick a chord of `num_degrees` notes from scale degree indices.

    Builds diatonic chords by stacking thirds (every other scale degree).
    Returns a list of scale-degree indices into `scale`.
    """
    n = len(scale)
    root_deg = random.choice(range(n))
    # Stack thirds: root, 3rd, 5th, 7th, 9th...
    chord_degs = []
    for i in range(num_degrees):
        chord_degs.append((root_deg + i * 2) % n)
    return chord_degs


def generate_random_sequence(root_note, octave, scale_type, num_notes, sound_type=None,
                              sr=SAMPLE_RATE, bpm=None, note_duration=None,
                              swing=0.0, velocity=0.8, velocity_variance=0.15,
                              note_variance=0.0, phrase_contour=0.0,
                              accent_strength=0.0):
    """
    Generate a melodic sequence constrained to a diatonic chord:
      1. Pick a random 4+ note chord (stacked thirds) from the scale
      2. Only use chord tones for the melody
      3. Stepwise motion through chord tones with octave variety
      4. Random double/dotted/rest variations for rhythmic interest
    """
    MAJOR = [0, 2, 4, 5, 7, 9, 11]
    MINOR = [0, 2, 3, 5, 7, 8, 10]
    scale = MAJOR if scale_type.lower() == 'major' else MINOR

    # Pick a 4-note chord (stacked thirds from a random root degree)
    chord_size = random.choice([4, 4, 4, 5])
    chord_degs = _pick_chord(scale, chord_size)
    chord_semitones = [scale[d] for d in chord_degs]

    root_freq_base = note_to_freq(root_note, octave)
    nc = len(chord_semitones)

    # Start on root of chord, biased to base octave
    chord_idx = 0
    cur_oct   = max(3, min(5, octave))

    notes = []
    for i in range(num_notes):
        sem  = chord_semitones[chord_idx] + (cur_oct - octave) * 12
        freq = root_freq_base * (2 ** (sem / 12))

        # Duration variation
        r_dur = random.random()
        if r_dur < 0.12:
            notes.append((None, 1.0))
        elif r_dur < 0.27:
            notes.append((freq, 2.0))
        elif r_dur < 0.40:
            notes.append((freq, 1.5))
        else:
            notes.append((freq, 1.0))

        # Move to next chord tone
        r = random.random()
        if r < 0.10:
            pass  # repeat same tone
        elif r < 0.65:
            # Step: ±1 chord tone
            step = random.choice([-1, 1])
            new_idx = chord_idx + step
            if new_idx < 0:
                if cur_oct > 3:
                    cur_oct -= 1
                    new_idx = nc - 1
                else:
                    new_idx = 0
            elif new_idx >= nc:
                if cur_oct < 5:
                    cur_oct += 1
                    new_idx = 0
                else:
                    new_idx = nc - 1
            chord_idx = new_idx
        elif r < 0.85:
            # Skip: ±2 chord tones
            step = random.choice([-2, 2])
            new_idx = chord_idx + step
            if new_idx < 0:
                if cur_oct > 3:
                    cur_oct -= 1
                    new_idx = nc + new_idx
                else:
                    new_idx = 0
            elif new_idx >= nc:
                if cur_oct < 5:
                    cur_oct += 1
                    new_idx = new_idx - nc
                else:
                    new_idx = nc - 1
            chord_idx = new_idx
        else:
            # Octave jump on same chord tone
            cur_oct = max(3, min(5, cur_oct + random.choice([-1, 1])))

    notes = notes[:num_notes]

    audio, _ = generate_sequence(notes, sound_type=sound_type, sr=sr, bpm=bpm,
                                 note_duration=note_duration,
                                 swing=swing, velocity=velocity,
                                 velocity_variance=velocity_variance,
                                 note_variance=note_variance,
                                 phrase_contour=phrase_contour,
                                 accent_strength=accent_strength)
    chord_name = f"deg{chord_degs[0]}"
    label = f"seq_{scale_type}_{root_note}{octave}_{chord_name}_{num_notes}"
    return audio, label
