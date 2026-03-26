"""
Melodic sequence generation: generate_sequence and generate_random_sequence.
"""

import numpy as np
import random

from generators import GENERATORS, generate_sound, SAMPLE_RATE, note_to_freq


def generate_sequence(notes_list, sound_type=None, sr=SAMPLE_RATE, bpm=None,
                      note_duration=None, swing=0.0, velocity=0.8,
                      velocity_variance=0.15, note_variance=0.0):
    if not notes_list:
        return np.array([], dtype=np.int16), "empty_sequence"

    if sound_type is None:
        sound_type = random.choice(list(GENERATORS.keys()))

    def _base_dur():
        if note_duration is not None:
            return note_duration
        if bpm is not None:
            beat = 60.0 / bpm
            candidates = [d for d in [beat/4, beat/3, beat/2, beat] if 0.04 <= d <= 1.0]
            return random.choice(candidates) if candidates else 0.25
        return random.uniform(0.15, 0.35)

    base_dur    = _base_dur()
    swing_delay = swing * 0.35 * base_dur
    fade_n      = min(int(0.008 * sr), 256)

    segments = []
    for i, entry in enumerate(notes_list):
        if isinstance(entry, (tuple, list)):
            freq, dur_mult = entry[0], entry[1]
        else:
            freq, dur_mult = entry, 1.0

        dur = max(0.04, base_dur * dur_mult
                  * (1.0 + note_variance * random.uniform(-0.5, 0.5)))

        if swing_delay > 0.001 and (i % 2 == 1):
            segments.append(np.zeros(int(swing_delay * sr), dtype=np.int16))

        if freq is None:
            segments.append(np.zeros(int(dur * sr), dtype=np.int16))
            continue

        if isinstance(freq, list):
            # Chord stack: mix audio for each frequency in the list
            freqs = [f for f in freq if f is not None]
            if not freqs:
                segments.append(np.zeros(int(dur * sr), dtype=np.int16))
                continue
            stack_signals = []
            for f in freqs:
                a, _ = generate_sound(sound_type, sr=sr, duration=dur,
                                      root_freq=f, bpm=bpm)
                stack_signals.append(a.astype(np.float32))
            max_len = max(len(s) for s in stack_signals)
            mixed = np.zeros(max_len, dtype=np.float32)
            for s in stack_signals:
                mixed[:len(s)] += s
            mixed /= len(stack_signals)
            note_audio_f = mixed
        else:
            a, _ = generate_sound(sound_type, sr=sr, duration=dur,
                                  root_freq=freq, bpm=bpm)
            note_audio_f = a.astype(np.float32)

        vel_spread = velocity_variance * 0.40
        vel = np.clip(velocity * (1.0 + random.uniform(-vel_spread, vel_spread)),
                      0.02, 1.0)
        sig = note_audio_f * vel

        n = len(sig)
        fn = min(fade_n, n // 4)
        if fn > 0:
            ramp = np.linspace(0.0, 1.0, fn, dtype=np.float32)
            sig[:fn]  *= ramp
            sig[-fn:] *= ramp[::-1]

        segments.append(np.clip(sig, -32767, 32767).astype(np.int16))

    audio = np.concatenate(segments)
    label = f"sequence_{sound_type}_{len(notes_list)}"
    return audio, label


def _apply_chord_stacks(notes, chord_semitones):
    """
    Post-process a notes list, converting some single-note entries to chord stacks.
    On strong beats (~50% chance) a note becomes ([root, 3rd, 5th], dur_mult),
    using the intervals derived from chord_semitones.
    """
    nc = len(chord_semitones)
    # Derive 3rd and 5th intervals from the chord's semitone pattern
    interval_3rd = (chord_semitones[1 % nc] - chord_semitones[0]) % 12 or 4
    interval_5th = (chord_semitones[2 % nc] - chord_semitones[0]) % 12
    if interval_5th == 0 or interval_5th == interval_3rd:
        interval_5th = 7

    result = []
    for i, note in enumerate(notes):
        if (isinstance(note, tuple) and note[0] is not None
                and not isinstance(note[0], list)
                and i % 2 == 0 and random.random() < 0.50):
            freq, dur_mult = note
            third = freq * 2 ** (interval_3rd / 12)
            fifth = freq * 2 ** (interval_5th / 12)
            result.append(([freq, third, fifth], dur_mult))
        else:
            result.append(note)
    return result


def _pick_chord(scale, num_degrees=4):
    n = len(scale)
    root_deg = random.choice(range(n))
    chord_degs = []
    for i in range(num_degrees):
        chord_degs.append((root_deg + i * 2) % n)
    return chord_degs


# ── Helpers for melodic improvements ──────────────────────────────────────────

def _sem_to_freq(root_freq_base, semitone, octave_offset):
    """Convert a semitone (relative to root) + octave offset to Hz."""
    return root_freq_base * (2 ** ((semitone + octave_offset * 12) / 12))


def _find_passing_tone(scale, sem_a, sem_b):
    """
    Find a scale tone that sits between two chord tones (by semitone value).
    Returns the passing semitone, or None if no scale tone fits between them.
    """
    lo, hi = min(sem_a, sem_b), max(sem_a, sem_b)
    # Look for scale tones strictly between the two chord tones
    candidates = [s for s in scale if lo < s < hi]
    # Also check an octave up for wrapping intervals (e.g. B→C = 11→0)
    if not candidates and lo != hi:
        candidates = [s for s in scale if lo < s + 12 < hi + 12 and s != sem_a and s != sem_b]
    if candidates:
        return random.choice(candidates)
    return None


def _pick_contour(phrase_len):
    """
    Return a list of directional biases (-1, 0, +1) for each note in a phrase,
    forming an arch or valley shape.

    arch:   rise in first half, fall in second half
    valley: fall then rise
    rise:   gradual ascent
    fall:   gradual descent
    """
    shape = random.choices(["arch", "valley", "rise", "fall"],
                           weights=[0.40, 0.25, 0.20, 0.15])[0]
    mid = phrase_len // 2
    contour = []
    for i in range(phrase_len):
        if shape == "arch":
            contour.append(1 if i < mid else -1)
        elif shape == "valley":
            contour.append(-1 if i < mid else 1)
        elif shape == "rise":
            contour.append(1)
        else:
            contour.append(-1)
    # Soften: the very first and last notes have no bias (landing points)
    contour[0] = 0
    contour[-1] = 0
    return contour


def _generate_motif(chord_semitones, scale, phrase_len, contour,
                    root_freq_base, base_octave, start_chord_idx, start_oct):
    """
    Generate a single phrase of (freq, dur_mult) tuples using chord tones,
    shaped by the given contour, with occasional passing tones on weak beats.

    Returns (notes_list, ending_chord_idx, ending_octave).
    """
    nc = len(chord_semitones)
    chord_idx = start_chord_idx
    cur_oct = start_oct
    notes = []

    for pos in range(phrase_len):
        oct_offset = cur_oct - base_octave
        sem = chord_semitones[chord_idx]
        freq = _sem_to_freq(root_freq_base, sem, oct_offset)

        is_phrase_end = (pos == phrase_len - 1)
        is_weak_beat = (pos % 2 == 1) and not is_phrase_end

        # ── Duration variation ────────────────────────────────────────────
        if is_phrase_end:
            # Phrase endings get longer notes for resolution feel
            dur_mult = random.choice([1.5, 2.0])
        else:
            r_dur = random.random()
            if r_dur < 0.08:
                # Rest — but not on the first note of a phrase
                if pos > 0:
                    notes.append((None, 1.0))
                    # Advance chord_idx for next iteration before continuing
                    chord_idx, cur_oct = _advance_chord_idx(
                        chord_idx, cur_oct, nc, contour[pos], base_octave)
                    continue
                else:
                    dur_mult = 1.0
            elif r_dur < 0.20:
                dur_mult = 1.5
            else:
                dur_mult = 1.0

        # ── Passing tone insertion on weak beats ──────────────────────────
        if is_weak_beat and random.random() < 0.30 and pos + 1 < phrase_len:
            # Figure out the next chord tone we're heading toward
            next_chord_idx, next_oct = _advance_chord_idx(
                chord_idx, cur_oct, nc, contour[pos], base_octave)
            next_sem = chord_semitones[next_chord_idx]
            # Adjust for octave difference
            sem_a = sem + oct_offset * 12
            sem_b = next_sem + (next_oct - base_octave) * 12
            passing = _find_passing_tone(scale, sem_a % 12, sem_b % 12)
            if passing is not None:
                # Use passing tone frequency in current octave
                freq = _sem_to_freq(root_freq_base, passing, oct_offset)
                notes.append((freq, 0.5))  # passing tones are short
                chord_idx, cur_oct = next_chord_idx, next_oct
                continue

        notes.append((freq, dur_mult))

        # ── Phrase ending: bias toward root (idx 0) or 5th (idx 2) ────────
        if is_phrase_end:
            # Don't advance — this is the landing note, next phrase picks up
            pass
        else:
            chord_idx, cur_oct = _advance_chord_idx(
                chord_idx, cur_oct, nc, contour[pos], base_octave)

    return notes, chord_idx, cur_oct


def _advance_chord_idx(chord_idx, cur_oct, nc, direction, base_octave):
    """
    Move to the next chord tone. `direction` biases movement:
      +1 = prefer stepping up, -1 = prefer stepping down, 0 = either.
    """
    if direction > 0:
        step_weights = {1: 0.60, 2: 0.15, 0: 0.15, -1: 0.10}
    elif direction < 0:
        step_weights = {-1: 0.60, -2: 0.15, 0: 0.15, 1: 0.10}
    else:
        step_weights = {1: 0.35, -1: 0.35, 0: 0.15, 2: 0.08, -2: 0.07}

    steps = list(step_weights.keys())
    weights = list(step_weights.values())
    step = random.choices(steps, weights=weights)[0]

    new_idx = chord_idx + step
    if new_idx < 0:
        if cur_oct > max(3, base_octave - 1):
            cur_oct -= 1
            new_idx = nc + new_idx
        else:
            new_idx = 0
    elif new_idx >= nc:
        if cur_oct < min(5, base_octave + 1):
            cur_oct += 1
            new_idx = new_idx - nc
        else:
            new_idx = nc - 1

    return new_idx, cur_oct


def _vary_motif(motif_notes, chord_semitones, scale, root_freq_base,
                base_octave, start_chord_idx, start_oct, variation_amount=0.3):
    """
    Create a variation of an existing motif by:
      - Keeping the rhythmic pattern (dur_mults) mostly intact
      - Transposing some notes by ±1 chord step
      - Occasionally replacing a note with a neighbor
    Returns (varied_notes, ending_chord_idx, ending_oct).
    """
    nc = len(chord_semitones)
    chord_idx = start_chord_idx
    cur_oct = start_oct
    varied = []

    for freq, dur_mult in motif_notes:
        if freq is None:
            # Keep rests
            varied.append((None, dur_mult))
            continue

        oct_offset = cur_oct - base_octave
        sem = chord_semitones[chord_idx]

        if random.random() < variation_amount:
            # Shift by ±1 chord step
            shift = random.choice([-1, 1])
            new_idx = chord_idx + shift
            if 0 <= new_idx < nc:
                chord_idx = new_idx
                sem = chord_semitones[chord_idx]
            # Occasional octave shift
            if random.random() < 0.15:
                cur_oct = max(3, min(5, cur_oct + random.choice([-1, 1])))
                oct_offset = cur_oct - base_octave

        new_freq = _sem_to_freq(root_freq_base, sem, oct_offset)

        # Slight rhythmic variation — occasionally stretch or compress
        new_dur = dur_mult
        if random.random() < variation_amount * 0.5:
            new_dur = random.choice([0.5, 1.0, 1.5]) if dur_mult == 1.0 else dur_mult

        varied.append((new_freq, new_dur))

        # Advance for next note
        step = random.choice([-1, 0, 1])
        new_idx = chord_idx + step
        if 0 <= new_idx < nc:
            chord_idx = new_idx

    return varied, chord_idx, cur_oct


def _resolve_to_chord_tone(chord_semitones, root_freq_base, base_octave, cur_oct):
    """Pick a resolution note — biased toward root (idx 0) or 5th (idx 2)."""
    nc = len(chord_semitones)
    # Weight root and 5th heavily
    weights = [1.0] * nc
    weights[0] = 5.0                     # root
    if nc > 2:
        weights[2] = 3.0                 # 5th (3rd stacked third)
    total = sum(weights)
    weights = [w / total for w in weights]

    idx = random.choices(range(nc), weights=weights)[0]
    sem = chord_semitones[idx]
    oct_offset = cur_oct - base_octave
    freq = _sem_to_freq(root_freq_base, sem, oct_offset)
    return freq, idx


# ── Main improved sequence generator ──────────────────────────────────────────

def generate_random_sequence(root_note, octave, scale_type, num_notes, sound_type=None,
                              sr=SAMPLE_RATE, bpm=None, note_duration=None,
                              swing=0.0, velocity=0.8, velocity_variance=0.15,
                              note_variance=0.0, chord_stacks=False):
    """
    Generate a melodic sequence with three improvements over pure random walk:

    1. **Phrase structure with resolution** — notes are grouped into 4-or-8-note
       phrases. Each phrase ends on a chord tone biased toward root or 5th,
       giving the melody periodic points of rest and resolution.

    2. **Motif repetition** — the first phrase generates a melodic/rhythmic motif.
       Subsequent phrases repeat and vary that motif (transposed, with small
       pitch and rhythm mutations), so the melody sounds intentional.

    3. **Passing tones** — on weak beats, there is a ~30 % chance of inserting a
       scale tone that sits between two consecutive chord tones, smoothing out
       the "hoppy" sound of pure chord-tone melodies.

    Contour shaping (arch, valley, rise, fall) biases each phrase's direction
    so melodies have audible shape rather than flat random movement.
    """
    MAJOR = [0, 2, 4, 5, 7, 9, 11]
    MINOR = [0, 2, 3, 5, 7, 8, 10]
    scale = MAJOR if scale_type.lower() == 'major' else MINOR

    chord_size = random.choice([4, 4, 4, 5])
    chord_degs = _pick_chord(scale, chord_size)
    chord_semitones = [scale[d] for d in chord_degs]

    root_freq_base = note_to_freq(root_note, octave)
    nc = len(chord_semitones)

    # ── Decide phrase structure ───────────────────────────────────────────
    phrase_len = 4 if num_notes <= 8 else random.choice([4, 4, 8])
    num_phrases = max(1, (num_notes + phrase_len - 1) // phrase_len)

    chord_idx = 0
    cur_oct = max(3, min(5, octave))
    all_notes = []
    motif = None

    for phrase_i in range(num_phrases):
        remaining = num_notes - len(all_notes)
        if remaining <= 0:
            break
        plen = min(phrase_len, remaining)
        contour = _pick_contour(plen)

        if phrase_i == 0:
            # ── First phrase: generate the seed motif ─────────────────────
            phrase_notes, chord_idx, cur_oct = _generate_motif(
                chord_semitones, scale, plen, contour,
                root_freq_base, octave, chord_idx, cur_oct)
            motif = phrase_notes
        else:
            # ── Later phrases: vary the motif ─────────────────────────────
            # Decide how to handle this phrase
            action = random.choices(
                ["repeat", "vary", "new"],
                weights=[0.25, 0.50, 0.25]
            )[0]

            if action == "repeat" and motif:
                # Transpose the motif to start from current position
                phrase_notes, chord_idx, cur_oct = _vary_motif(
                    motif, chord_semitones, scale, root_freq_base,
                    octave, chord_idx, cur_oct, variation_amount=0.10)
            elif action == "vary" and motif:
                phrase_notes, chord_idx, cur_oct = _vary_motif(
                    motif, chord_semitones, scale, root_freq_base,
                    octave, chord_idx, cur_oct, variation_amount=0.35)
            else:
                # Fresh phrase (still shaped by contour)
                phrase_notes, chord_idx, cur_oct = _generate_motif(
                    chord_semitones, scale, plen, contour,
                    root_freq_base, octave, chord_idx, cur_oct)

        # ── Force phrase ending toward resolution ─────────────────────────
        if len(phrase_notes) >= 2 and phrase_notes[-1][0] is not None:
            res_freq, res_idx = _resolve_to_chord_tone(
                chord_semitones, root_freq_base, octave, cur_oct)
            # Keep the duration from the generated note, replace pitch
            phrase_notes[-1] = (res_freq, phrase_notes[-1][1])
            chord_idx = res_idx

        all_notes.extend(phrase_notes)

    all_notes = all_notes[:num_notes]

    if chord_stacks:
        all_notes = _apply_chord_stacks(all_notes, chord_semitones)

    audio, _ = generate_sequence(all_notes, sound_type=sound_type, sr=sr, bpm=bpm,
                                 note_duration=note_duration,
                                 swing=swing, velocity=velocity,
                                 velocity_variance=velocity_variance,
                                 note_variance=note_variance)
    chord_name = f"deg{chord_degs[0]}"
    label = f"seq_{scale_type}_{root_note}{octave}_{chord_name}_{num_notes}"
    return audio, label