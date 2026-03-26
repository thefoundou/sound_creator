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
                      freq_hz may also be a list of frequencies (chord stack).
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
            candidates = [d for d in [beat/4, beat/3, beat/2, beat] if 0.04 <= d <= 1.0]
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
                dur *= 1.0 + 0.25 * phrase_contour
            elif phr_len > 2 and pos_in_phrase == phr_len - 2:
                dur *= 1.0 + 0.10 * phrase_contour

        if swing_delay > 0.001 and (i % 2 == 1):
            segments.append(np.zeros(int(swing_delay * sr), dtype=np.int16))

        if freq is None:
            cum_dur += dur
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

        # ── Velocity: random spread ───────────────────────────────────────
        vel_spread = velocity_variance * 0.40
        vel = velocity * (1.0 + random.uniform(-vel_spread, vel_spread))

        # ── Velocity: phrase contour (sine arc) ──────────────────────────
        if phrase_contour > 0:
            pos_in_phrase, phr_len = _phrase_of(i)
            if phr_len > 1:
                arc = math.sin(math.pi * pos_in_phrase / (phr_len - 1))
                contour_scale = 1.0 - phrase_contour * 0.35 * (1.0 - arc)
                vel *= contour_scale

        # ── Velocity: metric accent ──────────────────────────────────────
        if accent_strength > 0 and bar_len and bar_len > 0:
            beat_pos = (cum_dur % bar_len) / (bar_len / 4.0)
            beat_idx = int(beat_pos) % 4
            accent_w = _accent_pattern[beat_idx]
            vel *= 1.0 - accent_strength * (1.0 - accent_w)

        vel = np.clip(vel, 0.02, 1.0)
        sig = note_audio_f * vel

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


def _freq_to_midi(freq):
    """Convert Hz to a continuous MIDI note number (69 = A4 = 440 Hz)."""
    if freq <= 0:
        return 0
    return 69 + 12 * math.log2(freq / 440.0)


def _midi_to_freq(midi):
    """Convert MIDI note number back to Hz."""
    return 440.0 * (2 ** ((midi - 69) / 12))


def _build_chord_midi(root_midi, scale, best_deg, add_seventh):
    """
    Build a chord as a list of MIDI note numbers in root position
    (all voices above the root). Returns [root, 3rd, 5th] or
    [root, 3rd, 5th, 7th].
    """
    n = len(scale)
    semi_3rd = (scale[(best_deg + 2) % n] - scale[best_deg]) % 12
    semi_5th = (scale[(best_deg + 4) % n] - scale[best_deg]) % 12
    voices = [root_midi, root_midi + semi_3rd, root_midi + semi_5th]
    if add_seventh and n >= 7:
        semi_7th = (scale[(best_deg + 6) % n] - scale[best_deg]) % 12
        voices.append(root_midi + semi_7th)
    return voices


def _generate_inversions(chord_midi):
    """
    Generate all inversions of a chord within a reasonable register.

    For a triad [C4, E4, G4] this produces:
      root position: [C4, E4, G4]
      1st inversion: [E4, G4, C5]   (move root up an octave)
      2nd inversion: [G4, C5, E5]   (move root+3rd up an octave)
    Each inversion is then also shifted ±1 octave to give options in
    different registers.
    """
    n = len(chord_midi)
    inversions = []
    # Build each inversion by rotating which note is the bass
    current = list(chord_midi)
    for _ in range(n):
        inversions.append(sorted(current))
        # Move the lowest note up an octave
        current = [current[0] + 12] + current[1:]
        current.sort()

    # Also offer each inversion shifted ±12 semitones for register variety
    expanded = []
    for inv in inversions:
        for shift in [-12, 0, 12]:
            shifted = [m + shift for m in inv]
            # Keep within playable range (MIDI 36=C2 to MIDI 96=C7)
            if all(36 <= m <= 96 for m in shifted):
                expanded.append(shifted)
    return expanded


def _voice_leading_cost(prev_voices, candidate_voices):
    """
    Compute the total voice-movement cost between two chords.

    Uses nearest-voice assignment: for each voice in the candidate chord,
    find the closest voice in the previous chord and sum the distances.
    Penalises large leaps more than proportionally (squared distance)
    so the algorithm strongly prefers small, smooth movements.
    """
    if not prev_voices:
        # No previous chord — prefer voicings near the middle register
        center = 60  # middle C
        return sum((m - center) ** 2 for m in candidate_voices) * 0.01

    cost = 0
    prev_remaining = list(prev_voices)
    for voice in sorted(candidate_voices):
        if not prev_remaining:
            cost += 144  # penalty for extra voice (12^2)
            continue
        # Find nearest previous voice
        best_idx = min(range(len(prev_remaining)),
                       key=lambda j: abs(prev_remaining[j] - voice))
        dist = abs(prev_remaining[best_idx] - voice)
        cost += dist * dist  # squared to penalise large jumps
        prev_remaining.pop(best_idx)
    return cost


def _apply_chord_stacks(notes, scale, root_freq_base):
    """
    Convert every non-rest note to a diatonic chord stack with smooth
    voice leading between consecutive chords.

    Algorithm:
      1. Build a diatonic triad (+ optional 7th) for each note's scale degree.
      2. Generate all inversions of that chord in nearby registers.
      3. Pick the inversion that minimises total voice movement from the
         previous chord (squared-distance cost favours small, smooth steps).
      4. Common tones are naturally retained because they have zero cost.

    The first chord uses root position near the melody note's register.
    """
    n = len(scale)
    prev_voices = []  # MIDI note numbers of the previous chord
    result = []

    for note in notes:
        if not (isinstance(note, tuple) and note[0] is not None
                and not isinstance(note[0], list)):
            result.append(note)
            continue

        freq, dur_mult = note
        root_midi = _freq_to_midi(freq)

        # Find closest scale degree
        semitones_from_root = round(12 * math.log2(freq / root_freq_base)) % 12
        best_deg = min(
            range(n),
            key=lambda d: min(
                (scale[d] - semitones_from_root) % 12,
                (semitones_from_root - scale[d]) % 12,
            ),
        )

        # ~25 % chance of diatonic 7th
        add_seventh = (n >= 7 and random.random() < 0.25)

        # Build root-position chord in MIDI space
        chord_midi = _build_chord_midi(root_midi, scale, best_deg, add_seventh)

        # Generate all inversions across nearby registers
        candidates = _generate_inversions(chord_midi)
        if not candidates:
            candidates = [chord_midi]

        # Pick the inversion with minimum voice-leading cost
        best = min(candidates, key=lambda c: _voice_leading_cost(prev_voices, c))

        prev_voices = best
        freqs = [_midi_to_freq(m) for m in best]
        result.append((freqs, dur_mult))

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
    candidates = [s for s in scale if lo < s < hi]
    if not candidates and lo != hi:
        candidates = [s for s in scale if lo < s + 12 < hi + 12 and s != sem_a and s != sem_b]
    if candidates:
        return random.choice(candidates)
    return None


def _pick_contour(phrase_len):
    """
    Return a list of directional biases (-1, 0, +1) for each note in a phrase,
    forming an arch or valley shape.
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
            dur_mult = random.choice([1.5, 2.0])
        else:
            r_dur = random.random()
            if r_dur < 0.08:
                if pos > 0:
                    notes.append((None, 1.0))
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
            next_chord_idx, next_oct = _advance_chord_idx(
                chord_idx, cur_oct, nc, contour[pos], base_octave)
            next_sem = chord_semitones[next_chord_idx]
            sem_a = sem + oct_offset * 12
            sem_b = next_sem + (next_oct - base_octave) * 12
            passing = _find_passing_tone(scale, sem_a % 12, sem_b % 12)
            if passing is not None:
                freq = _sem_to_freq(root_freq_base, passing, oct_offset)
                notes.append((freq, 0.5))
                chord_idx, cur_oct = next_chord_idx, next_oct
                continue

        notes.append((freq, dur_mult))

        if is_phrase_end:
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
            varied.append((None, dur_mult))
            continue

        oct_offset = cur_oct - base_octave
        sem = chord_semitones[chord_idx]

        if random.random() < variation_amount:
            shift = random.choice([-1, 1])
            new_idx = chord_idx + shift
            if 0 <= new_idx < nc:
                chord_idx = new_idx
                sem = chord_semitones[chord_idx]
            if random.random() < 0.15:
                cur_oct = max(3, min(5, cur_oct + random.choice([-1, 1])))
                oct_offset = cur_oct - base_octave

        new_freq = _sem_to_freq(root_freq_base, sem, oct_offset)

        new_dur = dur_mult
        if random.random() < variation_amount * 0.5:
            new_dur = random.choice([0.5, 1.0, 1.5]) if dur_mult == 1.0 else dur_mult

        varied.append((new_freq, new_dur))

        step = random.choice([-1, 0, 1])
        new_idx = chord_idx + step
        if 0 <= new_idx < nc:
            chord_idx = new_idx

    return varied, chord_idx, cur_oct


def _resolve_to_chord_tone(chord_semitones, root_freq_base, base_octave, cur_oct):
    """Pick a resolution note — biased toward root (idx 0) or 5th (idx 2)."""
    nc = len(chord_semitones)
    weights = [1.0] * nc
    weights[0] = 5.0
    if nc > 2:
        weights[2] = 3.0
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
                              note_variance=0.0, chord_stacks=False,
                              phrase_contour=0.0, accent_strength=0.0):
    """
    Generate a melodic sequence with improvements over pure random walk:

    1. **Phrase structure with resolution** — notes are grouped into 4-or-8-note
       phrases. Each phrase ends on a chord tone biased toward root or 5th.

    2. **Motif repetition** — the first phrase generates a melodic/rhythmic motif.
       Subsequent phrases repeat and vary that motif.

    3. **Passing tones** — on weak beats, there is a ~30 % chance of inserting a
       scale tone between consecutive chord tones.

    Contour shaping (arch, valley, rise, fall) biases each phrase's direction.
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
            phrase_notes, chord_idx, cur_oct = _generate_motif(
                chord_semitones, scale, plen, contour,
                root_freq_base, octave, chord_idx, cur_oct)
            motif = phrase_notes
        else:
            action = random.choices(
                ["repeat", "vary", "new"],
                weights=[0.25, 0.50, 0.25]
            )[0]

            if action == "repeat" and motif:
                phrase_notes, chord_idx, cur_oct = _vary_motif(
                    motif, chord_semitones, scale, root_freq_base,
                    octave, chord_idx, cur_oct, variation_amount=0.10)
            elif action == "vary" and motif:
                phrase_notes, chord_idx, cur_oct = _vary_motif(
                    motif, chord_semitones, scale, root_freq_base,
                    octave, chord_idx, cur_oct, variation_amount=0.35)
            else:
                phrase_notes, chord_idx, cur_oct = _generate_motif(
                    chord_semitones, scale, plen, contour,
                    root_freq_base, octave, chord_idx, cur_oct)

        # ── Force phrase ending toward resolution ─────────────────────────
        if len(phrase_notes) >= 2 and phrase_notes[-1][0] is not None:
            res_freq, res_idx = _resolve_to_chord_tone(
                chord_semitones, root_freq_base, octave, cur_oct)
            phrase_notes[-1] = (res_freq, phrase_notes[-1][1])
            chord_idx = res_idx

        all_notes.extend(phrase_notes)

    all_notes = all_notes[:num_notes]

    if chord_stacks:
        all_notes = _apply_chord_stacks(all_notes, scale, root_freq_base)

    audio, _ = generate_sequence(all_notes, sound_type=sound_type, sr=sr, bpm=bpm,
                                 note_duration=note_duration,
                                 swing=swing, velocity=velocity,
                                 velocity_variance=velocity_variance,
                                 note_variance=note_variance,
                                 phrase_contour=phrase_contour,
                                 accent_strength=accent_strength)
    chord_name = f"deg{chord_degs[0]}"
    label = f"seq_{scale_type}_{root_note}{octave}_{chord_name}_{num_notes}"
    return audio, label
