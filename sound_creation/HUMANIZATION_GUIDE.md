# Humanizing Melodies: Code Alignment Guide

A comprehensive breakdown of what the research says about natural-sounding melodies and how our codebase (`sequence.py`, `generators.py`) can be updated to match.

---

## 1. Phrase-Aware Velocity Contour

### What the research says
Real performers shape velocity across phrases — notes at the start of a phrase are softer (pickup), the phrase peaks in the middle, and the final note tapers. This is fundamentally different from random per-note jitter. Academic research on expressive performance consistently shows that velocity follows an **arc** within each phrase, not a flat line with noise.

### What we do now
`sequence.py:67-69` — Each note gets an independent random velocity offset:
```python
vel_spread = velocity_variance * 0.40
vel = np.clip(velocity * (1.0 + random.uniform(-vel_spread, vel_spread)), 0.02, 1.0)
```
This is uniform random noise. Every note is equally likely to be loud or soft regardless of position. There's no sense of phrasing shape.

### What to change
Introduce a **phrase contour** that modulates velocity over groups of notes. Split the note list into phrases (e.g., 4–8 notes each) and apply a curve — gentle ramp up, peak around 60% through, taper off — on top of the existing random variance. Implementation:

- Add a `phrase_length` parameter (default 4–8, randomized per phrase).
- For each note in a phrase, compute a contour weight using a sine or parabolic arc: `contour = sin(pi * position / phrase_length)`.
- Multiply the base velocity by this contour before applying the existing random spread.
- The contour amplitude should be controllable — a `phrase_contour` parameter (0.0–1.0) that blends between flat (current behavior) and fully shaped.

This means the velocity line goes from `[random, random, random, ...]` to `[soft-random, medium-random, loud-random, medium-random, soft-random, ...]` — a massive perceptual improvement.

---

## 2. Phrase-Final Lengthening

### What the research says
One of the strongest findings in music performance research: musicians **lengthen the last note of a phrase**. This is analogous to how speakers slow down at the end of a sentence. It signals closure and gives the listener a parsing cue. The typical lengthening is 10–30% of the base duration.

### What we do now
`sequence.py:54-55` — Duration variation is purely random:
```python
dur = max(0.04, base_dur * dur_mult * (1.0 + note_variance * random.uniform(-0.5, 0.5)))
```
No awareness of phrase boundaries. The last note of a group is just as likely to be shortened as lengthened.

### What to change
When generating notes in phrases (same phrase grouping from Section 1):

- The **last note** of each phrase gets a duration multiplier boost: `dur *= 1.0 + phrase_final_stretch` where `phrase_final_stretch` defaults to ~0.15–0.25.
- The **second-to-last note** could get a mild stretch too (~0.05–0.10) for a smoother deceleration (ritardando).
- Optionally, the **first note** of the next phrase could be slightly delayed (a micro-breath) by inserting a tiny silence gap (10–40ms).

This is cheap to implement — it's a conditional multiplier on `dur` when `note_index == phrase_end`.

---

## 3. Expressive Microtiming (Not Just Swing)

### What the research says
Swing is one specific timing pattern (delaying offbeats). But real musicians apply **microtiming** more broadly: notes land slightly early when building tension, slightly late when relaxing. Vijay Iyer's research on jazz microtiming shows these shifts are 5–25ms and are *correlated with musical context*, not random. The research on "participatory discrepancies" shows that slight asynchronies between parts (or between notes in a melody) are what create the feeling of groove.

### What we do now
`sequence.py:44, 57-58` — Swing only, applied uniformly to all odd-indexed notes:
```python
swing_delay = swing * 0.35 * base_dur
if swing_delay > 0.001 and (i % 2 == 1):
    segments.append(np.zeros(int(swing_delay * sr), dtype=np.int16))
```
This is a fixed delay on every other note. No per-note timing jitter, no push/pull based on context.

### What to change
Add two layers of microtiming on top of swing:

**A. Random microtiming jitter:**
- Add a `timing_humanize` parameter (0.0–1.0).
- For each note, offset its start by `random.gauss(0, timing_humanize * 0.012 * base_dur)` seconds — Gaussian (not uniform) because real timing errors cluster near zero with occasional larger deviations.
- Clamp to ±20ms max to avoid audible flamming.
- Insert silence (for late notes) or trim the previous note's tail (for early notes).

**B. Phrase-position timing:**
- Notes near phrase ends should drift slightly late (laid-back feel, mirroring phrase-final lengthening).
- Notes at phrase starts can be slightly early (eager attack).
- Scale: ±5–10ms, controlled by the `phrase_contour` parameter.

This transforms timing from `[0, swing, 0, swing, ...]` to `[early, swing+jitter, jitter, swing+jitter+late, ...]` — far more organic.

---

## 4. Velocity Accents on Strong Beats

### What the research says
Across virtually all musical traditions, strong beats (beat 1, beat 3 in 4/4) receive higher velocity than weak beats. This isn't huge — typically 10–20% louder — but it's one of the primary cues that tells the listener where the downbeat is. Without it, the rhythm feels "flat" even if the timing is correct.

### What we do now
No metric accent awareness at all. Every note position is treated identically. The only velocity variation is random noise.

### What to change
- Track the current beat position within a bar (requires knowing the time signature — assume 4/4 default, expose as parameter).
- Apply a metric accent pattern. For 4/4: `accent_weights = [1.0, 0.7, 0.85, 0.7]` (beat 1 strongest, beat 3 secondary).
- Multiply each note's velocity by the accent weight for its beat position.
- The accent depth should be controllable via an `accent_strength` parameter (0.0 = no accent, 1.0 = full pattern). At 0.0 it's flat; at 0.5 the differences are subtle; at 1.0 they're pronounced.
- This layers with phrase contour and random variance: `final_vel = base_vel * phrase_contour * accent_weight * (1 + random_jitter)`.

Implementation detail: beat position can be derived from the cumulative duration of emitted notes relative to the bar length (`4 * 60/bpm` for 4/4).

---

## 5. Non-Uniform Duration Probabilities (Musical Rhythm Patterns)

### What the research says
The cross-cultural rhythm study (Nature, 2023) found that humans gravitate toward **small-integer-ratio rhythms** — durations relate to each other as 1:1, 2:1, 3:1, etc. Random duration multipliers can produce irrational ratios that sound unmusical. Real melodies also tend to use shorter notes during movement and longer notes at resting points (chord tones, phrase ends).

### What we do now
`sequence.py:134-142` — Duration multiplier is assigned by flat probability buckets:
```python
if r_dur < 0.12:       # rest
elif r_dur < 0.27:     # 2.0x (double)
elif r_dur < 0.40:     # 1.5x (dotted)
else:                   # 1.0x (normal)
```
Then `note_variance` adds a continuous random multiplier on top. The continuous variance can produce any ratio, including musically unnatural ones like 1.37x.

### What to change
**A. Snap duration variance to integer ratios:**
Instead of `1.0 + note_variance * random.uniform(-0.5, 0.5)`, snap the variance to musically meaningful multipliers:
```python
duration_ratios = [0.5, 0.667, 0.75, 1.0, 1.0, 1.0, 1.25, 1.333, 1.5, 2.0]
```
Weight toward 1.0 (normal) but allow occasional half-notes, triplets, dotted values. The `note_variance` parameter can control how far from 1.0 the selection reaches.

**B. Context-sensitive durations:**
- Stepwise motion → shorter notes (use 0.75x–1.0x multipliers).
- Skip or octave jump → longer notes (use 1.0x–1.5x).
- Phrase-ending notes → longest (1.5x–2.0x).
- After a rest → the returning note should be at least 1.0x (don't come back from silence with a truncated note).

This connects duration choices to the melodic movement logic already in `generate_random_sequence()`.

---

## 6. Dynamic Envelope Variation Per Note

### What the research says
Velocity in real instruments doesn't just change volume — it changes *timbre*. A soft piano note has a different harmonic spectrum than a loud one. A gentle guitar pluck has less high-frequency content. This is why randomizing a flat amplitude scalar sounds artificial — the timbre stays identical across all velocities.

### What we do now
`generators.py` — Each sound type uses fixed ADSR parameters regardless of velocity:
```python
return apply_adsr(sig, sr, attack=0.01, decay=0.05, sustain=0.8, release=0.15)
```
`sequence.py:70` — Velocity is applied as a pure amplitude scalar after generation:
```python
sig = note_audio.astype(np.float32) * vel
```

### What to change
Pass velocity into the generator functions and let it modulate the envelope and timbre:

- **Attack time:** Softer notes (low velocity) should have slightly longer attack (0.01–0.04s). Loud notes keep the sharp 0.01s attack. `attack = 0.01 + (1.0 - vel) * 0.03`.
- **Decay/Release:** Softer notes can have shorter decay (they die faster). `decay = 0.05 * vel + 0.02`.
- **Harmonic content:** For `pitched_pluck`, reduce the number of audible harmonics at low velocity — multiply higher harmonic weights by `vel`. A note at velocity 0.3 would have `harmonic_weights = [1.0, 0.55*0.3, 0.28*0.3, ...]` — fundamentally darker.
- **Brightness filter:** For saw/square, apply a simple low-pass roll-off that opens with velocity. This can be a one-pole filter: `cutoff_hz = 800 + vel * 4000`.

This means soft notes actually *sound* soft (dark, gentle attack) rather than just being quieter versions of the same timbre.

---

## 7. Legato and Articulation Variation

### What the research says
The gap (or overlap) between consecutive notes is a major expressive parameter. Staccato (short, separated) vs. legato (long, connected) vs. portato (slightly separated) — real players vary this constantly. Research shows articulation patterns often mirror phrase structure: legato within phrases, staccato at boundaries.

### What we do now
Notes are placed back-to-back in the `segments` array. The gap between notes is determined entirely by the linear fade-out applied at `sequence.py:76-77`:
```python
sig[-fn:] *= ramp[::-1]
```
This is a fixed 8ms fade. Every note has the same articulation.

### What to change
Add an `articulation` system:

- Introduce a `note_gap` parameter that controls the fraction of each note's duration that is silence at the end. Range: -0.1 (overlap/legato) to 0.3 (staccato).
- Within a phrase, vary articulation: notes within a phrase lean legato (gap = -0.05 to 0.05), notes at phrase boundaries lean staccato (gap = 0.1 to 0.2).
- For legato (negative gap), extend the note's audio slightly to overlap with the next note, crossfading in the overlap region.
- The existing 8ms fade-out should scale with the gap size — staccato notes get a faster fade, legato notes get a slower/no fade.

Implementation in `generate_sequence()`: after generating each note's audio, trim or extend it based on the computed gap, then append the appropriate silence.

---

## 8. Melodic Gravity and Tendency Tones

### What the research says
Melodies sound natural when they follow **melodic gravity** — large upward leaps are followed by stepwise descent, phrases tend to arc upward then come back down, and tendency tones (like the 7th degree) resolve. Currently the melodic movement is purely random within chord tones.

### What we do now
`sequence.py:144-184` — Movement is context-free. Each note's direction is chosen independently:
```python
r = random.random()
if r < 0.10:    pass           # repeat
elif r < 0.65:  step = ±1      # step
elif r < 0.85:  step = ±2      # skip
else:            octave jump    # jump
```
There's no memory of what just happened. A big upward leap can be followed by another big upward leap.

### What to change
Add **directional bias** based on the previous interval:

- Track `last_direction` (+1 for ascending, -1 for descending, 0 for repeat) and `last_interval_size` (0, 1, or 2 chord-tone steps).
- After a **large interval** (skip or octave jump): bias the next movement toward **stepwise motion in the opposite direction** (70% instead of current flat 55%). This implements the "leap-then-step-back" principle.
- After **repeated stepwise motion in one direction** (3+ notes ascending/descending): increase probability of direction change or repeat (rest point).
- Maintain a **register gravity** pull: when the melody is in octave 5, bias downward; when in octave 3, bias upward. This keeps the melody from getting stuck at range extremes.
- After a rest: the next note should favor **returning to a chord tone near the center of the range** rather than continuing from wherever the melody was before. This mimics how real players "reset" after a breath.

These are adjustments to the probability weights in the existing if/elif chain — structurally minimal changes with large perceptual impact.

---

## 9. Groove Template System (Beyond Swing)

### What the research says
The groove research shows that groove is "multiparameter" — it's not just swing. Different genres have specific timing templates. For example, hip-hop often pushes the snare slightly late while keeping hi-hats ahead of the beat. Bossa nova has a completely different timing feel from swing jazz. No single "swing" parameter captures this.

### What we do now
One swing parameter that delays every odd note by a fixed amount.

### What to change
Replace the single swing parameter with a **groove template** system:

```python
GROOVE_TEMPLATES = {
    'straight': [0.0] * 16,
    'swing':    [0.0, 0.33, 0.0, 0.33, 0.0, 0.33, 0.0, 0.33, ...],  # classic swing
    'shuffle':  [0.0, 0.25, 0.0, 0.25, 0.0, 0.25, 0.0, 0.25, ...],  # lighter swing
    'push':     [-0.05, 0.0, -0.05, 0.0, ...],                        # ahead-of-beat drive
    'lazy':     [0.05, 0.1, 0.05, 0.15, ...],                         # behind-the-beat feel
    'clave':    [0.0, 0.0, 0.08, 0.0, 0.0, -0.04, 0.0, 0.1, ...],   # son clave feel
}
```

Each template is a list of timing offsets (as fractions of the base note duration) for each step in a bar. The template repeats/wraps for the full sequence. Keep the `swing` parameter as a shortcut that selects/blends between templates.

---

## 10. Micro-Dynamics: Crescendo and Diminuendo

### What the research says
Performers don't just accent individual beats — they shape dynamics over multiple bars. Phrases often crescendo toward a climax and diminuendo toward resolution. This is distinct from phrase-contour velocity (Section 1), which operates within a single phrase. Here we're talking about **multi-phrase arcs**.

### What we do now
No awareness of large-scale dynamic shape. Every phrase, every section, operates at the same base velocity.

### What to change
Add a **global dynamic envelope** that modulates velocity across the entire sequence:

- Divide the sequence into sections (e.g., quarters of total length).
- Apply a macro contour: `[0.85, 1.0, 1.0, 0.9]` (start slightly soft, full energy in middle, gentle taper at end).
- This should be user-controllable or randomized from a set of shapes (arch, ramp-up, ramp-down, plateau).
- The macro contour multiplies the base velocity *before* phrase contour and accent weighting are applied.

This creates the sense of a performance that "goes somewhere" rather than a flat loop.

---

## Implementation Priority

| Priority | Change | Effort | Perceptual Impact |
|----------|--------|--------|-------------------|
| 1 | Phrase-aware velocity contour (§1) | Low | High |
| 2 | Phrase-final lengthening (§2) | Low | High |
| 3 | Metric accent on strong beats (§4) | Low | High |
| 4 | Expressive microtiming jitter (§3) | Medium | High |
| 5 | Melodic gravity / direction bias (§8) | Medium | High |
| 6 | Duration snapping to integer ratios (§5) | Low | Medium |
| 7 | Legato/staccato articulation (§7) | Medium | Medium |
| 8 | Dynamic envelope per velocity (§6) | Medium | Medium |
| 9 | Groove templates (§9) | Medium | Medium |
| 10 | Multi-phrase crescendo/diminuendo (§10) | Low | Low–Medium |

The first three items are low-effort, high-impact changes that can be done in `sequence.py` alone without touching the generator architecture. Items 4–5 require moderate refactoring of the note loop. Items 6–9 touch both `sequence.py` and `generators.py`.

---

## New Parameters Summary

| Parameter | Range | Default | Where |
|-----------|-------|---------|-------|
| `phrase_length` | 3–12 | 6 | `generate_sequence` |
| `phrase_contour` | 0.0–1.0 | 0.5 | `generate_sequence` |
| `phrase_final_stretch` | 0.0–0.5 | 0.2 | `generate_sequence` |
| `timing_humanize` | 0.0–1.0 | 0.3 | `generate_sequence` |
| `accent_strength` | 0.0–1.0 | 0.5 | `generate_sequence` |
| `articulation` | -0.1–0.3 | 0.05 | `generate_sequence` |
| `groove_template` | string | 'straight' | `generate_sequence` |
| `dynamic_shape` | string | 'arch' | `generate_sequence` |

These would also need corresponding UI knobs/sliders in `ui.py`.
