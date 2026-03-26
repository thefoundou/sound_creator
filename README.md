# Sound Creator

A dark-themed GUI synthesizer for generating and exporting procedural WAV audio. Built with Python and Tkinter.

---

## Overview

Sound Creator lets you generate synthesized sounds interactively or in batch, apply audio effects, and export WAV files. It also includes an intelligent melodic sequence generator that produces scale-aware, phrase-structured melodies with optional chord harmonisation.

---

## Running

**GUI (main mode):**
```bash
cd sound_creation
python ui.py
```

**CLI (batch generation):**
```bash
cd sound_creation
python generate_sounds.py -n 10 -o ~/Desktop/sounds
python generate_sounds.py --list-types
python generate_sounds.py -t sine --seed 42
```

CLI flags: `-n` count, `-o` output dir, `-t` sound type, `--seed` for reproducibility.

> **Note:** Audio playback uses `afplay` (macOS). On Linux substitute `aplay` or `paplay` in `ui.py`.

---

## Sound Types

| Type | Description |
|------|-------------|
| `pitched_pluck` | Karplus-Strong plucked string / kalimba |
| `sine` | Pure sine wave with ADSR envelope |
| `square` | Square wave with ADSR envelope |
| `saw` | Sawtooth wave with ADSR envelope |
| `triangle` | Triangle wave with ADSR envelope |

All output is 44.1 kHz, 16-bit mono WAV.

---

## UI Sections

### Sound Types
Toggle which synthesis types are active. Use **All** / **None** for quick selection. The active set is used for both random preview and sequence generation.

### Tuning
- **Key** — enable to lock generation to a root note and octave (displays Hz). Required for sequence generation.
- **BPM** — enable to set tempo (60–200 BPM). Displays beat duration in ms.
- **Bar Snap** — rounds the sound duration to the nearest full bar.

### Random Sequence
Generates an intelligent melodic sequence using the active sound type.

| Control | Description |
|---------|-------------|
| Scale | Major or Minor |
| Notes | Number of notes (1–32) |
| Note | Note value per step (1/16 → 1) |
| Chord Stacks | Stack each note into a diatonic triad (or 7th) |
| Swing | Timing offset on odd-numbered notes |
| Velocity | Base amplitude |
| Vel Var | Velocity randomness |
| Len Var | Note duration randomness |

#### How melodies are built
1. **Phrase structure** — notes are grouped into 4–8 note phrases, each ending on a resolution note biased toward the root or 5th.
2. **Motif repetition** — the first phrase sets a seed motif; later phrases repeat it (25%), vary it (50%), or generate fresh material (25%).
3. **Contour shaping** — each phrase gets an arch, valley, rise, or fall direction to give it audible shape.
4. **Passing tones** — ~30% chance on weak beats of inserting a scale tone between chord tones to smooth movement.

#### Chord Stacks
When enabled, every non-rest note becomes a diatonic chord. The voicing is determined by the note's scale degree:
- Degrees I/IV/V → major triad (4, 7 semitones)
- Degrees ii/iii/vi → minor triad (3, 7 semitones)
- Leading tone vii° → diminished triad (3, 6 semitones)

~25% of chords also receive a diatonic 7th (shell voicing).

### Preview Controls
| Button | Action |
|--------|--------|
| ▶ Preview | Generate and play a random sound |
| ■ Stop | Stop current playback |
| ↺ Replay | Replay the last previewed sound |
| Save Last | Export previewed audio to the output folder |
| Clear | Clear waveform and preview |
| ✦ Pleasantize | Apply gentle polish (fade, EQ, soft saturation, normalise) |

### Weirdify
Five knobs (0–100%) that warp the timbre:

| Knob | Effect |
|------|--------|
| Crush | Bit depth reduction (lo-fi / digital crunch) |
| Ring | Ring modulation (metallic, bell-like) |
| Warp | Hyperbolic tangent soft saturation |
| Stutter | Segment chopping with random reversals |
| Glitch | Random chunk copying / scrambling |

### Resample
Chops the audio into pieces and recombines them. **Chop Len** controls chunk size, **Chop Reuse** controls the probability of repeating a chunk.

### Live Effects
Applied to every preview in real time:

| Knob | Effect |
|------|--------|
| Reverb | Exponential impulse convolution (0.3–2.8 s decay) |
| Delay | Multi-tap feedback delay (80–530 ms) |
| Filter | FFT low-pass filter (200–18000 Hz cutoff) |

---

## Project Structure

```
sound_creation/
├── ui.py               — Main GUI application (SoundGenUI)
├── generators.py       — Synthesis primitives and sound type generators
├── sequence.py         — Melodic sequence generation
├── effects.py          — Weirdify, resample, live effects, pleasantize
├── widgets.py          — Custom Tkinter widgets (Knob, Toggle, WaveformView, DancingMan)
├── generate_sounds.py  — CLI entry point and re-export shim
└── assets/
    └── dancing-man-pngs/   — Pixel-art sprite frames for playback animation
```

---

## Dependencies

| Package | Use |
|---------|-----|
| `numpy` | DSP and synthesis |
| `Pillow` | Knob rendering and sprite loading (optional — falls back gracefully) |
| `tkinter` | GUI (Python standard library) |
| `wave` | WAV I/O (Python standard library) |
