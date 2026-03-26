"""
Random WAV Sound Generator — CLI entry point and re-export shim.

All logic lives in:
  generators.py  — synthesis primitives, sound type generators, generate_sound / mix_sounds
  sequence.py    — generate_sequence, generate_random_sequence
  effects.py     — apply_weirdify, resample_chop, apply_live_effects, pleasantize
"""

import argparse
import os
import random
import numpy as np

from generators import (SAMPLE_RATE, NOTE_SEMITONES, note_to_freq,
                         SOUND_TYPES, GENERATORS,
                         generate_sound, mix_sounds, write_wav)
from sequence import generate_sequence, generate_random_sequence
from effects import apply_weirdify, resample_chop, apply_live_effects, pleasantize

# Re-export everything so existing `from generate_sounds import ...` calls keep working
__all__ = [
    "SAMPLE_RATE", "NOTE_SEMITONES", "note_to_freq",
    "SOUND_TYPES", "GENERATORS",
    "generate_sound", "mix_sounds", "write_wav",
    "generate_sequence", "generate_random_sequence",
    "apply_weirdify", "resample_chop", "apply_live_effects", "pleasantize",
]


def main():
    parser = argparse.ArgumentParser(description="Generate random unique WAV sounds.")
    parser.add_argument("-n", "--count", type=int, default=5,
                        help="Number of WAV files to generate (default: 5)")
    parser.add_argument("-o", "--output", type=str, default="output",
                        help="Output directory (default: output/)")
    parser.add_argument("-t", "--type", type=str, choices=list(GENERATORS.keys()),
                        default=None, help="Force a specific sound type (default: random)")
    parser.add_argument("--seed", type=int, default=None,
                        help="Random seed for reproducibility")
    parser.add_argument("--list-types", action="store_true",
                        help="List all available sound types and exit")
    args = parser.parse_args()

    if args.list_types:
        print("Available sound types:")
        for name in GENERATORS:
            print(f"  {name}")
        return

    if args.seed is not None:
        random.seed(args.seed)
        np.random.seed(args.seed)

    os.makedirs(args.output, exist_ok=True)

    for i in range(1, args.count + 1):
        audio, sound_type = generate_sound(args.type)
        filename = f"sound_{i:03d}_{sound_type}.wav"
        filepath = os.path.join(args.output, filename)
        write_wav(filepath, audio)
        print(f"  [{i}/{args.count}] {filename}")

    print(f"\nDone. {args.count} file(s) saved to '{args.output}/'")


if __name__ == "__main__":
    main()
