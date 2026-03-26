"""
Sound Generator UI — Hardware Theme
"""

import tkinter as tk
from tkinter import ttk, scrolledtext
import threading
import subprocess
import tempfile
import os
import random
import numpy as np

from widgets import (
    BG, PANEL, SURFACE, BORDER, TEXT, MUTED, SUCCESS, WARN, DARK, KNOB_ARC,
    FONT_SECTION, FONT_LABEL, FONT_SMALL, FONT_TINY, FONT_MONO, FONT_BTN, FONT_TITLE,
    DancingMan, Knob, WeirdifyCanvas, Toggle, WaveformView, card, divider,
)
from generate_sounds import (
    GENERATORS, generate_sound,
    generate_sequence, generate_random_sequence,
    apply_weirdify, resample_chop, apply_live_effects,
    pleasantize, write_wav,
    note_to_freq, NOTE_SEMITONES, SAMPLE_RATE,
)

_ASSETS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                           "assets", "dancing-man-pngs")


# ── Main UI ────────────────────────────────────────────────────────────────────

class SoundGenUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Sound Generator")
        self.root.configure(bg=BG)
        self.root.resizable(True, True)

        self._preview_audio = None
        self._preview_type  = None
        self._preview_proc  = None
        self._preview_tmp   = None

        # Defaults (formerly in Settings panel)
        self.duration_var = tk.DoubleVar(value=2.0)
        self.folder_var   = tk.StringVar(value=os.path.abspath("output"))
        self.seed_var     = tk.StringVar(value="")
        self.count_var    = tk.IntVar(value=5)

        import threading as _threading
        self._stop_gen = _threading.Event()

        self._build()
        self.root.update_idletasks()
        self._center()

    def _center(self):
        sw = self.root.winfo_screenwidth()
        sh = self.root.winfo_screenheight()
        self.root.geometry(f"{sw}x{sh}+0+0")

    # ── Root layout ────────────────────────────────────────────────────────────

    def _build(self):
        bar = tk.Frame(self.root, bg=BG, pady=16)
        bar.pack(fill="x", padx=24)
        tk.Label(bar, text="Sound Generator", font=FONT_TITLE,
                 bg=BG, fg=TEXT).pack(side="left")
        tk.Label(bar, text="WAV Synthesizer", font=("SF Pro Text", 10),
                 bg=BG, fg=MUTED).pack(side="left", padx=(10, 0), pady=(4, 0))

        body = tk.Frame(self.root, bg=BG)
        body.pack(fill="both", padx=24, pady=(0, 24))

        col1 = tk.Frame(body, bg=BG)
        col2 = tk.Frame(body, bg=BG)
        col3 = tk.Frame(body, bg=BG)
        col1.pack(side="left", fill="y", padx=(0, 14))
        col2.pack(side="left", fill="y", padx=(0, 14))
        col3.pack(side="left", fill="both", expand=True)

        self._build_dancer(col1)
        self._build_types(col1)

        self._build_tuning(col2)
        self._build_preview(col2)
        self._build_log(col2)
        self._build_actions(col2)

        self._build_weirdify(col3)
        self._build_resample(col3)
        self._build_live_effects(col3)
        self._build_waveform(col3)

    # ── Dancing man ─────────────────────────────────────────────────────────────

    def _build_dancer(self, parent):
        c = card(parent, title=None, bg=BG, padx=0, pady=0)
        c.master.pack(fill="x", pady=(0, 14))
        self.dancer = DancingMan(c, _ASSETS_DIR, scale=8, bg=BG)
        self.dancer.pack(anchor="center", pady=8)

    # ── Sound types ────────────────────────────────────────────────────────────

    def _build_types(self, parent):
        c = card(parent, "Sound Types")
        c.master.pack(fill="x", pady=(0, 14))

        ctrl = tk.Frame(c, bg=PANEL)
        ctrl.pack(fill="x", pady=(0, 10))
        for txt, cmd in [("All", self._select_all), ("None", self._select_none)]:
            tk.Button(ctrl, text=txt, font=FONT_TINY, bg=SURFACE, fg=TEXT,
                      relief="flat", cursor="hand2", bd=0, padx=10, pady=3,
                      highlightthickness=1, highlightbackground=BORDER,
                      command=cmd).pack(side="left", padx=(0, 6))

        self.type_vars = {}
        labels = {
            "pitched_pluck": "Pitched Pluck",
            "sine":          "Sine",
            "square":        "Square",
            "saw":           "Saw",
            "triangle":      "Triangle",
        }
        for key, lbl in labels.items():
            var = tk.BooleanVar(value=True)
            self.type_vars[key] = var
            row = tk.Frame(c, bg=PANEL)
            row.pack(fill="x", pady=3)
            t = Toggle(row, var, bg=PANEL,
                       command=lambda v=var, r=row: self._on_toggle(v, r))
            t.pack(side="left", padx=(0, 8))
            lbl_w = tk.Label(row, text=lbl, font=FONT_LABEL,
                             bg=PANEL, fg=TEXT, cursor="hand2")
            lbl_w.pack(side="left")
            lbl_w.bind("<Button-1>",
                       lambda e, v=var, r=row: (v.set(not v.get()),
                                                self._on_toggle(v, r)))
            row._lbl = lbl_w

    def _on_toggle(self, var, row):
        row._lbl.config(fg=TEXT if var.get() else MUTED)

    def _select_all(self):
        for v in self.type_vars.values():
            v.set(True)

    def _select_none(self):
        for v in self.type_vars.values():
            v.set(False)

    def _style_menu(self, m):
        m.config(bg=SURFACE, fg=TEXT, activebackground=BORDER,
                 activeforeground=TEXT, relief="flat",
                 highlightthickness=1, highlightbackground=BORDER,
                 font=FONT_SMALL, bd=0, indicatoron=True)
        m["menu"].config(bg=SURFACE, fg=TEXT, activebackground=BORDER,
                         activeforeground=TEXT, font=FONT_SMALL,
                         relief="flat", bd=0)

    # ── Tuning panel ──────────────────────────────────────────────────────────

    def _build_tuning(self, parent):
        c = card(parent, "Tuning")
        c.master.pack(fill="x", pady=(0, 14))

        notes   = list(NOTE_SEMITONES.keys())
        octaves = [2, 3, 4, 5]

        key_hdr = tk.Frame(c, bg=PANEL)
        key_hdr.pack(fill="x", pady=(0, 6))

        self.key_enabled = tk.BooleanVar(value=False)
        Toggle(key_hdr, self.key_enabled, bg=PANEL,
               command=self._on_tuning_toggle).pack(side="left", padx=(0, 8))
        tk.Label(key_hdr, text="Key", font=FONT_LABEL, bg=PANEL, fg=TEXT).pack(side="left")
        self.key_freq_label = tk.Label(key_hdr, text="", font=FONT_TINY,
                                       bg=PANEL, fg=MUTED)
        self.key_freq_label.pack(side="right")

        key_row = tk.Frame(c, bg=PANEL)
        key_row.pack(fill="x", pady=(0, 12))

        self.key_note_var   = tk.StringVar(value="C")
        self.key_octave_var = tk.IntVar(value=4)

        self.key_note_menu = tk.OptionMenu(key_row, self.key_note_var, *notes,
                                           command=lambda _: self._update_freq_label())
        self._style_menu(self.key_note_menu)
        self.key_note_menu.pack(side="left", padx=(0, 6))

        self.key_octave_menu = tk.OptionMenu(key_row, self.key_octave_var, *octaves,
                                             command=lambda _: self._update_freq_label())
        self._style_menu(self.key_octave_menu)
        self.key_octave_menu.pack(side="left")

        bpm_hdr = tk.Frame(c, bg=PANEL)
        bpm_hdr.pack(fill="x", pady=(0, 6))

        self.bpm_enabled = tk.BooleanVar(value=False)
        Toggle(bpm_hdr, self.bpm_enabled, bg=PANEL,
               command=self._on_tuning_toggle).pack(side="left", padx=(0, 8))
        tk.Label(bpm_hdr, text="BPM", font=FONT_LABEL, bg=PANEL, fg=TEXT).pack(side="left")
        self.bpm_beat_label = tk.Label(bpm_hdr, text="", font=FONT_TINY,
                                       bg=PANEL, fg=MUTED)
        self.bpm_beat_label.pack(side="right")

        bpm_row = tk.Frame(c, bg=PANEL)
        bpm_row.pack(fill="x", pady=(0, 10))
        self.bpm_var   = tk.DoubleVar(value=120.0)
        self.bpm_label = tk.Label(bpm_row, text="120", width=4,
                                  font=FONT_LABEL, bg=PANEL, fg=TEXT, anchor="e")
        self.bpm_label.pack(side="right")
        ttk.Scale(bpm_row, from_=60, to=200, orient="horizontal",
                  variable=self.bpm_var,
                  command=self._on_bpm_change).pack(side="left", fill="x",
                                                    expand=True, padx=(0, 8))

        snap_row = tk.Frame(c, bg=PANEL)
        snap_row.pack(fill="x", pady=(0, 12))
        self.snap_bar_var = tk.BooleanVar(value=False)
        self.snap_bar_toggle = Toggle(snap_row, self.snap_bar_var, bg=PANEL)
        self.snap_bar_toggle.pack(side="left", padx=(0, 8))
        tk.Label(snap_row, text="Snap duration to bar", font=FONT_SMALL,
                 bg=PANEL, fg=MUTED).pack(side="left")

        # ── Sequence Generator ─────────────────────────────────────────────────
        seq_hdr = tk.Frame(c, bg=PANEL)
        seq_hdr.pack(fill="x", pady=(0, 6))
        tk.Label(seq_hdr, text="Random Sequence", font=FONT_LABEL, bg=PANEL, fg=TEXT).pack(side="left")

        seq_row = tk.Frame(c, bg=PANEL)
        seq_row.pack(fill="x", pady=(0, 8))

        tk.Label(seq_row, text="Scale:", font=FONT_SMALL, bg=PANEL, fg=TEXT).pack(side="left", padx=(0, 6))
        self.sequence_scale_var = tk.StringVar(value="Major")
        scale_menu = tk.OptionMenu(seq_row, self.sequence_scale_var, "Major", "Minor")
        self._style_menu(scale_menu)
        scale_menu.pack(side="left", padx=(0, 12))

        tk.Label(seq_row, text="Notes:", font=FONT_SMALL, bg=PANEL, fg=TEXT).pack(side="left", padx=(0, 6))
        self.sequence_notes_var = tk.IntVar(value=8)
        notes_spin = ttk.Spinbox(seq_row, from_=1, to=32, textvariable=self.sequence_notes_var,
                                 width=4, font=FONT_SMALL)
        notes_spin.pack(side="left", padx=(0, 12))

        tk.Label(seq_row, text="Note:", font=FONT_SMALL, bg=PANEL, fg=TEXT).pack(side="left", padx=(0, 6))
        self.sequence_note_value_var = tk.StringVar(value="1/8")
        note_menu = tk.OptionMenu(seq_row, self.sequence_note_value_var,
                                  "1/16", "1/8", "1/4", "1/2", "1")
        self._style_menu(note_menu)
        note_menu.pack(side="left", padx=(0, 12))

        self.generate_sequence_btn = tk.Button(seq_row, text="Generate Sequence",
                                               bg=SURFACE, fg=TEXT, font=FONT_BTN,
                                               activebackground=BORDER, activeforeground=TEXT,
                                               command=self._generate_sequence_preview)
        self.generate_sequence_btn.pack(side="left")

        knob_row = tk.Frame(c, bg=PANEL)
        knob_row.pack(fill="x", pady=(4, 0))
        self.seq_swing_var        = tk.DoubleVar(value=0.0)
        self.seq_velocity_var     = tk.DoubleVar(value=80.0)
        self.seq_vel_variance_var = tk.DoubleVar(value=15.0)
        self.seq_variance_var     = tk.DoubleVar(value=20.0)
        self.seq_phrase_var       = tk.DoubleVar(value=50.0)
        self.seq_accent_var       = tk.DoubleVar(value=40.0)
        for var, lbl in [(self.seq_swing_var,        "Swing"),
                         (self.seq_velocity_var,     "Velocity"),
                         (self.seq_vel_variance_var, "Vel Var"),
                         (self.seq_variance_var,     "Len Var"),
                         (self.seq_phrase_var,       "Phrase"),
                         (self.seq_accent_var,       "Accent")]:
            Knob(knob_row, var, size=62, label=lbl, bg=PANEL).pack(side="left", padx=4)

        self._on_tuning_toggle()
        self._update_freq_label()

    def _on_tuning_toggle(self):
        key_on = self.key_enabled.get()
        self.key_note_menu.config(state="normal" if key_on else "disabled")
        self.key_octave_menu.config(state="normal" if key_on else "disabled")
        self.key_freq_label.config(fg=TEXT if key_on else MUTED)
        self.generate_sequence_btn.config(state="normal" if key_on else "disabled")
        self._update_freq_label()
        self._on_bpm_change(self.bpm_var.get())

    def _update_freq_label(self):
        if self.key_enabled.get():
            freq = note_to_freq(self.key_note_var.get(), self.key_octave_var.get())
            self.key_freq_label.config(text=f"{freq:.1f} Hz")
        else:
            self.key_freq_label.config(text="")

    def _on_bpm_change(self, v):
        bpm = float(v)
        self.bpm_label.config(text=str(int(bpm)))
        if self.bpm_enabled.get():
            beat_ms = int(60000 / bpm)
            self.bpm_beat_label.config(text=f"{beat_ms} ms/beat")
        else:
            self.bpm_beat_label.config(text="")

    def _get_tuning(self):
        """Return (root_freq, bpm, duration) with bar-snapping applied."""
        root_freq = (note_to_freq(self.key_note_var.get(), self.key_octave_var.get())
                     if self.key_enabled.get() else None)
        bpm = float(self.bpm_var.get()) if self.bpm_enabled.get() else None
        dur = round(self.duration_var.get(), 1)
        if bpm and self.snap_bar_var.get():
            bar = 4 * 60.0 / bpm
            dur = max(bar, round(dur / bar) * bar)
            dur = round(dur, 2)
        return root_freq, bpm, dur

    # ── Weirdify ───────────────────────────────────────────────────────────────

    def _build_weirdify(self, parent):
        c = card(parent, title=None, bg=PANEL, padx=14, pady=14)
        c.master.pack(fill="x", pady=(0, 14))

        hdr = tk.Frame(c, bg=PANEL)
        hdr.pack(fill="x", pady=(0, 14))
        tk.Label(hdr, text="WEIRDIFY", font=("SF Pro Text", 10, "bold"),
                 bg=PANEL, fg=TEXT).pack(side="left")
        tk.Label(hdr, text="warp the timbre", font=FONT_TINY,
                 bg=PANEL, fg=MUTED).pack(side="left", padx=(8, 0))
        tk.Button(hdr, text="Reset", font=FONT_TINY, bg=SURFACE, fg=MUTED,
                  relief="flat", cursor="hand2", bd=0, padx=8, pady=2,
                  highlightthickness=1, highlightbackground=BORDER,
                  command=self._reset_weird).pack(side="right")

        knob_row = tk.Frame(c, bg=PANEL)
        knob_row.pack()

        effects = [
            ("crush",   "Crush"),
            ("ring",    "Ring"),
            ("warp",    "Warp"),
            ("stutter", "Stutter"),
            ("glitch",  "Glitch"),
        ]

        self.weird_vars = {}
        for key, _ in effects:
            self.weird_vars[key] = tk.DoubleVar(value=0.0)

        labels = [lbl for _, lbl in effects]
        WeirdifyCanvas(knob_row, self.weird_vars, labels, bg=PANEL).pack()

    def _reset_weird(self):
        for v in self.weird_vars.values():
            v.set(0.0)

    def _weird_params(self):
        return {k: v.get() for k, v in self.weird_vars.items()}

    # ── Resample ──────────────────────────────────────────────────────────────

    def _build_resample(self, parent):
        c = card(parent, title=None, bg=PANEL, padx=14, pady=14)
        c.master.pack(fill="x", pady=(0, 14))

        hdr = tk.Frame(c, bg=PANEL)
        hdr.pack(fill="x", pady=(0, 10))
        tk.Label(hdr, text="RESAMPLE", font=FONT_SECTION,
                 bg=PANEL, fg=MUTED).pack(side="left")

        knob_row = tk.Frame(c, bg=PANEL)
        knob_row.pack(fill="x")

        self.resample_chop_var  = tk.DoubleVar(value=25.0)
        self.resample_reuse_var = tk.DoubleVar(value=0.0)
        Knob(knob_row, self.resample_chop_var, size=62,
             label="Chop Len", bg=PANEL).pack(side="left", padx=4)
        Knob(knob_row, self.resample_reuse_var, size=62,
             label="Chop Reuse", bg=PANEL).pack(side="left", padx=4)

        self.resample_btn = tk.Button(
            knob_row, text="Resample", bg=SURFACE, fg=TEXT, font=FONT_BTN,
            activebackground=BORDER, activeforeground=TEXT,
            command=self._do_resample)
        self.resample_btn.pack(side="left", padx=(16, 0), pady=20)

    def _do_resample(self):
        if self._preview_audio is None:
            self.preview_status.config(text="Nothing to resample — preview first.", fg=WARN)
            return
        self.resample_btn.config(state="disabled", text="Chopping…")
        threading.Thread(target=self._run_resample, daemon=True).start()

    def _run_resample(self):
        chop_pct = self.resample_chop_var.get() / 100.0
        chop_len = 0.02 + chop_pct * 0.98
        reuse    = self.resample_reuse_var.get() / 100.0

        audio = resample_chop(self._preview_audio, chop_length=chop_len,
                              chop_reuse=reuse)
        audio = self._apply_live(audio)

        self._preview_audio = audio
        if self._preview_tmp:
            write_wav(self._preview_tmp, audio)

        self.root.after(0, self.waveform.set_audio, audio)
        self.root.after(0, self.preview_status.config,
                        {"text": f"Resampled — {chop_len:.2f}s chops, {int(reuse*100)}% reuse",
                         "fg": SUCCESS})
        self.root.after(0, self.resample_btn.config,
                        {"state": "normal", "text": "Resample"})

        self._preview_proc = subprocess.Popen(["afplay", self._preview_tmp])
        self.root.after(0, self.waveform.start_playback)
        self.root.after(0, self.dancer.start)
        self._preview_proc.wait()
        self._preview_proc = None
        self.root.after(0, self.waveform.stop_playback)
        self.root.after(0, self.dancer.stop)
        self.root.after(0, self._on_preview_done)

    # ── Live Effects ──────────────────────────────────────────────────────────

    def _build_live_effects(self, parent):
        c = card(parent, title=None, bg=PANEL, padx=14, pady=14)
        c.master.pack(fill="x", pady=(0, 14))

        hdr = tk.Frame(c, bg=PANEL)
        hdr.pack(fill="x", pady=(0, 10))
        tk.Label(hdr, text="LIVE EFFECTS", font=FONT_SECTION,
                 bg=PANEL, fg=MUTED).pack(side="left")

        knob_row = tk.Frame(c, bg=PANEL)
        knob_row.pack(fill="x")

        self.fx_reverb_var = tk.DoubleVar(value=0.0)
        self.fx_delay_var  = tk.DoubleVar(value=0.0)
        self.fx_filter_var = tk.DoubleVar(value=0.0)
        for var, lbl in [(self.fx_reverb_var, "Reverb"),
                         (self.fx_delay_var,  "Delay"),
                         (self.fx_filter_var, "Filter")]:
            Knob(knob_row, var, size=62, label=lbl, bg=PANEL).pack(side="left", padx=4)

    def _live_fx_params(self):
        return {
            "reverb":     self.fx_reverb_var.get(),
            "delay":      self.fx_delay_var.get(),
            "filter_amt": self.fx_filter_var.get(),
        }

    def _apply_live(self, audio):
        fx = self._live_fx_params()
        if any(v > 0 for v in fx.values()):
            return apply_live_effects(audio, **fx)
        return audio

    # ── Waveform ───────────────────────────────────────────────────────────────

    def _build_waveform(self, parent):
        c = card(parent, "Waveform")
        c.master.pack(fill="x", pady=(0, 14))
        self.waveform = WaveformView(c, height=90)
        self.waveform.pack(fill="x")

    # ── Preview ────────────────────────────────────────────────────────────────

    def _build_preview(self, parent):
        c = card(parent, "Preview")
        c.master.pack(fill="x", pady=(0, 14))

        self.preview_status = tk.Label(c, text="No sound previewed yet.",
                                       font=FONT_SMALL, bg=PANEL, fg=MUTED,
                                       anchor="w")
        self.preview_status.pack(fill="x", pady=(0, 10))

        btn_row = tk.Frame(c, bg=PANEL)
        btn_row.pack(fill="x")

        def dark_btn(parent, text, command, state="normal"):
            return tk.Button(parent, text=text, font=FONT_BTN,
                             bg=DARK, fg=TEXT, relief="flat",
                             cursor="hand2", bd=0, padx=14, pady=7,
                             state=state, command=command,
                             activebackground="#484848", activeforeground=TEXT)

        def light_btn(parent, text, command, state="normal"):
            return tk.Button(parent, text=text, font=FONT_BTN,
                             bg=SURFACE, fg=TEXT, relief="flat",
                             cursor="hand2", bd=0, padx=14, pady=7,
                             state=state, command=command,
                             highlightthickness=1, highlightbackground=BORDER,
                             activebackground=BORDER, activeforeground=TEXT)

        self.preview_btn = dark_btn(btn_row, "▶  Preview", self._start_preview)
        self.preview_btn.pack(side="left", padx=(0, 8))

        self.stop_btn = light_btn(btn_row, "■  Stop", self._stop_preview, state="disabled")
        self.stop_btn.pack(side="left", padx=(0, 8))

        self.replay_btn = light_btn(btn_row, "↺  Replay", self._replay_preview, state="disabled")
        self.replay_btn.pack(side="left", padx=(0, 8))

        self.save_last_btn = light_btn(btn_row, "Save Last", self._save_last, state="disabled")
        self.save_last_btn.pack(side="left", padx=(0, 8))

        self.clear_preview_btn = light_btn(btn_row, "Clear", self._clear_preview, state="disabled")
        self.clear_preview_btn.pack(side="left")

        btn_row2 = tk.Frame(c, bg=PANEL)
        btn_row2.pack(fill="x", pady=(8, 0))

        self.pleasantize_btn = tk.Button(
            btn_row2, text="✦  Pleasantize", font=FONT_BTN,
            bg="#2d6e45", fg=TEXT, relief="flat", cursor="hand2",
            bd=0, padx=14, pady=7, state="disabled",
            activebackground="#3a8a56", activeforeground=TEXT,
            command=self._do_pleasantize,
        )
        self.pleasantize_btn.pack(side="left")

    # ── Log ────────────────────────────────────────────────────────────────────

    def _build_log(self, parent):
        c = card(parent, "Output")
        c.master.pack(fill="x", pady=(0, 14))

        self.log = scrolledtext.ScrolledText(
            c, height=6, font=FONT_MONO,
            bg=SURFACE, fg=TEXT, insertbackground=TEXT,
            relief="flat", bd=0, state="disabled",
        )
        self.log.pack(fill="x")

    # ── Actions ────────────────────────────────────────────────────────────────

    def _build_actions(self, parent):
        row = tk.Frame(parent, bg=BG)
        row.pack(fill="x")

        self.gen_btn = tk.Button(
            row, text="Generate Sounds", font=FONT_BTN,
            bg=DARK, fg=TEXT, relief="flat", cursor="hand2",
            bd=0, padx=20, pady=10,
            activebackground="#484848", activeforeground=TEXT,
            command=self._start_generation,
        )
        self.gen_btn.pack(side="left", fill="x", expand=True, padx=(0, 8))

        self.cancel_gen_btn = tk.Button(
            row, text="Stop", font=FONT_BTN,
            bg=WARN, fg=TEXT, relief="flat", cursor="hand2",
            bd=0, padx=16, pady=10, state="disabled",
            activebackground="#c0392b", activeforeground=TEXT,
            command=self._cancel_generation,
        )
        self.cancel_gen_btn.pack(side="left", padx=(0, 8))

        tk.Button(
            row, text="Clear Log", font=FONT_BTN,
            bg=SURFACE, fg=TEXT, relief="flat", cursor="hand2",
            bd=0, padx=20, pady=10,
            highlightthickness=1, highlightbackground=BORDER,
            activebackground=BORDER, activeforeground=TEXT,
            command=self._clear_log,
        ).pack(side="left")

    # ── Helpers ────────────────────────────────────────────────────────────────

    def _log(self, msg, color=None):
        self.log.config(state="normal")
        tag = f"t{abs(hash(color or 'default'))}"
        self.log.tag_config(tag, foreground=color or TEXT)
        self.log.insert("end", msg + "\n", tag)
        self.log.see("end")
        self.log.config(state="disabled")

    def _clear_log(self):
        self.log.config(state="normal")
        self.log.delete("1.0", "end")
        self.log.config(state="disabled")

    def _enabled_types(self):
        return [k for k, v in self.type_vars.items() if v.get()]

    # ── Preview logic ──────────────────────────────────────────────────────────

    def _start_preview(self):
        enabled = self._enabled_types()
        if not enabled:
            self.preview_status.config(text="No sound types selected.", fg=WARN)
            return
        self._stop_preview()
        self.preview_btn.config(state="disabled", text="Generating…")
        threading.Thread(target=self._run_preview,
                         args=(enabled,), daemon=True).start()

    def _run_preview(self, enabled_types):
        root_freq, bpm, dur = self._get_tuning()
        sound_type = random.choice(enabled_types)
        audio, _ = generate_sound(sound_type, duration=dur,
                                  root_freq=root_freq, bpm=bpm)

        wp = self._weird_params()
        if any(v > 0 for v in wp.values()):
            audio = apply_weirdify(audio, **wp, bpm=bpm)

        audio = self._apply_live(audio)

        tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
        tmp.close()
        write_wav(tmp.name, audio)

        self._preview_audio = audio
        self._preview_type  = sound_type
        self._preview_tmp   = tmp.name

        label = sound_type.replace("_", " ").title()
        self.root.after(0, self.waveform.set_audio, audio)
        self.root.after(0, self.preview_status.config,
                        {"text": f"Playing: {label}", "fg": SUCCESS})
        self.root.after(0, self.preview_btn.config,
                        {"state": "normal", "text": "▶  Preview"})
        self.root.after(0, self.stop_btn.config, {"state": "normal"})
        self.root.after(0, self.replay_btn.config, {"state": "disabled"})
        self.root.after(0, self.save_last_btn.config, {"state": "normal"})
        self.root.after(0, self.pleasantize_btn.config, {"state": "normal"})
        self.root.after(0, self.clear_preview_btn.config, {"state": "normal"})

        self._preview_proc = subprocess.Popen(["afplay", tmp.name])
        self.root.after(0, self.waveform.start_playback)
        self.root.after(0, self.dancer.start)
        self._preview_proc.wait()
        self._preview_proc = None
        self.root.after(0, self.waveform.stop_playback)
        self.root.after(0, self.dancer.stop)
        self.root.after(0, self._on_preview_done)

    def _on_preview_done(self):
        if self._preview_type:
            label = self._preview_type.replace("_", " ").title()
            self.preview_status.config(
                text=f"Last: {label}  —  click Save Last to keep it", fg=MUTED)
        self.stop_btn.config(state="disabled")
        self.replay_btn.config(state="normal")

    def _generate_sequence_preview(self):
        if not self.key_enabled.get():
            self.preview_status.config(text="Enable Key tuning first.", fg=WARN)
            return

        enabled = self._enabled_types()
        if not enabled:
            self.preview_status.config(text="No sound types selected.", fg=WARN)
            return

        self._stop_preview()
        self.generate_sequence_btn.config(state="disabled", text="Generating…")
        threading.Thread(target=self._run_sequence_preview, daemon=True).start()

    def _run_sequence_preview(self):
        try:
            root_note  = self.key_note_var.get()
            octave     = self.key_octave_var.get()
            scale_type = self.sequence_scale_var.get()
            num_notes  = self.sequence_notes_var.get()
            note_value = self.sequence_note_value_var.get()

            _, bpm, _ = self._get_tuning()

            note_fractions = {"1/16": 0.25, "1/8": 0.5, "1/4": 1.0, "1/2": 2.0, "1": 4.0}
            beats = note_fractions.get(note_value, 0.5)
            effective_bpm = bpm if bpm else 120.0
            note_duration = beats * (60.0 / effective_bpm)

            enabled = self._enabled_types()
            sound_type = random.choice(enabled) if enabled else None

            swing          = self.seq_swing_var.get() / 100.0
            velocity       = self.seq_velocity_var.get() / 100.0
            vel_variance   = self.seq_vel_variance_var.get() / 100.0
            len_variance   = self.seq_variance_var.get() / 100.0
            phrase_contour = self.seq_phrase_var.get() / 100.0
            accent_str     = self.seq_accent_var.get() / 100.0

            audio, seq_label = generate_random_sequence(
                root_note, octave, scale_type, num_notes,
                sound_type=sound_type, bpm=effective_bpm,
                note_duration=note_duration,
                swing=swing, velocity=velocity,
                velocity_variance=vel_variance, note_variance=len_variance,
                phrase_contour=phrase_contour, accent_strength=accent_str)

            wp = self._weird_params()
            if any(v > 0 for v in wp.values()):
                audio = apply_weirdify(audio, **wp, bpm=bpm)

            audio = self._apply_live(audio)

            tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
            tmp.close()
            write_wav(tmp.name, audio)

            self._preview_audio = audio
            self._preview_type = seq_label
            self._preview_tmp = tmp.name

            label = f"{scale_type} {root_note}{octave} ({num_notes} notes)"
            self.root.after(0, self.waveform.set_audio, audio)
            self.root.after(0, self.preview_status.config,
                           {"text": f"Playing: {label}", "fg": SUCCESS})
            self.root.after(0, self.generate_sequence_btn.config,
                           {"state": "normal", "text": "Generate Sequence"})
            self.root.after(0, self.stop_btn.config, {"state": "normal"})
            self.root.after(0, self.replay_btn.config, {"state": "disabled"})
            self.root.after(0, self.save_last_btn.config, {"state": "normal"})
            self.root.after(0, self.pleasantize_btn.config, {"state": "normal"})
            self.root.after(0, self.clear_preview_btn.config, {"state": "normal"})

            self._preview_proc = subprocess.Popen(["afplay", tmp.name])
            self.root.after(0, self.waveform.start_playback)
            self.root.after(0, self.dancer.start)
            self._preview_proc.wait()
            self._preview_proc = None
            self.root.after(0, self.waveform.stop_playback)
            self.root.after(0, self.dancer.stop)
            self.root.after(0, self._on_preview_done)
        except Exception as e:
            self.root.after(0, self.preview_status.config,
                           {"text": f"Error: {str(e)}", "fg": WARN})
            self.root.after(0, self.generate_sequence_btn.config,
                           {"state": "normal", "text": "Generate Sequence"})

    def _stop_preview(self):
        if self._preview_proc and self._preview_proc.poll() is None:
            self._preview_proc.terminate()
        self.waveform.stop_playback()
        self.dancer.stop()
        self.stop_btn.config(state="disabled")
        self.replay_btn.config(state="normal" if self._preview_tmp else "disabled")

    def _replay_preview(self):
        if not self._preview_tmp or not os.path.exists(self._preview_tmp):
            return
        self._stop_preview()
        self.stop_btn.config(state="normal")
        self.replay_btn.config(state="disabled")
        threading.Thread(target=self._play_temp, daemon=True).start()

    def _play_temp(self):
        self._preview_proc = subprocess.Popen(["afplay", self._preview_tmp])
        self.root.after(0, self.waveform.start_playback)
        self.root.after(0, self.dancer.start)
        self._preview_proc.wait()
        self._preview_proc = None
        self.root.after(0, self.waveform.stop_playback)
        self.root.after(0, self.dancer.stop)
        self.root.after(0, self._on_preview_done)

    def _do_pleasantize(self):
        if self._preview_audio is None:
            return
        self.pleasantize_btn.config(state="disabled", text="Processing…")
        threading.Thread(target=self._run_pleasantize, daemon=True).start()

    def _run_pleasantize(self):
        audio = pleasantize(self._preview_audio)
        self._preview_audio = audio
        if self._preview_tmp:
            write_wav(self._preview_tmp, audio)
        self.root.after(0, self.waveform.set_audio, audio)
        self.root.after(0, self.preview_status.config,
                        {"text": "Pleasantized ✦  — sound smoothed and warmed",
                         "fg": SUCCESS})
        self.root.after(0, self.pleasantize_btn.config,
                        {"state": "normal", "text": "✦  Pleasantize"})

    def _save_last(self):
        if self._preview_audio is None:
            return
        folder = self.folder_var.get().strip() or "output"
        os.makedirs(folder, exist_ok=True)
        existing = [f for f in os.listdir(folder) if f.endswith(".wav")]
        idx      = len(existing) + 1
        filename = f"sound_{idx:03d}_{self._preview_type}.wav"
        write_wav(os.path.join(folder, filename), self._preview_audio)
        self._log(f"Saved → {filename}", SUCCESS)
        self.preview_status.config(text=f"Saved: {filename}", fg=SUCCESS)

    def _clear_preview(self):
        self._stop_preview()
        if self._preview_tmp and os.path.exists(self._preview_tmp):
            os.unlink(self._preview_tmp)
        self._preview_audio = None
        self._preview_type  = None
        self._preview_tmp   = None
        self.waveform.clear()
        self.preview_status.config(text="No sound previewed yet.", fg=MUTED)
        self.save_last_btn.config(state="disabled")
        self.replay_btn.config(state="disabled")
        self.pleasantize_btn.config(state="disabled")
        self.clear_preview_btn.config(state="disabled")

    # ── Batch generation ───────────────────────────────────────────────────────

    def _cancel_generation(self):
        self._stop_gen.set()

    def _start_generation(self):
        enabled = self._enabled_types()
        if not enabled:
            self._log("No sound types selected.", WARN)
            return
        self._stop_gen.clear()
        self.gen_btn.config(state="disabled", text="Generating…")
        self.cancel_gen_btn.config(state="normal")
        threading.Thread(target=self._run_generation,
                         args=(enabled,), daemon=True).start()

    def _run_generation(self, enabled_types):
        count  = int(self.count_var.get())
        folder = self.folder_var.get().strip() or "output"
        seed   = self.seed_var.get().strip()

        if seed:
            try:
                s = int(seed)
                random.seed(s)
                np.random.seed(s)
            except ValueError:
                self.root.after(0, self._log,
                                f"Invalid seed '{seed}', using random.", WARN)

        os.makedirs(folder, exist_ok=True)
        self.root.after(0, self._log, f"→ {folder}", MUTED)

        root_freq, bpm, dur = self._get_tuning()
        wp                  = self._weird_params()
        do_weird            = any(v > 0 for v in wp.values())

        saved = 0
        for i in range(1, count + 1):
            if self._stop_gen.is_set():
                self.root.after(0, self._log,
                                f"Stopped after {saved} file(s).", WARN)
                break

            sound_type = random.choice(enabled_types)
            audio, _   = generate_sound(sound_type, duration=dur,
                                        root_freq=root_freq, bpm=bpm)

            if do_weird:
                audio = apply_weirdify(audio, **wp, bpm=bpm)

            audio = self._apply_live(audio)

            filename = f"sound_{i:03d}_{sound_type}.wav"
            write_wav(os.path.join(folder, filename), audio)
            saved += 1
            self.root.after(0, self._log,
                            f"  [{i:>2}/{count}]  {filename}")
        else:
            self.root.after(0, self._log, f"Done. {saved} file(s) saved.", SUCCESS)

        self.root.after(0, self.gen_btn.config,
                        {"state": "normal", "text": "Generate Sounds"})
        self.root.after(0, self.cancel_gen_btn.config, {"state": "disabled"})


# ── Entry point ────────────────────────────────────────────────────────────────

def main():
    root = tk.Tk()
    style = ttk.Style(root)
    style.theme_use("clam")
    style.configure("Horizontal.TScale",
                    background=PANEL, troughcolor=BORDER,
                    sliderlength=14, sliderrelief="flat",
                    troughrelief="flat")
    SoundGenUI(root)
    root.mainloop()


if __name__ == "__main__":
    main()
