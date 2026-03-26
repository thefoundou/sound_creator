"""
Reusable UI widgets: DancingMan, Knob, WeirdifyCanvas, Toggle, WaveformView, card, divider.
Also defines the dark-mode palette and font constants used throughout the UI.
"""

import tkinter as tk
import math
import time
import os
import numpy as np

import sys

try:
    from PIL import Image as _PILImage, ImageDraw as _PILDraw
    from PIL import ImageFilter as _PILFilter, ImageTk as _PILImageTk
    _PIL_OK = True
except ImportError:
    _PIL_OK = False

# On macOS, tk.Button ignores bg/fg. tkmacosx.Button respects them.
_MACOS = sys.platform == "darwin"
try:
    from tkmacosx import Button as _MacButton
    _MACBTN_OK = _MACOS
except ImportError:
    _MACBTN_OK = False

from generators import SAMPLE_RATE

# ── Palette ────────────────────────────────────────────────────────────────────
BG      = "#141414"
PANEL   = "#1e1e1e"
SURFACE = "#282828"
BORDER  = "#383838"
TEXT    = "#e8e8e8"
MUTED   = "#707070"
SUCCESS = "#3dba6a"
WARN    = "#e74c3c"
DARK    = "#303030"
KNOB_ARC = "#e8e8e8"

FONT_SECTION = ("SF Pro Text",  8, "bold")
FONT_LABEL   = ("SF Pro Text", 11)
FONT_SMALL   = ("SF Pro Text", 10)
FONT_TINY    = ("SF Pro Text",  8)
FONT_MONO    = ("SF Mono",     10)
FONT_BTN     = ("SF Pro Text", 11, "bold")
FONT_TITLE   = ("SF Pro Display", 15, "bold")


# ── Dark-mode Button ──────────────────────────────────────────────────────────

def DarkButton(parent, text="", bg=DARK, fg=TEXT, font=FONT_BTN,
               activebackground=None, activeforeground=None,
               command=None, state="normal", padx=14, pady=7, **kw):
    """
    Create a button that actually respects bg/fg on macOS.
    Falls back to tk.Button on other platforms.
    """
    abg = activebackground or bg
    afg = activeforeground or fg
    if _MACBTN_OK:
        btn = _MacButton(
            parent, text=text, bg=bg, fg=fg, font=font,
            activebackground=abg, activeforeground=afg,
            borderless=True, command=command, state=state,
            cursor="hand2", padx=padx, pady=pady, **kw)
    else:
        btn = tk.Button(
            parent, text=text, bg=bg, fg=fg, font=font,
            activebackground=abg, activeforeground=afg,
            relief="flat", bd=0, command=command, state=state,
            cursor="hand2", padx=padx, pady=pady, **kw)
    return btn


# ── Dancing Man animation ──────────────────────────────────────────────────────

class DancingMan(tk.Canvas):
    """
    Pixel-art character that boomerangs through 3 sprite frames while audio plays.
    Frames cycle: 0 → 1 → 2 → 1 → 0 → 1 → …
    """
    FRAME_MS = 180   # ms between frame changes

    def __init__(self, parent, frame_dir, scale=8, bg=BG, **kw):
        self._frames = []
        frame_files = sorted(
            f for f in os.listdir(frame_dir) if f.lower().endswith(".png"))

        for fname in frame_files:
            path = os.path.join(frame_dir, fname)
            img  = _PILImage.open(path).convert("RGBA")
            w, h = img.size
            scaled = img.resize((w * scale, h * scale), _PILImage.NEAREST)
            bg_rgb = tuple(int(bg.lstrip("#")[i*2:i*2+2], 16) for i in range(3))
            bg_img = _PILImage.new("RGBA", scaled.size, (*bg_rgb, 255))
            composited = _PILImage.alpha_composite(bg_img, scaled)
            self._frames.append(_PILImageTk.PhotoImage(composited.convert("RGB")))

        W = self._frames[0].width()  if self._frames else 100
        H = self._frames[0].height() if self._frames else 100
        super().__init__(parent, width=W, height=H,
                         bg=bg, highlightthickness=0, **kw)

        self._sequence    = [0, 1, 2, 1]
        self._seq_idx     = 0
        self._playing     = False
        self._anim_id     = None
        self._img_id      = self.create_image(W // 2, H // 2,
                                               image=self._frames[0])

    def start(self):
        if self._playing:
            return
        self._playing = True
        self._seq_idx = 0
        self._animate()

    def stop(self):
        self._playing = False
        if self._anim_id:
            self.after_cancel(self._anim_id)
            self._anim_id = None
        if self._frames:
            self.itemconfig(self._img_id, image=self._frames[0])

    def _animate(self):
        if not self._playing or not self._frames:
            return
        idx = self._sequence[self._seq_idx % len(self._sequence)]
        self.itemconfig(self._img_id, image=self._frames[idx])
        self._seq_idx += 1
        self._anim_id = self.after(self.FRAME_MS, self._animate)


# ── Sphere shading kernel (shared by Knob and WeirdifyCanvas) ──────────────────

def _make_sphere_kernel(ri):
    """Phong-shaded metallic disc. Returns (rgb uint8 H×W×3, bool mask H×W)."""
    size = 2 * ri + 1
    y_idx, x_idx = np.ogrid[:size, :size]
    dx = (x_idx - ri).astype(np.float32)
    dy = (y_idx - ri).astype(np.float32)
    r2     = dx ** 2 + dy ** 2
    inside = r2 <= float(ri * ri)
    nz = np.where(inside, np.sqrt(np.maximum(0.0, 1.0 - r2 / (ri * ri))), 0.0).astype(np.float32)
    nx = (dx / max(ri, 1)).astype(np.float32)
    ny = (dy / max(ri, 1)).astype(np.float32)
    s3   = 0.5774
    diff = np.maximum(0.0, nx * (-s3) + ny * (-s3) + nz * s3)
    spec = diff ** 12
    shade = np.where(inside,
                     np.clip(155.0 + 75.0 * diff + 50.0 * spec, 0, 255),
                     0.0).astype(np.uint8)
    return np.stack([shade, shade, shade], axis=2), inside


# ── Knob widget ────────────────────────────────────────────────────────────────

class Knob(tk.Canvas):
    """
    Rotary knob — PIL bloom rendering matching WeirdifyCanvas aesthetic.
    Canvas is padded by _BM on each side so bloom overflows the knob body.
    """
    START  = 225
    SWEEP  = -270
    GREEN  = (45,  185,  80)
    YELLOW = (255, 200,   0)
    RED    = (220,  50,  50)
    _BM    = 28
    _PS    = 200

    @staticmethod
    def _hex_to_rgb(h):
        h = h.lstrip("#")
        return (int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16))

    def _arc_rgb(self, t):
        if t <= 0.5:
            u, A, B = t * 2, self.GREEN, self.YELLOW
        else:
            u, A, B = (t - 0.5) * 2, self.YELLOW, self.RED
        return (int(A[0] + u * (B[0] - A[0])),
                int(A[1] + u * (B[1] - A[1])),
                int(A[2] + u * (B[2] - A[2])))

    def __init__(self, parent, var, size=62, label="", bg=PANEL, **kw):
        bm = self._BM
        W  = size + 2 * bm
        H  = size + 22 + bm
        super().__init__(parent, width=W, height=H,
                         bg=bg, highlightthickness=0, cursor="hand2", **kw)
        self.var     = var
        self.size    = size
        self.label   = label
        self._bg     = bg
        self._bg_rgb = self._hex_to_rgb(bg)
        self._photo  = None
        self._y0 = self._v0 = None
        ri = size // 2 - 11
        self._sphere_rgb, self._sphere_mask = _make_sphere_kernel(ri)

        var.trace_add("write", lambda *_: self.after(0, self._draw))
        self.bind("<ButtonPress-1>",   self._press)
        self.bind("<B1-Motion>",       self._drag)
        self.bind("<Double-Button-1>", self._reset)
        self._draw()

    def _make_bloom(self, val):
        if val < 0.005:
            return None
        PS = self._PS
        hp = PS // 2
        ro = float(self.size // 2 - 3)
        arc_sigma = 1.5

        y_idx, x_idx = np.ogrid[0:PS, 0:PS]
        dx = (x_idx - hp).astype(np.float32)
        dy = (hp - y_idx).astype(np.float32)
        dist = np.sqrt(dx**2 + dy**2)

        ring_alpha = np.exp(-0.5 * ((dist - ro) / arc_sigma)**2).astype(np.float32)
        angle  = np.degrees(np.arctan2(dy, dx)).astype(np.float32) % 360.0
        rel    = (225.0 - angle) % 360.0
        in_lit = (rel <= val * 270.0).astype(np.float32)
        t      = np.where(in_lit > 0, rel / 270.0, 0.0).astype(np.float32)

        G = np.array(self.GREEN,  dtype=np.float32)
        Y = np.array(self.YELLOW, dtype=np.float32)
        R = np.array(self.RED,    dtype=np.float32)
        t3 = t[:, :, np.newaxis]
        u1 = np.clip(t3 * 2.0, 0.0, 1.0)
        u2 = np.clip((t3 - 0.5) * 2.0, 0.0, 1.0)
        color = np.where(t3 <= 0.5, G + u1*(Y-G), Y + u2*(R-Y)).astype(np.float32)

        alpha   = ring_alpha * in_lit
        arc_np  = color * alpha[:, :, np.newaxis]
        arc_pil = _PILImage.fromarray(np.clip(arc_np, 0, 255).astype(np.uint8))
        bloom   = arc_np * 2.2
        for radius, strength in [(3, 1.7), (16, 0.95), (55, 0.42)]:
            b = np.array(arc_pil.filter(_PILFilter.GaussianBlur(radius=radius)), dtype=np.float32)
            bloom += b * strength
        return bloom

    def _draw(self):
        self.delete("all")
        if not _PIL_OK:
            self._draw_tk()
            return
        s  = self.size
        bm = self._BM
        W  = s + 2 * bm
        H  = s + 22 + bm
        PS = self._PS
        val = max(0.0, min(1.0, self.var.get() / 100.0))
        cx  = W // 2
        cy  = bm + s // 2 - 2
        ro  = s // 2 - 3
        ri  = s // 2 - 11

        bg   = self._bg_rgb
        base = np.full((H, W, 3), bg, dtype=np.float32)

        bloom = self._make_bloom(val)
        if bloom is not None:
            b0_y = cy - PS // 2;  b1_y = b0_y + PS
            b0_x = cx - PS // 2;  b1_x = b0_x + PS
            sk_y0 = max(0, -b0_y);  sk_y1 = PS - max(0, b1_y - H)
            sk_x0 = max(0, -b0_x);  sk_x1 = PS - max(0, b1_x - W)
            c_y0  = max(0, b0_y);   c_y1  = min(H, b1_y)
            c_x0  = max(0, b0_x);   c_x1  = min(W, b1_x)
            if c_y1 > c_y0 and c_x1 > c_x0:
                base[c_y0:c_y1, c_x0:c_x1] = np.clip(
                    base[c_y0:c_y1, c_x0:c_x1] +
                    bloom[sk_y0:sk_y1, sk_x0:sk_x1] * 0.72, 0, 255)

        sph_rgb, sph_mask = self._sphere_rgb, self._sphere_mask
        si  = sph_rgb.shape[0]
        sy1 = cy - si // 2;  sy2 = sy1 + si
        sx1 = cx - si // 2;  sx2 = sx1 + si
        sk_y0 = max(0, -sy1);  sk_y1 = si - max(0, sy2 - H)
        sk_x0 = max(0, -sx1);  sk_x1 = si - max(0, sx2 - W)
        sy1 = max(0, sy1);  sy2 = min(H, sy2)
        sx1 = max(0, sx1);  sx2 = min(W, sx2)
        base[sy1:sy2, sx1:sx2] = np.where(
            sph_mask[sk_y0:sk_y1, sk_x0:sk_x1][:, :, np.newaxis],
            sph_rgb[sk_y0:sk_y1, sk_x0:sk_x1],
            base[sy1:sy2, sx1:sx2])

        result = _PILImage.fromarray(np.clip(base, 0, 255).astype(np.uint8))
        draw   = _PILDraw.Draw(result)

        bbox = [cx - ro, cy - ro, cx + ro, cy + ro]
        pil_s = (360.0 - self.START) % 360.0
        pil_e = (pil_s + abs(self.SWEEP)) % 360.0
        draw.arc(bbox, start=pil_s, end=pil_e, fill=(72, 72, 72), width=3)

        draw.ellipse([cx - ri - 1, cy - ri - 1, cx + ri + 1, cy + ri + 1],
                     outline=(90, 90, 90), width=1)

        if val > 0.01:
            rim = self._arc_rgb(min(val, 0.99))
            rim_dim = tuple(max(0, int(c * 0.4)) for c in rim)
            draw.ellipse([cx - ri, cy - ri, cx + ri, cy + ri],
                         outline=rim_dim, width=2)

        angle_rad = math.radians(self.START + self.SWEEP * val)
        dot_r = ri - 7
        dot_x = int(cx + dot_r * math.cos(angle_rad))
        dot_y = int(cy - dot_r * math.sin(angle_rad))
        draw.ellipse([dot_x - 3, dot_y - 3, dot_x + 3, dot_y + 3],
                     fill=(48, 48, 48))

        self._photo = _PILImageTk.PhotoImage(result)
        self.config(width=W, height=H)
        self.create_image(0, 0, anchor="nw", image=self._photo)
        self.create_text(cx, cy,
                         text=str(int(self.var.get())),
                         font=("SF Pro Text", 9, "bold"), fill=TEXT)
        self.create_text(cx, bm + s + 11,
                         text=self.label.upper(),
                         font=FONT_TINY, fill=MUTED)

    def _draw_tk(self):
        """Minimal fallback when PIL is unavailable."""
        self.delete("all")
        s = self.size;  bm = self._BM
        cx = bm + s // 2;  cy = bm + s // 2 - 2
        ro = s // 2 - 3;   ri = s // 2 - 11
        val = max(0.0, min(1.0, self.var.get() / 100.0))
        pad = bm + s // 2 - ro
        self.create_arc(pad, pad, bm + s - pad, bm + s - pad,
                        start=self.START, extent=self.SWEEP,
                        style="arc", outline=BORDER, width=3)
        self.create_oval(cx - ri, cy - ri, cx + ri, cy + ri,
                         fill=SURFACE, outline=BORDER, width=1)
        angle_rad = math.radians(self.START + self.SWEEP * val)
        dot_r = ri - 7
        self.create_oval(
            int(cx + dot_r * math.cos(angle_rad)) - 3,
            int(cy - dot_r * math.sin(angle_rad)) - 3,
            int(cx + dot_r * math.cos(angle_rad)) + 3,
            int(cy - dot_r * math.sin(angle_rad)) + 3,
            fill=DARK, outline="")
        self.create_text(cx, cy, text=str(int(self.var.get())),
                         font=("SF Pro Text", 9, "bold"), fill=TEXT)
        self.create_text(cx, bm + s + 11, text=self.label.upper(),
                         font=FONT_TINY, fill=MUTED)

    def _press(self, e):
        self._y0 = e.y_root
        self._v0 = self.var.get()

    def _drag(self, e):
        if self._y0 is None:
            return
        delta = (self._y0 - e.y_root) * 0.9
        self.var.set(max(0.0, min(100.0, self._v0 + delta)))

    def _reset(self, _):
        self.var.set(0.0)


# ── Weirdify knob canvas (PIL-based, bloom-aware) ──────────────────────────────

try:
    _LANCZOS = _PILImage.Resampling.LANCZOS
except AttributeError:
    _LANCZOS = _PILImage.LANCZOS  # type: ignore[attr-defined]


class WeirdifyCanvas(tk.Canvas):
    """
    Renders all Weirdify knobs on a single PIL surface so bloom radiates
    outward, overflows knob bounds, and blends additively between knobs.
    """
    START = 225
    SWEEP = -270

    GREEN  = (45,  185,  80)
    YELLOW = (255, 200,   0)
    RED    = (220,  50,  50)

    _BM = 28
    _SP = 84
    _PS = 280

    @staticmethod
    def _hex_to_rgb(h):
        h = h.lstrip("#")
        return (int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16))

    def _arc_rgb(self, t):
        if t <= 0.5:
            u, A, B = t * 2, self.GREEN, self.YELLOW
        else:
            u, A, B = (t - 0.5) * 2, self.YELLOW, self.RED
        return (int(A[0] + u * (B[0] - A[0])),
                int(A[1] + u * (B[1] - A[1])),
                int(A[2] + u * (B[2] - A[2])))

    def __init__(self, parent, vars_dict, labels, knob_size=62, bg=PANEL, **kw):
        self._n      = len(vars_dict)
        self._ks     = knob_size
        self._bg_rgb = self._hex_to_rgb(bg)

        W = 2 * self._BM + self._n * self._SP
        H = 2 * self._BM + self._ks + 22

        super().__init__(parent, width=W, height=H,
                         bg=bg, highlightthickness=0, cursor="hand2", **kw)

        self._vars       = list(vars_dict.values())
        self._labels     = labels
        self._photo      = None
        self._y0 = self._v0 = self._di = None
        self._render_job = None
        self._is_dragging = False

        self._cached_vals   = [None] * self._n
        self._cached_blooms = [None] * self._n

        self._sphere_data = self._make_sphere(knob_size // 2 - 11)

        for var in self._vars:
            var.trace_add("write", lambda *_: self._schedule_render())

        self.bind("<ButtonPress-1>",   self._press)
        self.bind("<B1-Motion>",       self._drag)
        self.bind("<ButtonRelease-1>", self._release)
        self.bind("<Double-Button-1>", self._reset)
        self.bind("<Configure>",       lambda _: self._schedule_render())
        self.after(80, self._render)

    @staticmethod
    def _make_sphere(ri):
        return _make_sphere_kernel(ri)

    def _schedule_render(self):
        if self._render_job:
            self.after_cancel(self._render_job)
        delay = 0 if self._is_dragging else 16
        self._render_job = self.after(delay, self._render)

    def _cx(self, i):
        return self._BM + i * self._SP + self._SP // 2

    def _cy(self):
        return self._BM + self._ks // 2 - 2

    def _nearest_knob(self, ex):
        best, best_d = 0, abs(ex - self._cx(0))
        for i in range(1, self._n):
            d = abs(ex - self._cx(i))
            if d < best_d:
                best, best_d = i, d
        return best, best_d

    def _press(self, e):
        idx, dist = self._nearest_knob(e.x)
        if dist <= self._ks // 2 + 8:
            self._di = idx
            self._y0 = e.y_root
            self._v0 = self._vars[idx].get()
            self._is_dragging = True

    def _drag(self, e):
        if self._di is None or self._y0 is None:
            return
        delta = (self._y0 - e.y_root) * 0.9
        self._vars[self._di].set(max(0.0, min(100.0, self._v0 + delta)))

    def _release(self, e):
        if self._is_dragging and self._di is not None:
            self._cached_vals[self._di] = None
        self._is_dragging = False
        self._di = None
        self._schedule_render()

    def _reset(self, e):
        idx, dist = self._nearest_knob(e.x)
        if dist <= self._ks // 2 + 8:
            self._vars[idx].set(0.0)

    def _make_knob_bloom(self, val, fast=False):
        if val < 0.005:
            return None

        PS = self._PS
        hp = PS // 2
        ro = float(self._ks // 2 - 3)
        arc_sigma = 1.5

        y_idx, x_idx = np.ogrid[0:PS, 0:PS]
        dx = (x_idx - hp).astype(np.float32)
        dy = (hp - y_idx).astype(np.float32)
        dist = np.sqrt(dx**2 + dy**2)

        ring_alpha = np.exp(-0.5 * ((dist - ro) / arc_sigma)**2).astype(np.float32)
        angle = np.degrees(np.arctan2(dy, dx)).astype(np.float32) % 360.0
        rel = (225.0 - angle) % 360.0
        in_lit = (rel <= val * 270.0).astype(np.float32)
        t = np.where(in_lit > 0, rel / 270.0, 0.0).astype(np.float32)

        G = np.array(self.GREEN,  dtype=np.float32)
        Y = np.array(self.YELLOW, dtype=np.float32)
        R = np.array(self.RED,    dtype=np.float32)

        t3 = t[:, :, np.newaxis]
        u1 = np.clip(t3 * 2.0, 0.0, 1.0)
        u2 = np.clip((t3 - 0.5) * 2.0, 0.0, 1.0)
        color = np.where(t3 <= 0.5,
                         G + u1 * (Y - G),
                         Y + u2 * (R - Y)).astype(np.float32)

        alpha = ring_alpha * in_lit
        arc_np = color * alpha[:, :, np.newaxis]

        arc_pil = _PILImage.fromarray(np.clip(arc_np, 0, 255).astype(np.uint8))

        bloom = arc_np * 2.2
        passes = [(3, 1.7), (16, 0.95)] if fast else [(3, 1.7), (16, 0.95), (55, 0.42)]
        for radius, strength in passes:
            b = np.array(arc_pil.filter(_PILFilter.GaussianBlur(radius=radius)),
                         dtype=np.float32)
            bloom += b * strength

        return bloom

    def _render(self):
        self._render_job = None
        if not _PIL_OK:
            return
        W = self.winfo_width()
        H = self.winfo_height()
        if W < 10 or H < 10:
            return

        ks  = self._ks
        ro  = ks // 2 - 3
        ri  = ks // 2 - 11
        N   = 60
        seg = self.SWEEP / N

        fast  = self._is_dragging
        bloom = np.zeros((H, W, 3), dtype=np.float32)
        hp    = self._PS // 2

        for i, var in enumerate(self._vars):
            val = round(max(0.0, min(1.0, var.get() / 100.0)), 3)

            if val != self._cached_vals[i]:
                self._cached_blooms[i] = self._make_knob_bloom(val, fast=fast)
                self._cached_vals[i]   = val

            patch = self._cached_blooms[i]
            if patch is None:
                continue

            cx, cy = self._cx(i), self._cy()
            y1, x1 = cy - hp, cx - hp
            y2, x2 = cy + hp, cx + hp
            iy1, ix1 = max(0, y1), max(0, x1)
            iy2, ix2 = min(H, y2), min(W, x2)
            py1, px1 = iy1 - y1, ix1 - x1
            py2 = py1 + (iy2 - iy1)
            px2 = px1 + (ix2 - ix1)
            bloom[iy1:iy2, ix1:ix2] += patch[py1:py2, px1:px2]

        bg_f     = np.array(self._bg_rgb, dtype=np.float32)
        base     = np.full((H, W, 3), bg_f, dtype=np.float32)
        result_np = np.clip(base + bloom * 0.78, 0, 255).astype(np.uint8)

        sphere_rgb, sphere_mask = self._sphere_data
        for i in range(self._n):
            cx, cy = self._cx(i), self._cy()
            ky1, kx1 = cy - ri, cx - ri
            ky2, kx2 = cy + ri + 1, cx + ri + 1
            iy1, ix1 = max(0, ky1), max(0, kx1)
            iy2, ix2 = min(H, ky2), min(W, kx2)
            sk_y1, sk_x1 = iy1 - ky1, ix1 - kx1
            sk_y2 = sk_y1 + (iy2 - iy1)
            sk_x2 = sk_x1 + (ix2 - ix1)
            result_np[iy1:iy2, ix1:ix2] = np.where(
                sphere_mask[sk_y1:sk_y2, sk_x1:sk_x2][:, :, np.newaxis],
                sphere_rgb [sk_y1:sk_y2, sk_x1:sk_x2],
                result_np  [iy1:iy2,     ix1:ix2])

        result = _PILImage.fromarray(result_np)
        draw   = _PILDraw.Draw(result)

        ps_track = (360.0 - self.START) % 360.0
        pe_track = ps_track + abs(self.SWEEP)

        for i, var in enumerate(self._vars):
            val = max(0.0, min(1.0, var.get() / 100.0))
            cx, cy = self._cx(i), self._cy()

            draw.arc([cx - ro, cy - ro, cx + ro, cy + ro],
                     start=ps_track, end=pe_track,
                     fill=(72, 72, 72), width=3)

            draw.ellipse([cx - ri, cy - ri, cx + ri, cy + ri],
                         outline=(90, 90, 90), width=1)

            if val > 0.01:
                n_lit = max(1, int(val * N))
                rim_r = ri - 1
                for j in range(0, n_lit, 3):
                    t   = j / max(N - 1, 1)
                    rgb = self._arc_rgb(t)
                    rim_c = (min(255, int(rgb[0] * 0.42 + 178)),
                             min(255, int(rgb[1] * 0.42 + 178)),
                             min(255, int(rgb[2] * 0.42 + 178)))
                    tk_s = self.START + j * seg
                    ps   = (360.0 - tk_s) % 360.0
                    pe   = ps + abs(seg * 3)
                    draw.arc([cx - rim_r, cy - rim_r, cx + rim_r, cy + rim_r],
                             start=ps, end=pe, fill=rim_c, width=2)

            angle_rad = math.radians(self.START + self.SWEEP * val)
            dot_r = ri - 7
            dot_x = cx + dot_r * math.cos(angle_rad)
            dot_y = cy - dot_r * math.sin(angle_rad)
            draw.ellipse([dot_x - 3, dot_y - 3, dot_x + 3, dot_y + 3],
                         fill=(26, 26, 26))

        self._photo = _PILImageTk.PhotoImage(result)
        self.delete("all")
        self.create_image(0, 0, anchor="nw", image=self._photo)

        for i, (var, lbl) in enumerate(zip(self._vars, self._labels)):
            cx, cy = self._cx(i), self._cy()
            self.create_text(cx, cy,
                             text=str(int(var.get())),
                             font=("SF Pro Text", 9, "bold"), fill=TEXT)
            self.create_text(cx, self._BM + self._ks + 11,
                             text=lbl.upper(), font=FONT_TINY, fill=MUTED)


# ── Toggle switch ──────────────────────────────────────────────────────────────

class Toggle(tk.Canvas):
    """iOS-style toggle — PIL bloom when active."""
    TW, TH = 34, 18
    _BM    = 9
    _GLOW  = (52, 200, 110)

    def __init__(self, parent, var, bg=PANEL, command=None, **kw):
        bm = self._BM
        super().__init__(parent, width=self.TW + 2 * bm, height=self.TH + 2 * bm,
                         bg=bg, highlightthickness=0, cursor="hand2", **kw)
        self.var     = var
        self._cmd    = command
        self._bg_rgb = tuple(int(bg.lstrip("#")[i*2:i*2+2], 16) for i in range(3))
        self._photo  = None
        var.trace_add("write", lambda *_: self.after(0, self._draw))
        self.bind("<Button-1>", self._click)
        self._draw()

    def _draw(self):
        self.delete("all")
        on = self.var.get()
        bm = self._BM
        TW, TH = self.TW, self.TH
        W, H   = TW + 2 * bm, TH + 2 * bm
        r      = TH // 2

        if on and _PIL_OK:
            bg = self._bg_rgb
            base = np.full((H, W, 3), bg, dtype=np.float32)

            bloom_src = _PILImage.new("RGB", (W, H), (0, 0, 0))
            bd = _PILDraw.Draw(bloom_src)
            gc = self._GLOW
            bd.ellipse([bm, bm, bm + TH, bm + TH], fill=gc)
            bd.ellipse([bm + TW - TH, bm, bm + TW, bm + TH], fill=gc)
            bd.rectangle([bm + r, bm, bm + TW - r, bm + TH], fill=gc)

            b_np  = np.array(bloom_src.filter(_PILFilter.GaussianBlur(radius=7)), dtype=np.float32)
            base  = np.clip(base + b_np * 2.0, 0, 255)

            result = _PILImage.fromarray(base.astype(np.uint8))
            draw   = _PILDraw.Draw(result)

            tc = (82, 82, 82)
            draw.ellipse([bm, bm, bm + TH, bm + TH], fill=tc)
            draw.ellipse([bm + TW - TH, bm, bm + TW, bm + TH], fill=tc)
            draw.rectangle([bm + r, bm, bm + TW - r, bm + TH], fill=tc)

            pad = 2
            x   = bm + TW - TH + pad
            draw.ellipse([x, bm + pad, x + TH - 2*pad, bm + TH - 2*pad],
                         fill=(220, 220, 220))

            self._photo = _PILImageTk.PhotoImage(result)
            self.config(width=W, height=H)
            self.create_image(0, 0, anchor="nw", image=self._photo)
        else:
            track_color = "#5a5a5a" if on else BORDER
            self.create_oval(bm, bm, bm + TH, bm + TH, fill=track_color, outline="")
            self.create_oval(bm + TW - TH, bm, bm + TW, bm + TH, fill=track_color, outline="")
            self.create_rectangle(bm + r, bm, bm + TW - r, bm + TH, fill=track_color, outline="")
            pad = 2
            x   = bm + TW - TH + pad if on else bm + pad
            self.create_oval(x, bm + pad, x + TH - 2*pad, bm + TH - 2*pad,
                             fill=SURFACE, outline="")

    def _click(self, _):
        self.var.set(not self.var.get())
        if self._cmd:
            self._cmd()


# ── Waveform view ─────────────────────────────────────────────────────────────

class WaveformView(tk.Canvas):
    """
    Displays a waveform envelope with an animated playhead.
    Envelope is computed once with vectorised numpy and cached as flat
    coordinate lists. Animation only moves existing canvas items.
    """

    WAVE_FILL    = "#3a3a3a"
    WAVE_STROKE  = "#909090"
    MARKER_COLOR = "#ff5555"

    def __init__(self, parent, sr=SAMPLE_RATE, **kw):
        kw.setdefault("bg", SURFACE)
        kw.setdefault("highlightthickness", 1)
        kw.setdefault("highlightbackground", BORDER)
        super().__init__(parent, **kw)
        self._sr           = sr
        self._audio        = None
        self._duration     = 0.0
        self._playing      = False
        self._start_time   = None
        self._anim_id      = None
        self._id_line      = None
        self._id_head      = None
        self._cached_size  = (0, 0)
        self._top_flat     = None
        self._bot_flat     = None

        self.bind("<Configure>", lambda _: self._render_waveform())

    def set_audio(self, audio_int16, sr=None):
        self._sr          = sr or self._sr
        self._audio       = audio_int16.astype(np.float32) / 32767.0
        self._duration    = len(audio_int16) / self._sr
        self._cached_size = (0, 0)
        self._render_waveform()

    def clear(self):
        self._audio       = None
        self._duration    = 0.0
        self._cached_size = (0, 0)
        self.stop_playback()

    def start_playback(self):
        self._playing    = True
        self._start_time = time.time()
        if self._id_line:
            self.itemconfig(self._id_line, state="normal")
            self.itemconfig(self._id_head, state="normal")
        self._animate()

    def stop_playback(self):
        self._playing = False
        if self._anim_id:
            self.after_cancel(self._anim_id)
            self._anim_id = None
        if self._id_line:
            self.itemconfig(self._id_line, state="hidden")
            self.itemconfig(self._id_head, state="hidden")

    def _render_waveform(self):
        w = self.winfo_width()
        h = self.winfo_height()
        if w < 2 or h < 2:
            return

        self.delete("all")
        self._id_line = None
        self._id_head = None
        cy = h // 2

        if self._audio is None:
            self.create_line(0, cy, w, cy, fill=BORDER, width=1)
            self.create_text(w // 2, cy,
                             text="Generate or preview a sound to see the waveform",
                             font=FONT_TINY, fill=MUTED)
            return

        n          = len(self._audio)
        cols       = w
        chunk_size = max(1, n // cols)
        n_use      = chunk_size * cols
        mat        = self._audio[:n_use].reshape(cols, chunk_size)
        maxes      = mat.max(axis=1)
        mins       = mat.min(axis=1)

        scale  = (h // 2 - 4) * 0.92
        xs     = np.arange(cols, dtype=np.float32)
        top_ys = (cy - maxes * scale).astype(np.float32)
        bot_ys = (cy - mins  * scale).astype(np.float32)

        top_flat = np.empty(cols * 2, dtype=np.float32)
        top_flat[0::2] = xs
        top_flat[1::2] = top_ys
        bot_flat = np.empty(cols * 2, dtype=np.float32)
        bot_flat[0::2] = xs
        bot_flat[1::2] = bot_ys

        self._top_flat    = top_flat
        self._bot_flat    = bot_flat
        self._cached_size = (w, h)

        bot_flat_rev = bot_flat.reshape(-1, 2)[::-1].ravel()
        poly = np.concatenate([top_flat, bot_flat_rev])
        self.create_polygon(poly.tolist(), fill=self.WAVE_FILL, outline="")

        self.create_line(top_flat.tolist(), fill=self.WAVE_STROKE, width=1)
        self.create_line(bot_flat.tolist(), fill=self.WAVE_STROKE, width=1)
        self.create_line(0, cy, w, cy, fill=BORDER, width=1, dash=(3, 5))

        self._id_line = self.create_line(0, 0, 0, h,
                                         fill=self.MARKER_COLOR, width=2,
                                         state="hidden")
        self._id_head = self.create_polygon(0, 0, 0, 0, 0, 0,
                                            fill=self.MARKER_COLOR, outline="",
                                            state="hidden")

    def _animate(self):
        if not self._playing or self._start_time is None:
            return

        w = self.winfo_width()
        h = self.winfo_height()

        if (w, h) != self._cached_size:
            self._render_waveform()
            if self._id_line:
                self.itemconfig(self._id_line, state="normal")
                self.itemconfig(self._id_head, state="normal")

        elapsed  = time.time() - self._start_time
        position = min(1.0, elapsed / self._duration) if self._duration > 0 else 0.0
        px       = int(position * w)

        if self._id_line:
            self.coords(self._id_line, px, 0, px, h)
            self.coords(self._id_head, px - 5, 0, px + 5, 0, px, 8)

        if position < 1.0 and self._playing:
            self._anim_id = self.after(33, self._animate)
        else:
            self._playing = False
            self.stop_playback()


# ── Card helper ────────────────────────────────────────────────────────────────

def card(parent, title=None, bg=PANEL, padx=14, pady=12):
    """Returns a framed card with an optional title label."""
    outer = tk.Frame(parent, bg=bg,
                     highlightthickness=1, highlightbackground=BORDER)
    inner = tk.Frame(outer, bg=bg, padx=padx, pady=pady)
    inner.pack(fill="both", expand=True)
    if title:
        tk.Label(inner, text=title.upper(), font=FONT_SECTION,
                 bg=bg, fg=MUTED).pack(anchor="w", pady=(0, 8))
    return inner


def divider(parent, bg=PANEL):
    tk.Frame(parent, bg=BORDER, height=1).pack(fill="x", pady=8)
