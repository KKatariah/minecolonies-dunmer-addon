#!/usr/bin/env python3
"""Minecraft skin folder reviewer: hue-window grey + approve/skip."""

from __future__ import annotations

import colorsys
import math
import tkinter as tk
from pathlib import Path
from tkinter import filedialog, messagebox, ttk

from PIL import Image, ImageTk


def _rgb_to_hue_deg(r: int, g: int, b: int) -> float:
    h, _s, _v = colorsys.rgb_to_hsv(r / 255.0, g / 255.0, b / 255.0)
    return h * 360.0


def _rgb_to_hsv_norm(r: int, g: int, b: int) -> tuple[float, float, float]:
    return colorsys.rgb_to_hsv(r / 255.0, g / 255.0, b / 255.0)


def _hsv_distance(
    h1: float, s1: float, v1: float, h2: float, s2: float, v2: float
) -> float:
    dh = abs(h1 - h2)
    dh = min(dh, 1.0 - dh)
    return math.sqrt((2.0 * dh) ** 2 + (s1 - s2) ** 2 + (v1 - v2) ** 2)


def _hue_arc_distance_deg(a: float, b: float) -> float:
    d = abs(a - b) % 360.0
    return min(d, 360.0 - d)


def apply_hue_window_grey(
    rgba: Image.Image,
    center_hue_deg: float,
    half_width_deg: float,
    protect_hsv_list: list[tuple[float, float, float]] | None = None,
    protect_radius: float = 0.18,
    blue_tint: float = 0.0,
    darken_amount: float = 0.0,
) -> Image.Image:
    """Pixels whose hue is within half_width_deg of center become luminance grey; alpha preserved.

    If protect_hsv_list is non-empty (each h,s,v in 0..1), pixels within protect_radius in HSV space
    (hue wraps) of any sample are left unchanged.
    """
    if rgba.mode != "RGBA":
        rgba = rgba.convert("RGBA")
    out = rgba.copy()
    px = out.load()
    w, h = out.size
    guards = protect_hsv_list or []
    tint = max(0.0, min(1.0, float(blue_tint)))
    darken = max(0.0, min(1.0, float(darken_amount)))
    for y in range(h):
        for x in range(w):
            r, g, b, a = px[x, y]
            if a == 0:
                continue
            hue = _rgb_to_hue_deg(r, g, b)
            if _hue_arc_distance_deg(hue, center_hue_deg) > half_width_deg:
                continue
            if guards:
                ch, cs, cv = _rgb_to_hsv_norm(r, g, b)
                if any(
                    _hsv_distance(ch, cs, cv, ph, ps, pv) <= protect_radius
                    for ph, ps, pv in guards
                ):
                    continue
            lum = int(0.299 * r + 0.587 * g + 0.114 * b + 0.5)
            lum = max(0, min(255, lum))
            if tint > 0.0:
                # Keep luminance detail while shifting the greyscale output towards blue.
                tr = int(lum * (1.0 - 0.55 * tint) + 0.5)
                tg = int(lum * (1.0 - 0.30 * tint) + 0.5)
                tb = int(min(255.0, lum * (1.0 + 0.45 * tint)) + 0.5)
                nr = max(0, min(255, tr))
                ng = max(0, min(255, tg))
                nb = max(0, min(255, tb))
            else:
                nr = lum
                ng = lum
                nb = lum

            if darken > 0.0:
                factor = 1.0 - darken
                nr = int(nr * factor + 0.5)
                ng = int(ng * factor + 0.5)
                nb = int(nb * factor + 0.5)

            px[x, y] = (nr, ng, nb, a)
    return out


PREVIEW_SCALE = 14


class SkinGreyApp(tk.Tk):
    def __init__(self) -> None:
        super().__init__()
        self.title("Minecraft skin greying")
        self.minsize(520, 480)

        self.folder: Path | None = None
        self.paths: list[Path] = []
        self.index = 0
        self.original: Image.Image | None = None
        self._photo: ImageTk.PhotoImage | None = None
        self._display_scale = float(PREVIEW_SCALE)
        self._resize_after_id: str | None = None

        self.center_hue_deg = tk.DoubleVar(value=15.0)
        self.half_width_deg = tk.DoubleVar(value=25.0)
        self.protect_radius = tk.DoubleVar(value=0.18)
        self.blue_tint = tk.DoubleVar(value=0.12)
        self.darken_amount = tk.DoubleVar(value=0.33)
        self.protect_list: list[tuple[float, float, float]] = []

        top = ttk.Frame(self, padding=8)
        top.pack(fill=tk.X)
        ttk.Button(top, text="Open folder…", command=self.open_folder).pack(side=tk.LEFT)

        self.status = ttk.Label(top, text="Open a folder of PNG skins.")
        self.status.pack(side=tk.LEFT, padx=12)

        self.readout = ttk.Label(self, text="Center: —°   Range: ±—°")
        self.readout.pack(pady=(0, 2))
        ttk.Label(
            self,
            text="Click = centre hue for grey  ·  Shift+click = add protected colour (e.g. clothes)",
            font=("TkDefaultFont", 10),
        ).pack(pady=(0, 4))

        mid = ttk.Frame(self, padding=(8, 4))
        mid.pack(fill=tk.BOTH, expand=True)
        row = ttk.Frame(mid)
        row.pack(fill=tk.BOTH, expand=True)

        self.canvas = tk.Canvas(
            row,
            highlightthickness=1,
            highlightbackground="#888",
            width=700,
            height=520,
        )
        self.canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, anchor=tk.N)
        self.canvas.bind("<Shift-Button-1>", self.on_protect_click)
        self.canvas.bind("<Button-1>", self.on_center_click)
        self.canvas.bind("<Configure>", self._on_canvas_resize)

        sidebar = ttk.Frame(row, width=112)
        sidebar.pack(side=tk.RIGHT, fill=tk.Y, padx=(10, 0), anchor=tk.N)
        sidebar.pack_propagate(False)
        ttk.Label(sidebar, text="Protected (click to remove)").pack(anchor=tk.NW)
        self.protect_swatch_frame = ttk.Frame(sidebar)
        self.protect_swatch_frame.pack(fill=tk.X, pady=(4, 6), anchor=tk.NW)
        self.clear_protect_btn = ttk.Button(
            sidebar, text="Clear all", command=self.clear_protect
        )
        self.clear_protect_btn.pack(anchor=tk.NW)

        ctrl = ttk.Frame(self, padding=(8, 0))
        ctrl.pack(fill=tk.X)
        ttk.Label(ctrl, text="Hue range (±°):").pack(anchor=tk.W)
        self.range_scale = ttk.Scale(
            ctrl,
            from_=0.0,
            to=90.0,
            variable=self.half_width_deg,
            command=lambda _v: self.refresh_preview(),
        )
        self.range_scale.pack(fill=tk.X, pady=4)

        pr = ttk.Frame(ctrl)
        pr.pack(fill=tk.X, pady=(4, 0))
        ttk.Label(
            pr,
            text="Protect zone (larger = more pixels like the Shift+click sample stay coloured):",
        ).pack(anchor=tk.W)
        self.protect_scale = ttk.Scale(
            pr,
            from_=0.05,
            to=0.45,
            variable=self.protect_radius,
            command=lambda _v: self.refresh_preview(),
        )
        self.protect_scale.pack(fill=tk.X, pady=2)

        bt = ttk.Frame(ctrl)
        bt.pack(fill=tk.X, pady=(4, 0))
        ttk.Label(
            bt,
            text="Blue tint on greyed area:",
        ).pack(anchor=tk.W)
        self.blue_tint_scale = ttk.Scale(
            bt,
            from_=0.0,
            to=1.0,
            variable=self.blue_tint,
            command=lambda _v: self.refresh_preview(),
        )
        self.blue_tint_scale.pack(fill=tk.X, pady=2)

        dk = ttk.Frame(ctrl)
        dk.pack(fill=tk.X, pady=(4, 0))
        ttk.Label(
            dk,
            text="Darken greyed area:",
        ).pack(anchor=tk.W)
        self.darken_scale = ttk.Scale(
            dk,
            from_=0.0,
            to=1.0,
            variable=self.darken_amount,
            command=lambda _v: self.refresh_preview(),
        )
        self.darken_scale.pack(fill=tk.X, pady=2)

        btns = ttk.Frame(self, padding=8)
        btns.pack(fill=tk.BOTH, expand=True)
        approve = ttk.Button(btns, text="APPROVE", command=self.approve)
        approve.pack(fill=tk.X, pady=4, ipady=12)
        skip = ttk.Button(btns, text="SKIP", command=self.skip)
        skip.pack(fill=tk.X, pady=4, ipady=12)

        self.bind("<Return>", lambda e: self.approve())
        self.bind("<space>", lambda e: self.skip())

        self._approve_btn = approve
        self._skip_btn = skip
        self._set_controls_enabled(False)

    def clear_protect(self) -> None:
        self.protect_list.clear()
        self._rebuild_protect_swatches()
        self.refresh_preview()

    def _remove_protect_at(self, index: int) -> None:
        if 0 <= index < len(self.protect_list):
            del self.protect_list[index]
            self._rebuild_protect_swatches()
            self.refresh_preview()

    def _rebuild_protect_swatches(self) -> None:
        for w in self.protect_swatch_frame.winfo_children():
            w.destroy()
        sw = 30
        for idx, (h, s, v) in enumerate(self.protect_list):
            rr, gg, bb = colorsys.hsv_to_rgb(h, s, v)
            hx = f"#{int(rr * 255 + 0.5):02x}{int(gg * 255 + 0.5):02x}{int(bb * 255 + 0.5):02x}"
            c = tk.Canvas(
                self.protect_swatch_frame,
                width=sw,
                height=sw,
                highlightthickness=1,
                highlightbackground="#666",
                bd=0,
                cursor="hand2",
            )
            c.pack(pady=(0, 4), anchor=tk.W)
            c.create_rectangle(0, 0, sw, sw, fill=hx, outline="")
            c.bind("<Button-1>", lambda _e, i=idx: self._remove_protect_at(i))

    def _pixel_from_event(self, event: tk.Event) -> tuple[int, int, int, int] | None:
        if self.original is None:
            return None
        scale = self._display_scale if self._display_scale > 0 else float(PREVIEW_SCALE)
        x = int(event.x / scale)
        y = int(event.y / scale)
        ow, oh = self.original.size
        if x < 0 or y < 0 or x >= ow or y >= oh:
            return None
        r, g, b, a = self.original.getpixel((x, y))
        if a == 0:
            return None
        return r, g, b, a

    def _on_canvas_resize(self, _event: tk.Event) -> None:
        if self._resize_after_id is not None:
            self.after_cancel(self._resize_after_id)
        # Debounce resize events to avoid excessive reprocessing while dragging window edges.
        self._resize_after_id = self.after(40, self._refresh_preview_after_resize)

    def _refresh_preview_after_resize(self) -> None:
        self._resize_after_id = None
        if self.original is not None:
            self.refresh_preview()

    def open_folder(self) -> None:
        path = filedialog.askdirectory(title="Folder of Minecraft skins")
        if not path:
            return
        self.folder = Path(path)
        self.paths = sorted(self.folder.glob("*.png"), key=lambda p: p.name.lower())
        self.index = 0
        if not self.paths:
            messagebox.showinfo("No PNGs", "No .png files in that folder.")
            self.original = None
            self.protect_list.clear()
            self._rebuild_protect_swatches()
            self._clear_canvas()
            self._set_controls_enabled(False)
            return
        self._set_controls_enabled(True)
        self.load_current()

    def _set_controls_enabled(self, on: bool) -> None:
        self.range_scale.state(["!disabled"] if on else ["disabled"])
        self.protect_scale.state(["!disabled"] if on else ["disabled"])
        self.blue_tint_scale.state(["!disabled"] if on else ["disabled"])
        self.darken_scale.state(["!disabled"] if on else ["disabled"])
        self.clear_protect_btn.state(["!disabled"] if on else ["disabled"])
        self._approve_btn.state(["!disabled"] if on else ["disabled"])
        self._skip_btn.state(["!disabled"] if on else ["disabled"])

    def _clear_canvas(self) -> None:
        self.canvas.delete("all")
        self._photo = None

    def load_current(self) -> None:
        if not self.paths or self.index >= len(self.paths):
            self._done_state()
            return
        p = self.paths[self.index]
        try:
            self.original = Image.open(p).convert("RGBA")
        except OSError as e:
            messagebox.showerror("Open failed", f"{p.name}: {e}")
            self.skip()
            return
        self.protect_list.clear()
        self._rebuild_protect_swatches()
        self.status.config(text=f"{p.name}  ({self.index + 1}/{len(self.paths)})")
        self.refresh_preview()

    def _done_state(self) -> None:
        self.original = None
        self.protect_list.clear()
        self._rebuild_protect_swatches()
        self._clear_canvas()
        self.status.config(text="Done. Open another folder or close.")
        self._set_controls_enabled(False)

    def refresh_preview(self) -> None:
        if self.original is None:
            return
        half = float(self.half_width_deg.get())
        center = float(self.center_hue_deg.get()) % 360.0
        self.center_hue_deg.set(center)
        rad = float(self.protect_radius.get())
        tint = float(self.blue_tint.get())
        darken = float(self.darken_amount.get())
        n = len(self.protect_list)
        extra = f"   ·   Protect: {n} colour{'s' if n != 1 else ''}" if n else "   ·   Protect: none"
        self.readout.config(
            text=(
                f"Center: {center:.1f}°   Range: ±{half:.1f}°   "
                f"Blue tint: {int(round(tint * 100))}%   Darken: {int(round(darken * 100))}%{extra}"
            )
        )
        processed = apply_hue_window_grey(
            self.original, center, half, self.protect_list, rad, tint, darken
        )
        w, h = processed.size

        natural_w = w * PREVIEW_SCALE
        natural_h = h * PREVIEW_SCALE
        avail_w = max(1, self.canvas.winfo_width() - 2)
        avail_h = max(1, self.canvas.winfo_height() - 2)
        fit = min(1.0, avail_w / natural_w, avail_h / natural_h)
        disp_w = max(1, int(round(natural_w * fit)))
        disp_h = max(1, int(round(natural_h * fit)))
        self._display_scale = PREVIEW_SCALE * fit

        disp = processed.resize((disp_w, disp_h), resample=Image.Resampling.NEAREST)
        self._photo = ImageTk.PhotoImage(disp)
        self.canvas.delete("all")
        self.canvas.create_image(0, 0, anchor=tk.NW, image=self._photo)

    def on_center_click(self, event: tk.Event) -> None:
        if (event.state & 0x0001) != 0:
            return
        pix = self._pixel_from_event(event)
        if pix is None:
            return
        r, g, b, _a = pix
        self.center_hue_deg.set(_rgb_to_hue_deg(r, g, b))
        self.refresh_preview()

    def on_protect_click(self, event: tk.Event) -> None:
        pix = self._pixel_from_event(event)
        if pix is None:
            return
        r, g, b, _a = pix
        hsv = _rgb_to_hsv_norm(r, g, b)
        if self.protect_list and any(
            _hsv_distance(hsv[0], hsv[1], hsv[2], ph, ps, pv) < 0.03
            for ph, ps, pv in self.protect_list
        ):
            return
        self.protect_list.append(hsv)
        self._rebuild_protect_swatches()
        self.refresh_preview()

    def approve(self) -> None:
        if self.folder is None or not self.paths or self.index >= len(self.paths):
            return
        if self.original is None:
            return
        out_dir = self.folder / "grey_export"
        out_dir.mkdir(parents=True, exist_ok=True)
        dest = out_dir / self.paths[self.index].name
        half = float(self.half_width_deg.get())
        center = float(self.center_hue_deg.get()) % 360.0
        rad = float(self.protect_radius.get())
        tint = float(self.blue_tint.get())
        darken = float(self.darken_amount.get())
        img = apply_hue_window_grey(
            self.original, center, half, self.protect_list, rad, tint, darken
        )
        try:
            img.save(dest, format="PNG")
        except OSError as e:
            messagebox.showerror("Save failed", str(e))
            return
        self.index += 1
        self.load_current()

    def skip(self) -> None:
        if not self.paths or self.index >= len(self.paths):
            return
        self.index += 1
        self.load_current()


def main() -> None:
    app = SkinGreyApp()
    app.mainloop()


if __name__ == "__main__":
    main()
