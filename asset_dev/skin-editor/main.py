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
    greying_enabled: bool = True,
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
            if greying_enabled:
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
            else:
                # Keep the original colour but bias the matched area toward blue.
                if tint > 0.0:
                    nr = int(r * (1.0 - 0.45 * tint) + 0.5)
                    ng = int(g * (1.0 - 0.25 * tint) + 0.5)
                    nb = int(min(255.0, b * (1.0 + 0.60 * tint) + 18.0 * tint) + 0.5)
                else:
                    nr = r
                    ng = g
                    nb = b

            if darken > 0.0:
                factor = 1.0 - darken
                nr = int(nr * factor + 0.5)
                ng = int(ng * factor + 0.5)
                nb = int(nb * factor + 0.5)

            px[x, y] = (nr, ng, nb, a)
    return out


def _is_near_white(r: int, g: int, b: int, threshold: int = 235) -> bool:
    return r >= threshold and g >= threshold and b >= threshold


def _is_red_eye_candidate(r: int, g: int, b: int) -> bool:
    if _is_near_white(r, g, b):
        return False
    _, saturation, value = colorsys.rgb_to_hsv(r / 255.0, g / 255.0, b / 255.0)
    luminance = 0.299 * r + 0.587 * g + 0.114 * b
    return luminance < 200.0 and (saturation >= 0.12 or luminance < 140.0 or value < 0.82)


def apply_red_eye_pass(
    source_rgba: Image.Image,
    target_rgba: Image.Image,
    sample_point: tuple[int, int] | None = None,
) -> Image.Image:
    if source_rgba.mode != "RGBA":
        source_rgba = source_rgba.convert("RGBA")
    if target_rgba.mode != "RGBA":
        target_rgba = target_rgba.convert("RGBA")
    if source_rgba.size != target_rgba.size:
        return target_rgba.copy()

    w, h = source_rgba.size
    out = target_rgba.copy()
    dst_px = out.load()
    TARGET_RED = (0x5A, 0x09, 0x09)
    anchor_x, anchor_y = sample_point if sample_point is not None else (10, 12)
    for x, y in ((anchor_x, anchor_y), (anchor_x + 3, anchor_y)):
        if 0 <= x < w and 0 <= y < h:
            _r, _g, _b, a = source_rgba.getpixel((x, y))
            if a > 0:
                dst_px[x, y] = (TARGET_RED[0], TARGET_RED[1], TARGET_RED[2], a)
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
        self.blue_tint = tk.DoubleVar(value=0.0)
        self.darken_amount = tk.DoubleVar(value=0.0)
        self.greying_enabled = tk.BooleanVar(value=False)
        self.red_eyes_enabled = tk.BooleanVar(value=False)
        self.protect_list: list[tuple[float, float, float]] = []
        self.eye_sample_point: tuple[int, int] | None = (10, 12)
        self._eye_pick_pending = False
        self.history: list[int] = []

        shell = ttk.Frame(self)
        shell.pack(fill=tk.BOTH, expand=True)
        self.body_canvas = tk.Canvas(shell, highlightthickness=0)
        body_scrollbar = ttk.Scrollbar(shell, orient=tk.VERTICAL, command=self.body_canvas.yview)
        self.body_canvas.configure(yscrollcommand=body_scrollbar.set)
        body_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.body_canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        self.body_content = ttk.Frame(self.body_canvas, padding=8)
        self.body_window = self.body_canvas.create_window((0, 0), window=self.body_content, anchor=tk.NW)
        self.body_content.bind("<Configure>", self._on_body_configure)
        self.body_canvas.bind("<Configure>", self._on_body_canvas_configure)

        top = ttk.Frame(self.body_content, padding=8)
        top.pack(fill=tk.X)
        ttk.Button(top, text="Open folder…", command=self.open_folder).pack(side=tk.LEFT)

        self.status = ttk.Label(top, text="Open a folder of PNG skins.")
        self.status.pack(side=tk.LEFT, padx=12)

        self.readout = ttk.Label(self.body_content, text="Center: —°   Range: ±—°")
        self.readout.pack(pady=(0, 2))
        ttk.Label(
            self.body_content,
            text="Click = centre hue for grey  ·  Shift+click = add protected colour (e.g. clothes)",
            font=("TkDefaultFont", 10),
        ).pack(pady=(0, 4))

        mid = ttk.Frame(self.body_content, padding=(8, 4))
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

        ctrl = ttk.Frame(self.body_content, padding=(8, 0))
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

        self.grey_toggle = ttk.Checkbutton(
            ctrl,
            text="Enable greying",
            variable=self.greying_enabled,
            command=self.refresh_preview,
        )
        self.grey_toggle.pack(anchor=tk.W, pady=(6, 0))

        eye_row = ttk.Frame(ctrl)
        eye_row.pack(fill=tk.X, pady=(8, 0))
        self.red_eye_toggle = ttk.Checkbutton(
            eye_row,
            text="Place red eyes",
            variable=self.red_eyes_enabled,
            command=self.refresh_preview,
        )
        self.red_eye_toggle.pack(side=tk.LEFT)
        self.pick_eye_btn = ttk.Button(eye_row, text="Pick eye sample", command=self.begin_eye_pick)
        self.pick_eye_btn.pack(side=tk.LEFT, padx=(8, 0))
        self.clear_eye_btn = ttk.Button(eye_row, text="Clear sample", command=self.clear_eye_sample)
        self.clear_eye_btn.pack(side=tk.LEFT, padx=(8, 0))

        self.eye_sample_label = ttk.Label(ctrl, text="Eye sample: none")
        self.eye_sample_label.pack(anchor=tk.W, pady=(2, 0))

        btns = ttk.Frame(self.body_content, padding=8)
        btns.pack(fill=tk.X)

        row1 = ttk.Frame(btns)
        row1.pack(fill=tk.X, pady=4)
        apply_layer = ttk.Button(row1, text="APPLY LAYER", command=self.apply_layer)
        apply_layer.pack(side=tk.LEFT, fill=tk.X, expand=True, ipady=10)
        approve = ttk.Button(row1, text="APPROVE", command=self.approve)
        approve.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(8, 0), ipady=10)

        row2 = ttk.Frame(btns)
        row2.pack(fill=tk.X, pady=4)
        back = ttk.Button(row2, text="BACK", command=self.back)
        back.pack(side=tk.LEFT, fill=tk.X, expand=True, ipady=10)
        skip = ttk.Button(row2, text="SKIP", command=self.skip)
        skip.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(8, 0), ipady=10)

        self.bind_all("<Return>", lambda e: (self.approve(), "break")[1])
        self.bind_all("<space>", lambda e: (self.skip(), "break")[1])
        self.bind_all("<Up>", self._move_red_eyes_up)
        self.bind_all("<Down>", self._move_red_eyes_down)
        self.bind_all("<Left>", self._move_red_eyes_left)
        self.bind_all("<Right>", self._move_red_eyes_right)

        self._apply_layer_btn = apply_layer
        self._approve_btn = approve
        self._skip_btn = skip
        self._back_btn = back
        self._set_controls_enabled(False)
        self._update_back_button_state()

    def _on_body_configure(self, _event: tk.Event) -> None:
        self.body_canvas.configure(scrollregion=self.body_canvas.bbox("all"))

    def _on_body_canvas_configure(self, event: tk.Event) -> None:
        self.body_canvas.itemconfigure(self.body_window, width=event.width)

    def clear_protect(self) -> None:
        self.protect_list.clear()
        self._rebuild_protect_swatches()
        self.refresh_preview()

    def _update_back_button_state(self) -> None:
        if hasattr(self, "_back_btn"):
            self._back_btn.state(["!disabled"] if self.folder is not None and bool(self.history) else ["disabled"])

    def _advance_to_next(self, push_history: bool) -> None:
        if self.folder is None or not self.paths or self.index >= len(self.paths):
            return
        if push_history:
            self.history.append(self.index)
        self.index += 1
        self.load_current()

    def back(self) -> None:
        if self.folder is None or not self.paths or not self.history:
            return
        self.index = self.history.pop()
        self.load_current()

    def begin_eye_pick(self) -> None:
        if self.original is None:
            return
        self._eye_pick_pending = True
        self.status.config(text="Click an eye pixel on the canvas to set the red-eye anchor.")

    def clear_eye_sample(self) -> None:
        self.eye_sample_point = (10, 12)
        self._eye_pick_pending = False
        self.eye_sample_label.config(text="Red eyes: default (10, 12) and (13, 12)")
        if self.original is not None:
            self.status.config(text=f"{self.paths[self.index].name}  ({self.index + 1}/{len(self.paths)})")
        self.refresh_preview()

    def _move_red_eyes(self, dx: int, dy: int) -> None:
        if self.original is None:
            return
        anchor_x, anchor_y = self.eye_sample_point if self.eye_sample_point is not None else (10, 12)
        new_x = anchor_x + dx
        new_y = anchor_y + dy
        width, height = self.original.size
        if new_x < 0 or new_y < 0 or new_x + 3 >= width or new_y >= height:
            return
        self.eye_sample_point = (new_x, new_y)
        self._eye_pick_pending = False
        self.eye_sample_label.config(text=f"Red eyes: ({new_x}, {new_y}) and ({new_x + 3}, {new_y})")
        self.refresh_preview()

    def _move_red_eyes_up(self, _event: tk.Event) -> str:
        self._move_red_eyes(0, -1)
        return "break"

    def _move_red_eyes_down(self, _event: tk.Event) -> str:
        self._move_red_eyes(0, 1)
        return "break"

    def _move_red_eyes_left(self, _event: tk.Event) -> str:
        self._move_red_eyes(-1, 0)
        return "break"

    def _move_red_eyes_right(self, _event: tk.Event) -> str:
        self._move_red_eyes(1, 0)
        return "break"

    def apply_layer(self) -> None:
        if self.original is None:
            return
        self.original = self._compose_processed_image()
        self.protect_list.clear()
        self._rebuild_protect_swatches()
        self.greying_enabled.set(False)
        self.blue_tint.set(0.0)
        self.darken_amount.set(0.0)
        self._eye_pick_pending = False
        self.status.config(text="Current layer applied. Pick another colour group if needed.")
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
        point = self._point_from_event(event)
        if point is None:
            return None
        x, y = point
        ow, oh = self.original.size
        if x < 0 or y < 0 or x >= ow or y >= oh:
            return None
        r, g, b, a = self.original.getpixel((x, y))
        if a == 0:
            return None
        return r, g, b, a

    def _point_from_event(self, event: tk.Event) -> tuple[int, int] | None:
        if self.original is None:
            return None
        scale = self._display_scale if self._display_scale > 0 else float(PREVIEW_SCALE)
        x = int(event.x / scale)
        y = int(event.y / scale)
        return x, y

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
        self.history.clear()
        if not self.paths:
            messagebox.showinfo("No PNGs", "No .png files in that folder.")
            self.original = None
            self.protect_list.clear()
            self._rebuild_protect_swatches()
            self.eye_sample_point = (10, 12)
            self._eye_pick_pending = False
            self.eye_sample_label.config(text="Red eyes: default (10, 12) and (13, 12)")
            self._clear_canvas()
            self._set_controls_enabled(False)
            self._update_back_button_state()
            return
        self._set_controls_enabled(True)
        self._update_back_button_state()
        self.load_current()

    def _set_controls_enabled(self, on: bool) -> None:
        self.range_scale.state(["!disabled"] if on else ["disabled"])
        self.protect_scale.state(["!disabled"] if on else ["disabled"])
        self.blue_tint_scale.state(["!disabled"] if on else ["disabled"])
        self.darken_scale.state(["!disabled"] if on else ["disabled"])
        self.grey_toggle.state(["!disabled"] if on else ["disabled"])
        self.red_eye_toggle.state(["!disabled"] if on else ["disabled"])
        self.pick_eye_btn.state(["!disabled"] if on else ["disabled"])
        self.clear_eye_btn.state(["!disabled"] if on else ["disabled"])
        self._apply_layer_btn.state(["!disabled"] if on else ["disabled"])
        self.clear_protect_btn.state(["!disabled"] if on else ["disabled"])
        self._approve_btn.state(["!disabled"] if on else ["disabled"])
        self._skip_btn.state(["!disabled"] if on else ["disabled"])
        self._update_back_button_state()

    def _clear_canvas(self) -> None:
        self.canvas.delete("all")
        self._photo = None

    def load_current(self) -> None:
        if not self.paths or self.index >= len(self.paths):
            self._done_state()
            return
        p = self.paths[self.index]
        display_path = p
        edited_path = self.folder / "grey_export" / p.name if self.folder is not None else None
        if edited_path is not None and edited_path.exists():
            display_path = edited_path
        try:
            self.original = Image.open(display_path).convert("RGBA")
        except OSError as e:
            if display_path != p:
                try:
                    self.original = Image.open(p).convert("RGBA")
                except OSError as original_error:
                    messagebox.showerror("Open failed", f"{p.name}: {original_error}")
                    self._advance_to_next(push_history=False)
                    return
            else:
                messagebox.showerror("Open failed", f"{p.name}: {e}")
                self._advance_to_next(push_history=False)
                return
        self.protect_list.clear()
        self._rebuild_protect_swatches()
        self.eye_sample_point = (10, 12)
        self._eye_pick_pending = False
        self.eye_sample_label.config(text="Red eyes: default (10, 12) and (13, 12)")
        suffix = "  [edited]" if display_path != p else ""
        self.status.config(text=f"{p.name}{suffix}  ({self.index + 1}/{len(self.paths)})")
        self._update_back_button_state()
        self.refresh_preview()

    def _done_state(self) -> None:
        self.original = None
        self.protect_list.clear()
        self._rebuild_protect_swatches()
        self.eye_sample_point = (10, 12)
        self._eye_pick_pending = False
        self.eye_sample_label.config(text="Red eyes: default (10, 12) and (13, 12)")
        self._clear_canvas()
        self.status.config(text="Done. Open another folder or close.")
        self._set_controls_enabled(False)
        self._update_back_button_state()

    def _compose_processed_image(self) -> Image.Image:
        if self.original is None:
            raise RuntimeError("No image loaded")
        half = float(self.half_width_deg.get())
        center = float(self.center_hue_deg.get()) % 360.0
        rad = float(self.protect_radius.get())
        tint = float(self.blue_tint.get())
        darken = float(self.darken_amount.get())
        processed = apply_hue_window_grey(
            self.original,
            center,
            half,
            self.protect_list,
            rad,
            tint,
            darken,
            self.greying_enabled.get(),
        )
        if self.red_eyes_enabled.get():
            processed = apply_red_eye_pass(self.original, processed, self.eye_sample_point)
        return processed

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
        mode = "on" if self.greying_enabled.get() else "off"
        eye_mode = "on" if self.red_eyes_enabled.get() else "off"
        sample_text = (
            f"   ·   Red eyes: ({self.eye_sample_point[0]}, {self.eye_sample_point[1]}) and ({self.eye_sample_point[0] + 3}, {self.eye_sample_point[1]})"
            if self.eye_sample_point is not None
            else "   ·   Red eyes: default (10, 12) and (13, 12)"
        )
        self.readout.config(
            text=(
                f"Center: {center:.1f}°   Range: ±{half:.1f}°   "
                f"Blue tint: {int(round(tint * 100))}%   Darken: {int(round(darken * 100))}%"
                f"{extra}   ·   Greying: {mode}   ·   Place red eyes: {eye_mode}{sample_text}"
            )
        )
        processed = self._compose_processed_image()
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
        if self._eye_pick_pending:
            point = self._point_from_event(event)
            if point is None:
                return
            self.eye_sample_point = point
            self._eye_pick_pending = False
            self.eye_sample_label.config(text=f"Red eyes: ({point[0]}, {point[1]}) and ({point[0] + 3}, {point[1]})")
            self.status.config(text=f"Red eyes set at ({point[0]}, {point[1]}) and ({point[0] + 3}, {point[1]}).")
            self.refresh_preview()
            return
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
        img = self._compose_processed_image()
        try:
            img.save(dest, format="PNG")
        except OSError as e:
            messagebox.showerror("Save failed", str(e))
            return
        self._advance_to_next(push_history=True)

    def skip(self) -> None:
        if not self.paths or self.index >= len(self.paths):
            return
        self._advance_to_next(push_history=True)


def main() -> None:
    app = SkinGreyApp()
    app.mainloop()


if __name__ == "__main__":
    main()
