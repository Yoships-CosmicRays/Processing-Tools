#!/usr/bin/env python3
"""
Transit Detector - Lucky Imaging Transit Detection Tool
Detects transiting objects (satellites, birds, etc.) in solar/lunar lucky imaging videos.

Supports AVI and SER input formats, outputs 16-bit uncompressed TIFF sequences.

Requirements:
    pip install numpy opencv-python tifffile scipy tkinter
    (tkinter is usually included with Python)
"""

import os
import sys
import gc
import struct
import datetime
import argparse
import re
import tkinter as tk
from tkinter import filedialog, messagebox
from pathlib import Path

import numpy as np
import cv2
import tifffile
from scipy import ndimage

# ─────────────────────────────────────────────
#  GLOBAL PARAMETERS  (edit here or via CLI)
# ─────────────────────────────────────────────
FRAMES_BEFORE_TRANSIT = 10   # Extra frames saved before first detection
FRAMES_AFTER_TRANSIT  = 10   # Extra frames saved after last detection
MIN_RESIDUAL_PIXELS   = 2    # Minimum connected pixels to count as a transit blob
THRESHOLD_MARGIN      = 1.2  # Multiplier on estimated noise level for threshold
CALM_FRAME_SAMPLE    = 30   # How many frames to sample to estimate noise threshold
ANCHOR_SIZE_DEFAULT   = 256  # Default anchor patch size (pixels)
GAP_MERGE_FRAMES      = 5    # If two detections are <= N frames apart, merge into one transit
MIN_TRANSIT_MOTION   = 100   # Reject group if total centroid displacement < this (px) [seeing filter]
HIGH_CONTRAST_SAMPLE = 100    # Frame pairs used to build the high-contrast mask
CONTRAST_MASK_SHIFT = 5     # Artificial mis-alignment (px) when building contrast mask


# ═══════════════════════════════════════════════════════════════════
#  SER FILE READER
# ═══════════════════════════════════════════════════════════════════

SER_COLOR_IDS = {
    0: 'MONO',
    8: 'BAYER_RGGB', 9: 'BAYER_GRBG', 10: 'BAYER_GBRG', 11: 'BAYER_BGGR',
    16: 'BAYER_CYYM', 17: 'BAYER_YCMY', 18: 'BAYER_YMCY', 19: 'BAYER_MYYC',
    100: 'RGB', 101: 'BGR',
}

class SERReader:
    """Reads SER (planetary imaging) video files."""

    def __init__(self, path):
        self.path = path
        self.f = open(path, 'rb')
        self._read_header()

    def _read_header(self):
        f = self.f
        f.seek(0)
        file_id = f.read(14).decode('ascii', errors='ignore')
        if not file_id.startswith('LUCAM-RECORDER'):
            raise ValueError(f"Not a valid SER file: {self.path}")

        self.lu_id        = struct.unpack('<i', f.read(4))[0]
        self.color_id     = struct.unpack('<i', f.read(4))[0]
        self.little_endian= struct.unpack('<i', f.read(4))[0]
        self.width        = struct.unpack('<i', f.read(4))[0]
        self.height       = struct.unpack('<i', f.read(4))[0]
        self.bit_depth    = struct.unpack('<i', f.read(4))[0]
        self.frame_count  = struct.unpack('<i', f.read(4))[0]
        self.observer     = f.read(40).decode('ascii', errors='ignore').strip('\x00')
        self.instrument   = f.read(40).decode('ascii', errors='ignore').strip('\x00')
        self.telescope    = f.read(40).decode('ascii', errors='ignore').strip('\x00')

        # Timestamps (UTC ticks since 01/01/0001 00:00:00)
        dt_start = struct.unpack('<q', f.read(8))[0]
        dt_end   = struct.unpack('<q', f.read(8))[0]

        # Convert .NET ticks to datetime
        self.start_time = self._ticks_to_dt(dt_start) if dt_start else None
        self.end_time   = self._ticks_to_dt(dt_end)   if dt_end   else None

        self.bytes_per_pixel = (self.bit_depth + 7) // 8
        color_name = SER_COLOR_IDS.get(self.color_id, 'UNKNOWN')
        self.is_color = color_name not in ('MONO',) and not color_name.startswith('BAYER_')
        self.is_bayer = color_name.startswith('BAYER_')
        self.bayer_pattern = color_name.replace('BAYER_', '') if self.is_bayer else None

        channels = 3 if self.is_color else 1
        self.frame_size = self.width * self.height * self.bytes_per_pixel * channels
        self.data_offset = 178  # fixed SER header size

    def _ticks_to_dt(self, ticks):
        """Convert .NET DateTime ticks (100ns since 0001-01-01) to datetime."""
        # .NET epoch = 0001-01-01; Python's datetime min = 0001-01-01
        try:
            epoch = datetime.datetime(1, 1, 1)
            return epoch + datetime.timedelta(microseconds=ticks // 10)
        except Exception:
            return None

    def get_frame(self, index):
        """Returns frame as numpy uint16 array (H, W) mono or (H, W, 3) color."""
        self.f.seek(self.data_offset + index * self.frame_size)
        raw = self.f.read(self.frame_size)

        dtype = np.uint16 if self.bytes_per_pixel == 2 else np.uint8

        if self.is_color:
            img = np.frombuffer(raw, dtype=dtype).reshape(self.height, self.width, 3)
        else:
            img = np.frombuffer(raw, dtype=dtype).reshape(self.height, self.width)

        # Fix endianness for 16-bit
        if self.bytes_per_pixel == 2 and self.little_endian == 0:
            img = img.byteswap()

        # Debayer if needed
        if self.is_bayer:
            codes = {
                'RGGB': cv2.COLOR_BAYER_RG2RGB,
                'GRBG': cv2.COLOR_BAYER_GR2RGB,
                'GBRG': cv2.COLOR_BAYER_GB2RGB,
                'BGGR': cv2.COLOR_BAYER_BG2RGB,
            }
            code = codes.get(self.bayer_pattern, cv2.COLOR_BAYER_RG2RGB)
            if self.bytes_per_pixel == 2:
                img = cv2.cvtColor(img, code + 4)  # 16-bit debayer
            else:
                img = cv2.cvtColor(img, code)

        return img

    def close(self):
        self.f.close()

    def __len__(self):
        return self.frame_count


# ═══════════════════════════════════════════════════════════════════
#  AVI READER
# ═══════════════════════════════════════════════════════════════════

class AVIReader:
    """Reads AVI files using OpenCV."""

    def __init__(self, path, txt_path=None):
        self.path = path
        self.cap = cv2.VideoCapture(str(path))
        if not self.cap.isOpened():
            raise ValueError(f"Cannot open AVI: {path}")

        self.frame_count = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.fps         = self.cap.get(cv2.CAP_PROP_FPS)
        self.width       = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height      = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        self.start_time  = None
        self.end_time    = None
        self._needs_flip = self._detect_flip()   # detect bottom-up AVI

        # Parse companion .txt if available
        if txt_path and os.path.exists(txt_path):
            self._parse_txt(txt_path)
        elif txt_path is None:
            # Auto-detect: same name, same folder
            auto = Path(path).with_suffix('.txt')
            if not auto.exists():
                auto = Path(str(path) + '.txt')
            if auto.exists():
                self._parse_txt(str(auto))

    def _parse_txt(self, txt_path):
        """Parse ZWO-style capture info .txt file."""
        with open(txt_path, 'r', encoding='utf-8', errors='ignore') as f:
            content = f.read()

        def extract(key):
            m = re.search(rf'^{key}\s*=\s*(.+)$', content, re.MULTILINE)
            return m.group(1).strip() if m else None

        start_str = extract('StartCapture')
        end_str   = extract('EndCapture')

        fmt = '%Y-%m-%dT%H:%M:%S.%fZ'
        if start_str:
            try:
                self.start_time = datetime.datetime.strptime(start_str, fmt)
            except ValueError:
                pass
        if end_str:
            try:
                self.end_time = datetime.datetime.strptime(end_str, fmt)
            except ValueError:
                pass

        # Also try to get fps from txt if OpenCV returns 0
        fps_str = extract('FPS')
        if fps_str and (self.fps == 0 or self.fps is None):
            try:
                self.fps = float(fps_str)
            except ValueError:
                pass

    def _detect_flip(self):
        """
        Detect whether the AVI is stored bottom-up (common with many capture
        software / codecs: SharpCap, YoshipsCap, FireCapture on Windows).

        Strategy: read frame 0 twice — normal and flipped vertically.
        Compute the NCC between each version and the next frame (frame 1).
        The orientation that gives the higher NCC is the correct one.
        If the two scores are indistinguishable (diff < 0.01), default to
        no flip (conservative).
        """
        import cv2 as _cv2
        self.cap.set(_cv2.CAP_PROP_POS_FRAMES, 0)
        ret0, f0 = self.cap.read()
        self.cap.set(_cv2.CAP_PROP_POS_FRAMES, 1)
        ret1, f1 = self.cap.read()
        if not ret0 or not ret1:
            return False   # cannot determine — default no flip

        # Work in grayscale float32 for NCC
        g0  = _cv2.cvtColor(f0, _cv2.COLOR_BGR2GRAY).astype('float32')
        g0f = _cv2.flip(g0, 0)
        g1  = _cv2.cvtColor(f1, _cv2.COLOR_BGR2GRAY).astype('float32')

        def ncc(a, b):
            a = a - a.mean(); b = b - b.mean()
            denom = (np.linalg.norm(a) * np.linalg.norm(b))
            return float(np.sum(a * b) / denom) if denom > 0 else 0.0

        score_normal = ncc(g0,  g1)
        score_flip   = ncc(g0f, g1)

        needs_flip = (score_flip - score_normal) > 0.01
        print(f"  AVI orientation detection: "
              f"normal NCC={score_normal:.4f}, flipped NCC={score_flip:.4f} "
              f"→ {'FLIP' if needs_flip else 'no flip'}")
        return needs_flip

    def get_frame(self, index):
        """Returns frame as numpy array (H, W) uint8/16 or (H, W, 3)."""
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, index)
        ret, frame = self.cap.read()
        if not ret:
            raise IOError(f"Cannot read frame {index}")
        if self._needs_flip:
            frame = cv2.flip(frame, 0)
        return frame

    def close(self):
        self.cap.release()

    def __len__(self):
        return self.frame_count


# ═══════════════════════════════════════════════════════════════════
#  UNIFIED VIDEO WRAPPER
# ═══════════════════════════════════════════════════════════════════

class VideoFile:
    """Unified interface over AVIReader / SERReader."""

    def __init__(self, path):
        self.path = Path(path)
        ext = self.path.suffix.lower()

        if ext == '.ser':
            self._reader = SERReader(str(path))
            self.frame_count = self._reader.frame_count
            self.fps         = None  # will be computed from timestamps
            self.start_time  = self._reader.start_time
            self.end_time    = self._reader.end_time
        elif ext in ('.avi', '.mov', '.mp4'):
            self._reader = AVIReader(str(path))
            self.frame_count = self._reader.frame_count
            self.fps         = self._reader.fps
            self.start_time  = self._reader.start_time
            self.end_time    = self._reader.end_time
        else:
            raise ValueError(f"Unsupported format: {ext}")

        # Compute fps from timestamps if not available
        if self.fps is None or self.fps == 0:
            if self.start_time and self.end_time and self.frame_count > 1:
                duration = (self.end_time - self.start_time).total_seconds()
                self.fps = self.frame_count / duration
            else:
                self.fps = 25.0  # fallback
                print("Warning: could not determine FPS, defaulting to 25")

        print(f"Loaded: {self.path.name}")
        print(f"  Frames: {self.frame_count}, FPS: {self.fps:.3f}")
        if self.start_time:
            print(f"  Start: {self.start_time.isoformat()}")
        if self.end_time:
            print(f"  End:   {self.end_time.isoformat()}")

    def get_frame(self, index):
        """Returns raw frame (may be color or mono, uint8 or uint16)."""
        return self._reader.get_frame(index)

    def get_frame_timestamp(self, index):
        """Returns datetime for a given frame index (linear interpolation)."""
        if self.start_time is None:
            return None
        offset = datetime.timedelta(seconds=index / self.fps)
        return self.start_time + offset

    def get_frame_mono_native(self, index):
        """Grayscale in native dtype (uint8 or uint16)."""
        return to_mono_native(self.get_frame(index))

    def get_frame_color_native(self, index):
        """BGR 3-channel in native dtype (uint8 or uint16)."""
        return to_color_native(self.get_frame(index))

    def get_frame_mono16(self, index):
        """Returns frame as uint16 mono (H, W)."""
        frame = self.get_frame(index)
        return to_mono16(frame)

    def get_frame_color16(self, index):
        """Returns frame as uint16 color BGR (H, W, 3)."""
        frame = self.get_frame(index)
        return to_color16(frame)

    def close(self):
        self._reader.close()

    def __len__(self):
        return self.frame_count

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()


# ═══════════════════════════════════════════════════════════════════
#  IMAGE CONVERSION UTILITIES
# ═══════════════════════════════════════════════════════════════════
#
# Philosophy: keep the native bit depth (uint8 or uint16) throughout
# all processing (alignment, diff, blob detection). Convert to uint16
# only at TIFF export time. 8-bit ADU values stay readable in diagnostic
# prints and thresholds are in the original ADU range.

def to_mono_native(frame):
    """Grayscale, preserving original dtype (uint8 stays uint8, uint16 stays uint16)."""
    if frame.ndim == 3:
        if frame.dtype == np.uint16:
            return (0.299 * frame[:,:,2] +
                    0.587 * frame[:,:,1] +
                    0.114 * frame[:,:,0]).astype(np.uint16)
        return cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)   # uint8
    return frame


def to_color_native(frame):
    """3-channel BGR, preserving original dtype."""
    if frame.ndim == 2:
        if frame.dtype == np.uint8:
            return cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
        return np.stack([frame, frame, frame], axis=-1)
    return frame


def to_mono16(frame):
    """Grayscale uint16 — used only for TIFF export."""
    mono = to_mono_native(frame)
    if mono.dtype == np.uint8:
        return (mono.astype(np.uint16) * 257)
    return mono


def to_color16(frame):
    """BGR uint16 3-channel — used only for TIFF export."""
    color = to_color_native(frame)
    if color.dtype == np.uint8:
        return (color.astype(np.uint16) * 257)
    return color


# ═══════════════════════════════════════════════════════════════════
#  TEMPLATE MATCHING ALIGNMENT  (AutoStakkert NCC method)
# ═══════════════════════════════════════════════════════════════════
#
# AutoStakkert aligns frames by searching for the reference patch
# inside each frame using Normalised Cross-Correlation (TM_CCOEFF_NORMED).
# The peak of the correlation map gives the integer-pixel shift directly.
# No FFT, no wrap-around ambiguity, robust on solar/lunar high-contrast patches.
#
# The anchor stores the EXACT pixel rectangle (y0,x0,y1,x1) from the
# median frame.  Each incoming frame is searched in a region enlarged by
# SEARCH_MARGIN pixels on each side to cover seeing motion.
# ───────────────────────────────────────────────────────────────────

SEARCH_MARGIN = 150   # pixels around anchor bbox searched in each direction


def extract_patch(img_mono, y0, x0, y1, x1):
    """Return the pixel rectangle [y0:y1, x0:x1] from img_mono."""
    return img_mono[y0:y1, x0:x1]


def _compute_shift_ncc(ref_patch, frame_mono, anchor_y0, anchor_x0,
                        anchor_y1, anchor_x1):
    """
    Find the best-match position of ref_patch inside frame_mono using NCC.
    Search is restricted to SEARCH_MARGIN around the anchor location.

    Returns (dy, dx): shift to apply to frame_mono to align it.
    dy > 0  => frame shifted down   => shift it up.
    dx > 0  => frame shifted right  => shift it left.
    """
    h_img, w_img = frame_mono.shape

    # Search window clipped to image bounds
    sy0 = max(0,     anchor_y0 - SEARCH_MARGIN)
    sy1 = min(h_img, anchor_y1 + SEARCH_MARGIN)
    sx0 = max(0,     anchor_x0 - SEARCH_MARGIN)
    sx1 = min(w_img, anchor_x1 + SEARCH_MARGIN)

    search_region = frame_mono[sy0:sy1, sx0:sx1].astype(np.float32)
    ref_f         = ref_patch.astype(np.float32)

    # matchTemplate requires search_region >= ref_patch on each dimension
    if search_region.shape[0] < ref_f.shape[0] or search_region.shape[1] < ref_f.shape[1]:
        return 0, 0   # cannot match — return no shift

    result = cv2.matchTemplate(search_region, ref_f, cv2.TM_CCOEFF_NORMED)
    _, _, _, max_loc = cv2.minMaxLoc(result)

    # max_loc is (x, y) top-left of best match inside search_region
    match_x = sx0 + max_loc[0]
    match_y = sy0 + max_loc[1]

    dy = match_y - anchor_y0
    dx = match_x - anchor_x0
    return int(dy), int(dx)


def _apply_shift(img, dy, dx):
    """
    Shift img by (dy, dx) pixels — pure translation, no scaling or rotation.
    Borders uncovered by the shift are filled with zeros (black).
    BORDER_CONSTANT avoids the mirror-reflection artefact that BORDER_REFLECT
    produces at large shifts, which would appear as a squished band and
    incorrectly trigger the blob detector.
    """
    h, w = img.shape[:2]
    M = np.float32([[1, 0, -dx], [0, 1, -dy]])
    return cv2.warpAffine(img.astype(np.float32), M, (w, h),
                          flags=cv2.INTER_NEAREST,
                          borderMode=cv2.BORDER_CONSTANT,
                          borderValue=0)


def align_frame(ref_patch, frame_mono, anchor_y0, anchor_x0, anchor_y1, anchor_x1):
    """
    Align frame_mono to the reference using NCC template matching.
    Returns (aligned_mono_uint16, dy, dx).
    """
    dy, dx = _compute_shift_ncc(ref_patch, frame_mono,
                                 anchor_y0, anchor_x0, anchor_y1, anchor_x1)
    aligned = _apply_shift(frame_mono, dy, dx).astype(frame_mono.dtype)
    return aligned, dy, dx


def align_frame_color(ref_patch, frame_color16, anchor_y0, anchor_x0,
                       anchor_y1, anchor_x1):
    """
    Align a uint16 color (H,W,3) frame using NCC on the mono channel.
    Returns (aligned_color_uint16, dy, dx).
    """
    mono = to_mono16(frame_color16)
    dy, dx = _compute_shift_ncc(ref_patch, mono,
                                 anchor_y0, anchor_x0, anchor_y1, anchor_x1)
    channels = [_apply_shift(frame_color16[:, :, c], dy, dx).astype(np.uint16)
                for c in range(3)]
    return np.stack(channels, axis=-1), dy, dx


# ═══════════════════════════════════════════════════════════════════
#  ANCHOR SELECTION UI
# ═══════════════════════════════════════════════════════════════════

class AnchorSelector:
    """
    Tkinter UI to select the anchor region on the median frame.
    User draws a rectangle; result is stored in self.result = (y0, x0, y1, x1).
    """

    def __init__(self, median_frame_mono):
        self.frame = median_frame_mono
        self.result = None
        self._rect_start = None
        self._rect_id = None

    def run(self):
        root = tk.Tk()
        root.title("Select Anchor Region — Draw rectangle then press ENTER")

        # Scale image for display (max 1200x800)
        h, w = self.frame.shape
        scale = min(1200 / w, 800 / h, 1.0)
        dw, dh = int(w * scale), int(h * scale)

        # Convert mono16 to displayable uint8
        disp = cv2.normalize(self.frame, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        disp_resized = cv2.resize(disp, (dw, dh))

        # Convert to PhotoImage via PIL or raw
        from PIL import Image, ImageTk
        pil_img = Image.fromarray(disp_resized)
        self._tk_img = ImageTk.PhotoImage(pil_img)
        self._scale = scale

        canvas = tk.Canvas(root, width=dw, height=dh, cursor="crosshair")
        canvas.pack()
        canvas.create_image(0, 0, anchor=tk.NW, image=self._tk_img)

        label = tk.Label(root, text="Draw a rectangle on a stable surface feature (NOT on a moving object). Press ENTER to confirm.")
        label.pack()

        self._canvas = canvas
        self._dw, self._dh = dw, dh

        canvas.bind("<ButtonPress-1>",   self._on_press)
        canvas.bind("<B1-Motion>",       self._on_drag)
        canvas.bind("<ButtonRelease-1>", self._on_release)
        root.bind("<Return>", lambda e: self._confirm(root))
        root.bind("<KP_Enter>", lambda e: self._confirm(root))

        root.mainloop()
        return self.result

    def _on_press(self, event):
        self._rect_start = (event.x, event.y)
        if self._rect_id:
            self._canvas.delete(self._rect_id)

    def _on_drag(self, event):
        if self._rect_start:
            x0, y0 = self._rect_start
            if self._rect_id:
                self._canvas.delete(self._rect_id)
            self._rect_id = self._canvas.create_rectangle(
                x0, y0, event.x, event.y,
                outline='red', width=2
            )
            self._rect_end = (event.x, event.y)

    def _on_release(self, event):
        self._rect_end = (event.x, event.y)

    def _confirm(self, root):
        if not hasattr(self, '_rect_end') or self._rect_start is None:
            messagebox.showwarning("No selection", "Please draw a rectangle first.")
            return

        x0, y0 = self._rect_start
        x1, y1 = self._rect_end
        scale = self._scale

        # Back to original image coords
        x0r, y0r = int(x0 / scale), int(y0 / scale)
        x1r, y1r = int(x1 / scale), int(y1 / scale)

        # Store exact rectangle (y0, x0, y1, x1) — NCC needs the real bbox
        ay0 = min(y0r, y1r)
        ax0 = min(x0r, x1r)
        ay1 = max(y0r, y1r)
        ax1 = max(x0r, x1r)

        # Enforce minimum patch size of 32 px on each side
        h_img, w_img = self.frame.shape
        if ay1 - ay0 < 32: ay1 = min(ay0 + 32, h_img)
        if ax1 - ax0 < 32: ax1 = min(ax0 + 32, w_img)

        self.result = (ay0, ax0, ay1, ax1)
        root.destroy()


# ═══════════════════════════════════════════════════════════════════
#  THRESHOLD ESTIMATION
# ═══════════════════════════════════════════════════════════════════

# ═══════════════════════════════════════════════════════════════════
#  LOW-CONTRAST ZONE SELECTOR  +  HIGH-CONTRAST MASK
# ═══════════════════════════════════════════════════════════════════

class LowContrastSelector:
    """
    Tkinter UI — asks the user to draw a rectangle over a homogeneous,
    low-contrast region. The p99 ADU of that region on the diff image
    becomes the threshold s2 for the high-contrast mask.
    result = (y0, x0, y1, x1) in full-frame coordinates.
    """

    def __init__(self, ref_frame_mono):
        self.frame = ref_frame_mono
        self.result = None
        self._rect_start = None
        self._rect_id    = None
        self._rect_end   = None

    def run(self):
        from PIL import Image, ImageTk
        root = tk.Tk()
        root.title("Select a LOW-CONTRAST homogeneous zone — press ENTER")
        root.attributes('-topmost', True)

        h, w = self.frame.shape
        scale = min(1200 / w, 800 / h, 1.0)
        dw, dh = int(w * scale), int(h * scale)

        disp = cv2.normalize(self.frame, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        disp_r = cv2.resize(disp, (dw, dh))
        pil_img = Image.fromarray(disp_r)
        self._tk_img = ImageTk.PhotoImage(pil_img)
        self._scale  = scale

        canvas = tk.Canvas(root, width=dw, height=dh, cursor="crosshair")
        canvas.pack()
        canvas.create_image(0, 0, anchor=tk.NW, image=self._tk_img)
        tk.Label(root, text="Draw a rectangle over a FLAT, LOW-CONTRAST area "
                            "(avoid sunspots, filaments, granulation borders). "
                            "Press ENTER to confirm.").pack()

        self._canvas = canvas
        canvas.bind("<ButtonPress-1>",   self._on_press)
        canvas.bind("<B1-Motion>",       self._on_drag)
        canvas.bind("<ButtonRelease-1>", self._on_release)
        root.bind("<Return>",   lambda e: self._confirm(root))
        root.bind("<KP_Enter>", lambda e: self._confirm(root))
        root.mainloop()
        return self.result

    def _on_press(self, event):
        self._rect_start = (event.x, event.y)
        if self._rect_id:
            self._canvas.delete(self._rect_id)

    def _on_drag(self, event):
        if self._rect_start:
            x0, y0 = self._rect_start
            if self._rect_id:
                self._canvas.delete(self._rect_id)
            self._rect_id = self._canvas.create_rectangle(
                x0, y0, event.x, event.y, outline='cyan', width=2)
            self._rect_end = (event.x, event.y)

    def _on_release(self, event):
        self._rect_end = (event.x, event.y)

    def _confirm(self, root):
        if self._rect_end is None or self._rect_start is None:
            messagebox.showwarning("No selection", "Please draw a rectangle first.")
            return
        x0, y0 = self._rect_start
        x1, y1 = self._rect_end
        s = self._scale
        ay0 = int(min(y0, y1) / s);  ay1 = int(max(y0, y1) / s)
        ax0 = int(min(x0, x1) / s);  ax1 = int(max(x0, x1) / s)
        h_img, w_img = self.frame.shape
        if ay1 - ay0 < 8:  ay1 = min(ay0 + 8, h_img)
        if ax1 - ax0 < 8:  ax1 = min(ax0 + 8, w_img)
        self.result = (ay0, ax0, ay1, ax1)
        root.destroy()


def build_contrast_mask(shifts, safe_crop, video, low_zone,
                        n_samples=HIGH_CONTRAST_SAMPLE,
                        shift_px=CONTRAST_MASK_SHIFT):
    """
    Build a binary mask (crop-space) where True = high-contrast pixel to ignore.

    For each of n_samples single aligned frames:
      1. Align + crop the frame.
      2. Apply an artificial mis-alignment of shift_px pixels in a random
         cardinal direction (up/down/left/right) to create a shifted copy.
         This highlights high-contrast edges without depending on consecutive
         frame content (so transits cannot contaminate the mask).
      3. Normalise luminosity of the shifted copy via median ratio.
      4. Compute absolute diff between original and shifted copy.
      5. s2 = p99 of diff inside the user-selected low-contrast zone.
      6. Pixels > s2 are flagged.
    Final mask = logical OR over all samples, then dilated 2 px.

    Note: estimate_threshold (called after) still uses consecutive frame pairs
    for the noise level — only the mask construction uses artificial shifts.
    """
    cy0, cx0, cy1, cx1 = safe_crop
    lz_y0, lz_x0, lz_y1, lz_x1 = low_zone
    n_frames = len(video)
    n = min(n_samples, n_frames)

    print(f"\nBuilding high-contrast mask from {n} frames "
          f"(artificial shift: {shift_px} px)...")
    rng = np.random.default_rng(seed=7)
    indices = sorted(rng.choice(n_frames, size=n, replace=False).tolist())

    directions = [(shift_px, 0), (-shift_px, 0), (0, shift_px), (0, -shift_px)]

    H_crop = cy1 - cy0
    W_crop = cx1 - cx0
    mask_acc = np.zeros((H_crop, W_crop), dtype=np.uint32)

    for k, i in enumerate(indices):
        m = video.get_frame_mono_native(i)
        dy, dx = shifts[i]
        a0 = _apply_shift(m, dy, dx).astype(m.dtype)[cy0:cy1, cx0:cx1]
        del m

        # Artificially shift a0 in a random cardinal direction
        art_dy, art_dx = directions[rng.integers(0, 4)]
        a1 = _apply_shift(a0, art_dy, art_dx).astype(a0.dtype)

        med0 = float(np.median(a0))
        med1 = float(np.median(a1))
        if med1 > 0:
            a1n = np.clip(a1.astype(np.float32) * (med0 / med1),
                          0, np.iinfo(a0.dtype).max).astype(a0.dtype)
        else:
            a1n = a1
        del a1

        diff = np.abs(a0.astype(np.int32) - a1n.astype(np.int32))
        del a0, a1n

        s2 = float(np.percentile(diff[lz_y0:lz_y1, lz_x0:lz_x1], 99))
        mask_acc += (diff > s2).astype(np.uint32)
        del diff

        if (k + 1) % 20 == 0:
            print(f"  {k+1}/{n} frames processed...", end='\r')

    mask = mask_acc > 0
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask = cv2.dilate(mask.astype(np.uint8), kernel, iterations=1).astype(bool)

    pct = 100.0 * mask.sum() / mask.size
    print(f"  High-contrast mask: {pct:.1f}% of crop area flagged.")
    gc.collect()
    return mask


def get_blob_centroid(diff_img, threshold, min_pixels=MIN_RESIDUAL_PIXELS,
                      contrast_mask=None):
    """
    Check if diff image contains a qualifying blob (>= min_pixels above threshold).
    If contrast_mask is provided (bool array), those pixels are zeroed before detection.
    Returns (cy, cx) centroid of the largest qualifying blob, or None if none found.
    """
    if contrast_mask is not None:
        diff_img = diff_img.copy()
        diff_img[contrast_mask] = 0
    mask = diff_img > threshold
    if not mask.any():
        return None
    labeled, num = ndimage.label(mask)
    if num == 0:
        return None

    best_size = 0
    best_centroid = None
    for region_id in range(1, num + 1):
        region_mask = labeled == region_id
        size = int(region_mask.sum())
        if size >= min_pixels and size > best_size:
            best_size = size
            cy, cx = ndimage.center_of_mass(region_mask)
            best_centroid = (int(round(cy)), int(round(cx)))

    return best_centroid  # None if no region qualifies


def has_bright_blob(diff_img, threshold, min_pixels=MIN_RESIDUAL_PIXELS):
    """Convenience wrapper — returns bool."""
    return get_blob_centroid(diff_img, threshold, min_pixels) is not None


def estimate_threshold(shifts, safe_crop, video,
                       n_samples=CALM_FRAME_SAMPLE, margin=THRESHOLD_MARGIN,
                       contrast_mask=None):
    """
    Estimate noise threshold on aligned + cropped diff frames.
    Same conditions as the blob scan: shift-corrected and cropped to safe zone.
    Samples n_samples evenly-spaced pairs, collects max ADU of each aligned diff,
    thresholds at 95th percentile x margin.
    """
    cy0, cx0, cy1, cx1 = safe_crop
    n_frames = len(video) - 1
    n = min(n_samples, n_frames)
    print(f"\nEstimating noise threshold from {n} aligned frames...")
    indices = np.linspace(0, n_frames - 1, n, dtype=int).tolist()
    maxima = []
    for i in indices:
        m0 = video.get_frame_mono_native(i)
        m1 = video.get_frame_mono_native(i + 1)
        dy0, dx0 = shifts[i]
        dy1, dx1 = shifts[i + 1]
        a0 = _apply_shift(m0, dy0, dx0).astype(m0.dtype)[cy0:cy1, cx0:cx1]
        a1 = _apply_shift(m1, dy1, dx1).astype(m1.dtype)[cy0:cy1, cx0:cx1]
        diff = np.abs(a0.astype(np.int32) - a1.astype(np.int32))
        if contrast_mask is not None:
            diff[contrast_mask] = 0
        maxima.append(float(diff.max()))
        del m0, m1, a0, a1, diff
    threshold = float(np.percentile(maxima, 95)) * margin
    print(f"  Raw maxima (95th pct): {np.percentile(maxima, 95):.1f} ADU")
    print(f"  Threshold (x{margin}): {threshold:.1f} ADU")
    return threshold


# ═══════════════════════════════════════════════════════════════════
#  TRANSIT DETECTION  (2-pass: shifts → safe crop → blob scan)
# ═══════════════════════════════════════════════════════════════════

def compute_safe_crop(shifts, frame_shape):
    """
    Given all (dy, dx) shifts applied via warpAffine M=[[1,0,-dx],[0,1,-dy]],
    compute the rectangle guaranteed free of black borders in every aligned frame.

    warpAffine with translation (-dx, -dy) moves pixel at (r,c) to (r-dy, c-dx).
    So:
      dy > 0 → content shifts UP   → last dy rows are black   → crop cy1 = H - max(dy>0)
      dy < 0 → content shifts DOWN → first |dy| rows are black → crop cy0 = max(|dy|<0)
      dx > 0 → content shifts LEFT → last dx cols are black   → crop cx1 = W - max(dx>0)
      dx < 0 → content shifts RIGHT→ first |dx| cols are black → crop cx0 = max(|dx|<0)
    """
    H, W = frame_shape
    dys = [dy for dy, dx in shifts]
    dxs = [dx for dy, dx in shifts]

    # rows: positive dy eats from bottom, negative dy eats from top
    cy0 = max(0,  max((-dy for dy in dys if dy < 0), default=0))
    cy1 = min(H,  H - max((dy for dy in dys if dy > 0), default=0))
    # cols: positive dx eats from right, negative dx eats from left
    cx0 = max(0,  max((-dx for dx in dxs if dx < 0), default=0))
    cx1 = min(W,  W - max((dx for dx in dxs if dx > 0), default=0))

    # Add 1-pixel safety margin to absorb any rounding
    cy0 = min(cy0 + 1, H // 2)
    cy1 = max(cy1 - 1, H // 2 + 1)
    cx0 = min(cx0 + 1, W // 2)
    cx1 = max(cx1 - 1, W // 2 + 1)

    if cy1 <= cy0 or cx1 <= cx0:
        print("  WARNING: safe crop degenerate (shifts too large) — using full frame.")
        return 0, 0, H, W

    return int(cy0), int(cx0), int(cy1), int(cx1)


def detect_transit_frames(video, ref_patch,
                           anchor_y0, anchor_x0, anchor_y1, anchor_x1,
                           n_samples=CALM_FRAME_SAMPLE,
                           margin=THRESHOLD_MARGIN,
                           low_zone=None):
    """
    Two-pass transit detection.

    Pass 1 — collect shifts:
        Align every frame via NCC, record (dy, dx). No blob detection yet.

    Safe crop:
        Compute the rectangle guaranteed free of black borders in all frames.

    Pass 2 — blob scan on cropped aligned frames:
        Compute |In - In+1| inside the safe crop only — no border artefacts.
        Centroid coordinates are stored in crop-space.

    Returns (detected, safe_crop, shifts):
        detected  : dict {frame_idx: (cy_crop, cx_crop)}
        safe_crop : (cy0, cx0, cy1, cx1)
        shifts    : list[(dy, dx)] one per frame
    """
    n = len(video)
    frame_shape = video.get_frame_mono16(0).shape   # (H, W)

    # ── Pass 1: collect all shifts ────────────────────────────────
    print(f"\nPass 1/2 \u2014 computing alignment shifts for {n} frames...")
    shifts = []
    for i in range(n):
        mono = video.get_frame_mono_native(i)
        _, dy, dx = align_frame(ref_patch, mono,
                                anchor_y0, anchor_x0, anchor_y1, anchor_x1)
        shifts.append((dy, dx))
        del mono
        if (i + 1) % 100 == 0:
            print(f"  {i+1}/{n} frames...", end='\r')

    gc.collect()
    dys = [d for d,_ in shifts]; dxs = [d for _,d in shifts]
    print(f"  Done. dy=[{min(dys)},{max(dys)}] dx=[{min(dxs)},{max(dxs)}]")

    # ── Safe crop ─────────────────────────────────────────────────
    cy0, cx0, cy1, cx1 = compute_safe_crop(shifts, frame_shape)
    print(f"  Safe crop: y=[{cy0}:{cy1}], x=[{cx0}:{cx1}]  ({cy1-cy0}\u00d7{cx1-cx0} px)")

    # ── High-contrast mask ─────────────────────────────
    if low_zone is not None:
        # Convert low_zone from full-frame coords to crop-space
        lz_y0 = max(low_zone[0] - cy0, 0)
        lz_x0 = max(low_zone[1] - cx0, 0)
        lz_y1 = min(low_zone[2] - cy0, cy1 - cy0)
        lz_x1 = min(low_zone[3] - cx0, cx1 - cx0)
        if lz_y1 > lz_y0 and lz_x1 > lz_x0:
            low_zone_crop = (lz_y0, lz_x0, lz_y1, lz_x1)
            contrast_mask = build_contrast_mask(
                shifts, (cy0, cx0, cy1, cx1), video, low_zone_crop)
        else:
            print("  WARNING: low-contrast zone outside safe crop — mask disabled.")
            contrast_mask = None
    else:
        contrast_mask = None

    # ── Threshold on aligned+cropped frames ─────────────
    threshold = estimate_threshold(shifts, (cy0, cx0, cy1, cx1), video,
                                   n_samples=n_samples, margin=margin,
                                   contrast_mask=contrast_mask)

    # ── Pass 2: blob scan on cropped frames ───────────────────────
    print(f"\nPass 2/2 \u2014 scanning for transit blobs in safe zone...")
    detected = {}

    prev_mono = video.get_frame_mono_native(0)
    dy0, dx0 = shifts[0]
    prev_al = _apply_shift(prev_mono, dy0, dx0).astype(prev_mono.dtype)[cy0:cy1, cx0:cx1]
    del prev_mono

    for i in range(1, n):
        curr_mono = video.get_frame_mono_native(i)
        dy, dx = shifts[i]
        curr_al = _apply_shift(curr_mono, dy, dx).astype(curr_mono.dtype)[cy0:cy1, cx0:cx1]
        del curr_mono

        diff = np.abs(curr_al.astype(np.int32) - prev_al.astype(np.int32)).astype(np.uint16)
        centroid = get_blob_centroid(diff, threshold,
                                     contrast_mask=contrast_mask)
        if centroid is not None:
            detected[i] = centroid
            if len(detected) % 10 == 0 or len(detected) == 1:
                print(f"  Blob at frame {i}, centroid={centroid} (crop coords)")

        del diff, prev_al
        prev_al = curr_al

        if (i + 1) % 100 == 0:
            print(f"  Scanned {i+1}/{n}...", end='\r')

    del prev_al
    gc.collect()
    print(f"\nTotal frames with transit blobs: {len(detected)}")
    return detected, (cy0, cx0, cy1, cx1), shifts

def group_transits(detected, gap=GAP_MERGE_FRAMES):
    """
    Group consecutive (or near-consecutive) detected frames into transit events.
    `detected` is a dict {frame_idx: (cy, cx)} as returned by detect_transit_frames.
    Returns list of lists of frame indices.
    """
    if not detected:
        return []

    frame_list = sorted(detected.keys())
    groups = []
    current = [frame_list[0]]
    for f in frame_list[1:]:
        if f - current[-1] <= gap:
            current.append(f)
        else:
            groups.append(current)
            current = [f]
    groups.append(current)
    return groups


# ═══════════════════════════════════════════════════════════════════
#  TRANSIT VALIDATION  (reject seeing artefacts)
# ═══════════════════════════════════════════════════════════════════

def validate_transit_group(group, detected_centroids,
                           min_move_px=MIN_TRANSIT_MOTION):
    """
    Reject a candidate group if the total displacement of the blob centroid
    between the first and last detected frame is below min_move_px.
    This eliminates seeing shimmers that stay at a fixed location.

    Returns (True, reason_str) if valid, (False, reason_str) if rejected.
    """
    detected_frames = [f for f in group if f in detected_centroids]
    if len(detected_frames) < 2:
        return False, "single detected frame — no displacement measurable"

    centroids = [detected_centroids[f] for f in detected_frames]
    cy0, cx0 = centroids[0]
    cyN, cxN = centroids[-1]
    total_disp = np.hypot(cyN - cy0, cxN - cx0)

    if total_disp < min_move_px:
        return False, (f"total displacement {total_disp:.1f} px "
                       f"< minimum {min_move_px} px (seeing shimmer)")

    # Compute mean speed just for the log message
    speeds = []
    for k in range(1, len(detected_frames)):
        dt = max(detected_frames[k] - detected_frames[k-1], 1)
        cy_p, cx_p = centroids[k-1]
        cy_c, cx_c = centroids[k]
        speeds.append(np.hypot(cy_c - cy_p, cx_c - cx_p) / dt)

    return True, (f"OK — displacement={total_disp:.1f} px, "
                  f"mean_speed={np.mean(speeds):.1f} px/frame")


# ═══════════════════════════════════════════════════════════════════
#  OUTPUT: SAVE TRANSIT SEQUENCES
# ═══════════════════════════════════════════════════════════════════

def timestamp_to_filename(ts):
    """Convert datetime to safe filename string."""
    if ts is None:
        return "unknown"
    return ts.strftime('%Y-%m-%dT%H-%M-%S_%f')[:-3]  # ms precision


# Output AVI FPS
OUTPUT_AVI_FPS = 8

# Arrow appearance
ARROW_OFFSET_PX  = 20    # pixels below centroid where arrow tip lands
ARROW_LENGTH_PX  = 18    # length of the arrow shaft in pixels
ARROW_ALPHA_DIFF = 0.55  # opacity of white arrow on diff video (0=invisible, 1=opaque)


def _draw_arrow_on_uint8(img_bgr, tip_y, tip_x, color_bgr, alpha):
    """
    Draw a downward-pointing arrow on a uint8 BGR image (in-place blend).
    The arrowhead tip sits at (tip_y, tip_x).
    """
    h, w = img_bgr.shape[:2]
    tip_y  = int(np.clip(tip_y,  0, h - 1))
    tip_x  = int(np.clip(tip_x,  0, w - 1))
    tail_y = int(np.clip(tip_y - ARROW_LENGTH_PX, 0, h - 1))
    tail_x = tip_x

    if alpha >= 1.0:
        cv2.arrowedLine(img_bgr, (tail_x, tail_y), (tip_x, tip_y),
                        color_bgr, thickness=2, tipLength=0.5)
    else:
        overlay = img_bgr.copy()
        cv2.arrowedLine(overlay, (tail_x, tail_y), (tip_x, tip_y),
                        color_bgr, thickness=2, tipLength=0.5)
        cv2.addWeighted(overlay, alpha, img_bgr, 1.0 - alpha, 0, img_bgr)
        del overlay


def _frame_to_uint8_bgr(frame_color):
    """Convert BGR frame (uint8 or uint16) to uint8 BGR for VideoWriter."""
    if frame_color.dtype == np.uint8:
        return frame_color
    return (frame_color >> 8).astype(np.uint8)


def _diff_to_uint8_bgr(diff_mono16):
    """Stretch mono uint16 diff image to full uint8 range, return 3-ch BGR."""
    stretched = cv2.normalize(diff_mono16, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    return cv2.cvtColor(stretched, cv2.COLOR_GRAY2BGR)


def save_transit_sequence(video, transit_group, transit_index, output_base_dir,
                          shifts, safe_crop,
                          detected_centroids,
                          frames_before=FRAMES_BEFORE_TRANSIT,
                          frames_after=FRAMES_AFTER_TRANSIT):
    """
    Save the full transit sequence (with padding) as:
      - 16-bit uncompressed TIFF files (one per frame, cropped to safe zone)
      - AVI normal : cropped color frames with a solid red arrow
      - AVI diff   : |In-In+1| stretched, with a semi-transparent white arrow

    shifts     : list of (dy, dx) for every frame in the video (from pass 1)
    safe_crop  : (cy0, cx0, cy1, cx1) black-border-free rectangle
    Centroids are already in crop coordinates.
    """
    cy0, cx0, cy1, cx1 = safe_crop
    first_frame = transit_group[0]
    last_frame  = transit_group[-1]

    start_idx = max(0, first_frame - frames_before)
    end_idx   = min(len(video) - 1, last_frame + frames_after)
    n_out     = end_idx - start_idx + 1

    first_centroid = detected_centroids[first_frame]
    last_centroid  = detected_centroids[last_frame]

    stem = Path(video.path).stem
    folder_name = f"{stem}-transit-{transit_index}"
    out_dir = Path(output_base_dir) / folder_name
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"\nSaving transit {transit_index}: frames {start_idx}–{end_idx} → {out_dir}")

    frame_size = (cx1 - cx0, cy1 - cy0)   # (W, H) for VideoWriter
    fourcc = cv2.VideoWriter_fourcc(*'MJPG')
    vw_normal = cv2.VideoWriter(
        str(out_dir / f"{stem}-transit-{transit_index}-normal.avi"),
        fourcc, OUTPUT_AVI_FPS, frame_size)
    vw_diff = cv2.VideoWriter(
        str(out_dir / f"{stem}-transit-{transit_index}-diff.avi"),
        fourcc, OUTPUT_AVI_FPS, frame_size)

    # Seed diff chain from frame just before start
    seed_idx = max(0, start_idx - 1)
    seed_mono = video.get_frame_mono_native(seed_idx)
    dy_s, dx_s = shifts[seed_idx]
    prev_mono_al = _apply_shift(seed_mono, dy_s, dx_s).astype(seed_mono.dtype)[cy0:cy1, cx0:cx1]
    del seed_mono

    for idx in range(start_idx, end_idx + 1):
        dy, dx = shifts[idx]

        frame_color = video.get_frame_color_native(idx)
        channels = [_apply_shift(frame_color[:, :, c], dy, dx).astype(frame_color.dtype)[cy0:cy1, cx0:cx1]
                    for c in range(3)]
        frame_al = np.stack(channels, axis=-1)
        del frame_color, channels

        curr_mono_al = to_mono_native(frame_al)

        ts = video.get_frame_timestamp(idx)
        frame_al16 = to_color16(frame_al)
        tifffile.imwrite(
            str(out_dir / (timestamp_to_filename(ts) + ".tif")),
            frame_al16, photometric='rgb', compression=None)
        del frame_al16

        if idx in detected_centroids:
            cy, cx = detected_centroids[idx]
        elif idx < first_frame:
            cy, cx = first_centroid
        else:
            cy, cx = last_centroid
        tip_y = cy + ARROW_OFFSET_PX

        normal_u8 = _frame_to_uint8_bgr(frame_al)
        del frame_al
        _draw_arrow_on_uint8(normal_u8, tip_y, cx, color_bgr=(0, 0, 255), alpha=1.0)
        vw_normal.write(normal_u8)
        del normal_u8

        diff_mono = np.abs(curr_mono_al.astype(np.int32) - prev_mono_al.astype(np.int32)).astype(np.uint16)
        diff_u8   = _diff_to_uint8_bgr(diff_mono)
        del diff_mono
        _draw_arrow_on_uint8(diff_u8, tip_y, cx,
                             color_bgr=(255, 255, 255), alpha=ARROW_ALPHA_DIFF)
        vw_diff.write(diff_u8)
        del diff_u8

        del prev_mono_al
        prev_mono_al = curr_mono_al

    del prev_mono_al
    vw_normal.release()
    vw_diff.release()
    gc.collect()
    print(f"  Saved {n_out} TIFFs + 2 AVI files.")

# ═══════════════════════════════════════════════════════════════════
#  MEDIAN FRAME COMPUTATION
# ═══════════════════════════════════════════════════════════════════

def compute_median_frame(video, max_samples=50):
    """Compute median frame from evenly-spaced samples (mono16)."""
    print("\nComputing median reference frame...")
    n = len(video)
    indices = np.linspace(0, n - 1, min(max_samples, n), dtype=int)
    stack = []
    for i in indices:
        stack.append(video.get_frame_mono_native(i))
    stacked = np.stack(stack, axis=0)
    del stack          # free the list of individual frames immediately
    median = np.median(stacked, axis=0).astype(np.uint16)
    del stacked        # free the 3-D stack (largest allocation here)
    gc.collect()
    print(f"  Median computed from {len(indices)} frames.")
    return median


# ═══════════════════════════════════════════════════════════════════
#  SPYDER / IDE DETECTION
# ═══════════════════════════════════════════════════════════════════

def _running_in_ide():
    """
    Returns True when the script is executed inside Spyder or another
    interactive IDE (Jupyter, PyCharm console, etc.).
    In those environments argparse must NOT be used because sys.argv
    is owned by the IDE and will cause a SystemExit on parse_args().
    """
    # Spyder sets this env var
    if os.environ.get('SPY_PYTHONPATH') or os.environ.get('SPYDER_ARGS'):
        return True
    # IPython kernel (covers Spyder, Jupyter, VS Code interactive)
    try:
        from IPython import get_ipython
        if get_ipython() is not None:
            return True
    except ImportError:
        pass
    # Fallback: if sys.argv[0] looks like an IDE launcher
    argv0 = sys.argv[0] if sys.argv else ''
    ide_markers = ('spyder', 'ipykernel', 'ipython', 'jupyter',
                   'pydev', 'pycharm', 'code',  # VS Code
                   'kernel', '-c')
    if any(m in argv0.lower() for m in ide_markers):
        return True
    return False


# ═══════════════════════════════════════════════════════════════════
#  CORE PIPELINE  (single file, anchor + ref_patch already known)
# ═══════════════════════════════════════════════════════════════════

def process_single_file(video, anchor_y0, anchor_x0, anchor_y1, anchor_x1,
                         ref_patch, output_dir,
                         frames_before=FRAMES_BEFORE_TRANSIT,
                         frames_after=FRAMES_AFTER_TRANSIT,
                         threshold_margin=THRESHOLD_MARGIN,
                         calm_samples=CALM_FRAME_SAMPLE,
                         low_zone=None):
    """
    Run the full detection pipeline on one already-opened VideoFile,
    using a pre-computed anchor (ref_patch + coordinates).
    Threshold is estimated independently for this file.
    Transit folders are numbered from 1 for each file.
    """
    # ── 2-pass transit detection (threshold estimated on aligned frames inside) ──
    detected, safe_crop, shifts = detect_transit_frames(
        video, ref_patch,
        anchor_y0, anchor_x0, anchor_y1, anchor_x1,
        n_samples=calm_samples,
        margin=threshold_margin,
        low_zone=low_zone
    )

    if not detected:
        print("  → No transits detected in this file.")
        return 0

    # ── Group transits ──
    transit_groups = group_transits(detected)
    print(f"  → {len(transit_groups)} candidate group(s) before validation.")

    # ── Validate groups ──
    valid_groups = []
    for g in transit_groups:
        ok, reason = validate_transit_group(g, detected)
        if ok:
            valid_groups.append(g)
            print(f"  ✓ Group frames {g[0]}–{g[-1]}: {reason}")
        else:
            print(f"  ✗ Group frames {g[0]}–{g[-1]} REJECTED: {reason}")

    if not valid_groups:
        print("  → All groups rejected by validation filters. No transits saved.")
        return 0

    print(f"  → {len(valid_groups)} confirmed transit(s) after validation.")

    # ── Save sequences (numbering resets per file) ──
    for n, group in enumerate(valid_groups, start=1):
        save_transit_sequence(
            video, group, n, output_dir,
            shifts=shifts,
            safe_crop=safe_crop,
            detected_centroids=detected,
            frames_before=frames_before,
            frames_after=frames_after
        )

    return len(valid_groups)

# ═══════════════════════════════════════════════════════════════════
#  BATCH PIPELINE  (multiple files, shared anchor)
# ═══════════════════════════════════════════════════════════════════

def run_batch(input_paths,
              output_dir=None,
              frames_before=FRAMES_BEFORE_TRANSIT,
              frames_after=FRAMES_AFTER_TRANSIT,
              threshold_margin=THRESHOLD_MARGIN,
              calm_samples=CALM_FRAME_SAMPLE):
    """
    Process a list of AVI/SER files of the same surface.

    Workflow:
      1. Open the first file → compute its median frame → show anchor UI once.
      2. Re-use those anchor coordinates for every subsequent file.
      3. Threshold is re-estimated independently for each file.
      4. Transit folder numbering resets to 1 for each file.

    Can be called directly from Spyder:
        from transit_detector import run_batch
        run_batch([r"C:/data/vid1.avi", r"C:/data/vid2.avi"])
    """
    if not input_paths:
        print("No files provided.")
        return

    input_paths = [Path(p) for p in input_paths]
    n_files = len(input_paths)
    print("=" * 60)
    print(f"  Transit Detector  —  BATCH MODE  ({n_files} file(s))")
    print("=" * 60)

    # ── Step 1: anchor selection on first file ──
    print(f"\n[1/{n_files}] Computing median frame for anchor selection: {input_paths[0].name}")
    first_video = VideoFile(input_paths[0])
    try:
        median_mono = compute_median_frame(first_video)
    finally:
        first_video.close()
        del first_video   # close file handle and free reader object

    print("\nOpening anchor selection window (used for ALL files in this batch)...")
    print("  → Draw a rectangle on a stable surface feature, then press ENTER.")
    selector = AnchorSelector(median_mono)
    result = selector.run()

    if result is None:
        print("No anchor selected. Aborting.")
        del median_mono
        gc.collect()
        return

    anchor_y0, anchor_x0, anchor_y1, anchor_x1 = result
    print(f"Anchor: y=[{anchor_y0}:{anchor_y1}], x=[{anchor_x0}:{anchor_x1}], "
          f"size={anchor_y1-anchor_y0}×{anchor_x1-anchor_x0} px")
    ref_patch = extract_patch(median_mono, anchor_y0, anchor_x0, anchor_y1, anchor_x1)
    print(f"Reference patch shape: {ref_patch.shape}")

    # ── Low-contrast zone selection ──
    print("\nOpening low-contrast zone selection (for high-contrast mask)...")
    print("  → Draw a rectangle over a FLAT, HOMOGENEOUS region.")
    print("  → Avoid sunspots, filaments, granulation edges, etc.")
    lc_selector = LowContrastSelector(median_mono)
    lc_result = lc_selector.run()
    del median_mono
    gc.collect()

    if lc_result is None:
        print("No low-contrast zone selected — contrast mask disabled.")
        low_zone_fullframe = None
    else:
        lz_y0, lz_x0, lz_y1, lz_x1 = lc_result
        low_zone_fullframe = (lz_y0, lz_x0, lz_y1, lz_x1)
        print(f"Low-contrast zone: y=[{lz_y0}:{lz_y1}], x=[{lz_x0}:{lz_x1}]")

    # ── Step 2: process each file ──
    total_transits = 0
    for file_idx, path in enumerate(input_paths, start=1):
        print(f"\n{'─' * 60}")
        print(f"[{file_idx}/{n_files}] Processing: {path.name}")
        print(f"{'─' * 60}")

        file_output_dir = Path(output_dir) if output_dir else path.parent

        video = VideoFile(path)
        try:
            n_found = process_single_file(
                video,
                anchor_y0, anchor_x0, anchor_y1, anchor_x1,
                ref_patch,
                file_output_dir,
                frames_before=frames_before,
                frames_after=frames_after,
                threshold_margin=threshold_margin,
                calm_samples=calm_samples,
                low_zone=low_zone_fullframe,
            )
            total_transits += n_found
        except Exception as e:
            print(f"  ERROR processing {path.name}: {e}")
        finally:
            video.close()
            del video     # release file handle + all internal buffers
            gc.collect()  # force-free numpy arrays from this file before next one

    print(f"\n{'=' * 60}")
    print(f"  Batch complete. {total_transits} transit sequence(s) total across {n_files} file(s).")
    print(f"{'=' * 60}")


# ═══════════════════════════════════════════════════════════════════
#  MAIN ENTRY POINT
# ═══════════════════════════════════════════════════════════════════

def _pick_files():
    """Open a multi-file selection dialog. Returns list of paths or []."""
    root = tk.Tk()
    root.withdraw()
    root.attributes('-topmost', True)
    paths = filedialog.askopenfilenames(
        title="Select AVI / SER video file(s) — same surface, any order",
        filetypes=[
            ("Video files", "*.avi *.AVI *.ser *.SER"),
            ("AVI files",   "*.avi *.AVI"),
            ("SER files",   "*.ser *.SER"),
            ("All files",   "*.*"),
        ]
    )
    root.destroy()
    return list(paths)


def main():
    # ── Detect execution context ──
    if _running_in_ide():
        # ────────────────────────────────────────────────────────
        #  IDE / Spyder mode
        #  Parameters are taken from the GLOBAL CONSTANTS above.
        #  Multi-file selection via Windows dialog.
        # ────────────────────────────────────────────────────────
        print("=" * 60)
        print("  Transit Detector  —  running in IDE / Spyder mode")
        print("  Edit GLOBAL PARAMETERS at the top of the script to")
        print("  change frames_before, frames_after, threshold, etc.")
        print("  Hold Ctrl (or Shift) in the file dialog to select")
        print("  multiple files.")
        print("=" * 60)

        input_paths = _pick_files()
        if not input_paths:
            print("No file selected. Exiting.")
            return

        run_batch(
            input_paths,
            frames_before=FRAMES_BEFORE_TRANSIT,
            frames_after=FRAMES_AFTER_TRANSIT,
            threshold_margin=THRESHOLD_MARGIN,
            calm_samples=CALM_FRAME_SAMPLE,
        )

    else:
        # ────────────────────────────────────────────────────────
        #  CLI mode — supports multiple positional file arguments
        #  python transit_detector.py vid1.avi vid2.avi vid3.ser
        # ────────────────────────────────────────────────────────
        parser = argparse.ArgumentParser(
            description='Transit Detector — detect transiting objects in lucky imaging videos.'
        )
        parser.add_argument('input', nargs='*',
                            help='One or more AVI/SER files (same surface). '
                                 'Omit to open a file dialog.')
        parser.add_argument('--output', '-o', default=None,
                            help='Output directory (default: same folder as each input file)')
        parser.add_argument('--frames-before', type=int, default=FRAMES_BEFORE_TRANSIT,
                            help=f'Frames saved before transit (default: {FRAMES_BEFORE_TRANSIT})')
        parser.add_argument('--frames-after', type=int, default=FRAMES_AFTER_TRANSIT,
                            help=f'Frames saved after transit (default: {FRAMES_AFTER_TRANSIT})')
        parser.add_argument('--threshold-margin', type=float, default=THRESHOLD_MARGIN,
                            help=f'Noise multiplier for threshold (default: {THRESHOLD_MARGIN})')
        parser.add_argument('--calm-samples', type=int, default=CALM_FRAME_SAMPLE,
                            help=f'Frames sampled for threshold (default: {CALM_FRAME_SAMPLE})')

        args = parser.parse_args()

        input_paths = args.input
        if not input_paths:
            input_paths = _pick_files()
            if not input_paths:
                print("No file selected. Exiting.")
                sys.exit(0)

        run_batch(
            input_paths,
            output_dir=args.output,
            frames_before=args.frames_before,
            frames_after=args.frames_after,
            threshold_margin=args.threshold_margin,
            calm_samples=args.calm_samples,
        )


if __name__ == '__main__':
    main()
