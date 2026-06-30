#!/usr/bin/env python3
"""
Planetary Gradient Removal

Supported input/output:
    AVI RGB/MONO 8-bit   -> AVI compatible
    SER MONO 8/16-bit    -> SER MONO 8/16-bit
    SER RGB 8/16-bit     -> SER RGB 8/16-bit
"""

import csv
import gc
import os
import struct
import shutil
import tempfile
from pathlib import Path
import tkinter as tk
from tkinter import filedialog, messagebox
from concurrent.futures import ThreadPoolExecutor, as_completed, wait, FIRST_COMPLETED

import cv2
import numpy as np

# ═══════════════════════════════════════════════════════════════════
#  HEAD PARAMETERS
# ═══════════════════════════════════════════════════════════════════

NORMALIZE_FRAMES = False
NORMALIZE_TO_PERCENT = 60.0

SIGNAL_PERCENTILE = 99.5
MIN_SIGNAL_THRESHOLD = 5.0
MIN_SIGNAL_PIXELS = 20

OUTPUT_GAIN = 1.0
OUTPUT_GAMMA = 1.0

# ═══════════════════════════════════════════════════════════════════
# ═══════════════════════════════════════════════════════════════════

DISPLAY_SCALE_MAX = 1100

PROCESSING_THREADS = max(1, min(8, (os.cpu_count() or 2) - 1))
CACHE_SAVE_QUEUE_LIMIT = PROCESSING_THREADS * 4

OUTPUT_SUFFIX = "_GR"
GRADIENT_TIFF_SUFFIX = "-gradient"
CSV_SUFFIX = "_log"

SER_COLOR_IDS = {
    0: "MONO",
    100: "RGB",
}


class SERReader:
    def __init__(self, path):
        self.path = Path(path)
        self.f = open(path, "rb")
        self._read_header()

    def _read_header(self):
        f = self.f
        f.seek(0)

        self.header_bytes = f.read(178)
        f.seek(0)

        file_id = f.read(14).decode("ascii", errors="ignore")
        if not file_id.startswith("LUCAM-RECORDER"):
            raise ValueError(f"Not a valid SER file: {self.path}")

        self.lu_id = struct.unpack("<i", f.read(4))[0]
        self.color_id = struct.unpack("<i", f.read(4))[0]
        self.little_endian = struct.unpack("<i", f.read(4))[0]
        self.width = struct.unpack("<i", f.read(4))[0]
        self.height = struct.unpack("<i", f.read(4))[0]
        self.bit_depth = struct.unpack("<i", f.read(4))[0]
        self.frame_count = struct.unpack("<i", f.read(4))[0]

        self.observer = f.read(40)
        self.instrument = f.read(40)
        self.telescope = f.read(40)
        self.datetime = f.read(8)
        self.datetime_utc = f.read(8)

        if self.color_id not in SER_COLOR_IDS:
            raise ValueError(
                f"Unsupported SER color type: {self.color_id}. "
                "Only MONO and RGB SER files are supported."
            )

        self.color_name = SER_COLOR_IDS[self.color_id]
        self.is_mono = self.color_name == "MONO"
        self.is_color = self.color_name == "RGB"

        self.bytes_per_pixel = (self.bit_depth + 7) // 8
        self.channels = 3 if self.is_color else 1
        self.frame_size = self.width * self.height * self.bytes_per_pixel * self.channels
        self.data_offset = 178

        self._dtype = np.uint8
        self._effective_endian = None

        if self.bytes_per_pixel == 2:
            self._effective_endian = "<" if self.little_endian != 0 else ">"
            self._dtype = np.dtype(self._effective_endian + "u2")
            self._auto_detect_uint16_endian()

        self.data_end_offset = self.data_offset + self.frame_count * self.frame_size

    def _uint16_smoothness_score(self, arr):
        a = arr.astype(np.int32, copy=False)
        step_y = max(1, a.shape[0] // 256)
        step_x = max(1, a.shape[1] // 256)
        a = a[::step_y, ::step_x]

        if a.ndim == 3:
            a = a[..., 0]

        dy = np.abs(np.diff(a, axis=0)).mean() if a.shape[0] > 1 else 0.0
        dx = np.abs(np.diff(a, axis=1)).mean() if a.shape[1] > 1 else 0.0

        return float(dx + dy)

    def _auto_detect_uint16_endian(self):
        if self.bytes_per_pixel != 2 or self.frame_count <= 0:
            return

        pos = self.f.tell()

        try:
            self.f.seek(self.data_offset)
            raw = self.f.read(self.frame_size)
        finally:
            self.f.seek(pos)

        if len(raw) != self.frame_size:
            return

        shape = (
            (self.height, self.width, 3)
            if self.is_color
            else (self.height, self.width)
        )

        try:
            le = np.frombuffer(raw, dtype="<u2").reshape(shape)
            be = np.frombuffer(raw, dtype=">u2").reshape(shape)
        except ValueError:
            return

        score_le = self._uint16_smoothness_score(le)
        score_be = self._uint16_smoothness_score(be)

        header_endian = "<" if self.little_endian != 0 else ">"
        chosen = header_endian

        if score_le > 0 and score_be > 0:
            ratio = max(score_le, score_be) / max(min(score_le, score_be), 1e-9)
            if ratio >= 1.35:
                chosen = "<" if score_le < score_be else ">"

        self._effective_endian = chosen
        self._dtype = np.dtype(chosen + "u2")

        if chosen != header_endian:
            print(
                f"  SER endian warning: header says "
                f"{'little' if header_endian == '<' else 'big'}-endian, "
                f"but frame data looks "
                f"{'little' if chosen == '<' else 'big'}-endian. "
                f"Using detected order."
            )

    def get_frame_rgb_or_mono(self, index):
        self.f.seek(self.data_offset + index * self.frame_size)
        raw = self.f.read(self.frame_size)

        if len(raw) != self.frame_size:
            raise IOError(f"Cannot read SER frame {index}")

        shape = (
            (self.height, self.width, 3)
            if self.is_color
            else (self.height, self.width)
        )

        frame = np.frombuffer(raw, dtype=self._dtype).reshape(shape)

        if self.bytes_per_pixel == 2:
            frame = frame.astype(np.uint16, copy=False)

        return frame

    def get_trailing_bytes(self):
        self.f.seek(0, os.SEEK_END)
        file_size = self.f.tell()

        if file_size <= self.data_end_offset:
            return b""

        self.f.seek(self.data_end_offset)
        return self.f.read(file_size - self.data_end_offset)

    def close(self):
        self.f.close()


class SERWriter:
    def __init__(self, output_path, source_reader):
        self.output_path = Path(output_path)
        self.source = source_reader
        self.f = open(self.output_path, "wb")

        self.f.write(self.source.header_bytes)

        self.frame_count = self.source.frame_count
        self.frames_written = 0

        if self.source.bytes_per_pixel == 1:
            self.write_dtype = np.uint8
        else:
            endian = self.source._effective_endian
            self.write_dtype = np.dtype(endian + "u2")

    def write(self, frame_native_order):
        if self.frames_written >= self.frame_count:
            raise RuntimeError("Too many frames written to SER output.")

        frame = np.asarray(frame_native_order)

        expected_shape = (
            (self.source.height, self.source.width, 3)
            if self.source.is_color
            else (self.source.height, self.source.width)
        )

        if frame.shape != expected_shape:
            raise RuntimeError(
                f"Invalid SER frame shape: got {frame.shape}, "
                f"expected {expected_shape}"
            )

        if self.source.bytes_per_pixel == 1:
            frame_out = np.clip(frame, 0, 255).astype(np.uint8)
        else:
            frame_out = np.clip(frame, 0, 65535).astype(self.write_dtype)

        self.f.write(frame_out.tobytes(order="C"))
        self.frames_written += 1

    def close(self):
        if self.frames_written != self.frame_count:
            print(
                f"  Warning: SER writer wrote {self.frames_written}/"
                f"{self.frame_count} frames."
            )

        trailing = self.source.get_trailing_bytes()

        if trailing:
            self.f.write(trailing)

        self.f.close()


class VideoFile:
    def __init__(self, path):
        self.path = Path(path)
        self.ext = self.path.suffix.lower()

        if self.ext == ".ser":
            self.reader = SERReader(self.path)
            self.frame_count = self.reader.frame_count
            self.width = self.reader.width
            self.height = self.reader.height
            self.fps = None
            self.bit_depth = self.reader.bit_depth
            self.max_value = 255.0 if self.bit_depth <= 8 else 65535.0
            self.is_color = self.reader.is_color
            self.color_name = self.reader.color_name

        elif self.ext == ".avi":
            self.reader = cv2.VideoCapture(str(self.path))

            if not self.reader.isOpened():
                raise RuntimeError(f"Cannot open AVI: {self.path}")

            self.frame_count = int(self.reader.get(cv2.CAP_PROP_FRAME_COUNT))
            self.fps = float(self.reader.get(cv2.CAP_PROP_FPS))
            self.width = int(self.reader.get(cv2.CAP_PROP_FRAME_WIDTH))
            self.height = int(self.reader.get(cv2.CAP_PROP_FRAME_HEIGHT))
            self.bit_depth = 8
            self.max_value = 255.0
            self.color_name = "RGB"

            if self.fps <= 0:
                raise RuntimeError(f"Invalid frame rate in AVI file: {self.path}")

            first = self.get_frame_rgb_or_mono(0)
            self.is_color = not is_effectively_mono(first)

        else:
            raise RuntimeError(f"Unsupported input format: {self.ext}")

        if self.frame_count <= 0:
            raise RuntimeError(f"Invalid frame count: {self.path}")

    def get_frame_rgb_or_mono(self, index):
        if self.ext == ".ser":
            return self.reader.get_frame_rgb_or_mono(index)

        self.reader.set(cv2.CAP_PROP_POS_FRAMES, index)
        ok, frame = self.reader.read()

        if not ok:
            raise IOError(f"Cannot read AVI frame {index}: {self.path}")

        if is_effectively_mono(frame):
            return frame[:, :, 0]

        # OpenCV decodes AVI color as BGR, but the program works internally in RGB.
        return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    def close(self):
        if self.ext == ".ser":
            self.reader.close()
        else:
            self.reader.release()


def select_input_videos():
    root = tk.Tk()
    root.withdraw()
    root.attributes("-topmost", True)

    paths = filedialog.askopenfilenames(
        title="Select AVI or SER videos to process",
        filetypes=[
            ("Video files", "*.avi *.ser"),
            ("AVI files", "*.avi"),
            ("SER files", "*.ser"),
            ("All files", "*.*"),
        ]
    )

    root.destroy()
    return [Path(p) for p in paths]


def is_effectively_mono(frame):
    if frame.ndim == 2:
        return True

    if frame.ndim != 3 or frame.shape[2] != 3:
        return False

    return (
        np.array_equal(frame[:, :, 0], frame[:, :, 1]) and
        np.array_equal(frame[:, :, 1], frame[:, :, 2])
    )


def ensure_rgb(frame):
    if frame.ndim == 2:
        return np.stack([frame, frame, frame], axis=-1)

    return frame


def luminance_rgb(img):
    img = ensure_rgb(img)

    return (
        0.299 * img[:, :, 0] +
        0.587 * img[:, :, 1] +
        0.114 * img[:, :, 2]
    )


def frame_to_display_u8(frame):
    frame = ensure_rgb(frame)

    if frame.dtype == np.uint8:
        return cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

    frame8 = np.clip(frame / 257.0, 0, 255).astype(np.uint8)
    return cv2.cvtColor(frame8, cv2.COLOR_RGB2BGR)


def resize_for_display(img, max_size=DISPLAY_SCALE_MAX):
    h, w = img.shape[:2]
    scale = min(max_size / max(w, h), 1.0)

    if scale < 1.0:
        resized = cv2.resize(
            img,
            (int(w * scale), int(h * scale)),
            interpolation=cv2.INTER_AREA
        )
    else:
        resized = img.copy()

    return resized, scale


def choose_background_roi(reference_frame):
    display = frame_to_display_u8(reference_frame)
    display, scale = resize_for_display(display)

    base = display.copy()
    preview = display.copy()

    roi_data = {
        "drawing": False,
        "start": None,
        "end": None,
        "confirmed": False,
        "cancelled": False,
    }

    window_name = "Select BACKGROUND / GRADIENT ROI - ENTER to confirm, ESC to cancel"

    def redraw():
        preview[:] = base

        if roi_data["start"] is not None and roi_data["end"] is not None:
            x0, y0 = roi_data["start"]
            x1, y1 = roi_data["end"]

            x0, x1 = sorted((x0, x1))
            y0, y1 = sorted((y0, y1))

            overlay = preview.copy()

            cv2.rectangle(
                overlay,
                (x0, y0),
                (x1, y1),
                (210, 210, 210),
                thickness=-1
            )

            cv2.addWeighted(overlay, 0.35, preview, 0.65, 0, preview)

            cv2.rectangle(
                preview,
                (x0, y0),
                (x1, y1),
                (230, 230, 230),
                thickness=2
            )

        cv2.imshow(window_name, preview)

    def mouse_callback(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            roi_data["drawing"] = True
            roi_data["start"] = (x, y)
            roi_data["end"] = (x, y)
            redraw()

        elif event == cv2.EVENT_MOUSEMOVE and roi_data["drawing"]:
            roi_data["end"] = (x, y)
            redraw()

        elif event == cv2.EVENT_LBUTTONUP:
            roi_data["drawing"] = False
            roi_data["end"] = (x, y)
            redraw()

    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.setMouseCallback(window_name, mouse_callback)
    redraw()

    while True:
        key = cv2.waitKey(20) & 0xFF

        if key in (13, 10):
            roi_data["confirmed"] = True
            break

        if key == 27:
            roi_data["cancelled"] = True
            break

        if cv2.getWindowProperty(window_name, cv2.WND_PROP_VISIBLE) < 1:
            roi_data["cancelled"] = True
            break

    cv2.destroyWindow(window_name)

    if roi_data["cancelled"] or not roi_data["confirmed"]:
        raise RuntimeError("No ROI selected.")

    if roi_data["start"] is None or roi_data["end"] is None:
        raise RuntimeError("No ROI selected.")

    x0, y0 = roi_data["start"]
    x1, y1 = roi_data["end"]

    x0, x1 = sorted((x0, x1))
    y0, y1 = sorted((y0, y1))

    w = x1 - x0
    h = y1 - y0

    if w == 0 or h == 0:
        raise RuntimeError("No ROI selected.")

    return (
        int(x0 / scale),
        int(y0 / scale),
        int(w / scale),
        int(h / scale),
    )


def mean_rgb_in_roi(frame, roi):
    frame = ensure_rgb(frame)
    x, y, w, h = roi
    patch = frame[y:y+h, x:x+w].astype(np.float32)
    return patch.reshape(-1, 3).mean(axis=0)


def subtract_background_gradient(frame, roi):
    frame_rgb = ensure_rgb(frame).astype(np.float32)
    gradient_rgb = mean_rgb_in_roi(frame, roi)

    corrected = frame_rgb - gradient_rgb
    corrected = np.clip(corrected, 0, None)

    return corrected, gradient_rgb


def build_signal_mask(reference_corrected):
    lum = luminance_rgb(reference_corrected)

    threshold = np.percentile(lum, SIGNAL_PERCENTILE)
    threshold = max(threshold, MIN_SIGNAL_THRESHOLD)

    mask = lum >= threshold

    if np.count_nonzero(mask) < MIN_SIGNAL_PIXELS:
        threshold = np.percentile(lum, 99.0)
        threshold = max(threshold, MIN_SIGNAL_THRESHOLD)
        mask = lum >= threshold

    if np.count_nonzero(mask) < MIN_SIGNAL_PIXELS:
        raise RuntimeError(
            "Cannot build a reliable signal mask for brightness normalization."
        )

    return mask


def normalize_frame_brightness(corrected_frame, signal_mask, max_value):
    lum = luminance_rgb(corrected_frame)
    values = lum[signal_mask]

    if values.size == 0:
        return corrected_frame, 1.0, 0.0

    current_signal = float(np.mean(values))

    if current_signal <= 0:
        return corrected_frame, 1.0, current_signal

    target_signal = max_value * (NORMALIZE_TO_PERCENT / 100.0)
    scale = target_signal / current_signal

    normalized = corrected_frame * scale

    return normalized, scale, current_signal


def apply_output_gain_gamma(img, max_value):
    img = np.clip(img * OUTPUT_GAIN, 0, max_value)

    if OUTPUT_GAMMA != 1.0:
        img_norm = img / max_value
        img_norm = np.power(img_norm, 1.0 / OUTPUT_GAMMA)
        img = img_norm * max_value

    return np.clip(img, 0, max_value)


def make_avi_frame_for_writer(frame):
    frame = np.asarray(frame)

    if frame.ndim == 2:
        frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
    else:
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

    if frame.dtype != np.uint8:
        frame = np.clip(frame, 0, 255).astype(np.uint8)

    return np.ascontiguousarray(frame)


def create_avi_writer(output_path, fps, width, height):
    codec_candidates = [
        ("HFYU", "HuffYUV lossless"),
        ("FFV1", "FFV1 lossless"),
        ("LAGS", "Lagarith lossless"),
        ("MJPG", "Motion JPEG fallback"),
    ]

    dummy = np.zeros((height, width, 3), dtype=np.uint8)

    for fourcc_str, label in codec_candidates:
        test_path = output_path.with_name(output_path.stem + f"_codec_test_{fourcc_str}.avi")

        try:
            fourcc = cv2.VideoWriter_fourcc(*fourcc_str)
            writer = cv2.VideoWriter(
                str(test_path),
                fourcc,
                fps,
                (width, height),
                True
            )

            if not writer.isOpened():
                writer.release()
                continue

            try:
                writer.write(dummy)
                writer.release()
            except Exception:
                writer.release()
                if test_path.exists():
                    test_path.unlink()
                continue

            if test_path.exists():
                test_path.unlink()

            writer = cv2.VideoWriter(
                str(output_path),
                fourcc,
                fps,
                (width, height),
                True
            )

            if writer.isOpened():
                return writer

            writer.release()

        except Exception:
            try:
                if test_path.exists():
                    test_path.unlink()
            except Exception:
                pass

    raise RuntimeError(
        "Cannot create AVI output with available codecs. "
        "Try outputting SER instead or install a lossless AVI codec."
    )


def save_mean_gradient_tiff(output_path, mean_gradient_rgb, width, height, max_value):
    if max_value <= 255:
        mean_16 = np.clip(mean_gradient_rgb * 257.0, 0, 65535).astype(np.uint16)
    else:
        mean_16 = np.clip(mean_gradient_rgb, 0, 65535).astype(np.uint16)

    img16 = np.zeros((height, width, 3), dtype=np.uint16)
    img16[:, :, 0] = mean_16[0]
    img16[:, :, 1] = mean_16[1]
    img16[:, :, 2] = mean_16[2]

    ok = cv2.imwrite(str(output_path), cv2.cvtColor(img16, cv2.COLOR_RGB2BGR))

    if not ok:
        raise RuntimeError(f"Cannot write TIFF: {output_path}")


def make_output_paths(video_path):
    video_path = Path(video_path)
    output_ext = video_path.suffix.lower()

    output_video_path = video_path.with_name(
        video_path.stem + OUTPUT_SUFFIX + output_ext
    )

    output_tiff_path = video_path.with_name(
        video_path.stem + GRADIENT_TIFF_SUFFIX + ".tif"
    )

    output_csv_path = video_path.with_name(
        video_path.stem + CSV_SUFFIX + ".csv"
    )

    return output_video_path, output_tiff_path, output_csv_path


def frame_to_output_native_for_params(frame_rgb, video_params):
    is_color = video_params["is_color"]
    max_value = video_params["max_value"]

    if not is_color:
        frame = luminance_rgb(frame_rgb)
    else:
        frame = frame_rgb

    if max_value <= 255:
        return np.clip(frame, 0, 255).astype(np.uint8)

    return np.clip(frame, 0, 65535).astype(np.uint16)


def save_cache_frame(cache_path, frame):
    np.save(cache_path, frame, allow_pickle=False)
    return cache_path


def cache_video_frames(video, cache_dir):
    cache_dir.mkdir(parents=True, exist_ok=True)

    print("\nCaching frames to temporary disk cache...")

    cache_paths = []
    pending = set()

    with ThreadPoolExecutor(max_workers=PROCESSING_THREADS) as executor:
        for i in range(video.frame_count):
            frame = video.get_frame_rgb_or_mono(i).copy()
            cache_path = cache_dir / f"frame_{i:08d}.npy"
            cache_paths.append(cache_path)

            pending.add(
                executor.submit(
                    save_cache_frame,
                    cache_path,
                    frame
                )
            )

            if len(pending) >= CACHE_SAVE_QUEUE_LIMIT:
                done, pending = wait(pending, return_when=FIRST_COMPLETED)
                for future in done:
                    future.result()

            if (i + 1) % 100 == 0:
                print(f"  Cached {i + 1}/{video.frame_count} frames...", end="\r")

        for future in as_completed(pending):
            future.result()

    print(f"  Cached {video.frame_count}/{video.frame_count} frames.")
    return cache_paths


def process_frame_worker(cache_path, roi, signal_mask, video_params):
    index = int(cache_path.stem.split("_")[-1])

    frame = np.load(cache_path, allow_pickle=False)

    max_value = video_params["max_value"]

    corrected, gradient_rgb = subtract_background_gradient(frame, roi)

    if NORMALIZE_FRAMES:
        corrected, scale, current_signal = normalize_frame_brightness(
            corrected,
            signal_mask,
            max_value
        )
    else:
        scale = 1.0
        current_signal = 0.0

    corrected = apply_output_gain_gamma(corrected, max_value)
    output_frame = frame_to_output_native_for_params(corrected, video_params)

    del frame, corrected

    return {
        "index": index,
        "output_frame": output_frame,
        "gradient_rgb": gradient_rgb,
        "current_signal": current_signal,
        "scale": scale,
    }


def process_single_video(video_path, roi, signal_mask):
    video = VideoFile(video_path)
    cache_dir = None
    writer = None

    try:
        output_video_path, output_tiff_path, output_csv_path = make_output_paths(video.path)

        if video.ext == ".avi":
            writer = create_avi_writer(
                output_video_path,
                video.fps,
                video.width,
                video.height
            )
        else:
            writer = SERWriter(output_video_path, video.reader)

        video_params = {
            "ext": video.ext,
            "is_color": video.is_color,
            "max_value": video.max_value,
        }

        print(f"\nProcessing: {video.path.name}")
        print(f"  Format:    {video.ext.upper().replace('.', '')}")
        print(f"  Frames:    {video.frame_count}")

        if video.ext == ".avi":
            print(f"  FPS:       {video.fps:.6f}")
        else:
            print("  FPS:       SER native header/timestamps preserved")

        print(f"  Size:      {video.width} x {video.height}")
        print(f"  Bit depth: {video.bit_depth}")
        print(f"  Color:     {'RGB' if video.is_color else 'MONO'}")
        print(f"  Threads:   {PROCESSING_THREADS}")
        print(f"  Output:    {output_video_path.name}")

        cache_root = Path(tempfile.gettempdir())
        cache_dir = cache_root / f"{video.path.stem}_PGR_cache_{os.getpid()}"

        if cache_dir.exists():
            shutil.rmtree(cache_dir)

        cache_paths = cache_video_frames(video, cache_dir)

        gradient_sum = np.zeros(3, dtype=np.float64)
        processed_frames = 0

        print("\nProcessing frames...")

        with open(output_csv_path, "w", newline="") as f:
            csv_writer = csv.writer(f)

            csv_writer.writerow([
                "frame",
                "gradient_R",
                "gradient_G",
                "gradient_B",
                "signal_before_normalization",
                "normalization_scale",
                "output_gain",
                "output_gamma",
                "normalize_to_percent",
                "native_max_value"
            ])

            results_by_index = {}

            with ThreadPoolExecutor(max_workers=PROCESSING_THREADS) as executor:
                futures = [
                    executor.submit(
                        process_frame_worker,
                        cache_path,
                        roi,
                        signal_mask,
                        video_params
                    )
                    for cache_path in cache_paths
                ]

                next_to_write = 0

                for future in as_completed(futures):
                    result = future.result()
                    results_by_index[result["index"]] = result

                    while next_to_write in results_by_index:
                        item = results_by_index.pop(next_to_write)

                        output_frame = item["output_frame"]

                        if video.ext == ".avi":
                            output_frame = make_avi_frame_for_writer(output_frame)

                        writer.write(output_frame)

                        gradient_rgb = item["gradient_rgb"]
                        gradient_sum += gradient_rgb
                        processed_frames += 1

                        csv_writer.writerow([
                            processed_frames,
                            gradient_rgb[0],
                            gradient_rgb[1],
                            gradient_rgb[2],
                            item["current_signal"],
                            item["scale"],
                            OUTPUT_GAIN,
                            OUTPUT_GAMMA,
                            NORMALIZE_TO_PERCENT,
                            video.max_value
                        ])

                        next_to_write += 1

                        if processed_frames % 100 == 0:
                            print(
                                f"  {processed_frames}/{video.frame_count} frames processed...",
                                end="\r"
                            )

                        del item, output_frame

        writer.close() if video.ext == ".ser" else writer.release()
        writer = None

        if processed_frames == 0:
            raise RuntimeError(f"No frames processed: {video.path}")

        mean_gradient = gradient_sum / processed_frames

        save_mean_gradient_tiff(
            output_tiff_path,
            mean_gradient,
            video.width,
            video.height,
            video.max_value
        )

        print(f"  {processed_frames}/{video.frame_count} frames processed.")
        print("  Done.")
        print(f"  Output video: {output_video_path}")
        print(f"  Mean gradient TIFF: {output_tiff_path}")
        print(f"  CSV log: {output_csv_path}")

    finally:
        if writer is not None:
            try:
                writer.close() if video.ext == ".ser" else writer.release()
            except Exception:
                pass

        video.close()

        if cache_dir is not None and cache_dir.exists():
            print("\nRemoving temporary cache...")
            shutil.rmtree(cache_dir, ignore_errors=True)
            print("  Cache removed.")

        gc.collect()


def validate_batch_geometry(video_paths):
    ref = VideoFile(video_paths[0])

    try:
        ref_width = ref.width
        ref_height = ref.height

        for path in video_paths[1:]:
            v = VideoFile(path)

            try:
                if v.width != ref_width or v.height != ref_height:
                    raise RuntimeError(
                        "Batch videos do not have the same resolution.\n"
                        f"Reference: {video_paths[0].name} = {ref_width} x {ref_height}\n"
                        f"Mismatch:  {Path(path).name} = {v.width} x {v.height}"
                    )
            finally:
                v.close()

    finally:
        ref.close()


def run_batch(video_paths):
    if not video_paths:
        print("No files selected.")
        return

    video_paths = [Path(p) for p in video_paths]
    n_files = len(video_paths)

    print("=" * 68)
    print("  Planetary Gradient Removal")
    print("=" * 68)
    print(f"  Files selected:   {n_files}")
    print(f"  Normalize frames: {NORMALIZE_FRAMES}")
    print(f"  Normalize target: {NORMALIZE_TO_PERCENT:.1f}%")
    print(f"  Output gain:      {OUTPUT_GAIN}")
    print(f"  Output gamma:     {OUTPUT_GAMMA}")
    print(f"  Threads:          {PROCESSING_THREADS}")
    print("=" * 68)

    validate_batch_geometry(video_paths)

    first_video = VideoFile(video_paths[0])

    try:
        reference_frame = first_video.get_frame_rgb_or_mono(first_video.frame_count - 1)
    finally:
        first_video.close()

    print("\nOpening ROI selection window...")
    print("  Select only background / sky gradient.")
    print("  Avoid planet, rings, moons, dust spots, and useful signal.")

    roi = choose_background_roi(reference_frame)

    print(
        f"\nShared background ROI:"
        f" x={roi[0]}, y={roi[1]}, w={roi[2]}, h={roi[3]}"
    )

    reference_corrected, _ = subtract_background_gradient(reference_frame, roi)
    signal_mask = build_signal_mask(reference_corrected)

    print("\nBrightness normalization mask:")
    print(f"  Signal pixels:     {np.count_nonzero(signal_mask)}")
    print(f"  Signal percentile: {SIGNAL_PERCENTILE}")
    print(f"  Target level:      {NORMALIZE_TO_PERCENT:.1f}%")

    del reference_frame, reference_corrected
    gc.collect()

    total_ok = 0
    total_failed = 0

    for index, path in enumerate(video_paths, start=1):
        print("\n" + "─" * 68)
        print(f"[{index}/{n_files}] {path.name}")
        print("─" * 68)

        try:
            process_single_video(path, roi, signal_mask)
            total_ok += 1
        except Exception as exc:
            total_failed += 1
            print(f"  ERROR: {exc}")

    print("\n" + "=" * 68)
    print("  Batch complete.")
    print(f"  Successful files: {total_ok}")
    print(f"  Failed files:     {total_failed}")
    print("=" * 68)


def main():
    video_paths = select_input_videos()

    if not video_paths:
        return

    try:
        run_batch(video_paths)
    except Exception as exc:
        print(f"\nFATAL ERROR: {exc}")

        root = tk.Tk()
        root.withdraw()
        root.attributes("-topmost", True)
        messagebox.showerror("Planetary Gradient Removal - Error", str(exc))
        root.destroy()
        return

    root = tk.Tk()
    root.withdraw()
    root.attributes("-topmost", True)
    messagebox.showinfo(
        "Planetary Gradient Removal",
        "Batch processing complete."
    )
    root.destroy()


if __name__ == "__main__":
    main()