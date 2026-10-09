import csv
import queue
from collections import deque
import threading
import time
import tkinter as tk
from datetime import datetime
from tkinter import filedialog, messagebox, ttk

import numpy as np
import pyaudio
from scipy.signal import butter, sosfilt


class WatchTicker:
    """Records audio from the microphone and detects watch ticks."""

    # Waveform-fold rate estimate: the high-passed energy envelope is stored in
    # BIN-sample bins and folded at candidate periods; the sharpest fold wins.
    BIN = 32
    FOLD_MIN_SECONDS = 30.0
    FOLD_SETTLED_SECONDS = 60.0
    FOLD_MAX_SECONDS = 600.0

    def __init__(self, chunk_size=512, sample_rate=44100):
        self.chunk_size = chunk_size
        self.sample_rate = sample_rate
        self.threshold = 0.024
        self.audio = None
        self.stream = None
        self.running = False
        self.counting = False
        self.origin = 0.0
        self.t_now = 0.0
        self.thread = None
        self.events = queue.Queue()
        self.live_ticks = deque(maxlen=500)
        self.calibrate = True
        self.cal_points = deque(maxlen=3600)
        self.cal_samples = 0
        self.cal_next = 0.0
        self.listen = False
        self.volume = 1.0
        self.out_stream = None
        self.play_queue = queue.Queue(maxsize=8)
        self.play_thread = None
        self.hp_sos = butter(4, 6000, "hp", fs=sample_rate, output="sos")
        self.hp_zi = np.zeros((self.hp_sos.shape[0], 2))
        self.fold_busy = False
        self._fold_started = 0.0
        self.reset()

    def reset(self):
        self.tick_times = []
        self.ticks = []
        self.env_bins = []
        self.fold_result = None

    def begin_counting(self):
        self.reset()
        self.origin = self.t_now
        self.counting = True

    def end_counting(self):
        self.counting = False

    def estimate_fold(self, nominal_bph):
        """Estimate the beat period (sample-clock seconds) by folding the envelope.

        Folding uses the whole run, so missed or extra ticks and low-frequency
        room noise do not bias it the way per-tick threshold timing does.
        """
        if self.fold_busy or not self.env_bins:
            return
        self.fold_busy = True
        try:
            dt = self.BIN / self.sample_rate
            env = np.concatenate(list(self.env_bins)).astype(np.float64)
            env = env[-int(self.FOLD_MAX_SECONDS / dt):]
            duration = len(env) * dt
            if duration < self.FOLD_MIN_SECONDS:
                return
            env = np.clip(env - np.median(env), 0, None)

            def make_score(values, step):
                t = np.arange(len(values)) * step

                def score(period):
                    nb = int(round(period / step))
                    phase = ((t % period) * (nb / period)).astype(np.int64) % nb
                    return np.bincount(phase, weights=values, minlength=nb).var()
                return score

            # The fold peak is only about (tick width / run length) wide, so the
            # coarse grid must be that fine or the search can step over it.
            group = 6
            coarse = env[: len(env) // group * group].reshape(-1, group).sum(1)
            coarse_score = make_score(coarse, dt * group)
            previous = self.fold_result
            centre = previous[0] if previous else 3600.0 / nominal_bph
            span = 0.001 if previous else 0.02
            step = 0.3 * dt * group / duration
            count = int(np.clip(2 * span / step, 81, 20000))
            cand = centre * (1 + np.linspace(-span, span, count))
            best = cand[int(np.argmax([coarse_score(p) for p in cand]))]

            coarse_step = cand[1] / cand[0] - 1
            fine_score = make_score(env, dt)
            for span in (2 * coarse_step, 0.05 * coarse_step):
                cand = best * (1 + np.linspace(-span, span, 81))
                best = cand[int(np.argmax([fine_score(p) for p in cand]))]
            self.fold_result = (best, duration)
        finally:
            self.fold_busy = False

    @staticmethod
    def list_inputs():
        """Return [(device index, name)] for all input devices and the default index."""
        p = pyaudio.PyAudio()
        try:
            devices = [
                (i, p.get_device_info_by_index(i)["name"])
                for i in range(p.get_device_count())
                if p.get_device_info_by_index(i)["maxInputChannels"] > 0
            ]
            try:
                default = p.get_default_input_device_info()["index"]
            except IOError:
                default = None
        finally:
            p.terminate()
        return devices, default

    def start(self, device_index=None):
        self.audio = pyaudio.PyAudio()
        self.stream = self.audio.open(
            format=pyaudio.paFloat32,
            channels=1,
            rate=self.sample_rate,
            input=True,
            input_device_index=device_index,
            frames_per_buffer=self.chunk_size,
        )
        self.reset()
        self.live_ticks.clear()
        self.cal_points.clear()
        self.cal_next = 0.0
        self.t_now = 0.0
        self.hp_zi[:] = 0.0
        self.running = True
        self.thread = threading.Thread(target=self._loop, daemon=True)
        self.thread.start()
        self.play_thread = threading.Thread(target=self._play_loop, daemon=True)
        self.play_thread.start()

    def set_listen(self, enabled):
        if enabled and self.out_stream is None and self.audio:
            self.out_stream = self.audio.open(
                format=pyaudio.paFloat32,
                channels=1,
                rate=self.sample_rate,
                output=True,
                frames_per_buffer=self.chunk_size,
            )
        self.listen = enabled

    def _play_loop(self):
        # Playback runs on its own thread so a slow speaker never stalls tick detection.
        while self.running:
            try:
                data = self.play_queue.get(timeout=0.2)
            except queue.Empty:
                continue
            stream = self.out_stream
            if stream is not None and self.listen:
                try:
                    stream.write(data)
                except Exception:
                    pass

    def stop(self):
        self.running = False
        self.counting = False
        self.listen = False
        for t in (self.thread, self.play_thread):
            if t:
                t.join(timeout=2)
        self.thread = self.play_thread = None
        if self.out_stream:
            self.out_stream.stop_stream()
            self.out_stream.close()
            self.out_stream = None
        if self.stream:
            self.stream.stop_stream()
            self.stream.close()
            self.stream = None
        if self.audio:
            self.audio.terminate()
            self.audio = None

    def _loop(self):
        chunk_index = 0
        above = False
        while self.running:
            try:
                data = self.stream.read(self.chunk_size, exception_on_overflow=False)
            except Exception as exc:
                self.running = False
                self.counting = False
                self.events.put(("error", str(exc)))
                return
            samples = np.frombuffer(data, dtype=np.float32)
            # The filter state must see every chunk, even when not counting.
            filtered, self.hp_zi = sosfilt(self.hp_sos, samples.astype(np.float64), zi=self.hp_zi)
            if self.counting:
                bins = filtered.reshape(-1, self.BIN)
                self.env_bins.append(np.sqrt((bins ** 2).mean(1)).astype(np.float32))
            if self.listen:
                out = np.clip(samples * self.volume, -1.0, 1.0).astype(np.float32)
                try:
                    self.play_queue.put_nowait(out.tobytes())
                except queue.Full:
                    pass
            rms = float(np.sqrt(np.mean(samples ** 2)))
            # Time from the sample count avoids jitter from chunk scheduling.
            t = chunk_index * self.chunk_size / self.sample_rate
            chunk_index += 1
            self.t_now = t
            self._log_clock(chunk_index)
            tick = rms > self.threshold and not above
            above = rms > self.threshold
            if tick:
                self.live_ticks.append(t)
                if self.counting:
                    self.tick_times.append(t - self.origin)
                    self.ticks.append(rms)
            self.events.put(("level", t, rms))

    CAL_MIN_SECONDS = 120.0

    def _log_clock(self, chunk_index):
        # Pair the audio sample count with the system clock (NTP/GPS-disciplined).
        now = time.time()
        if now >= self.cal_next:
            self.cal_points.append((chunk_index * self.chunk_size, now))
            self.cal_next = now + 1.0

    def clock_report(self):
        """Return (measured sample rate, ppm error, seconds observed) or None."""
        pts = list(self.cal_points)
        if len(pts) < 10:
            return None
        n = np.array([p[0] for p in pts], dtype=np.float64)
        w = np.array([p[1] for p in pts], dtype=np.float64)
        span = w[-1] - w[0]
        # Samples vs wall time: the slope is the true sample rate.
        rate = float(np.polyfit(w - w[0], n - n[0], 1)[0])
        return rate, (rate / self.sample_rate - 1.0) * 1e6, span

    def clock_factor(self):
        """Multiplier turning nominal sample-clock seconds into real seconds."""
        report = self.clock_report()
        if not self.calibrate or report is None or report[2] < self.CAL_MIN_SECONDS:
            return 1.0
        return self.sample_rate / report[0]

    STANDARD_BPH = (14400, 18000, 19800, 21600, 25200, 28800, 36000)

    def stats(self):
        """Return a dict of tick analysis metrics, or None if too few ticks."""
        if len(self.tick_times) < 4:
            return None
        factor = self.clock_factor()
        times = np.array(self.tick_times) * factor
        intervals = np.diff(times)
        # Drop outliers (missed ticks or noise) using the median.
        median = np.median(intervals)
        mask = (intervals > 0.7 * median) & (intervals < 1.3 * median)
        good = intervals[mask]
        if len(good) < 3:
            return None
        mean = float(good.mean())
        bph = 3600.0 / mean
        nominal = min(self.STANDARD_BPH, key=lambda s: abs(s - bph))
        # Fit tick time against beat number. Missed ticks keep their slot, so
        # one dropped beat does not skew the rate and timing error averages out.
        beats = np.round((times - times[0]) / mean)
        beats_ok = np.concatenate(([True], mask))
        if beats_ok.sum() >= 3 and np.ptp(beats[beats_ok]) > 0:
            slope = np.polyfit(beats[beats_ok], times[beats_ok], 1)[0]
            bph = 3600.0 / slope
            nominal = min(self.STANDARD_BPH, key=lambda s: abs(s - bph))
        rate_basis = "tick fit"
        now = time.time()
        if now - self._fold_started > 3.0 and not self.fold_busy:
            self._fold_started = now
            threading.Thread(target=self.estimate_fold, args=(float(nominal),), daemon=True).start()
        fold = self.fold_result
        if fold is not None:
            bph = 3600.0 / (fold[0] * factor)
            rate_basis = f"waveform fold, {fold[1]:.0f} s"
            if fold[1] < self.FOLD_SETTLED_SECONDS:
                rate_basis += " (settling)"
        # Alternating intervals correspond to tick and tock.
        tick_iv, tock_iv = intervals[0::2], intervals[1::2]
        tick_iv = tick_iv[(tick_iv > 0.7 * median) & (tick_iv < 1.3 * median)]
        tock_iv = tock_iv[(tock_iv > 0.7 * median) & (tock_iv < 1.3 * median)]
        beat_error = (
            abs(tick_iv.mean() - tock_iv.mean()) / 2 * 1000.0
            if len(tick_iv) and len(tock_iv) else 0.0
        )
        volumes = np.array(self.ticks)
        return {
            "bph": bph,
            "rate_basis": rate_basis,
            "nominal": nominal,
            "rate_error": (bph / nominal - 1.0) * 86400.0,
            "beat_error": beat_error,
            "jitter": float(good.std()) * 1000.0,
            "mean": mean * 1000.0,
            "min": float(good.min()) * 1000.0,
            "max": float(good.max()) * 1000.0,
            "outliers": int(len(intervals) - len(good)),
            "volume_mean": float(volumes.mean()),
            "volume_peak": float(volumes.max()),
            "intervals": intervals,
        }

    def save_data(self, filename):
        with open(filename, "w", newline="") as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(["Tick", "Time_s", "Interval_s", "Volume"])
            previous = None
            factor = self.clock_factor()
            for i, (t, volume) in enumerate(zip(self.tick_times, self.ticks), 1):
                t *= factor
                interval = "" if previous is None else f"{t - previous:.6f}"
                writer.writerow([i, f"{t:.6f}", interval, f"{volume:.6f}"])
                previous = t


class WatchTimerApp(tk.Tk):
    GRAPH_SECONDS = 4.0
    GRAPH_MAX = 0.08
    GRAPH_W = 396
    GRAPH_H = 170
    IV_W = 400
    IV_H = 200

    def __init__(self):
        super().__init__()
        self.title("Watch Timer")
        self.geometry("860x750")
        self.resizable(False, False)
        self.ticker = WatchTicker()
        self.started_at = None
        self.history = deque()
        self._build()
        self.protocol("WM_DELETE_WINDOW", self._on_close)
        self._start_monitoring()
        self.after(50, self._poll)

    def _start_monitoring(self):
        devices, default = WatchTicker.list_inputs()
        self.devices = devices
        names = [name for _, name in devices]
        # Prefer the built-in mic over the system default (often AirPods or a phone).
        choice = next((i for i, n in devices if "macbook" in n.lower() and "microphone" in n.lower()), default)
        if choice is None and devices:
            choice = devices[0][0]
        self.device_combo.config(values=names)
        if choice is not None:
            self.device_combo.set(dict(devices)[choice])
        self._open_device(choice)

    def _open_device(self, index):
        self.ticker.stop()
        self.history.clear()
        self.listen_var.set(False)
        self.started_at = None
        try:
            self.ticker.start(index)
        except Exception as exc:
            self.ticker.stop()
            self.status.config(text="Microphone unavailable")
            self.start_btn.config(state="disabled")
            messagebox.showerror("Microphone error", str(exc))
            return
        self.start_btn.config(state="normal")
        self.stop_btn.config(state="disabled")
        self.status.config(text="Live - press Start to count ticks")

    def _on_device(self, _event=None):
        name = self.device_combo.get()
        index = next((i for i, n in self.devices if n == name), None)
        self._open_device(index)

    def _build(self):
        pad = {"padx": 12, "pady": 6}
        container = ttk.Frame(self)
        container.pack(fill="both", expand=True)
        frame = ttk.Frame(container)
        frame.pack(side="left", fill="y", **pad)
        side = ttk.Frame(container)
        side.pack(side="left", fill="both", expand=True, **pad)

        self.status = ttk.Label(frame, text="Ready", font=("Helvetica", 14))
        self.status.pack(anchor="w", **pad)

        ttk.Label(frame, text="Input device").pack(anchor="w")
        self.device_combo = ttk.Combobox(frame, state="readonly", width=48)
        self.device_combo.pack(fill="x", pady=(0, 8))
        self.device_combo.bind("<<ComboboxSelected>>", self._on_device)
        ttk.Label(frame, text="Input level").pack(anchor="w")
        self.level = ttk.Progressbar(frame, maximum=self.GRAPH_MAX, length=380)
        self.level.pack(fill="x", pady=(0, 8))

        ttk.Label(frame, text="Tick threshold").pack(anchor="w")
        self.threshold_var = tk.DoubleVar(value=self.ticker.threshold)
        thr_row = ttk.Frame(frame)
        thr_row.pack(fill="x", pady=(0, 8))
        ttk.Scale(
            thr_row, from_=0.001, to=0.1, variable=self.threshold_var,
            command=lambda _v: self._set_threshold(self.threshold_var.get()),
        ).pack(side="left", fill="x", expand=True)
        ttk.Button(thr_row, text="-", width=2,
                   command=lambda: self._nudge_threshold(-0.001)).pack(side="left", padx=(8, 0))
        self.threshold_spin = ttk.Spinbox(
            thr_row, from_=0.001, to=0.2, increment=0.001, width=7, format="%.3f",
            command=self._on_spin,
        )
        self.threshold_spin.set(f"{self.ticker.threshold:.3f}")
        self.threshold_spin.pack(side="left", padx=2)
        self.threshold_spin.bind("<Return>", lambda _e: self._on_spin())
        self.threshold_spin.bind("<FocusOut>", lambda _e: self._on_spin())
        ttk.Button(thr_row, text="+", width=2,
                   command=lambda: self._nudge_threshold(0.001)).pack(side="left")

        self.listen_var = tk.BooleanVar(value=False)
        listen_row = ttk.Frame(frame)
        listen_row.pack(fill="x", pady=(0, 8))
        ttk.Checkbutton(
            listen_row, text="Listen (use headphones)", variable=self.listen_var,
            command=self._on_listen,
        ).pack(side="left")
        self.volume_var = tk.DoubleVar(value=1.0)
        ttk.Scale(
            listen_row, from_=0.0, to=5.0, variable=self.volume_var,
            command=self._on_volume,
        ).pack(side="left", fill="x", expand=True, padx=(10, 0))
        ttk.Label(frame, text="Live waveform (last 4 s)  |  red = detected ticks").pack(anchor="w")
        self.graph = tk.Canvas(
            frame, width=self.GRAPH_W, height=self.GRAPH_H,
            bg="#101418", highlightthickness=1, highlightbackground="#888",
        )
        self.graph.pack(pady=(0, 8))

        grid = ttk.Frame(frame)
        grid.pack(fill="x", pady=6)
        self.values = {}
        for row, name in enumerate(["Live ticks/sec", "Ticks", "Elapsed", "Beat rate", "Interval jitter"]):
            ttk.Label(grid, text=name + ":").grid(row=row, column=0, sticky="w")
            label = ttk.Label(grid, text="-", font=("Helvetica", 13, "bold"))
            label.grid(row=row, column=1, sticky="w", padx=12)
            self.values[name] = label

        buttons = ttk.Frame(frame)
        buttons.pack(fill="x", pady=10)
        self.start_btn = ttk.Button(buttons, text="Start", command=self.start)
        self.start_btn.pack(side="left", expand=True, fill="x", padx=2)
        self.stop_btn = ttk.Button(buttons, text="Stop", command=self.stop, state="disabled")
        self.stop_btn.pack(side="left", expand=True, fill="x", padx=2)
        self.save_btn = ttk.Button(buttons, text="Save CSV", command=self.save, state="disabled")
        self.save_btn.pack(side="left", expand=True, fill="x", padx=2)

        ttk.Label(side, text="Tick analysis", font=("Helvetica", 14, "bold")).pack(anchor="w", pady=(6, 6))
        self.metrics = {}
        mgrid = ttk.Frame(side)
        mgrid.pack(fill="x")
        for row, name in enumerate([
            "Beat rate", "Nearest standard", "Rate error", "Rate basis", "Beat error",
            "Interval mean", "Interval min / max", "Interval jitter",
            "Outliers dropped", "Tick volume (mean / peak)", "Quality",
        ]):
            ttk.Label(mgrid, text=name + ":").grid(row=row, column=0, sticky="w", pady=2)
            label = ttk.Label(mgrid, text="-", font=("Helvetica", 12, "bold"))
            label.grid(row=row, column=1, sticky="w", padx=10)
            self.metrics[name] = label

        ttk.Label(side, text="Tick intervals (ms) per tick  |  yellow = mean").pack(anchor="w", pady=(12, 0))
        self.interval_graph = tk.Canvas(
            side, width=self.IV_W, height=self.IV_H,
            bg="#101418", highlightthickness=1, highlightbackground="#888",
        )
        self.interval_graph.pack(pady=(0, 8))

        self.cal_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(
            side, text="Calibrate audio clock to system time", variable=self.cal_var,
            command=lambda: setattr(self.ticker, "calibrate", self.cal_var.get()),
        ).pack(anchor="w")
        self.cal_label = ttk.Label(side, text="Audio clock: measuring...")
        self.cal_label.pack(anchor="w")
    def _on_listen(self):
        try:
            self.ticker.set_listen(self.listen_var.get())
        except Exception as exc:
            self.listen_var.set(False)
            messagebox.showerror("Speaker error", str(exc))

    def _on_volume(self, _value):
        self.ticker.volume = self.volume_var.get()

    def _set_threshold(self, value):
        value = round(min(max(value, 0.001), 0.2), 3)
        self.ticker.threshold = value
        self.threshold_var.set(value)
        self.threshold_spin.set(f"{value:.3f}")

    def _nudge_threshold(self, delta):
        self._set_threshold(self.ticker.threshold + delta)

    def _on_spin(self):
        try:
            self._set_threshold(float(self.threshold_spin.get()))
        except ValueError:
            self.threshold_spin.set(f"{self.ticker.threshold:.3f}")

    def start(self):
        self.ticker.begin_counting()
        self.started_at = time.time()
        self.status.config(text="Recording ticks...")
        self.start_btn.config(state="disabled")
        self.stop_btn.config(state="normal")
        self.save_btn.config(state="disabled")

    def stop(self):
        self.ticker.end_counting()
        self.started_at = None
        self.start_btn.config(state="normal" if self.ticker.running else "disabled")
        self.stop_btn.config(state="disabled")
        if self.ticker.running:
            self.status.config(text="Live - press Start to count ticks")
        if self.ticker.tick_times:
            self.save_btn.config(state="normal")

    def save(self):
        default = f"watch_ticks_{datetime.now():%Y%m%d_%H%M%S}.csv"
        filename = filedialog.asksaveasfilename(
            defaultextension=".csv", initialfile=default,
            filetypes=[("CSV files", "*.csv")],
        )
        if filename:
            self.ticker.save_data(filename)
            self.status.config(text=f"Saved {filename}")

    def _poll(self):
        error = None
        try:
            while True:
                event = self.ticker.events.get_nowait()
                if event[0] == "level":
                    _, t, rms = event
                    self.history.append((t, rms))
                    self.level.config(value=rms)
                elif event[0] == "error":
                    error = event[1]
        except queue.Empty:
            pass
        if error:
            self.stop()
            self.status.config(text="Audio error")
            messagebox.showerror("Audio error", error)
        self._draw_graph()
        self._refresh_stats()
        self.after(50, self._poll)

    def _draw_graph(self):
        g = self.graph
        g.delete("all")
        w, h = self.GRAPH_W, self.GRAPH_H
        if not self.history:
            return
        t_end = self.history[-1][0]
        t_start = t_end - self.GRAPH_SECONDS
        while self.history and self.history[0][0] < t_start:
            self.history.popleft()

        def x_of(t):
            return (t - t_start) / self.GRAPH_SECONDS * w

        def y_of(v):
            return h - min(v / self.GRAPH_MAX, 1.0) * (h - 4) - 2

        ty = y_of(self.ticker.threshold)
        g.create_line(0, ty, w, ty, fill="#e6b800", dash=(4, 3))
        for t in list(self.ticker.live_ticks):
            if t >= t_start:
                g.create_line(x_of(t), 0, x_of(t), h, fill="#ff4d4d")
        points = []
        for t, v in self.history:
            points.extend((x_of(t), y_of(v)))
        if len(points) >= 4:
            g.create_line(*points, fill="#4dd2ff")

    def _show_clock(self):
        report = self.ticker.clock_report()
        if report is None:
            return
        rate, ppm, span = report
        need = self.ticker.CAL_MIN_SECONDS
        if not self.ticker.calibrate:
            state = "correction off"
        elif span < need:
            state = f"calibrating {span:.0f}/{need:.0f} s"
        else:
            state = f"applied ({-ppm * 0.0864:+.1f} s/day correction)"
        self.cal_label.config(text=f"Audio clock: {rate:,.1f} Hz ({ppm:+.0f} ppm) - {state}")

    def _show_live_rate(self):
        window = 5.0
        now = self.ticker.t_now
        if now < window:
            self.values["Live ticks/sec"].config(text="measuring...")
            return
        n = sum(1 for t in list(self.ticker.live_ticks) if t >= now - window)
        self.values["Live ticks/sec"].config(text=f"{n / window:.1f}  (target 6.0 = 21,600 bph)")

    def _refresh_stats(self):
        self._show_live_rate()
        self._show_clock()
        self.values["Ticks"].config(text=str(len(self.ticker.tick_times)))
        if self.ticker.counting and self.started_at:
            self.values["Elapsed"].config(text=f"{time.time() - self.started_at:.1f} s")
        stats = self.ticker.stats()
        self._show_analysis(stats)

    def _show_analysis(self, stats):
        m = self.metrics
        if not stats:
            for label in m.values():
                label.config(text="-")
            self.values["Beat rate"].config(text="-")
            self.values["Interval jitter"].config(text="-")
            self.interval_graph.delete("all")
            return
        bph = stats["bph"]
        self.values["Beat rate"].config(text=f"{bph:,.0f} bph ({bph / 3600:.2f} Hz)")
        self.values["Interval jitter"].config(text=f"{stats['jitter']:.2f} ms")
        m["Beat rate"].config(text=f"{bph:,.1f} bph")
        m["Nearest standard"].config(text=f"{stats['nominal']:,} bph")
        m["Rate error"].config(text=f"{stats['rate_error']:+.1f} s/day")
        m["Rate basis"].config(text=stats["rate_basis"])
        m["Beat error"].config(text=f"{stats['beat_error']:.2f} ms")
        m["Interval mean"].config(text=f"{stats['mean']:.2f} ms")
        m["Interval min / max"].config(text=f"{stats['min']:.1f} / {stats['max']:.1f} ms")
        m["Interval jitter"].config(text=f"{stats['jitter']:.2f} ms")
        m["Outliers dropped"].config(text=str(stats["outliers"]))
        m["Tick volume (mean / peak)"].config(
            text=f"{stats['volume_mean']:.3f} / {stats['volume_peak']:.3f}")
        m["Quality"].config(text=self._quality(stats))
        self._draw_intervals(stats)

    @staticmethod
    def _quality(stats):
        rate, beat, jitter = abs(stats["rate_error"]), stats["beat_error"], stats["jitter"]
        if rate <= 10 and beat <= 1.0 and jitter <= 2.0:
            return "Excellent"
        if rate <= 30 and beat <= 2.5 and jitter <= 5.0:
            return "Good"
        return "Needs service / noisy"

    def _draw_intervals(self, stats):
        g = self.interval_graph
        g.delete("all")
        w, h = self.IV_W, self.IV_H
        ivs = stats["intervals"][-80:] * 1000.0
        mean = stats["mean"]
        lo, hi = mean * 0.5, mean * 1.5

        def y_of(v):
            return h - (min(max(v, lo), hi) - lo) / (hi - lo) * (h - 4) - 2

        g.create_line(0, y_of(mean), w, y_of(mean), fill="#e6b800", dash=(4, 3))
        step = w / max(len(ivs) - 1, 1)
        points = []
        for i, v in enumerate(ivs):
            x, y = i * step, y_of(v)
            points.extend((x, y))
            color = "#4dd2ff" if lo < v < hi else "#ff4d4d"
            g.create_oval(x - 2, y - 2, x + 2, y + 2, fill=color, outline=color)
        if len(points) >= 4:
            g.create_line(*points, fill="#2a7f9e")

    def _on_close(self):
        self.ticker.stop()
        self.destroy()


if __name__ == "__main__":
    WatchTimerApp().mainloop()
