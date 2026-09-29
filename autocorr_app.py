#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Наглядное построение автокорреляционной функции стационарного процесса.

Верхний график  : исходный процесс x(t) (генерируется, выбирается из меню или рисуется мышью)
Второй график   : тот же процесс, сдвинутый на лаг τ
Третий график   : оба процесса, наложенные друг на друга; область совпадения закрашена
                  (зелёный - совпадение одного знака, красный - разного знака у центрированного сигнала)
Нижний график   : столбики R(τ) = среднее x(t)*x(t-τ) по области перекрытия; растут с увеличением лага

Зависимости:  pip install numpy matplotlib   (tkinter входит в стандартный Python)
Запуск:       python autocorr_app.py
"""
import tkinter as tk
from tkinter import ttk

import numpy as np
import matplotlib

matplotlib.use("TkAgg")
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk

PRESETS = [
    "Импульсы: экспоненциальные интервалы (Пуассон)",
    "Импульсы: гамма-интервалы (k=4)",
    "Импульсы: равномерные интервалы U(m−s, m+s)",
    "Импульсы: нормальные интервалы N(m, s) (почти периодические)",
    "Телеграфный сигнал ±1",
    "Синус + шум",
    "Белый шум",
    "Нарисовать самому",
]
GAP_KIND = {
    PRESETS[0]: "exp", PRESETS[1]: "gamma", PRESETS[2]: "uniform", PRESETS[3]: "periodic",
}


def draw_gap(kind, mean, spread, rng):
    if kind == "exp":
        return rng.exponential(mean)
    if kind == "gamma":
        return rng.gamma(4.0, mean / 4.0)
    if kind == "uniform":
        return max(0.0, rng.uniform(mean - spread, mean + spread))
    return max(0.0, rng.normal(mean, spread))


def make_pulses(N, kind, mean, width, spread, rng):
    """Прямоугольные импульсы фиксированной ширины; интервалы между ними случайны."""
    x = np.zeros(N)
    t = -rng.uniform(0, mean + width)  # случайная фаза старта (приближение к стационарности)
    while t < N:
        t += draw_gap(kind, mean, spread, rng)
        s = t
        t += width
        a, b = max(0, int(round(s))), min(N, int(round(t)))
        if b > a:
            x[a:b] = 1.0
    return x


def make_telegraph(N, mean, rng):
    x = np.zeros(N)
    t, v = 0.0, rng.choice([-1.0, 1.0])
    while t < N:
        d = max(1.0, rng.exponential(mean))
        x[int(t):int(t + d)] = v
        t += d
        v = -v
    return x


def compute_R(sig, maxlag):
    """R[k] = (1/(N-k)) * sum x[t]*x[t+k]  (через FFT)."""
    N = len(sig)
    n = 1 << (2 * N - 1).bit_length()
    F = np.fft.rfft(sig, n)
    c = np.fft.irfft(F * np.conj(F), n)[: maxlag + 1]
    return c / (N - np.arange(maxlag + 1))


def geti(var, default, lo, hi):
    try:
        return int(min(max(int(var.get()), lo), hi))
    except (tk.TclError, ValueError):
        return default


def getf(var, default, lo, hi):
    try:
        return float(min(max(float(var.get()), lo), hi))
    except (tk.TclError, ValueError):
        return default


class App:
    def __init__(self, root):
        self.root = root
        root.title("Автокорреляционная функция — наглядное построение")
        self.rng = np.random.default_rng()
        self.raw = np.zeros(10)
        self.sig = self.raw
        self.R = np.zeros(1)
        self.maxlag = 1
        self.lag = 0
        self.playing = False
        self.drawing = False
        self.last = None
        self._sync = False
        self.dyn = []      # артисты, перерисовываемые при каждом лаге
        self.static = []   # артисты верхнего графика
        self.build_ui()
        self.generate()

    # ------------------------------------------------------------------ UI
    def build_ui(self):
        self.preset = tk.StringVar(value=PRESETS[0])
        self.n_var = tk.IntVar(value=4000)
        self.gap_var = tk.DoubleVar(value=40.0)
        self.w_var = tk.DoubleVar(value=10.0)
        self.spread_var = tk.DoubleVar(value=8.0)
        self.center = tk.BooleanVar(value=False)
        self.norm = tk.BooleanVar(value=False)
        self.draw_var = tk.BooleanVar(value=False)
        self.snap = tk.BooleanVar(value=True)
        self.step_var = tk.IntVar(value=5)
        self.speed = tk.IntVar(value=20)

        top = ttk.Frame(self.root, padding=4)
        top.pack(fill="x")
        ttk.Label(top, text="Процесс:").pack(side="left")
        cb = ttk.Combobox(top, textvariable=self.preset, values=PRESETS, width=44, state="readonly")
        cb.pack(side="left", padx=4)
        cb.bind("<<ComboboxSelected>>", lambda e: self.generate())
        for text, var, w in (("N отсчётов", self.n_var, 7), ("средний интервал", self.gap_var, 6),
                             ("ширина импульса", self.w_var, 6),
                             ("разброс s", self.spread_var, 6)):
            ttk.Label(top, text=text).pack(side="left", padx=(8, 2))
            ttk.Entry(top, textvariable=var, width=w).pack(side="left")
        ttk.Button(top, text="Новая реализация", command=self.generate).pack(side="left", padx=8)
        ttk.Button(top, text="Очистить", command=self.clear).pack(side="left")

        row2 = ttk.Frame(self.root, padding=4)
        row2.pack(fill="x")
        ttk.Checkbutton(row2, text="Рисовать мышью на верхнем графике", variable=self.draw_var).pack(side="left")
        ttk.Checkbutton(row2, text="уровни 0/1", variable=self.snap).pack(side="left", padx=6)
        ttk.Checkbutton(row2, text="вычесть среднее", variable=self.center, command=self.recompute).pack(side="left", padx=6)
        ttk.Checkbutton(row2, text="нормировать на R(0)", variable=self.norm, command=self.recompute).pack(side="left", padx=6)
        ttk.Button(row2, text="Показать всё", command=self.show_all).pack(side="left", padx=6)
        ttk.Label(row2, text="окно:").pack(side="left")
        self.pos_scale = ttk.Scale(row2, from_=0, to=1, length=220, command=self.on_pos)
        self.pos_scale.pack(side="left", padx=4)
        ttk.Label(row2, text="(колесо мыши — масштаб, лупа/рука на панели — детали)").pack(side="left")

        row3 = ttk.Frame(self.root, padding=4)
        row3.pack(fill="x")
        self.play_btn = ttk.Button(row3, text="▶ Пуск", command=self.toggle)
        self.play_btn.pack(side="left")
        ttk.Button(row3, text="⟲ Сброс", command=self.reset_lag).pack(side="left", padx=4)
        ttk.Label(row3, text="лаг τ:").pack(side="left", padx=(8, 2))
        self.lag_scale = ttk.Scale(row3, from_=0, to=1, length=420, command=self.on_lag)
        self.lag_scale.pack(side="left")
        ttk.Label(row3, text="шаг лага").pack(side="left", padx=(8, 2))
        ttk.Spinbox(row3, from_=1, to=500, textvariable=self.step_var, width=5).pack(side="left")
        ttk.Label(row3, text="скорость").pack(side="left", padx=(8, 2))
        ttk.Scale(row3, from_=1, to=60, variable=self.speed, length=100).pack(side="left")

        self.fig = Figure(figsize=(11, 8.5), constrained_layout=True)
        axs = self.fig.subplots(4, 1, gridspec_kw={"height_ratios": [1, 1, 1.2, 1.4]})
        self.ax1, self.ax2, self.ax3, self.ax4 = axs
        self.ax2.sharex(self.ax1)
        self.ax3.sharex(self.ax1)
        self.ax1.set_title("x(t) — исходный процесс", fontsize=10, loc="left")
        self.ax2.set_title("тот же процесс, сдвинутый на τ", fontsize=10, loc="left")
        self.ax3.set_xlabel("t")
        self.ax4.set_xlabel("лаг τ")
        self.ax4.set_ylabel("R(τ)")
        for a in (self.ax1, self.ax2):
            a.tick_params(labelbottom=False)
        for a in axs:
            a.grid(alpha=0.25)

        holder = ttk.Frame(self.root)
        holder.pack(fill="both", expand=True)
        self.canvas = FigureCanvasTkAgg(self.fig, master=holder)
        self.toolbar = NavigationToolbar2Tk(self.canvas, holder)
        self.toolbar.update()
        self.canvas.get_tk_widget().configure(width=640, height=420)
        self.canvas.get_tk_widget().pack(fill="both", expand=True)
        c = self.canvas
        c.mpl_connect("button_press_event", self.on_press)
        c.mpl_connect("motion_notify_event", self.on_motion)
        c.mpl_connect("button_release_event", self.on_release)
        c.mpl_connect("scroll_event", self.on_scroll)

    # ------------------------------------------------------------ генерация
    def generate(self):
        N = geti(self.n_var, 4000, 200, 50000)
        self.n_var.set(N)
        mean = getf(self.gap_var, 40.0, 1.0, 1e6)
        w = getf(self.w_var, 10.0, 1.0, 1e6)
        spread = getf(self.spread_var, 8.0, 0.0, 1e6)
        p = self.preset.get()
        rng = self.rng
        if p in GAP_KIND:
            x = make_pulses(N, GAP_KIND[p], mean, w, spread, rng)
        elif p == PRESETS[4]:
            x = make_telegraph(N, mean, rng)
        elif p == PRESETS[5]:
            t = np.arange(N)
            x = np.sin(2 * np.pi * t / (mean + w) + rng.uniform(0, 6.28)) + 0.5 * rng.standard_normal(N)
        elif p == PRESETS[6]:
            x = rng.standard_normal(N)
        else:  # рисовать самому
            x = np.zeros(N)
            self.draw_var.set(True)
        self.raw = x
        self.maxlag = max(1, N // 2)
        self.lag_scale.configure(to=self.maxlag)
        self.pos_scale.configure(to=N)
        self.lag = 0
        self.lag_scale.set(0)
        self.ax1.set_xlim(0, min(N, 1000))
        self.ax4.set_xlim(-1, self.maxlag * 1.02)
        self.pos_scale.set(0)
        self.recompute()

    def clear(self):
        self.raw = np.zeros(len(self.raw))
        self.preset.set(PRESETS[-1])
        self.draw_var.set(True)
        self.recompute()

    def recompute(self):
        raw = self.raw
        self.sig = raw - raw.mean() if self.center.get() else raw.copy()
        R = compute_R(self.sig, self.maxlag)
        if self.norm.get() and R[0] > 1e-12:
            R = R / R[0]
        self.R = R
        self.ax4.set_ylim(*self.span(np.concatenate([R, [0.0]]), zero_floor=True))
        self.ax2.set_ylim(*self.span(self.sig))
        self.ax3.set_ylim(*self.span(self.sig))
        self.ax1.set_ylim(*self.span(raw))
        self.refresh_static()
        self.update()

    @staticmethod
    def span(v, zero_floor=False):
        lo, hi = min(float(np.min(v)), 0.0), max(float(np.max(v)), 0.0)
        if hi - lo < 1e-9:
            hi = lo + 1.0
        pad = 0.12 * (hi - lo)
        return (0.0 if (zero_floor and lo >= 0) else lo - pad), hi + pad

    # ------------------------------------------------------------ отрисовка
    def refresh_static(self):
        for a in self.static:
            a.remove()
        N = len(self.raw)
        t = np.arange(N)
        self.static = list(self.ax1.step(t, self.raw, where="post", color="tab:blue", lw=1))
        self.static.append(self.ax1.fill_between(t, 0, self.raw, step="post", color="tab:blue", alpha=0.15))
        self.canvas.draw_idle()

    def update(self, now=False):
        for a in self.dyn:
            try:
                a.remove()
            except (ValueError, NotImplementedError):
                pass
        self.dyn = []
        sig, lag = self.sig, self.lag
        N = len(sig)
        t = np.arange(N)
        xs = np.zeros(N)
        if lag < N:
            xs[lag:] = sig[: N - lag]

        # 2: сдвинутая копия
        self.dyn += self.ax2.step(t, xs, where="post", color="tab:orange", lw=1)
        self.dyn.append(self.ax2.axvline(lag, ls="--", color="gray", lw=1))

        # 3: наложение и область совпадения
        self.dyn += self.ax3.step(t, sig, where="post", color="tab:blue", lw=0.9, alpha=0.8)
        self.dyn += self.ax3.step(t, xs, where="post", color="tab:orange", lw=0.9, alpha=0.8)
        same = (sig * xs) > 0
        m = np.where(same, np.sign(sig) * np.minimum(np.abs(sig), np.abs(xs)), 0.0)
        self.dyn.append(self.ax3.fill_between(t, 0, m, where=m > 0, step="post", color="tab:green", alpha=0.75))
        self.dyn.append(self.ax3.fill_between(t, 0, m, where=m < 0, step="post", color="tab:red", alpha=0.6))
        Rk = self.R[min(lag, len(self.R) - 1)]
        self.ax3.set_title(f"наложение при τ = {lag}: закрашено — совпадение   (R = {Rk:.4f})",
                           fontsize=10, loc="left")

        # 4: столбики R(τ) для всех пройденных лагов
        step = geti(self.step_var, 5, 1, 500)
        lags = np.arange(0, lag + 1, step)
        if lags[-1] != lag:
            lags = np.append(lags, lag)
        width = step * 0.85
        self.dyn.append(self.ax4.bar(lags[:-1], self.R[lags[:-1]], width=width, color="tab:blue"))
        self.dyn.append(self.ax4.bar([lag], [self.R[lag]], width=width, color="tab:orange"))
        self.ax4.set_title("R(τ) — среднее произведение x(t)·x(t−τ) по области перекрытия",
                           fontsize=10, loc="left")
        (self.canvas.draw if now else self.canvas.draw_idle)()

    # ------------------------------------------------------------ анимация
    def on_lag(self, v):
        if self._sync:  # значение выставлено программой — перерисовку делает set_lag
            return
        self.lag = min(self.maxlag, max(0, int(float(v))))
        self.update()

    def set_lag(self, n, now=False):
        """Программная установка лага: не зависит от того, вызывает ли Tk command у Scale."""
        self.lag = min(self.maxlag, max(0, int(n)))
        self._sync = True
        try:
            self.lag_scale.set(self.lag)
        finally:
            self._sync = False
        self.update(now)

    def reset_lag(self):
        self.playing = False
        self.play_btn.configure(text="▶ Пуск")
        self.set_lag(0)

    def toggle(self):
        self.playing = not self.playing
        self.play_btn.configure(text="⏸ Пауза" if self.playing else "▶ Пуск")
        if self.playing:
            if self.lag >= self.maxlag:
                self.set_lag(0)
            self.tick()

    def tick(self):
        if not self.playing:
            return
        if self.lag >= self.maxlag:
            self.playing = False
            self.play_btn.configure(text="▶ Пуск")
            return
        self.set_lag(self.lag + geti(self.step_var, 5, 1, 500), now=True)
        # следующий кадр планируется ПОСЛЕ отрисовки, чтобы кнопки успевали реагировать
        self.root.after(max(5, int(1000 / getf(self.speed, 20, 1, 60))), self.tick)

    # ------------------------------------------------------- масштаб / вид
    def show_all(self):
        self.ax1.set_xlim(0, len(self.raw))
        self.ax4.set_xlim(-1, self.maxlag * 1.02)
        self.canvas.draw_idle()

    def on_pos(self, v):
        x0, x1 = self.ax1.get_xlim()
        s = float(v)
        self.ax1.set_xlim(s, s + (x1 - x0))
        self.canvas.draw_idle()

    def on_scroll(self, e):
        if e.inaxes is None or e.xdata is None:
            return
        f = 0.8 if e.button == "up" else 1.25
        full = (0, self.maxlag) if e.inaxes is self.ax4 else (0, len(self.raw))
        x0, x1 = e.inaxes.get_xlim()
        n0, n1 = e.xdata - (e.xdata - x0) * f, e.xdata + (x1 - e.xdata) * f
        if n1 - n0 < 5:
            return
        if n1 - n0 > full[1] - full[0]:
            n0, n1 = full
        e.inaxes.set_xlim(n0, n1)
        self.canvas.draw_idle()

    # ----------------------------------------------------- рисование мышью
    def mode_none(self):
        m = self.toolbar.mode
        return getattr(m, "value", m) == ""

    def on_press(self, e):
        if (self.draw_var.get() and e.inaxes is self.ax1 and e.button == 1 and self.mode_none()):
            self.drawing, self.last = True, None
            self.paint(e)

    def on_motion(self, e):
        if self.drawing and e.inaxes is self.ax1:
            self.paint(e)

    def on_release(self, e):
        if self.drawing:
            self.drawing = False
            self.recompute()

    def paint(self, e):
        if e.xdata is None or e.ydata is None:
            return
        N = len(self.raw)
        i = min(max(int(round(e.xdata)), 0), N - 1)
        y = float(e.ydata)
        if self.snap.get():
            y = 1.0 if y > 0.5 else 0.0
        if self.last is None:
            idx, vals = np.array([i]), np.array([y])
        else:
            i0, y0 = self.last
            n = abs(i - i0) + 1
            idx = np.linspace(i0, i, n).round().astype(int)
            vals = np.full(n, y) if self.snap.get() else np.linspace(y0, y, n)
        self.raw[idx] = vals
        self.last = (i, y)
        self.refresh_static()


def main():
    root = tk.Tk()
    sw, sh = root.winfo_screenwidth(), root.winfo_screenheight()
    root.geometry(f"{min(1200, sw - 60)}x{min(950, sh - 120)}+20+20")
    root.minsize(900, 600)
    App(root)
    root.mainloop()


if __name__ == "__main__":
    main()
