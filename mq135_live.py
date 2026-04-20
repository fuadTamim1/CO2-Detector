"""
╔══════════════════════════════════════════════════════════════════════╗
║       MQ-135  LIVE  Simulation Dashboard  —  CO₂ Detection          ║
║       Real-time mock data  |  Interactive controls  |  Multi-panel   ║
╚══════════════════════════════════════════════════════════════════════╝

Controls (keyboard):
    SPACE     — Pause / Resume simulation
    R         — Reset and restart
    +  /  =   — Speed UP  (faster data stream)
    -         — Speed DOWN
    1 … 5     — Focus single panel  (full-screen that chart)
    0         — Back to 5-panel grid view
    M         — Cycle mock mode  (realistic → sine → step → spike)
    N         — Toggle Gaussian noise ON / OFF
    S         — Save current frame as PNG
    Q / Esc   — Quit

Mouse:
    Hover any panel header button  →  click to isolate that panel
"""

import sys
import time
import numpy as np
import matplotlib
matplotlib.use("TkAgg")          # interactive backend (falls back gracefully)
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.widgets import Button, Slider
from matplotlib.animation import FuncAnimation
from dataclasses import dataclass, field
from pathlib import Path
from scipy.stats import norm
import warnings
import pygame
warnings.filterwarnings("ignore")

OUTPUT_PNG = Path(__file__).parent / "mq135_live_snapshot.png"
ALERT_FILE = Path(__file__).parent / "alert.mp3"

# ══════════════════════════════════════════════════════════════════════
# CONFIG
# ══════════════════════════════════════════════════════════════════════

@dataclass
class MQ135Config:
    a:    float = 110.47
    b:    float = -2.862
    Ro:   float = 36_526
    RL:   float = 10_000
    VCC:  float = 5.0
    ADC_BITS:  int   = 12
    ADC_VREF:  float = 3.3
    ADC_MAX:   int   = 4095
    ALARM_PPM:           float = 1000.0
    ALARM_CONFIRM_COUNT: int   = 3
    NOISE_STD: float = 0.005

# ══════════════════════════════════════════════════════════════════════
# SIGNAL CHAIN
# ══════════════════════════════════════════════════════════════════════

class SignalChain:
    def __init__(self, cfg: MQ135Config):
        self.cfg = cfg

    def ppm_to_voltage(self, ppm: float) -> float:
        rs_ro   = np.clip((ppm / self.cfg.a) ** (1.0 / self.cfg.b), 0.1, 10.0)
        Rs      = rs_ro * self.cfg.Ro
        voltage = self.cfg.VCC * self.cfg.RL / (Rs + self.cfg.RL)
        return float(np.clip(voltage, 0.0, self.cfg.VCC))

    def add_noise(self, voltage: float, enabled: bool = True) -> float:
        if not enabled:
            return voltage
        noise = np.random.normal(0.0, self.cfg.NOISE_STD)
        return float(np.clip(voltage + noise, 0.0, self.cfg.ADC_VREF))

    def voltage_to_adc(self, voltage: float) -> int:
        return int(np.clip(
            round(np.clip(voltage, 0, self.cfg.ADC_VREF)
                  / self.cfg.ADC_VREF * self.cfg.ADC_MAX),
            0, self.cfg.ADC_MAX))

    def adc_to_ppm(self, adc: int) -> float:
        v   = adc / self.cfg.ADC_MAX * self.cfg.ADC_VREF
        v   = max(v, 0.01)
        Rs  = np.clip(self.cfg.RL * (self.cfg.VCC / v - 1.0), 100, 1e6)
        return float(np.clip(self.cfg.a * ((Rs / self.cfg.Ro) ** self.cfg.b), 0, 10000))

# ══════════════════════════════════════════════════════════════════════
# MOCK DATA GENERATORS  (infinite iterators — yield one sample at a time)
# ══════════════════════════════════════════════════════════════════════

def gen_realistic(dt: float = 1.0):
    """Piecewise office-room scenario, loops every 300 s."""
    rng = np.random.default_rng()
    t   = 0.0
    while True:
        tc = t % 300          # position inside one 300-s cycle
        if   tc < 60:   base = 400  + 50  * (tc / 60)
        elif tc < 120:  base = 450  + 600 * ((tc - 60)  / 60)
        elif tc < 150:  base = 1050 + 200 * ((tc - 120) / 30)
        elif tc < 180:  base = 1250 - 100 * ((tc - 150) / 30)
        elif tc < 240:  base = 1150 - 700 * ((tc - 180) / 60)
        else:           base = 450  - 50  * ((tc - 240) / 60)
        ppm = base + rng.normal(0, 15) + 20 * np.sin(2 * np.pi * tc / 120)
        yield t, float(np.clip(ppm, 300, 5000))
        t += dt

def gen_sine(dt: float = 1.0):
    """Sine wave centred at 750 PPM ± 400 PPM, period 100 s."""
    rng = np.random.default_rng(seed=7)
    t   = 0.0
    while True:
        ppm = 750 + 400 * np.sin(2 * np.pi * t / 100) + rng.normal(0, 25)
        yield t, float(np.clip(ppm, 300, 2500))
        t += dt

def gen_step(dt: float = 1.0):
    """Repeating step: 500 → 1200 → 600 PPM."""
    t = 0.0
    while True:
        tc  = t % 300
        ppm = 500.0 if tc < 90 else (1200.0 if tc < 210 else 600.0)
        yield t, ppm
        t += dt

def gen_spike(dt: float = 1.0):
    """Baseline 450 PPM with random short spikes and one long spike."""
    rng = np.random.default_rng(seed=3)
    t   = 0.0
    while True:
        tc  = t % 300
        ppm = 450.0 + rng.normal(0, 10)
        # short 2-s spikes at fixed positions
        for spike_t in [40, 80, 130, 170]:
            if spike_t <= tc < spike_t + 2:
                ppm = 1100.0
        # sustained 10-s spike
        if 220 <= tc < 230:
            ppm = 1150.0
        yield t, float(np.clip(ppm, 300, 2000))
        t += dt

GENERATORS = {
    "realistic": gen_realistic,
    "sine":      gen_sine,
    "step":      gen_step,
    "spike":     gen_spike,
}
MOCK_CYCLE = list(GENERATORS.keys())

# ══════════════════════════════════════════════════════════════════════
# ROLLING BUFFER
# ══════════════════════════════════════════════════════════════════════

class RingBuffer:
    def __init__(self, maxlen: int):
        self.maxlen = maxlen
        self._t    = np.full(maxlen, np.nan)
        self._data = {k: np.full(maxlen, np.nan)
                      for k in ("ppm_true", "ppm_meas", "voltage_clean",
                                "voltage_noisy", "adc", "error", "alarm", "rsro")}
        self._idx  = 0
        self._full = False

    def push(self, t, **kwargs):
        i = self._idx % self.maxlen
        self._t[i] = t
        for k, v in kwargs.items():
            self._data[k][i] = v
        self._idx += 1
        if self._idx >= self.maxlen:
            self._full = True

    def get(self):
        if not self._full:
            n = self._idx
            t = self._t[:n]
            d = {k: v[:n] for k, v in self._data.items()}
        else:
            # rotate so oldest is first
            i = self._idx % self.maxlen
            t = np.roll(self._t, -i)
            d = {k: np.roll(v, -i) for k, v in self._data.items()}
        return t, d

# ══════════════════════════════════════════════════════════════════════
# ALARM STATE
# ══════════════════════════════════════════════════════════════════════

class AlarmState:
    def __init__(self, cfg: MQ135Config):
        self.cfg     = cfg
        self.counter = 0
        self.active  = False

    def update(self, ppm: float) -> bool:
        if ppm >= self.cfg.ALARM_PPM:
            self.counter += 1
            if self.counter >= self.cfg.ALARM_CONFIRM_COUNT:
                self.active = True
        else:
            self.counter = 0
            self.active  = False
        return self.active

    def reset(self):
        self.counter = 0
        self.active  = False

# ══════════════════════════════════════════════════════════════════════
# COLOUR SCHEME
# ══════════════════════════════════════════════════════════════════════

C = dict(
    bg     = "#0a0e17",
    panel  = "#111827",
    border = "#1e293b",
    grid   = "#1e293b",
    text   = "#e2e8f0",
    dim    = "#64748b",
    cyan   = "#38bdf8",
    green  = "#4ade80",
    red    = "#f87171",
    yellow = "#fbbf24",
    orange = "#fb923c",
    purple = "#a78bfa",
    alarm_bg = "#1f0a0a",
)

# ══════════════════════════════════════════════════════════════════════
# LIVE DASHBOARD
# ══════════════════════════════════════════════════════════════════════

class LiveDashboard:
    WINDOW   = 120    # seconds shown in scrolling window
    BUF_SIZE = 3000   # total ring-buffer capacity

    PANEL_TITLES = [
        "[1]  CO₂ Concentration  (PPM)",
        "[2]  Resistance Ratio  Rs/Ro",
        "[3]  Analog Voltage  (V)",
        "[4]  ADC Output  (12-bit)",
        "[5]  Measurement Error  (PPM)",
    ]

    def __init__(self):
        self.cfg      = MQ135Config()
        self.chain    = SignalChain(self.cfg)
        self.alarm    = AlarmState(self.cfg)
        self.buf      = RingBuffer(self.BUF_SIZE)

        self.mock_idx = 0                          # index into MOCK_CYCLE
        self.gen      = self._make_gen()
        self.noise_on = True
        self.paused   = False
        self.focus    = 0                          # 0 = grid, 1-5 = single panel
        self.speed    = 1                          # samples per animation frame
        self.alarm_events: list[dict] = []
        self.total_samples = 0

        # Audio setup
        pygame.mixer.init()
        try:
            self.alert_sound = pygame.mixer.Sound(str(ALERT_FILE))
            self.audio_enabled = True
        except Exception as e:
            print(f"Warning: Could not load alert sound: {e}")
            self.audio_enabled = False
        self.alert_playing = False

        self._build_figure()
        self._connect_events()

    # ── generator factory ────────────────────────────────────────
    def _make_gen(self):
        name = MOCK_CYCLE[self.mock_idx]
        return GENERATORS[name]()

    def _reset(self):
        self.gen   = self._make_gen()
        self.buf   = RingBuffer(self.BUF_SIZE)
        self.alarm.reset()
        self.alarm_events.clear()
        self.total_samples = 0
        if self.audio_enabled and self.alert_playing:
            self.alert_sound.stop()
            self.alert_playing = False

    # ── figure & axes ────────────────────────────────────────────
    def _build_figure(self):
        plt.rcParams.update({
            "figure.facecolor":  C["bg"],
            "text.color":        C["text"],
            "axes.facecolor":    C["panel"],
            "axes.edgecolor":    C["border"],
            "axes.labelcolor":   C["dim"],
            "xtick.color":       C["dim"],
            "ytick.color":       C["dim"],
            "grid.color":        C["grid"],
            "grid.linewidth":    0.6,
            "font.family":       "monospace",
        })

        self.fig = plt.figure(figsize=(17, 10))
        self.fig.patch.set_facecolor(C["bg"])
        self.fig.canvas.manager.set_window_title("MQ-135 Live Simulation Dashboard")

        self._layout_grid()

    def _layout_grid(self):
        """5-panel grid layout + control strip."""
        self.fig.clear()

        # outer layout: top header | main | bottom controls
        outer = gridspec.GridSpec(
            3, 1, figure=self.fig,
            height_ratios=[0.045, 1, 0.13],
            hspace=0.04, left=0.05, right=0.98,
            top=0.97, bottom=0.02,
        )

        # ── header ───────────────────────────────────────────────
        self.ax_hdr = self.fig.add_subplot(outer[0])
        self.ax_hdr.axis("off")
        self._draw_header()

        # ── 5-panel grid ─────────────────────────────────────────
        inner = gridspec.GridSpecFromSubplotSpec(
            2, 3, subplot_spec=outer[1],
            hspace=0.45, wspace=0.32,
        )
        # top row: [1] CO2 spans 2 cols, [2] Rs/Ro
        # bottom row: [3] Voltage, [4] ADC, [5] Error
        self.axes = [
            self.fig.add_subplot(inner[0, :2]),   # [0] CO2  — wide
            self.fig.add_subplot(inner[0, 2]),     # [1] Rs/Ro
            self.fig.add_subplot(inner[1, 0]),     # [2] Voltage
            self.fig.add_subplot(inner[1, 1]),     # [3] ADC
            self.fig.add_subplot(inner[1, 2]),     # [4] Error
        ]

        for ax, title in zip(self.axes, self.PANEL_TITLES):
            self._style_ax(ax, title)

        # ── initialise line objects ───────────────────────────────
        # Panel 0 — CO2
        self.ln_true,  = self.axes[0].plot([], [], color=C["cyan"],  lw=1.5, label="True PPM", zorder=3)
        self.ln_meas,  = self.axes[0].plot([], [], color=C["green"], lw=0.9, alpha=0.8, label="Measured PPM", zorder=2)
        self.alarm_span= self.axes[0].axhspan(self.cfg.ALARM_PPM, 9999, alpha=0.07, color=C["red"], zorder=0)
        self.threshold = self.axes[0].axhline(self.cfg.ALARM_PPM, color=C["red"], lw=1.1, ls="--", alpha=0.8,
                                              label=f"Alarm  ({self.cfg.ALARM_PPM:.0f} PPM)")
        self.fill_alarm= self.axes[0].fill_between([], [], [], alpha=0.2, color=C["red"])
        self.axes[0].legend(loc="upper left", fontsize=7, facecolor=C["panel"],
                            labelcolor=C["text"], framealpha=0.9, edgecolor=C["border"])
        self.axes[0].set_ylim(200, 1600)

        # live value annotation
        self.txt_ppm = self.axes[0].text(
            0.99, 0.92, "— PPM", transform=self.axes[0].transAxes,
            ha="right", va="top", fontsize=11, fontweight="bold",
            color=C["cyan"], fontfamily="monospace")
        self.txt_alarm = self.axes[0].text(
            0.99, 0.78, "", transform=self.axes[0].transAxes,
            ha="right", va="top", fontsize=9, fontweight="bold",
            color=C["red"], fontfamily="monospace")

        # Panel 1 — Rs/Ro
        self.ln_rsro,  = self.axes[1].plot([], [], color=C["orange"], lw=1.3)
        self.axes[1].set_ylim(0.3, 0.9)

        # Panel 2 — Voltage
        self.ln_vclean, = self.axes[2].plot([], [], color=C["cyan"],   lw=1.3, label="Clean")
        self.ln_vnoisy, = self.axes[2].plot([], [], color=C["yellow"], lw=0.6, alpha=0.75, label="Noisy")
        self.axes[2].axhline(self.cfg.ADC_VREF, color=C["red"], lw=0.7, ls=":", alpha=0.7,
                             label=f"Vref={self.cfg.ADC_VREF}V")
        self.axes[2].legend(fontsize=6, facecolor=C["panel"], labelcolor=C["text"],
                            framealpha=0.9, edgecolor=C["border"])
        self.axes[2].set_ylim(0.5, 3.5)

        # Panel 3 — ADC
        self.ln_adc,   = self.axes[3].step([], [], color=C["purple"], lw=0.9, where="mid")
        self.axes[3].set_ylim(0, 4400)
        self.axes[3].axhline(self.cfg.ADC_MAX, color=C["red"], lw=0.6, ls=":", alpha=0.5)

        # Panel 4 — Error
        self.ln_err,   = self.axes[4].plot([], [], color=C["text"], lw=0.7, alpha=0.6)
        self.axes[4].axhline(0, color=C["dim"], lw=0.8)
        self.txt_mae   = self.axes[4].text(
            0.02, 0.95, "MAE: —", transform=self.axes[4].transAxes,
            color=C["yellow"], fontsize=7, va="top", fontfamily="monospace")

        # ── control strip ────────────────────────────────────────
        ctrl = gridspec.GridSpecFromSubplotSpec(
            1, 8, subplot_spec=outer[2],
            wspace=0.25,
        )

        btn_cfg = dict(color=C["border"], hovercolor="#2d3748")

        ax_pause = self.fig.add_subplot(ctrl[0, 0])
        ax_reset = self.fig.add_subplot(ctrl[0, 1])
        ax_mode  = self.fig.add_subplot(ctrl[0, 2])
        ax_noise = self.fig.add_subplot(ctrl[0, 3])
        ax_save  = self.fig.add_subplot(ctrl[0, 4])
        ax_speed = self.fig.add_subplot(ctrl[0, 5:7])
        ax_focus = self.fig.add_subplot(ctrl[0, 7])

        self.btn_pause = Button(ax_pause, "⏸  Pause",  **btn_cfg)
        self.btn_reset = Button(ax_reset, "↺  Reset",  **btn_cfg)
        self.btn_mode  = Button(ax_mode,  "◈  Mode",   **btn_cfg)
        self.btn_noise = Button(ax_noise, "〜  Noise ✓", **btn_cfg)
        self.btn_save  = Button(ax_save,  "💾  Save",  **btn_cfg)
        self.btn_focus = Button(ax_focus, "⊞  Grid",   **btn_cfg)

        for btn in (self.btn_pause, self.btn_reset, self.btn_mode,
                    self.btn_noise, self.btn_save, self.btn_focus):
            btn.label.set_fontfamily("monospace")
            btn.label.set_fontsize(8)
            btn.label.set_color(C["text"])

        self.sld_speed = Slider(
            ax_speed, "Speed", 1, 10, valinit=self.speed, valstep=1,
            color=C["cyan"],
        )
        ax_speed.set_facecolor(C["panel"])
        self.sld_speed.label.set_color(C["text"])
        self.sld_speed.label.set_fontfamily("monospace")
        self.sld_speed.label.set_fontsize(8)
        self.sld_speed.valtext.set_color(C["yellow"])
        self.sld_speed.valtext.set_fontfamily("monospace")

        # ── status bar ───────────────────────────────────────────
        self.ax_status = self.axes[0]   # borrow axis for status text
        self.txt_status = self.fig.text(
            0.5, 0.004, self._status_str(),
            ha="center", va="bottom", fontsize=7,
            color=C["dim"], fontfamily="monospace"
        )

        self._connect_buttons()

    def _layout_focus(self, panel_idx: int):
        """Single-panel focus layout."""
        self.fig.clear()

        outer = gridspec.GridSpec(
            3, 1, figure=self.fig,
            height_ratios=[0.045, 1, 0.13],
            hspace=0.04, left=0.06, right=0.97,
            top=0.97, bottom=0.02,
        )
        self.ax_hdr = self.fig.add_subplot(outer[0])
        self.ax_hdr.axis("off")
        self._draw_header()

        ax_main = self.fig.add_subplot(outer[1])
        self._style_ax(ax_main, self.PANEL_TITLES[panel_idx])
        self.axes = [ax_main] * 5      # all panels point to same axis

        # rebuild lines for the focused panel only
        self.ln_true,  = ax_main.plot([], [], color=C["cyan"],  lw=2.0, label="True PPM")
        self.ln_meas,  = ax_main.plot([], [], color=C["green"], lw=1.2, alpha=0.85, label="Measured")
        self.ln_rsro,  = ax_main.plot([], [], color=C["orange"],lw=2.0)
        self.ln_vclean,= ax_main.plot([], [], color=C["cyan"],  lw=2.0, label="Clean")
        self.ln_vnoisy,= ax_main.plot([], [], color=C["yellow"],lw=1.0, alpha=0.8, label="Noisy")
        self.ln_adc,   = ax_main.step( [], [], color=C["purple"],lw=1.3, where="mid")
        self.ln_err,   = ax_main.plot([], [], color=C["text"],  lw=1.2, alpha=0.7)

        self.alarm_span = ax_main.axhspan(self.cfg.ALARM_PPM, 9999, alpha=0.07, color=C["red"])
        self.threshold  = ax_main.axhline(self.cfg.ALARM_PPM, color=C["red"], lw=1.1, ls="--", alpha=0.8)
        self.fill_alarm = ax_main.fill_between([], [], [], alpha=0.2, color=C["red"])
        self.txt_ppm    = ax_main.text(0.99, 0.92, "— PPM", transform=ax_main.transAxes,
                                       ha="right", va="top", fontsize=14, fontweight="bold",
                                       color=C["cyan"])
        self.txt_alarm  = ax_main.text(0.99, 0.83, "", transform=ax_main.transAxes,
                                       ha="right", va="top", fontsize=10, fontweight="bold",
                                       color=C["red"])
        self.txt_mae    = ax_main.text(0.02, 0.95, "MAE: —", transform=ax_main.transAxes,
                                       color=C["yellow"], fontsize=9, va="top")

        # Y-axis limits per panel
        ylims = [(200, 1600), (0.3, 0.9), (0.5, 3.5), (0, 4400), (-80, 80)]
        ax_main.set_ylim(*ylims[panel_idx])

        if panel_idx == 0:
            ax_main.legend(loc="upper left", fontsize=9, facecolor=C["panel"],
                           labelcolor=C["text"], framealpha=0.9, edgecolor=C["border"])
        if panel_idx == 2:
            ax_main.axhline(self.cfg.ADC_VREF, color=C["red"], lw=0.7, ls=":")
            ax_main.legend(loc="upper left", fontsize=9, facecolor=C["panel"],
                           labelcolor=C["text"], framealpha=0.9, edgecolor=C["border"])

        # controls strip
        ctrl = gridspec.GridSpecFromSubplotSpec(
            1, 8, subplot_spec=outer[2], wspace=0.25)

        btn_cfg = dict(color=C["border"], hovercolor="#2d3748")
        ax_pause = self.fig.add_subplot(ctrl[0, 0])
        ax_reset = self.fig.add_subplot(ctrl[0, 1])
        ax_mode  = self.fig.add_subplot(ctrl[0, 2])
        ax_noise = self.fig.add_subplot(ctrl[0, 3])
        ax_save  = self.fig.add_subplot(ctrl[0, 4])
        ax_speed = self.fig.add_subplot(ctrl[0, 5:7])
        ax_focus = self.fig.add_subplot(ctrl[0, 7])

        self.btn_pause = Button(ax_pause, "⏸  Pause",  **btn_cfg)
        self.btn_reset = Button(ax_reset, "↺  Reset",  **btn_cfg)
        self.btn_mode  = Button(ax_mode,  "◈  Mode",   **btn_cfg)
        self.btn_noise = Button(ax_noise, "〜  Noise ✓", **btn_cfg)
        self.btn_save  = Button(ax_save,  "💾  Save",  **btn_cfg)
        self.btn_focus = Button(ax_focus, f"[{panel_idx+1}]→⊞ Grid", **btn_cfg)

        for btn in (self.btn_pause, self.btn_reset, self.btn_mode,
                    self.btn_noise, self.btn_save, self.btn_focus):
            btn.label.set_fontfamily("monospace")
            btn.label.set_fontsize(8)
            btn.label.set_color(C["text"])

        self.sld_speed = Slider(
            ax_speed, "Speed", 1, 10, valinit=self.speed, valstep=1,
            color=C["cyan"])
        ax_speed.set_facecolor(C["panel"])
        self.sld_speed.label.set_color(C["text"])
        self.sld_speed.label.set_fontfamily("monospace")
        self.sld_speed.label.set_fontsize(8)
        self.sld_speed.valtext.set_color(C["yellow"])
        self.sld_speed.valtext.set_fontfamily("monospace")

        self.txt_status = self.fig.text(
            0.5, 0.004, self._status_str(),
            ha="center", va="bottom", fontsize=7,
            color=C["dim"], fontfamily="monospace"
        )
        self._connect_buttons()

    def _draw_header(self):
        self.ax_hdr.set_xlim(0, 1)
        self.ax_hdr.set_ylim(0, 1)
        self.ax_hdr.axis("off")
        self.ax_hdr.text(0.01, 0.5,
            "MQ-135  LIVE  CO₂  SENSOR  SIMULATION",
            fontsize=11, fontweight="bold", color=C["cyan"],
            va="center", fontfamily="monospace")
        mode_name = MOCK_CYCLE[self.mock_idx].upper()
        self.ax_hdr.text(0.99, 0.5,
            f"MODE: {mode_name}   |   "
            f"KEYS:  SPACE=pause  R=reset  M=mode  N=noise  +=faster  -=slower  1-5=focus  0=grid  S=save  Q=quit",
            fontsize=6.5, color=C["dim"],
            ha="right", va="center", fontfamily="monospace")

    def _style_ax(self, ax, title: str):
        ax.set_facecolor(C["panel"])
        ax.set_title(title, color=C["text"], fontsize=8,
                     fontweight="bold", pad=5)
        ax.grid(True, color=C["grid"], lw=0.6, alpha=0.9)
        ax.tick_params(colors=C["dim"], labelsize=7)
        for sp in ax.spines.values():
            sp.set_color(C["border"])
        ax.xaxis.label.set_color(C["dim"])
        ax.yaxis.label.set_color(C["dim"])
        ax.set_xlabel("Time (s)", fontsize=7)

    def _status_str(self) -> str:
        mode  = MOCK_CYCLE[self.mock_idx]
        noise = "ON" if self.noise_on else "OFF"
        state = "PAUSED" if self.paused else "RUNNING"
        focus = f"Panel {self.focus}" if self.focus else "Grid"
        return (f"Mode: {mode}  |  Speed: ×{self.speed}  |  Noise: {noise}  |  "
                f"View: {focus}  |  State: {state}  |  Samples: {self.total_samples}")

    # ── button & keyboard wiring ──────────────────────────────────
    def _connect_buttons(self):
        self.btn_pause.on_clicked(self._on_pause)
        self.btn_reset.on_clicked(self._on_reset)
        self.btn_mode .on_clicked(self._on_mode)
        self.btn_noise.on_clicked(self._on_noise)
        self.btn_save .on_clicked(self._on_save)
        self.btn_focus.on_clicked(self._on_focus_btn)
        self.sld_speed.on_changed(self._on_speed)

    def _connect_events(self):
        self.fig.canvas.mpl_connect("key_press_event", self._on_key)

    def _on_key(self, event):
        k = event.key
        if k == " ":             self._on_pause(None)
        elif k == "r":           self._on_reset(None)
        elif k == "m":           self._on_mode(None)
        elif k == "n":           self._on_noise(None)
        elif k == "s":           self._on_save(None)
        elif k in ("+", "="):   self._set_speed(min(10, self.speed + 1))
        elif k == "-":           self._set_speed(max(1,  self.speed - 1))
        elif k == "0":           self._set_focus(0)
        elif k in "12345":       self._set_focus(int(k))
        elif k in ("q", "escape"): plt.close("all")

    def _on_pause(self, _):
        self.paused = not self.paused
        lbl = "▶  Resume" if self.paused else "⏸  Pause"
        self.btn_pause.label.set_text(lbl)

    def _on_reset(self, _):
        self._reset()

    def _on_mode(self, _):
        self.mock_idx = (self.mock_idx + 1) % len(MOCK_CYCLE)
        self._reset()
        # rebuild header text
        self._draw_header()

    def _on_noise(self, _):
        self.noise_on = not self.noise_on
        lbl = "〜  Noise ✓" if self.noise_on else "〜  Noise ✗"
        self.btn_noise.label.set_text(lbl)

    def _on_save(self, _):
        self.fig.savefig(OUTPUT_PNG, dpi=150, bbox_inches="tight",
                         facecolor=C["bg"])
        print(f"  Snapshot saved --> {OUTPUT_PNG}")

    def _on_focus_btn(self, _):
        if self.focus == 0:
            self._set_focus(1)
        else:
            self._set_focus(0)

    def _on_speed(self, val):
        self.speed = int(val)

    def _set_speed(self, v: int):
        self.speed = v
        self.sld_speed.set_val(v)

    def _set_focus(self, panel: int):
        self.focus = panel
        if panel == 0:
            self._layout_grid()
        else:
            self._layout_focus(panel - 1)
        self.fig.canvas.draw_idle()

    # ── animation frame ──────────────────────────────────────────
    def _update(self, _frame):
        if self.paused:
            return

        # ingest N samples per frame (controlled by speed slider)
        for _ in range(self.speed):
            sim_t, ppm_true = next(self.gen)
            self.total_samples += 1

            # signal chain
            v_clean = self.chain.ppm_to_voltage(ppm_true)
            v_noisy = self.chain.add_noise(v_clean, self.noise_on)
            adc_val = self.chain.voltage_to_adc(v_noisy)
            ppm_meas= self.chain.adc_to_ppm(adc_val)
            alarm   = self.alarm.update(ppm_meas)
            rsro    = (ppm_true / self.cfg.a) ** (1.0 / self.cfg.b)
            error   = ppm_meas - ppm_true

            self.buf.push(
                sim_t,
                ppm_true=ppm_true, ppm_meas=ppm_meas,
                voltage_clean=v_clean, voltage_noisy=v_noisy,
                adc=float(adc_val), error=error,
                alarm=float(alarm), rsro=float(rsro),
            )

        t_arr, d = self.buf.get()
        if len(t_arr) < 2:
            return

        # rolling window
        t_now = t_arr[-1]
        t_min = t_now - self.WINDOW
        mask  = t_arr >= t_min
        t_w   = t_arr[mask]

        def w(key): return d[key][mask]

        # Audio alert control
        if self.audio_enabled:
            if self.alarm.active and not self.alert_playing:
                self.alert_sound.play(loops=-1)
                self.alert_playing = True
            elif not self.alarm.active and self.alert_playing:
                self.alert_sound.stop()
                self.alert_playing = False

        # Panel 0 — CO2
        if self.focus == 0 or self.focus == 1:
            ax = self.axes[0]
            self.ln_true.set_data(t_w, w("ppm_true"))
            self.ln_meas.set_data(t_w, w("ppm_meas"))
            
            # Optimized alarm fill: only recreate if alarm was or is active
            alarm_mask = w("alarm").astype(bool)
            if alarm_mask.any():
                self.fill_alarm.remove()
                alarm_vals = np.where(alarm_mask, w("ppm_meas"), np.nan)
                self.fill_alarm = ax.fill_between(
                    t_w, 0, alarm_vals, alpha=0.2, color=C["red"])
            else:
                # hide fill if no alarm
                self.fill_alarm.remove()
                self.fill_alarm = ax.fill_between([], [], [], alpha=0)

            ax.set_xlim(t_min, t_now)

            ppm_now  = d["ppm_true"][-1]
            self.txt_ppm.set_text(f"{ppm_now:.1f} PPM")
            self.txt_ppm.set_color(C["red"] if self.alarm.active else C["cyan"])
            if self.alarm.active:
                self.txt_alarm.set_text("⚠ CO2 HIGH")
            else:
                self.txt_alarm.set_text("")

        if self.focus == 0 or self.focus == 2:
            ax = self.axes[1]
            self.ln_rsro.set_data(t_w, w("rsro"))
            ax.set_xlim(t_min, t_now)

        if self.focus == 0 or self.focus == 3:
            ax = self.axes[2]
            self.ln_vclean.set_data(t_w, w("voltage_clean"))
            self.ln_vnoisy.set_data(t_w, w("voltage_noisy"))
            ax.set_xlim(t_min, t_now)

        if self.focus == 0 or self.focus == 4:
            ax = self.axes[3]
            self.ln_adc.set_data(t_w, w("adc"))
            ax.set_xlim(t_min, t_now)

        if self.focus == 0 or self.focus == 5:
            ax  = self.axes[4]
            err = w("error")
            self.ln_err.set_data(t_w, err)
            ax.set_xlim(t_min, t_now)
            
            # Dynamic Y-axis for error panel
            span = max(40, np.abs(err).max() * 1.3)
            ax.set_ylim(-span, span)
            
            mae = np.abs(err).mean()
            self.txt_mae.set_text(f"MAE: {mae:.2f} PPM")

        # update status bar
        self.txt_status.set_text(self._status_str())

        # flash background red when alarm active
        self.fig.patch.set_facecolor(C["alarm_bg"] if self.alarm.active else C["bg"])

    # ── run ───────────────────────────────────────────────────────
    def run(self):
        """Start the animation loop."""
        # interval in ms; reduced for smoother "video-like" experience
        self.anim = FuncAnimation(
            self.fig, self._update,
            interval=33,          # ~30 fps
            cache_frame_data=False,
        )
        plt.tight_layout(pad=0)
        print("=" * 64)
        print("  MQ-135 Live Simulation Dashboard")
        print("=" * 64)
        print("  SPACE   Pause / Resume")
        print("  R       Reset")
        print("  M       Cycle mock mode (realistic→sine→step→spike)")
        print("  N       Toggle Gaussian noise")
        print("  + / -   Speed up / down")
        print("  1-5     Focus single panel")
        print("  0       Back to grid view")
        print("  S       Save snapshot PNG")
        print("  Q/Esc   Quit")
        print("=" * 64)
        plt.show()
        
        # Cleanup audio on exit
        if self.audio_enabled:
            pygame.mixer.quit()


# ══════════════════════════════════════════════════════════════════════
# ENTRY POINT
# ══════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    # Allow --mock <mode> flag to pre-select starting mode
    args = sys.argv[1:]
    dash = LiveDashboard()
    if "--mock" in args:
        idx = args.index("--mock")
        mode = args[idx + 1] if idx + 1 < len(args) else "realistic"
        if mode in MOCK_CYCLE:
            dash.mock_idx = MOCK_CYCLE.index(mode)
            dash.gen = dash._make_gen()
    dash.run()
