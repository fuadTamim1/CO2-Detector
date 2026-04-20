"""
╔══════════════════════════════════════════════════════════════════╗
║          MQ-135 Gas Sensor Simulator  —  CO₂ Detection          ║
║          Hardware-Free Testing with Mock & Sine-Wave Data        ║
╚══════════════════════════════════════════════════════════════════╝

Core equation:   PPM = a × (Rs/Ro)^b

Where:
    PPM  → Gas concentration in parts-per-million
    Rs   → Sensor resistance when exposed to gas (Ohm)
    Ro   → Sensor resistance in clean air (reference, Ohm)
    a, b → Calibration constants from the MQ-135 datasheet

Signal chain simulated end-to-end:
    [1] Mock CO2 profile  (choose one of four generators below)
    [2] PPM  ->  Rs/Ro  ->  analog voltage  (voltage divider)
    [3] Gaussian noise added to the analog voltage
    [4] 12-bit ADC quantisation  (as on ESP32, Vref = 3.3 V)
    [5] ADC value  ->  reconstructed PPM  (on-chip firmware math)
    [6] Alarm logic: "CO2 HIGH" with consecutive-reading confirmation

Mock / test-signal generators (hardware-free validation):
    generate_co2_profile()   - realistic piecewise ramp  (default)
    mock_sine_co2()          - sine wave + noise  (unit-test signal)
    mock_step_co2()          - ideal step function  (alarm edge test)
    mock_multi_spike_co2()   - short spikes  (false-alarm rejection test)

Usage:
    python mq135_simulation.py                   # default realistic profile
    python mq135_simulation.py --mock sine        # sine-wave test
    python mq135_simulation.py --mock step        # step-function test
    python mq135_simulation.py --mock spike       # multi-spike test
    python mq135_simulation.py --no-show          # skip interactive window
"""

import sys
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from dataclasses import dataclass
from typing import Tuple
from pathlib import Path
from scipy.stats import norm
import warnings
warnings.filterwarnings("ignore")

# PNG is saved in the same folder as this script
OUTPUT_PNG = Path(__file__).parent / "mq135_simulation.png"


# ══════════════════════════════════════════════════════════════════
# SECTION 1 — Sensor configuration  (MQ-135 datasheet constants)
# ══════════════════════════════════════════════════════════════════

@dataclass
class MQ135Config:
    """
    MQ-135 sensor parameters for CO2 detection.

    Calibration constants (a, b) are read off the CO2 curve in the
    official Hanwei MQ-135 datasheet (log-log plot, linear regression).
    Ro is computed so that Vout sits comfortably within the ESP32 ADC
    input range (0 - 3.3 V) across the expected PPM range.
    """
    # Calibration constants (CO2 curve from datasheet)
    a: float = 110.47    # Scaling factor
    b: float = -2.862    # Exponent (negative: resistance drops as PPM rises)

    # Resistances
    Ro: float = 36_526   # Clean-air reference resistance (Ohm)
                         # Chosen so Vout ~ 1.5 V at 400 PPM (clean air)
    RL: float = 10_000   # Load resistor in the voltage-divider circuit (Ohm)

    # Supply voltage
    VCC: float = 5.0     # Circuit supply (V)

    # ESP32 ADC specifications
    ADC_BITS: int   = 12     # Resolution: 2^12 = 4096 discrete levels
    ADC_VREF: float = 3.3    # Reference voltage (V) — ESP32 maximum input
    ADC_MAX:  int   = 4095   # Maximum digital value (2^12 - 1)

    # Alarm settings
    ALARM_PPM:           float = 1000.0  # CO2 alarm threshold (PPM)
    ALARM_CONFIRM_COUNT: int   = 3       # Consecutive readings to confirm alarm

    # Gaussian noise model
    NOISE_STD: float = 0.005  # Voltage noise std-dev (5 mV — realistic for ESP32)


# ══════════════════════════════════════════════════════════════════
# SECTION 2 — Signal conversion chain
# ══════════════════════════════════════════════════════════════════

class MQ135Simulator:
    """
    Full end-to-end MQ-135 signal chain:
        PPM -> Rs/Ro -> Vout -> Vout+noise -> ADC -> PPM* -> Alarm

    Works with any CO2 profile (real hardware or mock data).
    """

    def __init__(self, config: MQ135Config = None):
        self.cfg           = config or MQ135Config()
        self.alarm_counter = 0      # consecutive readings above threshold
        self.alarm_active  = False  # current alarm state
        self.alarm_log     = []     # list of {type, time, ppm} events

    # ------------------------------------------------------------------
    def ppm_to_rs_ro(self, ppm: np.ndarray) -> np.ndarray:
        """
        Invert the sensor equation to obtain the resistance ratio Rs/Ro.

            PPM = a * (Rs/Ro)^b
            Rs/Ro = (PPM / a)^(1/b)

        Result clamped to the physically plausible range [0.1, 10].
        """
        rs_ro = (ppm / self.cfg.a) ** (1.0 / self.cfg.b)
        return np.clip(rs_ro, 0.1, 10.0)

    # ------------------------------------------------------------------
    def rs_ro_to_voltage(self, rs_ro: np.ndarray) -> np.ndarray:
        """
        Voltage divider circuit:  VCC -- Rs --[Vout]-- RL -- GND

            Vout = VCC * RL / (Rs + RL),   where Rs = rs_ro * Ro

        Output clamped to [0, VCC].
        """
        Rs      = rs_ro * self.cfg.Ro
        voltage = self.cfg.VCC * self.cfg.RL / (Rs + self.cfg.RL)
        return np.clip(voltage, 0.0, self.cfg.VCC)

    # ------------------------------------------------------------------
    def add_gaussian_noise(self, voltage: np.ndarray,
                           seed: int = None) -> np.ndarray:
        """
        Simulate real-world electrical noise:  Vnoise ~ N(0, sigma^2)

        Noise sources modelled:
          * Thermal (Johnson-Nyquist) noise in resistors
          * Electromagnetic interference (EMI)
          * Power-supply ripple
          * ADC quantisation noise

        Output clamped to [0, ADC_VREF].
        """
        rng   = np.random.default_rng(seed)
        noise = rng.normal(loc=0.0, scale=self.cfg.NOISE_STD,
                           size=voltage.shape)
        return np.clip(voltage + noise, 0.0, self.cfg.ADC_VREF)

    # ------------------------------------------------------------------
    def voltage_to_adc(self, voltage: np.ndarray) -> np.ndarray:
        """
        Simulate ESP32 12-bit ADC quantisation:

            ADC_value = round( Vin / Vref * (2^bits - 1) )

        Voltages above Vref saturate at ADC_MAX = 4095.
        """
        adc = np.round(
            np.clip(voltage, 0, self.cfg.ADC_VREF)
            / self.cfg.ADC_VREF
            * self.cfg.ADC_MAX
        ).astype(int)
        return np.clip(adc, 0, self.cfg.ADC_MAX)

    # ------------------------------------------------------------------
    def adc_to_ppm(self, adc_values: np.ndarray) -> np.ndarray:
        """
        Reverse conversion performed inside the ESP32 firmware:

            ADC -> Voltage -> Rs -> Rs/Ro -> PPM

        Voltage-divider inversion:
            Rs = RL * (VCC / Vout - 1)
        """
        voltage      = adc_values / self.cfg.ADC_MAX * self.cfg.ADC_VREF
        voltage_safe = np.where(voltage < 0.01, 0.01, voltage)
        Rs    = self.cfg.RL * (self.cfg.VCC / voltage_safe - 1.0)
        Rs    = np.clip(Rs, 100, 1_000_000)
        rs_ro = Rs / self.cfg.Ro
        ppm   = self.cfg.a * (rs_ro ** self.cfg.b)
        return np.clip(ppm, 0, 10_000)

    # ------------------------------------------------------------------
    def check_alarm(self, ppm_value: float, timestamp: float) -> dict:
        """
        Smart alarm with consecutive-reading confirmation.

        Rules:
          * PPM >= threshold for N readings in a row  -> ALARM ON
          * PPM drops below threshold                  -> ALARM OFF
          * Avoids false triggers caused by momentary noise spikes
        """
        was_active = self.alarm_active

        if ppm_value >= self.cfg.ALARM_PPM:
            self.alarm_counter += 1
            if self.alarm_counter >= self.cfg.ALARM_CONFIRM_COUNT:
                self.alarm_active = True
        else:
            self.alarm_counter = 0
            self.alarm_active  = False

        event = None
        if self.alarm_active and not was_active:
            event = {"type": "ALARM_ON",  "time": timestamp, "ppm": ppm_value}
            self.alarm_log.append(event)
            print(f"  [t={timestamp:6.1f}s]  *** CO2 HIGH alarm TRIGGERED ***  "
                  f"{ppm_value:.1f} PPM")
        elif not self.alarm_active and was_active:
            event = {"type": "ALARM_OFF", "time": timestamp, "ppm": ppm_value}
            self.alarm_log.append(event)
            print(f"  [t={timestamp:6.1f}s]  Alarm CLEARED  "
                  f"{ppm_value:.1f} PPM")

        return {"alarm_active": self.alarm_active,
                "counter":      self.alarm_counter,
                "event":        event}


# ══════════════════════════════════════════════════════════════════
# SECTION 3 — Mock / test-signal generators
#             Use these to validate the pipeline without hardware.
# ══════════════════════════════════════════════════════════════════

def generate_co2_profile(duration_s: int = 300, dt: float = 1.0,
                          seed: int = 42) -> Tuple[np.ndarray, np.ndarray]:
    """
    MOCK A — Realistic piecewise CO2 profile  (office-room scenario).

    Stages:
        0 -  60 s : Clean background air (~400 PPM, slight rise)
       60 - 120 s : Rapid rise  (people entering / increased activity)
      120 - 180 s : Dangerous peak  (crosses 1000 PPM alarm threshold)
      180 - 240 s : Ventilation opened -> CO2 drops
      240 - 300 s : Recovery to moderate level

    Superimposed:
        * Gaussian random variation  (breath-to-breath fluctuation, sigma=15 PPM)
        * Slow sinusoidal drift      (HVAC cycling, period ~120 s, amplitude 20 PPM)
    """
    rng  = np.random.default_rng(seed)
    t    = np.arange(0, duration_s, dt)
    n    = len(t)
    base = np.zeros(n)

    for i, ti in enumerate(t):
        if   ti < 60:   base[i] = 400  + 50  * (ti / 60)
        elif ti < 120:  base[i] = 450  + 600 * ((ti - 60)  / 60)
        elif ti < 150:  base[i] = 1050 + 200 * ((ti - 120) / 30)
        elif ti < 180:  base[i] = 1250 - 100 * ((ti - 150) / 30)
        elif ti < 240:  base[i] = 1150 - 700 * ((ti - 180) / 60)
        else:           base[i] = 450  - 50  * ((ti - 240) / 60)

    natural_variation = rng.normal(0, 15, n)
    slow_drift        = 20 * np.sin(2 * np.pi * t / 120)

    return t, np.clip(base + natural_variation + slow_drift, 300, 5000)


def mock_sine_co2(duration_s: int = 300, dt: float = 1.0,
                   seed: int = 7) -> Tuple[np.ndarray, np.ndarray]:
    """
    MOCK B — Sine wave + Gaussian noise.

        PPM(t) = centre + amplitude * sin(2*pi*t / period) + noise

    Useful for:
        * Unit-testing the ADC quantisation step in isolation
        * Verifying that PPM reconstruction is invertible
        * Checking that the noise model does not distort the signal shape

    Parameters are chosen so the signal crosses the 1000 PPM alarm
    threshold twice per cycle, giving a clean benchmark for the alarm logic.
    """
    rng       = np.random.default_rng(seed)
    t         = np.arange(0, duration_s, dt)
    centre    = 750.0    # PPM midpoint
    amplitude = 400.0    # PPM peak deviation
    period    = 100.0    # seconds per cycle
    noise_std = 25.0     # PPM random noise

    ppm = (centre
           + amplitude * np.sin(2 * np.pi * t / period)
           + rng.normal(0, noise_std, len(t)))
    return t, np.clip(ppm, 300, 2500)


def mock_step_co2(duration_s: int = 300,
                   dt: float = 1.0) -> Tuple[np.ndarray, np.ndarray]:
    """
    MOCK C — Ideal step function straddling the alarm threshold.

    Useful for:
        * Testing the consecutive-reading confirmation window precisely
        * Verifying alarm ON / OFF edge detection
        * Checking for off-by-one errors in the counter logic

    Profile:
          0 -  90 s : 500 PPM  (safe zone)
         90 - 210 s : 1200 PPM (alarm zone -- should trigger after N readings)
        210 - 300 s : 600 PPM  (recovery)
    """
    t   = np.arange(0, duration_s, dt)
    ppm = np.where(t < 90, 500.0,
          np.where(t < 210, 1200.0, 600.0))
    return t, ppm


def mock_multi_spike_co2(duration_s: int = 300, dt: float = 1.0,
                          seed: int = 3) -> Tuple[np.ndarray, np.ndarray]:
    """
    MOCK D — Baseline with multiple short spikes above the alarm threshold.

    Useful for:
        * Testing false-alarm rejection (spikes < confirmation window)
        * Verifying that a sustained spike DOES trigger the alarm

    Profile:
        Baseline  ~450 PPM
        Short spikes (2 s) at 1100 PPM  -> should NOT trigger (below N-reading window)
        One long spike (10 s) at 1150 PPM -> SHOULD trigger alarm
    """
    rng = np.random.default_rng(seed)
    t   = np.arange(0, duration_s, dt)
    ppm = np.full(len(t), 450.0) + rng.normal(0, 10, len(t))

    # Short 2-second spikes -- each below the confirmation threshold
    for spike_t in [40, 80, 130, 170]:
        mask = (t >= spike_t) & (t < spike_t + 2)
        ppm[mask] = 1100.0

    # One sustained 10-second spike -- exceeds confirmation threshold
    mask = (t >= 220) & (t < 230)
    ppm[mask] = 1150.0

    return t, np.clip(ppm, 300, 2000)


# ══════════════════════════════════════════════════════════════════
# SECTION 4 — Run the full simulation pipeline
# ══════════════════════════════════════════════════════════════════

def run_simulation(mock_mode: str = "realistic") -> dict:
    """
    Execute the complete signal chain and return all intermediate arrays.

    mock_mode options:
        "realistic"  -- piecewise office-room CO2 profile  (default)
        "sine"       -- sine wave + noise
        "step"       -- ideal step function
        "spike"      -- baseline with multiple short spikes
    """
    print("=" * 64)
    print("  MQ-135 CO2 Sensor Simulation")
    print(f"  Mock mode : {mock_mode.upper()}")
    print("=" * 64)

    cfg = MQ135Config()
    sim = MQ135Simulator(cfg)

    # [1] Generate CO2 profile ----------------------------------------
    print("\n[1] Generating CO2 time-series ...")
    profile_fn = {
        "realistic": generate_co2_profile,
        "sine":      mock_sine_co2,
        "step":      mock_step_co2,
        "spike":     mock_multi_spike_co2,
    }.get(mock_mode, generate_co2_profile)

    t, ppm_true = profile_fn()
    print(f"     Samples  : {len(t)}")
    print(f"     Duration : {t[-1]:.0f} s")
    print(f"     PPM range: {ppm_true.min():.1f} -- {ppm_true.max():.1f}")

    # [2] PPM -> Rs/Ro -> clean voltage --------------------------------
    print("\n[2] Converting PPM -> Rs/Ro -> analog voltage ...")
    rs_ro         = sim.ppm_to_rs_ro(ppm_true)
    voltage_clean = sim.rs_ro_to_voltage(rs_ro)
    print(f"     Voltage range (clean): {voltage_clean.min():.3f} -- "
          f"{voltage_clean.max():.3f} V")

    # [3] Add Gaussian noise -------------------------------------------
    print("\n[3] Adding Gaussian noise ...")
    voltage_noisy = sim.add_gaussian_noise(voltage_clean, seed=99)
    noise         = voltage_noisy - voltage_clean
    print(f"     Noise std  : {noise.std()*1000:.2f} mV  "
          f"(target {cfg.NOISE_STD*1000:.1f} mV)")
    print(f"     Peak noise : +/-{np.abs(noise).max()*1000:.2f} mV")

    # [4] ADC quantisation --------------------------------------------
    print("\n[4] Simulating 12-bit ADC (ESP32) ...")
    adc_values = sim.voltage_to_adc(voltage_noisy)
    print(f"     ADC output range: {adc_values.min()} -- "
          f"{adc_values.max()}  (max = {cfg.ADC_MAX})")

    # [5] Reconstruct PPM from ADC ------------------------------------
    print("\n[5] Reconstructing PPM from ADC readings ...")
    ppm_measured = sim.adc_to_ppm(adc_values)
    error        = ppm_measured - ppm_true
    print(f"     Mean Absolute Error : {np.abs(error).mean():.2f} PPM")
    print(f"     Peak error          : {np.abs(error).max():.2f} PPM")

    # [6] Alarm evaluation --------------------------------------------
    print(f"\n[6] Evaluating alarm (threshold={cfg.ALARM_PPM:.0f} PPM, "
          f"confirm={cfg.ALARM_CONFIRM_COUNT} readings) ...")
    alarm_states = np.zeros(len(t), dtype=bool)
    for i, (ti, ppm_val) in enumerate(zip(t, ppm_measured)):
        result          = sim.check_alarm(float(ppm_val), float(ti))
        alarm_states[i] = result["alarm_active"]

    print(f"\n     Total alarm events  : {len(sim.alarm_log)}")
    print(f"     Time in alarm state : {alarm_states.mean()*100:.1f}%")

    return {
        "t":             t,
        "ppm_true":      ppm_true,
        "ppm_measured":  ppm_measured,
        "rs_ro":         rs_ro,
        "voltage_clean": voltage_clean,
        "voltage_noisy": voltage_noisy,
        "adc_values":    adc_values,
        "alarm_states":  alarm_states,
        "alarm_log":     sim.alarm_log,
        "config":        cfg,
        "error":         error,
        "mock_mode":     mock_mode,
    }


# ══════════════════════════════════════════════════════════════════
# SECTION 5 — Analysis dashboard
# ══════════════════════════════════════════════════════════════════

def plot_results(data: dict, show_window: bool = True):
    """
    Render a 7-panel analysis dashboard.
    Saves a PNG next to this script, then optionally opens an
    interactive Matplotlib window (show_window=True by default).

    Panels:
        [1]  CO2 concentration  (true vs measured, alarm overlay)
        [2]  Rs/Ro ratio
        [3]  Analog voltage     (clean vs noisy)
        [4]  ADC digital output (12-bit staircase)
        [5]  Measurement error  (PPM_measured - PPM_true)
        [6]  Gaussian noise histogram + theoretical PDF
        [7]  Alarm state timeline
    """
    cfg  = data["config"]
    t    = data["t"]
    mode = data["mock_mode"].upper()

    # Figure layout
    fig = plt.figure(figsize=(16, 14), facecolor="#0d1117")
    fig.suptitle(
        f"MQ-135 CO2 Sensor Simulation   |   Mock mode: {mode}",
        fontsize=15, fontweight="bold", color="#e6edf3",
        y=0.985, fontfamily="monospace"
    )
    gs = gridspec.GridSpec(4, 2, figure=fig,
                           hspace=0.48, wspace=0.35,
                           left=0.08, right=0.96, top=0.94, bottom=0.06)

    # Colour palette
    DARK   = "#0d1117"
    PANEL  = "#161b22"
    GRID   = "#21262d"
    TEXT   = "#e6edf3"
    DIM    = "#7d8590"
    CYAN   = "#79c0ff"
    GREEN  = "#56d364"
    RED    = "#f85149"
    YELLOW = "#e3b341"
    ORANGE = "#d29922"
    PURPLE = "#bc8cff"

    def style_ax(ax, title):
        ax.set_facecolor(PANEL)
        ax.tick_params(colors=DIM, labelsize=8)
        ax.set_title(title, color=TEXT, fontsize=9, fontweight="bold",
                     fontfamily="monospace", pad=8)
        ax.grid(True, color=GRID, linewidth=0.7, alpha=0.8)
        for sp in ax.spines.values():
            sp.set_color(GRID)
        ax.xaxis.label.set_color(DIM)
        ax.yaxis.label.set_color(DIM)

    # Panel [1] — CO2 concentration (full width) ----------------------
    ax1 = fig.add_subplot(gs[0, :])
    style_ax(ax1, "[1]  CO2 Concentration  —  Ground Truth  vs  ADC-Measured")

    ax1.axhspan(cfg.ALARM_PPM, 6000, alpha=0.07, color=RED, zorder=0)
    ax1.axhline(cfg.ALARM_PPM, color=RED, linewidth=1.2, linestyle="--",
                alpha=0.85,
                label=f"Alarm threshold  ({cfg.ALARM_PPM:.0f} PPM)")
    ax1.plot(t, data["ppm_true"],     color=CYAN,  linewidth=1.6,
             label="True PPM  (simulated ground truth)", zorder=3)
    ax1.plot(t, data["ppm_measured"], color=GREEN, linewidth=1.0,
             alpha=0.75, label="Measured PPM  (via ADC + noise)", zorder=2)

    alarm_fill = np.where(data["alarm_states"], data["ppm_measured"], np.nan)
    ax1.fill_between(t, 0, alarm_fill, alpha=0.18, color=RED,
                     label="Active alarm region")

    for evt in data["alarm_log"]:
        c   = RED   if evt["type"] == "ALARM_ON"  else GREEN
        mk  = "^"   if evt["type"] == "ALARM_ON"  else "v"
        lbl = "ON"  if evt["type"] == "ALARM_ON"  else "OFF"
        ax1.annotate(f'ALARM {lbl}\n{evt["ppm"]:.0f} PPM',
                     xy=(evt["time"], evt["ppm"]),
                     xytext=(evt["time"] + 4, evt["ppm"] + 90),
                     color=c, fontsize=7, fontfamily="monospace",
                     arrowprops=dict(arrowstyle="->", color=c, lw=0.8))

    ax1.set_ylabel("CO2 Concentration (PPM)", fontsize=8)
    ax1.set_xlabel("Time (s)", fontsize=8)
    ax1.legend(loc="upper left", fontsize=7, facecolor=PANEL,
               labelcolor=TEXT, framealpha=0.9, edgecolor=GRID)
    ax1.set_xlim(0, t[-1])
    ax1.set_ylim(200, max(1600, data["ppm_true"].max() * 1.12))
    ax1.text(0.99, 0.95, f"PPM_max = {data['ppm_true'].max():.0f}",
             transform=ax1.transAxes, ha="right", va="top",
             color=YELLOW, fontsize=8, fontfamily="monospace")

    # Panel [2] — Rs/Ro ratio -----------------------------------------
    ax2 = fig.add_subplot(gs[1, 0])
    style_ax(ax2, "[2]  Sensor Resistance Ratio  Rs / Ro")
    ax2.plot(t, data["rs_ro"], color=ORANGE, linewidth=1.4)
    ax2.fill_between(t, data["rs_ro"].min(), data["rs_ro"],
                     alpha=0.15, color=ORANGE)
    ax2.set_ylabel("Rs / Ro", fontsize=8)
    ax2.set_xlabel("Time (s)", fontsize=8)
    ax2.set_xlim(0, t[-1])

    # Panel [3] — Analog voltage --------------------------------------
    ax3 = fig.add_subplot(gs[1, 1])
    style_ax(ax3, "[3]  Analog Output Voltage  —  Clean  vs  Noisy")
    ax3.plot(t, data["voltage_clean"], color=CYAN,   linewidth=1.4,
             label="Vout  (clean)", zorder=3)
    ax3.plot(t, data["voltage_noisy"], color=YELLOW, linewidth=0.7,
             alpha=0.8, label="Vout + Gaussian noise")
    ax3.axhline(cfg.ADC_VREF, color=RED, linewidth=0.8, linestyle=":",
                label=f"ESP32 ADC limit  ({cfg.ADC_VREF} V)")
    ax3.set_ylabel("Voltage (V)", fontsize=8)
    ax3.set_xlabel("Time (s)", fontsize=8)
    ax3.legend(fontsize=7, facecolor=PANEL, labelcolor=TEXT,
               framealpha=0.9, edgecolor=GRID)
    ax3.set_xlim(0, t[-1])

    # Panel [4] — ADC output ------------------------------------------
    ax4 = fig.add_subplot(gs[2, 0])
    style_ax(ax4, "[4]  ADC Digital Output  (12-bit,  0 – 4095)")
    ax4.step(t, data["adc_values"], color=PURPLE, linewidth=0.9,
             where="mid", label="ADC reading")
    ax4.fill_between(t, 0, data["adc_values"], alpha=0.10,
                     color=PURPLE, step="mid")
    ax4.axhline(cfg.ADC_MAX, color=RED, linewidth=0.7, linestyle=":",
                alpha=0.6, label=f"ADC saturation  ({cfg.ADC_MAX})")
    ax4.set_ylabel("ADC Value", fontsize=8)
    ax4.set_xlabel("Time (s)", fontsize=8)
    ax4.set_ylim(0, 4400)
    ax4.legend(fontsize=7, facecolor=PANEL, labelcolor=TEXT,
               framealpha=0.9, edgecolor=GRID)
    ax4.set_xlim(0, t[-1])

    # Panel [5] — Measurement error -----------------------------------
    ax5 = fig.add_subplot(gs[2, 1])
    style_ax(ax5, "[5]  Measurement Error  (PPM_measured − PPM_true)")
    ax5.fill_between(t, data["error"], 0,
                     where=data["error"] >= 0, color=GREEN, alpha=0.4,
                     label="Positive error")
    ax5.fill_between(t, data["error"], 0,
                     where=data["error"] <  0, color=RED,   alpha=0.4,
                     label="Negative error")
    ax5.plot(t, data["error"], color=TEXT, linewidth=0.6, alpha=0.5)
    ax5.axhline(0, color=DIM, linewidth=0.8)
    ax5.set_ylabel("Error (PPM)", fontsize=8)
    ax5.set_xlabel("Time (s)", fontsize=8)
    ax5.legend(fontsize=7, facecolor=PANEL, labelcolor=TEXT,
               framealpha=0.9, edgecolor=GRID)
    ax5.set_xlim(0, t[-1])
    mae = np.abs(data["error"]).mean()
    ax5.text(0.02, 0.95, f"MAE = {mae:.2f} PPM",
             transform=ax5.transAxes, color=YELLOW, fontsize=8,
             fontfamily="monospace", va="top")

    # Panel [6] — Noise histogram -------------------------------------
    ax6 = fig.add_subplot(gs[3, 0])
    style_ax(ax6, "[6]  Gaussian Noise Distribution  (Histogram + Theoretical PDF)")
    noise_mv = (data["voltage_noisy"] - data["voltage_clean"]) * 1000
    ax6.hist(noise_mv, bins=40, color=CYAN, alpha=0.70,
             edgecolor=DARK, linewidth=0.4, density=True,
             label="Measured noise")
    x_rng = np.linspace(noise_mv.min(), noise_mv.max(), 200)
    pdf   = norm.pdf(x_rng, 0, cfg.NOISE_STD * 1000)
    ax6.plot(x_rng, pdf, color=YELLOW, linewidth=1.8,
             label=f"N(0, {cfg.NOISE_STD*1000:.0f} mV) PDF")
    ax6.set_xlabel("Noise (mV)", fontsize=8)
    ax6.set_ylabel("Probability Density", fontsize=8)
    ax6.legend(fontsize=7, facecolor=PANEL, labelcolor=TEXT,
               framealpha=0.9, edgecolor=GRID)
    ax6.text(0.02, 0.95, f"sigma = {noise_mv.std():.2f} mV",
             transform=ax6.transAxes, color=YELLOW, fontsize=8,
             fontfamily="monospace", va="top")

    # Panel [7] — Alarm state timeline --------------------------------
    ax7 = fig.add_subplot(gs[3, 1])
    style_ax(ax7, "[7]  Alarm State Timeline  —  CO2 HIGH")
    ax7.fill_between(t, data["alarm_states"].astype(int),
                     step="mid", color=RED, alpha=0.50,
                     label="Alarm active")
    ax7.step(t, data["alarm_states"].astype(int),
             color=RED, linewidth=1.3, where="mid")
    ax7.set_yticks([0, 1])
    ax7.set_yticklabels(["SAFE  (OK)", "DANGER  (CO2 HIGH)"],
                        color=TEXT, fontsize=8)
    ax7.set_xlabel("Time (s)", fontsize=8)
    ax7.set_ylim(-0.1, 1.45)
    ax7.set_xlim(0, t[-1])
    pct = data["alarm_states"].mean() * 100
    ax7.text(0.98, 0.90, f"Alarm time: {pct:.1f}%",
             transform=ax7.transAxes, ha="right", color=RED,
             fontsize=8, fontfamily="monospace")

    # Save PNG
    plt.savefig(OUTPUT_PNG, dpi=150, bbox_inches="tight", facecolor=DARK)
    print(f"\n  Plot saved  -->  {OUTPUT_PNG}")

    # Open interactive window (blocks until closed)
    if show_window:
        print("  Opening interactive window  (close the window to exit) ...")
        plt.show()
    else:
        plt.close()


# ══════════════════════════════════════════════════════════════════
# Entry point
# ══════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    # ── Parse CLI flags ───────────────────────────────────────────
    # Usage:
    #   python mq135_simulation.py                  (realistic, window on)
    #   python mq135_simulation.py --mock sine
    #   python mq135_simulation.py --mock step
    #   python mq135_simulation.py --mock spike
    #   python mq135_simulation.py --no-show        (save PNG only)
    args      = sys.argv[1:]
    mock_mode = "realistic"
    show_win  = True

    if "--mock" in args:
        idx       = args.index("--mock")
        mock_mode = args[idx + 1] if idx + 1 < len(args) else "realistic"
    if "--no-show" in args:
        show_win = False

    # ── Run pipeline ──────────────────────────────────────────────
    data = run_simulation(mock_mode=mock_mode)
    cfg  = data["config"]

    # ── Print summary ─────────────────────────────────────────────
    print("\n" + "=" * 64)
    print("  SIMULATION SUMMARY")
    print("=" * 64)
    print(f"  Mock mode          : {mock_mode}")
    print(f"  ADC resolution     : {cfg.ADC_BITS}-bit  ({cfg.ADC_MAX + 1} levels)")
    print(f"  ADC reference      : {cfg.ADC_VREF} V  (ESP32)")
    print(f"  Alarm threshold    : {cfg.ALARM_PPM:.0f} PPM")
    print(f"  Confirmation count : {cfg.ALARM_CONFIRM_COUNT} consecutive readings")
    print(f"  Noise std-dev      : {cfg.NOISE_STD*1000:.1f} mV")
    print(f"  Mean Absolute Error: {np.abs(data['error']).mean():.2f} PPM")
    print(f"  Alarm events       : {len(data['alarm_log'])}")
    print("=" * 64)

    # ── Plot dashboard ────────────────────────────────────────────
    plot_results(data, show_window=show_win)
    print("  Done.")
