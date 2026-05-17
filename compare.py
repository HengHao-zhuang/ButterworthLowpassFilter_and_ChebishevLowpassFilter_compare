import argparse
import os
import sys
from pathlib import Path

import numpy as np

os.environ.setdefault("MPLCONFIGDIR", str(Path(".matplotlib-cache").resolve()))

if "--no-show" in sys.argv:
    import matplotlib

    matplotlib.use("Agg")

import matplotlib.pyplot as plt
from scipy import signal


STYLE = {
    "butter": "#1f77b4",
    "cheby": "#d62728",
    "raw": "#404040",
    "grid": "#b0b0b0",
}


def db(value):
    return 20 * np.log10(np.maximum(np.abs(value), 1e-15))


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Aggressively compare non-ideal Butterworth and Chebyshev LC "
            "power filters for side-channel mitigation."
        )
    )
    parser.add_argument("--no-show", action="store_true")
    parser.add_argument("--output-dir", default="plots")
    return parser.parse_args()


def configure_plot_style():
    plt.rcParams.update(
        {
            "figure.dpi": 120,
            "savefig.dpi": 300,
            "font.size": 11,
            "axes.titlesize": 13,
            "axes.labelsize": 11,
            "legend.fontsize": 10,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
            "axes.linewidth": 1.0,
            "lines.linewidth": 2.2,
            "grid.linewidth": 0.7,
            "grid.alpha": 0.35,
        }
    )


def zpk_add_esr_and_parasitics(z, p, k, cutoff_hz, kind):
    """Add simple non-ideal power-delivery behavior to the analog prototype.

    This is not a full SPICE model. It intentionally captures the qualitative
    effects that matter here: capacitor ESR zero, high-frequency ESL pole, and
    a load-network resonance caused by non-ideal inductors, capacitors, vias,
    and package parasitics.
    """
    wz_esr = 2 * np.pi * 17e6
    wp_esl = 2 * np.pi * 420e6
    z = list(z) + [-wz_esr]
    p = list(p) + [-wp_esl]
    k = k * (wp_esl / wz_esr)

    if kind == "butter":
        resonance_hz = 1.32 * cutoff_hz
        resonance_q = 0.85
        extra_pole_hz = 180e6
    else:
        resonance_hz = 1.12 * cutoff_hz
        resonance_q = 3.8
        extra_pole_hz = 145e6

    w0 = 2 * np.pi * resonance_hz
    sigma = -w0 / (2 * resonance_q)
    wd = w0 * np.sqrt(max(0.0, 1 - 1 / (4 * resonance_q**2)))
    p.extend([sigma + 1j * wd, sigma - 1j * wd])
    k *= w0**2

    wp_extra = 2 * np.pi * extra_pole_hz
    p.append(-wp_extra)
    k *= wp_extra
    return np.array(z, dtype=complex), np.array(p, dtype=complex), np.real_if_close(k)


def make_filters(order, cutoff_hz, ripple_db):
    wc = 2 * np.pi * cutoff_hz
    butter_z, butter_p, butter_k = signal.butter(
        order, wc, btype="low", analog=True, output="zpk"
    )
    cheby_z, cheby_p, cheby_k = signal.cheby1(
        order, ripple_db, wc, btype="low", analog=True, output="zpk"
    )

    butter = zpk_add_esr_and_parasitics(
        butter_z, butter_p, butter_k, cutoff_hz, "butter"
    )
    cheby = zpk_add_esr_and_parasitics(
        cheby_z, cheby_p, cheby_k, cutoff_hz, "cheby"
    )
    return {"Butterworth": butter, "Chebyshev": cheby}


def zpk_freq_response(filter_zpk, freq_hz):
    z, p, k = filter_zpk
    _, h = signal.freqs_zpk(z, p, k, worN=2 * np.pi * freq_hz)
    return h


def zpk_to_ss(filter_zpk):
    z, p, k = filter_zpk
    return signal.ZerosPolesGain(z, p, k).to_ss()


def group_delay_seconds(filter_zpk, freq_hz):
    h = zpk_freq_response(filter_zpk, freq_hz)
    phase = np.unwrap(np.angle(h))
    omega = 2 * np.pi * freq_hz
    return -np.gradient(phase, omega)


def ringing_metrics(time_s, response, target=1.0):
    peak = float(np.max(response))
    overshoot = max(0.0, (peak - target) / target * 100)
    lower = target * 0.98
    upper = target * 1.02
    outside = np.where((response < lower) | (response > upper))[0]
    settling_time = float(time_s[outside[-1]]) if outside.size else 0.0
    steady = response[-max(20, response.size // 20) :]
    ripple_pp = float(np.max(steady) - np.min(steady))
    return overshoot, settling_time, ripple_pp


def synthetic_side_channel_trace(time_s):
    rng = np.random.default_rng(2026)
    trace = np.full_like(time_s, 0.02)

    event_times = np.array(
        [0.28, 0.43, 0.60, 0.84, 1.05, 1.31, 1.55, 1.76, 2.04, 2.33, 2.58, 2.83]
    ) * 1e-6
    sensitive_bits = np.array([1, 0, 1, 1, 0, 1, 0, 0, 1, 0, 1, 1])
    sigma_fast = 3.8e-9
    sigma_slow = 18e-9

    for event_time, bit in zip(event_times, sensitive_bits):
        amp = 0.06 + 0.085 * bit
        trace += amp * np.exp(-0.5 * ((time_s - event_time) / sigma_fast) ** 2)
        trace += 0.030 * bit * np.exp(-0.5 * ((time_s - event_time - 22e-9) / sigma_slow) ** 2)

    regulator_freq = 42e6
    regulator_ripple = 0.030 * signal.square(2 * np.pi * regulator_freq * time_s, duty=0.42)
    regulator_ripple += 0.018 * np.sin(2 * np.pi * 84e6 * time_s + 0.35)
    regulator_ripple += 0.011 * np.sin(2 * np.pi * 126e6 * time_s + 1.2)

    clock_feedthrough = 0.016 * np.sin(2 * np.pi * 168e6 * time_s)
    random_noise = 0.0045 * rng.normal(size=time_s.size)
    return trace + regulator_ripple + clock_feedthrough + random_noise


def pulsed_load_current(time_s):
    load = np.zeros_like(time_s)
    load[(time_s >= 0.18e-6) & (time_s < 0.92e-6)] = 1.0
    load[(time_s >= 1.30e-6) & (time_s < 1.42e-6)] = -0.55
    load[(time_s >= 1.70e-6) & (time_s < 2.35e-6)] = 0.70
    load[(time_s >= 2.62e-6) & (time_s < 2.74e-6)] = -0.75

    edge_sigma = 3.0e-9
    kernel_t = np.linspace(-18e-9, 18e-9, 121)
    kernel = np.exp(-0.5 * (kernel_t / edge_sigma) ** 2)
    kernel /= np.sum(kernel)
    return np.convolve(load, kernel, mode="same")


def simulate(filter_zpk, input_signal, time_s):
    sys = zpk_to_ss(filter_zpk)
    _, y, _ = signal.lsim(sys, U=input_signal, T=time_s)
    return y


def plot_bode(filters, cutoff_hz, ripple_db, output_dir):
    freq_hz = np.logspace(4, 9.2, 7000)

    fig, axes = plt.subplots(3, 1, figsize=(7.2, 9.0), sharex=True)
    for name, filt in filters.items():
        h = zpk_freq_response(filt, freq_hz)
        color = STYLE["butter"] if name == "Butterworth" else STYLE["cheby"]
        label = name if name == "Butterworth" else f"{name} ({ripple_db:g} dB ripple)"
        axes[0].semilogx(freq_hz, db(h), color=color, label=label)
        axes[1].semilogx(freq_hz, np.unwrap(np.angle(h)) * 180 / np.pi, color=color)
        axes[2].semilogx(freq_hz, group_delay_seconds(filt, freq_hz) * 1e9, color=color)

    for ax in axes:
        ax.axvline(cutoff_hz, color="black", linestyle="--", linewidth=1.0)
        ax.grid(True, which="both")

    axes[0].set_title("Non-Ideal 8th-Order LC Power Filter Response")
    axes[0].set_ylabel("Magnitude (dB)")
    axes[0].set_ylim(-120, 8)
    axes[0].legend(loc="lower left")
    axes[1].set_ylabel("Phase (deg)")
    axes[2].set_ylabel("Group delay (ns)")
    axes[2].set_xlabel("Frequency (Hz)")
    axes[2].set_ylim(0, 620)

    fig.tight_layout()
    path = output_dir / "paper_bode_group_delay.png"
    fig.savefig(path)
    return path


def plot_transition_zoom(filters, cutoff_hz, ripple_db, output_dir):
    freq_hz = np.linspace(0.35 * cutoff_hz, 2.2 * cutoff_hz, 3500)

    fig, axes = plt.subplots(2, 1, figsize=(7.2, 6.6), sharex=True)
    for name, filt in filters.items():
        h = zpk_freq_response(filt, freq_hz)
        color = STYLE["butter"] if name == "Butterworth" else STYLE["cheby"]
        axes[0].plot(freq_hz / 1e6, db(h), color=color, label=name)
        axes[1].plot(freq_hz / 1e6, group_delay_seconds(filt, freq_hz) * 1e9, color=color)

    axes[0].axhspan(-2, 0, color=STYLE["cheby"], alpha=0.08, label="2 dB Chebyshev ripple band")
    for ax in axes:
        ax.axvline(cutoff_hz / 1e6, color="black", linestyle="--", linewidth=1.0)
        ax.grid(True)

    axes[0].annotate(
        "Butterworth passband flatness",
        xy=(0.60 * cutoff_hz / 1e6, -0.03),
        xytext=(2.3, 2.2),
        arrowprops={"arrowstyle": "->", "color": STYLE["butter"]},
        color=STYLE["butter"],
    )
    axes[0].annotate(
        "Chebyshev sharper roll-off",
        xy=(1.25 * cutoff_hz / 1e6, -44),
        xytext=(6.7, -18),
        arrowprops={"arrowstyle": "->", "color": STYLE["cheby"]},
        color=STYLE["cheby"],
    )
    axes[0].set_title("Transition Region Zoom Around Cutoff")
    axes[0].set_ylabel("Magnitude (dB)")
    axes[0].set_ylim(-70, 5)
    axes[0].legend(loc="lower left")
    axes[1].set_ylabel("Group delay (ns)")
    axes[1].set_xlabel("Frequency (MHz)")
    axes[1].set_ylim(0, 700)

    fig.tight_layout()
    path = output_dir / "paper_transition_zoom.png"
    fig.savefig(path)
    return path


def plot_impulse_step(filters, cutoff_hz, output_dir):
    time_s = np.linspace(0, 1.8e-6, 3200)
    fig_impulse, ax_impulse = plt.subplots(1, 1, figsize=(7.2, 3.9))
    fig_step, ax_step = plt.subplots(1, 1, figsize=(7.2, 4.2))
    metric_lines = []

    for name, filt in filters.items():
        color = STYLE["butter"] if name == "Butterworth" else STYLE["cheby"]
        sys = zpk_to_ss(filt)
        _, impulse = signal.impulse(sys, T=time_s)
        _, step = signal.step(sys, T=time_s)
        ax_impulse.plot(
            time_s * 1e6,
            impulse / np.max(np.abs(impulse)),
            color=color,
            label=name,
        )
        ax_step.plot(time_s * 1e6, step, color=color, label=name)
        overshoot, settling, ripple_pp = ringing_metrics(time_s, step)
        metric_lines.append(
            f"{name}: overshoot={overshoot:.1f}%, settle~{settling * 1e6:.2f} us, late ripple={ripple_pp:.3f}"
        )

    ax_impulse.grid(True)
    ax_impulse.legend(loc="upper right")
    ax_impulse.set_title("Impulse Response: Ringing Energy")
    ax_impulse.set_xlabel("Time (us)")
    ax_impulse.set_ylabel("Normalized amplitude")

    ax_step.axhline(1.0, color="black", linestyle="--", linewidth=1.0)
    ax_step.grid(True)
    ax_step.legend(loc="upper right")
    ax_step.set_title("Step Response: Overshoot and Settling")
    ax_step.set_xlabel("Time (us)")
    ax_step.set_ylabel("Normalized output")
    ax_step.text(
        0.98,
        0.05,
        "\n".join(metric_lines),
        transform=ax_step.transAxes,
        ha="right",
        va="bottom",
        bbox={"boxstyle": "round,pad=0.35", "facecolor": "white", "alpha": 0.85, "edgecolor": "#777777"},
    )

    fig_impulse.tight_layout()
    fig_step.tight_layout()
    impulse_path = output_dir / "paper_impulse_response.png"
    step_path = output_dir / "paper_step_ringing.png"
    fig_impulse.savefig(impulse_path)
    fig_step.savefig(step_path)
    return [impulse_path, step_path]


def plot_load_transient(filters, output_dir):
    time_s = np.linspace(0, 3.0e-6, 6500)
    load = pulsed_load_current(time_s)

    fig, axes = plt.subplots(2, 1, figsize=(7.2, 6.8), sharex=True)
    axes[0].plot(time_s * 1e6, load, color=STYLE["raw"], linewidth=1.7, label="Pulsed digital load")
    axes[0].set_title("Fast Digital Load Current")
    axes[0].set_ylabel("Normalized load")
    axes[0].grid(True)
    axes[0].legend()

    for name, filt in filters.items():
        color = STYLE["butter"] if name == "Butterworth" else STYLE["cheby"]
        droop = simulate(filt, load, time_s)
        ac_droop = droop - np.mean(droop[-900:])
        axes[1].plot(time_s * 1e6, ac_droop, color=color, label=name)

    axes[1].set_title("Supply Response to Load Transients")
    axes[1].set_xlabel("Time (us)")
    axes[1].set_ylabel("Voltage-noise proxy")
    axes[1].grid(True)
    axes[1].legend(loc="upper right")
    axes[1].annotate(
        "Chebyshev ringing after fast load edges",
        xy=(0.95, 0.40),
        xytext=(1.20, 0.78),
        textcoords="data",
        arrowprops={"arrowstyle": "->", "color": STYLE["cheby"]},
        color=STYLE["cheby"],
    )

    fig.tight_layout()
    path = output_dir / "paper_load_transient.png"
    fig.savefig(path)
    return path


def plot_switching_noise_and_sca(filters, output_dir):
    time_s = np.linspace(0, 3.2e-6, 9000)
    trace = synthetic_side_channel_trace(time_s)

    fig, axes = plt.subplots(3, 1, figsize=(7.2, 9.0), sharex=True)
    axes[0].plot(time_s * 1e6, trace, color=STYLE["raw"], linewidth=1.0)
    axes[0].set_title("Injected Switching-Regulator Noise + Synthetic SCA Leakage")
    axes[0].set_ylabel("Raw trace")
    axes[0].grid(True)

    outputs = {}
    for name, filt in filters.items():
        color = STYLE["butter"] if name == "Butterworth" else STYLE["cheby"]
        outputs[name] = simulate(filt, trace, time_s)
        axes[1].plot(time_s * 1e6, outputs[name], color=color, label=name)
    axes[1].set_title("Filtered Side-Channel Trace")
    axes[1].set_ylabel("Filtered trace")
    axes[1].grid(True)
    axes[1].legend()

    butter_residual = trace - outputs["Butterworth"]
    cheby_residual = trace - outputs["Chebyshev"]
    axes[2].plot(time_s * 1e6, butter_residual, color=STYLE["butter"], label="Removed by Butterworth")
    axes[2].plot(time_s * 1e6, cheby_residual, color=STYLE["cheby"], label="Removed by Chebyshev")
    axes[2].set_title("High-Frequency Leakage Suppressed by Each Filter")
    axes[2].set_xlabel("Time (us)")
    axes[2].set_ylabel("Residual")
    axes[2].grid(True)
    axes[2].legend()

    fig.tight_layout()
    path = output_dir / "paper_switching_noise_sca_trace.png"
    fig.savefig(path)
    return path


def plot_component_sensitivity(filters, cutoff_hz, ripple_db, order, output_dir):
    rng = np.random.default_rng(44)
    freq_hz = np.linspace(0.45 * cutoff_hz, 2.0 * cutoff_hz, 1700)

    fig, axes = plt.subplots(1, 2, figsize=(7.6, 3.8), sharey=True)
    for ax, name in zip(axes, ["Butterworth", "Chebyshev"]):
        color = STYLE["butter"] if name == "Butterworth" else STYLE["cheby"]
        for _ in range(80):
            fc = cutoff_hz * (1 + rng.normal(0, 0.12))
            rp = max(0.2, ripple_db * (1 + rng.normal(0, 0.22)))
            varied = make_filters(order, fc, rp)[name]
            h = zpk_freq_response(varied, freq_hz)
            ax.plot(freq_hz / 1e6, db(h), color=color, alpha=0.08, linewidth=1.0)

        h_nom = zpk_freq_response(filters[name], freq_hz)
        ax.plot(freq_hz / 1e6, db(h_nom), color="black", linewidth=2.0, label="Nominal")
        ax.axvline(cutoff_hz / 1e6, color="black", linestyle="--", linewidth=1.0)
        ax.set_title(name)
        ax.set_xlabel("Frequency (MHz)")
        ax.grid(True)
        ax.legend()

    axes[0].set_ylabel("Magnitude (dB)")
    axes[0].set_ylim(-75, 8)
    fig.suptitle("Component Sensitivity: L/C Tolerance, ESR Zero, Parasitic Resonance")
    fig.tight_layout()
    path = output_dir / "paper_component_sensitivity.png"
    fig.savefig(path)
    return path


def main():
    args = parse_args()
    configure_plot_style()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    order = 8
    cutoff_hz = 5e6
    ripple_db = 2.0
    filters = make_filters(order, cutoff_hz, ripple_db)

    paths = [
        plot_bode(filters, cutoff_hz, ripple_db, output_dir),
        plot_transition_zoom(filters, cutoff_hz, ripple_db, output_dir),
        *plot_impulse_step(filters, cutoff_hz, output_dir),
        plot_load_transient(filters, output_dir),
        plot_switching_noise_and_sca(filters, output_dir),
        plot_component_sensitivity(filters, cutoff_hz, ripple_db, order, output_dir),
    ]

    if not args.no_show:
        plt.show()

    for path in paths:
        print(f"Saved {path}")


if __name__ == "__main__":
    main()
