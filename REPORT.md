# Butterworth and Chebyshev LC Low-Pass Filters for Power SCA Mitigation

This note compares Butterworth and Chebyshev Type I LC low-pass filters used on
the supply path of an MCU as hardware countermeasures against power side-channel
analysis (SCA). The Python simulation intentionally uses aggressive parameters
to make practical differences visible: 8th-order filters, 2 dB Chebyshev ripple,
capacitor ESR/ESL effects, non-ideal inductor/load resonance, pulsed digital
loads, fast leakage bursts, and injected switching-regulator noise.

The model is still a qualitative research model rather than a replacement for
SPICE. A final PCB design needs real capacitor ESR/ESL, inductor DCR/saturation,
regulator impedance, package parasitics, and board layout.

## Comparison Table

| Topic | Butterworth LC low-pass | Chebyshev Type I LC low-pass |
|---|---|---|
| Attenuation near cutoff | Smooth roll-off; less aggressive transition band | Much sharper transition for the same order |
| Stopband suppression | Good, but usually needs higher order for very steep suppression | Stronger high-frequency suppression at the same order |
| Passband ripple | Maximally flat magnitude; no intentional passband ripple | Designed passband ripple; simulation uses 2 dB to emphasize the effect |
| Transient response | Better damping, less overshoot and ringing | More overshoot and ringing due to higher-Q sections and parasitic resonance |
| Phase distortion | Nonlinear phase, but usually smoother than Chebyshev | Larger phase and group-delay variation near cutoff |
| Power trace smoothing | Smooths high-frequency switching leakage while preserving DC stability | Strong smoothing above cutoff, but ripple/ringing can create new trace artifacts |
| Deep-learning SCA effect | Reduces high-frequency features and lowers SNR; may force models toward slower leakage | Can suppress more high-frequency leakage, but ringing may produce learnable filter-shaped features |
| Signal integrity | Generally safer for supply integrity | Higher risk of supply ripple, load-step overshoot, and resonance |
| MCU operation stability | Usually easier to stabilize across PVT and load changes | Needs more damping and tolerance control |
| PCB implementation risk | Moderate | Higher |

## Frequency-Domain Analysis

For an order-N low-pass filter, both filters eventually roll off at about
`20N dB/decade`. The practical difference is near the cutoff frequency.

Butterworth is maximally flat in the passband. Its magnitude response has no
intentional ripple, so the MCU receives a cleaner low-frequency supply profile.
This is why Butterworth-like power filters are common in power integrity work:
they trade transition sharpness for predictable supply behavior.

Chebyshev Type I places controlled ripple in the passband to obtain a steeper
transition band. For SCA mitigation, this is attractive because high-frequency
components from digital switching can be attenuated more strongly without
increasing filter order. The cost is that the ripple exists exactly in the band
that may include real load-current variation and regulator-control dynamics.

The updated plots explicitly zoom the transition region and include group delay:

- `plots/paper_bode_group_delay.png`
- `plots/paper_transition_zoom.png`
- `plots/paper_component_sensitivity.png`

## Time-Domain Analysis

Power SCA attacks exploit correlations between processed data and measured
current or voltage traces. A supply low-pass filter can reduce trace bandwidth,
blur short switching events, and lower the amplitude of fast data-dependent
components.

Butterworth filters usually give a cleaner time-domain response. Step response
overshoot is lower and settling is more predictable. In power delivery, this is
important because the MCU does not draw a perfectly sinusoidal current; it draws
bursty current from clocked logic, memory, GPIO, buses, and peripherals.

Chebyshev filters generally produce stronger smoothing of fast edges, but their
higher-Q behavior causes more ringing after load steps. In an SCA context, this
can be double-edged: high-frequency leakage is reduced, but the ringing may
spread one sensitive event into a longer waveform. A deep-learning model may
learn those repeated impulse-response shapes if they remain data-correlated.

The updated transient plots include impulse response, step response, load
transient response, and synthetic side-channel trace filtering:

- `plots/paper_impulse_response.png`
- `plots/paper_step_ringing.png`
- `plots/paper_load_transient.png`
- `plots/paper_switching_noise_sca_trace.png`

## Influence on Deep-Learning-Based SCA

Deep-learning SCA models can exploit features that classical correlation power
analysis may miss. Filtering changes the feature space rather than eliminating
all leakage.

Butterworth filtering tends to remove high-frequency details smoothly. This can
reduce model accuracy when the leakage is carried by sharp transitions, but it
may leave lower-frequency envelope leakage intact. It is usually a robust
defensive layer, not a standalone protection.

Chebyshev filtering can remove more high-frequency information for the same
order, which may hurt convolutional models that rely on fast local features.
However, passband ripple and ringing may introduce deterministic distortion. If
the filtered waveform is consistent across encryptions, a trained model may
adapt to it, especially when the attacker has profiling traces from the same
board.

Filtering should therefore be combined with algorithmic hiding/masking,
randomized timing, balanced firmware activity, careful decoupling, and physical
measurement hardening.

## Signal Integrity and MCU Stability

The supply filter is in the power path, so it must satisfy two goals at once:
reduce externally observable leakage and maintain a low-impedance, stable MCU
supply. These goals can conflict.

Butterworth filters are commonly selected in power integrity because they are
well-behaved: flat passband, moderate Q, less ringing, and less sensitivity to
component tolerance. They are easier to integrate with decoupling capacitors and
voltage regulators.

Chebyshev filters provide stronger high-frequency suppression, but their
intentional passband ripple and higher-Q poles make them more sensitive. Without
damping, an LC network can resonate with the regulator output impedance, MCU
package inductance, vias, planes, and decoupling capacitors. That resonance may
increase supply noise instead of reducing it.

## Practical PCB Implementation Considerations

Place the filter close to the protected power domain, but keep enough local
decoupling directly at every STM32F4 VDD/VDDA pin. The filter should not replace
local decoupling. It should shape the impedance seen from the outside while the
MCU still sees a stable local charge reservoir.

Use a layout with short current loops, low-inductance ground return, and a
continuous ground plane. Avoid routing sensitive measurement points near noisy
clock, GPIO, SWD, USB, or DC/DC switching nodes. If using a series inductor or
ferrite bead, check its impedance at the leakage frequencies of interest, not
only its nominal value at one test frequency.

Add damping when the LC response rings. Common options are capacitor ESR,
small series resistance, RC snubbers, lossy ferrite beads, or intentionally
choosing a less aggressive filter response. Damping reduces peak attenuation but
often improves real security because it avoids creating a clean resonant marker.

## Component Sensitivity

Real LC filters are sensitive to:

- Inductor tolerance and DC bias shift
- Inductor DCR and saturation current
- Capacitor tolerance, voltage coefficient, and temperature coefficient
- Capacitor ESR/ESL and self-resonant frequency
- Load impedance changes caused by MCU clock mode and peripheral activity
- Regulator output impedance and control-loop stability
- PCB parasitic inductance and resistance

Butterworth responses usually degrade gracefully under tolerance spread.
Chebyshev responses can shift ripple peaks and resonant behavior more visibly,
because the design relies on higher-Q pole placement.

## Capacitor and Inductor Selection

For capacitors, use several values in parallel only when their combined
impedance has been checked. A typical MCU network may include local 100 nF MLCCs
near each VDD pin, bulk capacitance nearby, and optional domain-specific filtering
for analog or security-critical rails. Choose voltage rating high enough that
DC-bias capacitance loss is acceptable.

For inductors, check saturation current against worst-case MCU current plus load
steps. DCR creates voltage drop and heat. High-Q inductors can ring strongly; for
power filtering, a lossy ferrite bead or damped inductor network may be more
stable than an ideal high-Q inductor.

The cutoff estimate for a simple second-order LC section is:

```text
fc = 1 / (2*pi*sqrt(L*C))
```

This formula is only a starting point. The final response depends on source
impedance, load impedance, ESR, ESL, and layout.

## STM32F4 Power Filtering Implications

For STM32F4 boards, keep the basic ST-style power integrity discipline:
decouple every VDD pin locally, treat VDDA carefully, and keep the ground return
low impedance. A side-channel filter placed upstream of the MCU supply domain
should not starve the MCU during clock edges, flash accesses, DMA bursts, GPIO
switching, or peripheral startup.

A Butterworth-like damped LC network is usually the safer first choice for the
main digital supply because it reduces high-frequency leakage while preserving
supply stability. A Chebyshev-like network may be considered when stronger
high-frequency attenuation is required, but it should be validated for startup,
reset behavior, brownout margin, PLL/clock stability, ADC noise, and worst-case
load steps.

For VDDA or sensitive analog domains, avoid aggressive ripple. A quieter,
well-damped filter is normally better than a sharper filter that rings.

## Simulation Examples

Run:

```bash
python3 compare.py --no-show
```

Generated plots:

- `plots/paper_bode_group_delay.png`: wideband Bode magnitude, phase, and group delay
- `plots/paper_transition_zoom.png`: transition-region zoom around cutoff
- `plots/paper_impulse_response.png`: impulse response and ringing energy
- `plots/paper_step_ringing.png`: step response, overshoot, and settling
- `plots/paper_load_transient.png`: pulsed digital load response
- `plots/paper_switching_noise_sca_trace.png`: switching-regulator noise plus synthetic SCA leakage
- `plots/paper_component_sensitivity.png`: approximate tolerance, ESR, and parasitic sensitivity

The simulation uses:

- 8th-order Butterworth analog low-pass prototype
- 8th-order Chebyshev Type I analog low-pass prototype
- 5 MHz cutoff frequency
- 2 dB Chebyshev passband ripple
- capacitor ESR zero and ESL/high-frequency pole
- non-ideal load-network resonance representing inductor, capacitor, package,
  via, and regulator parasitics
- fast digital load steps and short data-dependent current bursts
- injected switching-regulator ripple at tens of MHz plus harmonics

## Conclusions for SCA Mitigation

Butterworth LC filters are commonly preferred for power integrity because they
are flat, predictable, and relatively stable under real PCB conditions. They
provide useful SCA bandwidth reduction while minimizing the risk of introducing
new supply artifacts.

Chebyshev LC filters can suppress high-frequency leakage more strongly for the
same order. This can help against attacks that rely on fast switching features,
including deep-learning models trained on high-bandwidth traces. The tradeoff is
passband ripple, phase distortion, ringing, higher component sensitivity, and
greater MCU stability risk.

For STM32F4 SCA mitigation, a practical recommendation is to start with a
well-damped Butterworth-like LC or ferrite-plus-capacitor network, verify supply
integrity on an oscilloscope, then evaluate whether the residual leakage justifies
a sharper Chebyshev-like response. Security should be measured with actual traces
and attacks, because filtering can reshape leakage rather than remove it.
