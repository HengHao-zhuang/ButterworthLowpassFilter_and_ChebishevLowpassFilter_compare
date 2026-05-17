# Butterworth vs Chebyshev LC Low-Pass Filters for Power SCA Mitigation

This repo compares aggressive non-ideal Butterworth and Chebyshev Type I LC
low-pass power-filter models as hardware countermeasures against power
side-channel attacks.

## Setup

```bash
python3 -m pip install -r requirements.txt
```

## Run

```bash
python3 compare.py
```

For saving the figures without opening plot windows:

```bash
python3 compare.py --no-show
```

The script generates:

- `plots/paper_bode_group_delay.png`
- `plots/paper_transition_zoom.png`
- `plots/paper_impulse_response.png`
- `plots/paper_step_ringing.png`
- `plots/paper_load_transient.png`
- `plots/paper_switching_noise_sca_trace.png`
- `plots/paper_component_sensitivity.png`

See `REPORT.md` for the engineering discussion, comparison table, and STM32F4
power-filtering implications.
