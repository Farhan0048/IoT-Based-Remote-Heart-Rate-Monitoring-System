import numpy as np

from baselines import PanTompkinsStyleDetector
from research_ecg import (
    AdaptiveEnergyDetector,
    FixedThresholdDetector,
    detection_metrics,
    quantize_uniform,
    resample_ecg,
)
from study_utils import paired_hr_metrics, telemetry_costs


def synthetic_ecg(fs=250, seconds=20):
    rng = np.random.default_rng(7)
    n = fs * seconds
    t = np.arange(n) / fs
    signal = 0.03 * np.sin(2 * np.pi * 0.3 * t) + 0.005 * rng.standard_normal(n)
    peaks = np.arange(fs, n, fs)
    width = max(1, int(0.015 * fs))
    for peak in peaks:
        indices = np.arange(max(0, peak - 4 * width), min(n, peak + 4 * width))
        signal[indices] += np.exp(-0.5 * ((indices - peak) / width) ** 2)
    return signal, peaks


signal, reference = synthetic_ecg()
detectors = {
    "fixed_threshold": FixedThresholdDetector(0.5),
    "adaptive_energy_v0": AdaptiveEnergyDetector(threshold_k=2.0),
    "pan_tompkins_style": PanTompkinsStyleDetector(),
}

for name, detector in detectors.items():
    detected = detector.detect(signal, 250)
    metrics = detection_metrics(reference, detected, 250, 120)
    print(name, metrics)
    assert metrics["f1"] > 0.95, (name, metrics)

hr = paired_hr_metrics(
    reference,
    detectors["adaptive_energy_v0"].detect(signal, 250),
    250,
    120,
)
assert hr["hr_mae_bpm"] < 1.0, hr
assert abs(len(resample_ecg(signal, 250, 100)) - 2000) <= 1
assert len(np.unique(quantize_uniform(signal, 10))) <= 1024

cost = telemetry_costs(600, 100, event_times_s=[100, 105, 400])
assert 0 < cost["adaptive_reduction_vs_continuous"] < 1
print("PASS telemetry", cost)
