import numpy as np

from research_ecg import (
    AdaptiveEnergyDetector,
    FixedThresholdDetector,
    detection_metrics,
    quantize_uniform,
    resample_ecg,
)


def synthetic_ecg(fs=250, seconds=10):
    n = fs * seconds
    t = np.arange(n) / fs
    x = 0.03 * np.sin(2 * np.pi * 0.3 * t)
    peaks = np.arange(fs, n, fs)
    width = max(1, int(0.015 * fs))
    for p in peaks:
        idx = np.arange(max(0, p - 4 * width), min(n, p + 4 * width))
        x[idx] += np.exp(-0.5 * ((idx - p) / width) ** 2)
    return x, peaks


x, ref = synthetic_ecg()
fixed = FixedThresholdDetector(0.5).detect(x, 250)
adaptive = AdaptiveEnergyDetector(threshold_k=2.0).detect(x, 250)
fm = detection_metrics(ref, fixed, 250, 100)
am = detection_metrics(ref, adaptive, 250, 120)
assert fm["f1"] > 0.99, fm
assert am["f1"] > 0.95, am
assert abs(len(resample_ecg(x, 250, 100)) - 1000) <= 1
assert len(np.unique(quantize_uniform(x, 10))) <= 1024
print("PASS", {"fixed_f1": fm["f1"], "adaptive_f1": am["f1"]})
