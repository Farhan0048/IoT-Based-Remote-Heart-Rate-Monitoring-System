from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path
import numpy as np
from scipy.signal import butter, sosfiltfilt, find_peaks, resample_poly


@dataclass
class FixedThresholdDetector:
    """Project-inspired rising-threshold detector.

    The historical Processing implementation used a literal ADC threshold of 620.
    That value is meaningful only on the original 0..1023 ADC scale. For external
    ECG datasets this detector should use one threshold calibrated on development
    data and frozen before test evaluation.
    """

    threshold: float
    refractory_s: float = 0.20

    def detect(self, signal: np.ndarray, fs: float) -> np.ndarray:
        x = np.asarray(signal, dtype=float)
        if x.ndim != 1:
            raise ValueError("signal must be one-dimensional")
        above = x > self.threshold
        crossings = np.flatnonzero(above[1:] & ~above[:-1]) + 1
        if crossings.size == 0:
            return crossings.astype(int)
        refractory = max(1, int(round(self.refractory_s * fs)))
        kept = [int(crossings[0])]
        for idx in crossings[1:]:
            if int(idx) - kept[-1] >= refractory:
                kept.append(int(idx))
        return np.asarray(kept, dtype=int)


@dataclass
class AdaptiveEnergyDetector:
    """Initial lightweight adaptive QRS detector research prototype.

    This is a reproducible Stage-2 starting point, not yet a novelty claim.
    """

    low_hz: float = 5.0
    high_hz: float = 18.0
    integration_ms: float = 120.0
    refractory_ms: float = 250.0
    threshold_k: float = 3.0
    refinement_ms: float = 80.0

    def _bandpass(self, x: np.ndarray, fs: float) -> np.ndarray:
        nyq = fs / 2.0
        low = max(0.5, min(self.low_hz, 0.45 * fs))
        high = min(self.high_hz, 0.90 * nyq)
        if high <= low:
            raise ValueError(f"sampling rate {fs} Hz is too low for configured bandpass")
        sos = butter(2, [low / nyq, high / nyq], btype="bandpass", output="sos")
        return sosfiltfilt(sos, x)

    def detect(self, signal: np.ndarray, fs: float) -> np.ndarray:
        x = np.asarray(signal, dtype=float)
        if x.ndim != 1:
            raise ValueError("signal must be one-dimensional")
        if x.size < max(32, int(fs)):
            return np.array([], dtype=int)

        y = self._bandpass(x, fs)
        d = np.diff(y, prepend=y[0])
        energy = d * d
        win = max(1, int(round(self.integration_ms * fs / 1000.0)))
        integ = np.convolve(energy, np.ones(win) / win, mode="same")

        median = float(np.median(integ))
        mad = float(np.median(np.abs(integ - median)))
        robust_sigma = 1.4826 * mad + np.finfo(float).eps
        threshold = median + self.threshold_k * robust_sigma
        distance = max(1, int(round(self.refractory_ms * fs / 1000.0)))
        candidates, _ = find_peaks(integ, height=threshold, distance=distance)

        refine = max(1, int(round(self.refinement_ms * fs / 1000.0)))
        refined: list[int] = []
        for c in candidates:
            lo = max(0, int(c) - refine)
            hi = min(x.size, int(c) + refine + 1)
            local = x[lo:hi]
            if local.size == 0:
                continue
            baseline = np.median(local)
            r = lo + int(np.argmax(np.abs(local - baseline)))
            if not refined or r - refined[-1] >= distance:
                refined.append(r)
            elif abs(x[r] - baseline) > abs(x[refined[-1]] - baseline):
                refined[-1] = r
        return np.asarray(refined, dtype=int)


def match_peaks(reference, detected, fs: float, tolerance_ms: float = 150.0):
    """One-to-one greedy matching of detected peaks to reference annotations."""
    ref = np.sort(np.asarray(reference, dtype=int))
    det = np.sort(np.asarray(detected, dtype=int))
    tol = int(round(tolerance_ms * fs / 1000.0))

    i = j = 0
    matched_ref, matched_det, errors = [], [], []
    while i < len(ref) and j < len(det):
        delta = int(det[j]) - int(ref[i])
        if abs(delta) <= tol:
            matched_ref.append(int(ref[i]))
            matched_det.append(int(det[j]))
            errors.append(delta)
            i += 1
            j += 1
        elif det[j] < ref[i] - tol:
            j += 1
        else:
            i += 1

    tp = len(matched_ref)
    return {
        "tp": tp,
        "fp": len(det) - tp,
        "fn": len(ref) - tp,
        "matched_reference": np.asarray(matched_ref, dtype=int),
        "matched_detected": np.asarray(matched_det, dtype=int),
        "timing_error_samples": np.asarray(errors, dtype=int),
    }


def detection_metrics(reference, detected, fs: float, tolerance_ms: float = 150.0):
    m = match_peaks(reference, detected, fs=fs, tolerance_ms=tolerance_ms)
    tp, fp, fn = m["tp"], m["fp"], m["fn"]
    sensitivity = tp / (tp + fn) if tp + fn else np.nan
    ppv = tp / (tp + fp) if tp + fp else np.nan
    f1 = 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else np.nan
    der = (fp + fn) / (tp + fn) if tp + fn else np.nan
    errors_ms = m["timing_error_samples"] * 1000.0 / fs
    return {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "sensitivity": sensitivity,
        "ppv": ppv,
        "f1": f1,
        "der": der,
        "median_abs_timing_error_ms": float(np.median(np.abs(errors_ms))) if errors_ms.size else np.nan,
        "p95_abs_timing_error_ms": float(np.percentile(np.abs(errors_ms), 95)) if errors_ms.size else np.nan,
    }


def resample_ecg(x, fs_in: int, fs_out: int):
    """Polyphase resampling with anti-alias filtering."""
    frac = Fraction(fs_out, fs_in).limit_denominator()
    return resample_poly(np.asarray(x, dtype=float), frac.numerator, frac.denominator)


def quantize_uniform(x, bits: int):
    """Uniformly quantize while retaining the original numeric scale."""
    if bits < 2:
        raise ValueError("bits must be >=2")
    x = np.asarray(x, dtype=float)
    lo, hi = float(np.min(x)), float(np.max(x))
    if hi == lo:
        return x.copy()
    levels = 2**bits - 1
    q = np.round((x - lo) / (hi - lo) * levels) / levels
    return lo + q * (hi - lo)


def load_wfdb_record(record_path: str | Path, channel: int = 0):
    """Load local PhysioNet/WFDB data using the optional `wfdb` package."""
    try:
        import wfdb
    except ImportError as exc:
        raise RuntimeError(
            "The optional 'wfdb' package is required to read PhysioNet .dat/.atr files. "
            "Install it with: pip install wfdb"
        ) from exc

    p = Path(record_path)
    rec = wfdb.rdrecord(str(p), physical=True)
    ann = wfdb.rdann(str(p), "atr")
    if channel >= rec.p_signal.shape[1]:
        raise ValueError(f"channel {channel} unavailable; record has {rec.p_signal.shape[1]} channels")
    return {
        "signal": np.asarray(rec.p_signal[:, channel], dtype=float),
        "fs": float(rec.fs),
        "reference_peaks": np.asarray(ann.sample, dtype=int),
        "symbols": np.asarray(ann.symbol),
        "signal_name": rec.sig_name[channel] if rec.sig_name else f"ch{channel}",
    }
