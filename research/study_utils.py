from __future__ import annotations

import numpy as np
from research_ecg import match_peaks


def rescale_peak_indices(peaks, fs_in: float, fs_out: float):
    """Map annotation indices after resampling."""
    return np.rint(np.asarray(peaks, dtype=float) * fs_out / fs_in).astype(int)


def paired_hr_metrics(reference_peaks, detected_peaks, fs: float, tolerance_ms: float = 150.0):
    """Compute HR error from consecutive one-to-one matched beats."""
    matched = match_peaks(reference_peaks, detected_peaks, fs, tolerance_ms)
    ref = matched["matched_reference"]
    det = matched["matched_detected"]
    if len(ref) < 2:
        return {
            "hr_pairs": 0,
            "hr_mae_bpm": np.nan,
            "hr_rmse_bpm": np.nan,
            "hr_mape_pct": np.nan,
        }

    ref_rr = np.diff(ref) / float(fs)
    det_rr = np.diff(det) / float(fs)
    valid = (ref_rr > 0) & (det_rr > 0)
    ref_hr = 60.0 / ref_rr[valid]
    det_hr = 60.0 / det_rr[valid]
    error = det_hr - ref_hr
    return {
        "hr_pairs": int(len(error)),
        "hr_mae_bpm": float(np.mean(np.abs(error))) if len(error) else np.nan,
        "hr_rmse_bpm": float(np.sqrt(np.mean(error * error))) if len(error) else np.nan,
        "hr_mape_pct": float(100.0 * np.mean(np.abs(error) / ref_hr)) if len(error) else np.nan,
    }


def telemetry_costs(
    duration_s: float,
    fs: float,
    sample_bytes: int = 2,
    raw_packet_samples: int = 25,
    feature_period_s: float = 10.0,
    feature_payload_bytes: int = 24,
    transport_overhead_bytes: int = 50,
    event_times_s=None,
    event_window_s: float = 12.0,
):
    """Estimate communication bytes for continuous, feature-only and adaptive modes.

    Event timestamps represent generic monitoring triggers, not diagnoses.
    Overlapping raw-waveform event windows are merged before byte accounting.
    """
    if duration_s <= 0 or fs <= 0:
        raise ValueError("duration_s and fs must be positive")

    raw_samples = int(np.ceil(duration_s * fs))
    continuous_payload = raw_samples * sample_bytes
    continuous_packets = int(np.ceil(raw_samples / raw_packet_samples))
    continuous_total = continuous_payload + continuous_packets * transport_overhead_bytes

    feature_packets = int(np.ceil(duration_s / feature_period_s))
    feature_payload = feature_packets * feature_payload_bytes
    feature_total = feature_payload + feature_packets * transport_overhead_bytes

    half_window = event_window_s / 2.0
    intervals: list[tuple[float, float]] = []
    for time_s in sorted(float(t) for t in (event_times_s or [])):
        start = max(0.0, time_s - half_window)
        stop = min(duration_s, time_s + half_window)
        if stop <= start:
            continue
        if intervals and start <= intervals[-1][1]:
            intervals[-1] = (intervals[-1][0], max(intervals[-1][1], stop))
        else:
            intervals.append((start, stop))

    raw_event_s = sum(stop - start for start, stop in intervals)
    event_samples = int(np.ceil(raw_event_s * fs))
    event_payload = event_samples * sample_bytes
    event_packets = int(np.ceil(event_samples / raw_packet_samples)) if event_samples else 0
    adaptive_payload = feature_payload + event_payload
    adaptive_packets = feature_packets + event_packets
    adaptive_total = adaptive_payload + adaptive_packets * transport_overhead_bytes

    return {
        "continuous_payload_bytes": continuous_payload,
        "continuous_total_bytes": continuous_total,
        "feature_payload_bytes": feature_payload,
        "feature_total_bytes": feature_total,
        "adaptive_payload_bytes": adaptive_payload,
        "adaptive_total_bytes": adaptive_total,
        "adaptive_raw_seconds": raw_event_s,
        "adaptive_reduction_vs_continuous": 1.0 - adaptive_total / continuous_total,
    }
