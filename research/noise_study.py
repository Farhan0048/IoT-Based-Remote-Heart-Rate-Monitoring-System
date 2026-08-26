#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
import numpy as np
import pandas as pd

from baselines import PanTompkinsStyleDetector
from research_ecg import AdaptiveEnergyDetector, detection_metrics, load_wfdb_record
from study_runner import beat_annotations
from study_utils import paired_hr_metrics

RECORDS = {
    "118e24": 24,
    "118e18": 18,
    "118e12": 12,
    "118e06": 6,
    "118e00": 0,
    "118e_6": -6,
    "119e24": 24,
    "119e18": 18,
    "119e12": 12,
    "119e06": 6,
    "119e00": 0,
    "119e_6": -6,
}


def in_noisy_intervals(samples, fs: float, duration_samples: int):
    """Mask samples inside NSTDB's alternating noisy segments.

    Noise starts after minute 5 and alternates 2 minutes noisy / 2 minutes clean
    until the end of the 30-minute record.
    """
    sample_array = np.asarray(samples, dtype=int)
    seconds = sample_array / float(fs)
    duration_s = duration_samples / float(fs)
    mask = np.zeros(len(sample_array), dtype=bool)
    start = 5.0 * 60.0
    while start < duration_s:
        stop = min(start + 2.0 * 60.0, duration_s)
        mask |= (seconds >= start) & (seconds < stop)
        start += 4.0 * 60.0
    return mask


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", default="data/physionet/nstdb")
    parser.add_argument("--tolerance-ms", type=float, default=150.0)
    parser.add_argument("--output", default="results/noise_stress_results.csv")
    args = parser.parse_args()

    detectors = {
        "pan_tompkins_style": PanTompkinsStyleDetector(),
        "adaptive_energy_v0": AdaptiveEnergyDetector(),
    }
    rows = []

    for record_id, snr_db in RECORDS.items():
        item = load_wfdb_record(Path(args.data_root) / record_id, channel=0)
        references = beat_annotations(item["reference_peaks"], item["symbols"])
        ref_mask = in_noisy_intervals(references, item["fs"], len(item["signal"]))
        noisy_references = references[ref_mask]

        for name, detector in detectors.items():
            detected = detector.detect(item["signal"], item["fs"])
            det_mask = in_noisy_intervals(detected, item["fs"], len(item["signal"]))
            noisy_detected = detected[det_mask]
            metrics = detection_metrics(
                noisy_references,
                noisy_detected,
                item["fs"],
                args.tolerance_ms,
            )
            hr = paired_hr_metrics(
                noisy_references,
                noisy_detected,
                item["fs"],
                args.tolerance_ms,
            )
            rows.append(
                {
                    "record": record_id,
                    "source_record": record_id[:3],
                    "snr_db": snr_db,
                    "detector": name,
                    "reference_beats_noisy_segments": int(len(noisy_references)),
                    "detected_beats_noisy_segments": int(len(noisy_detected)),
                    **metrics,
                    **hr,
                }
            )

    frame = pd.DataFrame(rows)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output, index=False)
    summary = frame.groupby(["detector", "snr_db"])[
        ["sensitivity", "ppv", "f1", "hr_mae_bpm"]
    ].median()
    print(summary.to_string())
    print(f"\nSaved {output}")


if __name__ == "__main__":
    main()
