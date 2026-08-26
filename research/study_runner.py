#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
import numpy as np
import pandas as pd

from baselines import PanTompkinsStyleDetector
from research_ecg import (
    AdaptiveEnergyDetector,
    FixedThresholdDetector,
    detection_metrics,
    load_wfdb_record,
    quantize_uniform,
    resample_ecg,
)
from study_utils import paired_hr_metrics, rescale_peak_indices

DS1 = [
    "101", "106", "108", "109", "112", "114", "115", "116", "118", "119", "122", "124",
    "201", "203", "205", "207", "208", "209", "215", "220", "223", "230",
]
DS2 = [
    "100", "103", "105", "111", "113", "117", "121", "123", "200", "202", "210", "212",
    "213", "214", "219", "221", "222", "228", "231", "232", "233", "234",
]
BEAT_SYMBOLS = {
    "N", "L", "R", "B", "A", "a", "J", "S", "V", "r", "F",
    "e", "j", "n", "E", "/", "f", "Q", "?",
}


def beat_annotations(samples, symbols):
    keep = np.array([symbol in BEAT_SYMBOLS for symbol in symbols], dtype=bool)
    return np.asarray(samples, dtype=int)[keep]


def load_records(root: Path, records: list[str]):
    loaded = []
    for record_id in records:
        item = load_wfdb_record(root / record_id, channel=0)
        item["record"] = record_id
        item["beats"] = beat_annotations(item["reference_peaks"], item["symbols"])
        loaded.append(item)
    return loaded


def frozen_threshold(dev_records, percentile: float = 99.0):
    """One global fixed threshold calibrated from development records only."""
    record_thresholds = [
        float(np.percentile(item["signal"], percentile)) for item in dev_records
    ]
    return float(np.median(record_thresholds))


def evaluate(detector, signal, references, fs, tolerance_ms):
    detected = detector.detect(signal, fs)
    return {
        **detection_metrics(references, detected, fs, tolerance_ms),
        **paired_hr_metrics(references, detected, fs, tolerance_ms),
        "detected_beats": int(len(detected)),
        "reference_beats": int(len(references)),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", default="data/physionet/mitdb")
    parser.add_argument(
        "--mode", choices=["baseline", "sampling", "quantization"], default="baseline"
    )
    parser.add_argument("--test-records", nargs="*", default=None)
    parser.add_argument("--dev-records", nargs="*", default=None)
    parser.add_argument("--tolerance-ms", type=float, default=150.0)
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    root = Path(args.data_root)
    development = load_records(root, args.dev_records or DS1)
    test = load_records(root, args.test_records or DS2)
    threshold = frozen_threshold(development)

    detectors = {
        "fixed_threshold": FixedThresholdDetector(threshold),
        "pan_tompkins_style": PanTompkinsStyleDetector(),
        "adaptive_energy_v0": AdaptiveEnergyDetector(),
    }

    sampling_rates = [360, 250, 200, 125, 100, 50] if args.mode == "sampling" else [None]
    bit_depths = [16, 12, 10, 8] if args.mode == "quantization" else [None]
    rows = []

    for item in test:
        for fs_target in sampling_rates:
            if fs_target is None or fs_target == int(round(item["fs"])):
                signal = item["signal"]
                references = item["beats"]
                fs = item["fs"]
            else:
                signal = resample_ecg(
                    item["signal"], int(round(item["fs"])), fs_target
                )
                references = rescale_peak_indices(
                    item["beats"], item["fs"], fs_target
                )
                fs = float(fs_target)

            for bits in bit_depths:
                evaluated_signal = (
                    quantize_uniform(signal, bits)
                    if bits is not None and bits < 16
                    else signal
                )
                for name, detector in detectors.items():
                    metrics = evaluate(
                        detector,
                        evaluated_signal,
                        references,
                        fs,
                        args.tolerance_ms,
                    )
                    rows.append(
                        {
                            "record": item["record"],
                            "detector": name,
                            "mode": args.mode,
                            "sampling_hz": fs,
                            "bits": bits if bits is not None else np.nan,
                            "frozen_fixed_threshold": (
                                threshold if name == "fixed_threshold" else np.nan
                            ),
                            **metrics,
                        }
                    )

    frame = pd.DataFrame(rows)
    output = Path(args.output or f"results/{args.mode}_results.csv")
    output.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output, index=False)
    summary = frame.groupby(["detector", "sampling_hz"], dropna=False)[
        ["sensitivity", "ppv", "f1", "hr_mae_bpm"]
    ].median()
    print(summary.to_string())
    print(f"\nSaved {output}")


if __name__ == "__main__":
    main()
