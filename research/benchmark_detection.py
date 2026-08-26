#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
import numpy as np
import pandas as pd

from research_ecg import (
    AdaptiveEnergyDetector,
    FixedThresholdDetector,
    detection_metrics,
    load_wfdb_record,
)

BEAT_SYMBOLS = set("NLRBAaJSEjVFe/Qf") | {"!"}


def physiological_beat_annotations(samples, symbols):
    keep = [i for i, s in enumerate(symbols) if s in BEAT_SYMBOLS]
    return samples[np.asarray(keep, dtype=int)]


def frozen_threshold_from_development_signal(x, percentile=99.0):
    """Calibrate once on development data; never per test record."""
    return float(np.percentile(np.asarray(x, dtype=float), percentile))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", default="data/physionet/mitdb")
    ap.add_argument("--records", nargs="+", default=["100"])
    ap.add_argument("--fixed-threshold", type=float, default=None)
    ap.add_argument("--tolerance-ms", type=float, default=150.0)
    ap.add_argument("--output", default="results/detection_baseline.csv")
    args = ap.parse_args()

    loaded = [
        load_wfdb_record(Path(args.data_root) / r, channel=0) | {"record": r}
        for r in args.records
    ]

    threshold = args.fixed_threshold
    if threshold is None:
        threshold = frozen_threshold_from_development_signal(loaded[0]["signal"])
        print(
            f"Development threshold frozen at {threshold:.6f} "
            f"from record {loaded[0]['record']}"
        )

    detectors = {
        "fixed_threshold": FixedThresholdDetector(threshold=threshold),
        "adaptive_energy_v0": AdaptiveEnergyDetector(),
    }

    rows = []
    for item in loaded:
        refs = physiological_beat_annotations(
            item["reference_peaks"], item["symbols"]
        )
        for name, detector in detectors.items():
            detected = detector.detect(item["signal"], item["fs"])
            m = detection_metrics(
                refs,
                detected,
                fs=item["fs"],
                tolerance_ms=args.tolerance_ms,
            )
            rows.append(
                {
                    "record": item["record"],
                    "detector": name,
                    "threshold": threshold if name == "fixed_threshold" else np.nan,
                    **m,
                }
            )

    df = pd.DataFrame(rows)
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    print(df.to_string(index=False))
    print(f"\nSaved {out}")


if __name__ == "__main__":
    main()
