# IoT ECG Research Framework (Stage 2)

This directory is the reproducible research layer for the historical IoT-Based Remote Heart Rate Monitoring System.

## Research rule

The original project is treated as archival prototype evidence. New performance claims come from reproducible experiments against annotated public ECG databases. No missing hardware measurements are fabricated.

## Current implementation

- project-inspired fixed-threshold beat detector;
- lightweight adaptive-energy detector v0 (research starting point, **not yet claimed as novel**);
- one-to-one R-peak matching and Sensitivity/PPV/F1/DER metrics;
- timing-error metrics;
- anti-aliased sampling-rate transformation;
- uniform ADC-resolution simulation;
- PhysioNet downloader;
- MIT-BIH benchmark runner;
- synthetic sanity tests.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install numpy scipy pandas wfdb
```

## Sanity test

```bash
cd research
python sanity_check.py
```

## Download first MIT-BIH records

```bash
cd research
python fetch_physionet.py --database mitdb --records 100 101 103 105
```

Datasets belong under `research/data/` and should not be committed.

## Run baseline benchmark

```bash
cd research
python benchmark_detection.py --records 100 101 103 105
```

The first record is currently used only to establish the frozen project-inspired fixed threshold. The final study will use an explicit development/test split at the subject/record level.

## Next experiments

1. Validate the baseline on a defined MIT-BIH development/test split.
2. Implement Pan-Tompkins and Hamilton baselines.
3. Freeze AdaptiveEnergyDetector parameters on development records.
4. Evaluate held-out MIT-BIH records.
5. Run sampling-rate and quantization ablations.
6. Run MIT-BIH Noise Stress Test experiments.
7. Add event-driven telemetry simulation.
