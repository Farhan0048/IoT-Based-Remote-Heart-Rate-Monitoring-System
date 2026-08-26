# IoT ECG Research Framework (Stage 2)

This directory is the reproducible research layer for the historical **IoT-Based Remote Heart Rate Monitoring System**.

## Research rule

The original project is treated as archival prototype evidence. New performance claims must come from reproducible experiments against annotated public ECG databases. Missing hardware measurements must not be fabricated or inferred from software timing.

## Current implementation

- project-inspired fixed-threshold beat detector;
- lightweight adaptive-energy detector v0 (**research starting point, not yet claimed as novel**);
- transparent Pan-Tompkins-style baseline;
- one-to-one R-peak matching;
- Sensitivity, PPV, F1 and detection-error-rate metrics;
- R-peak timing-error metrics;
- beat-to-beat heart-rate MAE/RMSE/MAPE;
- anti-aliased sampling-rate transformation;
- uniform ADC-resolution simulation;
- communication-cost model for continuous, feature-only and event-driven telemetry;
- PhysioNet downloader;
- DS1-development / DS2-held-out MIT-BIH experiment runner;
- synthetic sanity tests.

## Environment

```bash
cd research
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Sanity test

```bash
python sanity_check.py
```

The sanity test is only a software-integrity check. Synthetic performance is **not** a paper result.

## Download MIT-BIH

```bash
python fetch_physionet.py --database mitdb --all-records
```

Public datasets belong under `research/data/` and should not be committed.

## Experimental split

The main MIT-BIH experiment uses the established inter-patient split encoded in `config_stage2.yaml` and `study_runner.py`:

- **DS1:** development/calibration only;
- **DS2:** held-out final testing only.

The project-inspired fixed threshold is calibrated once from DS1 and frozen before DS2 evaluation. No per-record test-set tuning is permitted.

## Run experiments

Baseline detector comparison:

```bash
python study_runner.py --mode baseline
```

Sampling-rate study:

```bash
python study_runner.py --mode sampling
```

Quantization study:

```bash
python study_runner.py --mode quantization
```

For a small smoke test, `--dev-records` and `--test-records` can override the full split. Those overrides must not be used for final reported results unless documented in the manuscript.

## Planned next work

1. Obtain and execute the official MIT-BIH records in an environment with working PhysioNet/WFDB access.
2. Freeze adaptive-detector parameters using DS1 only.
3. Run DS2 baseline, sampling and quantization studies.
4. Add MIT-BIH Noise Stress Test evaluation.
5. Add independent INCART validation.
6. Finalize signal-quality index and event-trigger policy.
7. Generate publication figures and statistical summaries.

## Integrity boundary

The historical volunteer measurements remain preliminary feasibility evidence. The new software benchmark does not create new claims about physical ESP8266 power, battery life, wireless latency, electrode behavior, or clinical diagnostic performance.
