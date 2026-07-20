# MoTE ImageNet-R B0-Inc20 Five-Seed Reproduction

## Per-seed results

| Seed | Avg Acc CNN | Last Acc CNN | Forgetting CNN |
|---:|---:|---:|---:|
| 1991 | 81.1090 | 74.8000 | 6.8644 |
| 1992 | 80.8020 | 73.6700 | 7.6767 |
| 1993 | 81.1750 | 74.6000 | 7.3411 |
| 1994 | 81.2150 | 74.1700 | 7.4244 |
| 1995 | 80.0750 | 74.4000 | 7.4911 |

## Mean +/- std

- Average Accuracy CNN: 80.8752 +/- 0.4758
- Last Accuracy CNN: 74.3280 +/- 0.4359
- Forgetting CNN: 7.3596 +/- 0.3031

## Curves

### Seed 1991
- Top1 curve: [91.63, 86.64, 85.5, 84.06, 81.54, 78.21, 76.89, 76.29, 75.53, 74.8]
- Top5 curve: [98.67, 95.96, 94.84, 94.12, 92.72, 91.16, 90.63, 90.14, 89.35, 88.68]

### Seed 1992
- Top1 curve: [91.49, 87.08, 83.67, 81.56, 80.03, 80.17, 78.02, 77.14, 75.19, 73.67]
- Top5 curve: [98.07, 96.81, 94.63, 93.44, 91.96, 91.52, 90.57, 90.29, 89.4, 88.75]

### Seed 1993
- Top1 curve: [92.89, 86.98, 85.0, 82.83, 80.37, 78.96, 77.76, 76.84, 75.52, 74.6]
- Top5 curve: [98.4, 96.59, 95.07, 93.07, 91.59, 91.17, 90.51, 89.76, 88.88, 88.33]

### Seed 1994
- Top1 curve: [92.38, 88.3, 84.85, 82.69, 80.72, 79.15, 78.2, 76.73, 74.96, 74.17]
- Top5 curve: [98.24, 96.05, 94.51, 93.56, 92.8, 91.2, 90.31, 89.89, 88.89, 88.7]

### Seed 1995
- Top1 curve: [91.9, 85.15, 83.47, 80.89, 79.06, 78.78, 76.57, 76.07, 74.46, 74.4]
- Top5 curve: [98.06, 95.73, 94.34, 92.62, 92.04, 91.19, 90.03, 89.72, 88.62, 88.5]

## Paper comparison

| Metric | Reproduction | Reported MoTE | Difference |
|---|---:|---:|---:|
| Average Accuracy CNN | 80.8752 +/- 0.4758 | 81.93 +/- 0.53 | -1.0548 |
| Forgetting CNN | 7.3596 +/- 0.3031 | 6.91 +/- 0.41 | +0.4496 |

The five-seed run is complete and stable. It is a valid paper-level reproduction run, but the remaining systematic metric gap must be retained in later analysis rather than described as an exact numerical match.

## Reproduction evidence and limitations

- Official MoTE source commit: `93d20b3`.
- Model: MoTE with `vit_base_patch16_224_in21k_mote`.
- Setting: B0-Inc20, seeds 1991-1995, 20 epochs per task.
- Initial total parameters: 86,988,288.
- Trainable parameters reported during training: 1,189,632.
- Hardware: two NVIDIA GeForce RTX 5090 GPUs; each seed used one GPU.
- Runtime: approximately 30.5 minutes per seed.
- GPU-memory observation: approximately 2636 MiB observed during a seed-1993 snapshot; this was not instrumented peak memory.
- Raw logs, actual configs, compatibility patches, environment details, and parser are included in this directory.
- Formal peak-memory instrumentation, inference timing, adapter-growth accounting, and prototype-memory accounting remain separate checklist items.

- Seed 1995 ran manually on GPU 0 while seed 1994 ran on GPU 1. Its stdout capture was accidentally truncated after successful completion; the archived seed-1995 log was restored from MoTE's complete `FileHandler` log and contains all final curves and metrics.
