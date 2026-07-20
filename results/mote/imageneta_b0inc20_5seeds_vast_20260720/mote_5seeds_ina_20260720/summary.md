# MoTE ImageNet-A B0-Inc20 Five-Seed Reproduction

## Per-seed results

| Seed | Avg Acc CNN | Last Acc CNN | Forgetting CNN |
|---:|---:|---:|---:|
| 1991 | 65.3690 | 55.5000 | 14.9433 |
| 1992 | 65.1330 | 55.1700 | 12.3556 |
| 1993 | 66.4910 | 54.7100 | 12.2278 |
| 1994 | 65.1260 | 53.9200 | 9.4222 |
| 1995 | 63.5560 | 54.5100 | 11.1078 |

## Mean +/- std

- Average Accuracy CNN: 65.1350 +/- 1.0472
- Last Accuracy CNN: 54.7620 +/- 0.6096
- Forgetting CNN: 12.0113 +/- 2.0175

## Curves

### Seed 1991
- Top1 curve: [82.14, 74.71, 71.36, 68.35, 65.99, 61.77, 59.96, 57.55, 56.36, 55.5]
- Top5 curve: [96.43, 93.1, 90.93, 89.06, 86.42, 84.01, 82.3, 79.72, 79.39, 78.54]

### Seed 1992
- Top1 curve: [84.62, 73.67, 69.82, 64.53, 63.48, 63.27, 62.09, 58.47, 56.21, 55.17]
- Top5 curve: [98.72, 95.02, 91.3, 86.24, 85.81, 85.51, 84.67, 82.65, 80.26, 79.66]

### Seed 1993
- Top1 curve: [82.86, 77.22, 73.95, 70.63, 67.3, 64.1, 60.57, 58.01, 55.56, 54.71]
- Top5 curve: [98.29, 94.17, 92.86, 90.83, 87.51, 86.26, 83.68, 81.97, 80.22, 79.0]

### Seed 1994
- Top1 curve: [80.0, 72.56, 72.28, 70.5, 66.19, 62.08, 59.85, 58.29, 55.59, 53.92]
- Top5 curve: [97.14, 93.98, 92.9, 90.6, 88.52, 85.03, 82.21, 81.28, 79.0, 77.62]

### Seed 1995
- Top1 curve: [80.59, 72.12, 67.49, 63.82, 61.12, 61.51, 60.39, 58.21, 55.8, 54.51]
- Top5 curve: [97.65, 93.27, 86.68, 85.32, 84.0, 83.58, 83.41, 81.81, 79.84, 77.68]

## Paper comparison

| Metric | Reproduction | Reported MoTE | Difference |
|---|---:|---:|---:|
| Average Accuracy CNN | 65.1350 +/- 1.0472 | 67.26 +/- 0.96 | -2.1250 |
| Forgetting CNN | 12.0113 +/- 2.0175 | 11.19 +/- 1.69 | +0.8213 |

The five-seed run is complete and stable. It is a valid paper-level reproduction run, but the remaining systematic metric gap must be retained in later analysis rather than described as an exact numerical match.

## Reproduction evidence and limitations

- Official MoTE source commit: `93d20b3`.
- Model: MoTE with `vit_base_patch16_224_in21k_mote`.
- Setting: B0-Inc20, seeds 1991-1995, 20 epochs per task.
- Initial total parameters: 86,988,288.
- Trainable parameters reported during training: 1,189,632.
- Hardware: two NVIDIA GeForce RTX 5090 GPUs; each seed used one GPU.
- Runtime: approximately 8 minutes per seed.
- GPU-memory observation: approximately 4154 MiB observed during a seed-1993 snapshot; this was not instrumented peak memory.
- Raw logs, actual configs, compatibility patches, environment details, and parser are included in this directory.
- Formal peak-memory instrumentation, inference timing, adapter-growth accounting, and prototype-memory accounting remain separate checklist items.
