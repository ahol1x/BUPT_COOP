# MoTE Reproduction Summary

Last updated: 2026-07-20

## Project context

This document summarizes the completed and pending MoTE reproduction work for the BUPT_COOP continual-learning project.

The current research goal is to reproduce four Linjie-related PTM-CIL methods, understand how each method modifies and uses the ViT backbone, and then design an adaptive strategy-selection algorithm that chooses the most suitable method for each incoming continual-learning task.

The authoritative execution order is:

1. Finish the complete MoTE paper-level reproduction.
2. Finish a paper-exact LDEPrompt reproduction.
3. Reproduce QKD.
4. Reproduce official TaE.
5. Compare the four codebases and design the adaptive model-selection algorithm.

C3Box and LAMDA-PILOT are auxiliary environment/testing tools, not primary research targets.

## Current reproduced method

- Method: MoTE, Mixture of Task-Specific Experts
- Backbone: ViT-B/16-based pretrained transformer
- Setting: exemplar-free class-incremental learning
- Completed main-result seeds: 1991, 1992, 1993, 1994, 1995
- Metrics saved: Average Accuracy, Last Accuracy, Forgetting, Top-1 curve, Top-5 curve
- Additional records: configs, logs, patches, scripts, dataset notes, and environment information

## Completed reproduction results

| Method | Dataset | Split | Seeds | Avg Acc | Last Acc | Forgetting | Commit |
|---|---|---|---|---:|---:|---:|---|
| MoTE | CIFAR100 / cifar224 | B0-Inc10 | 1991-1995 | 92.7050 +/- 0.2010 | 88.5540 +/- 0.2061 | 4.9800 +/- 0.3175 | becebdf |
| MoTE | CUB-200-2011 | B0-Inc20 | 1991-1995 | 91.3962 +/- 0.4214 | 87.1720 +/- 0.0879 | 4.6464 +/- 0.5002 | c3819e1 |
| MoTE | ImageNet-A | B0-Inc20 | 1991-1995 | 65.1350 +/- 1.0472 | 54.7620 +/- 0.6096 | 12.0113 +/- 2.0175 | this update |
| MoTE | ImageNet-R | B0-Inc20 | 1991-1995 | 80.8752 +/- 0.4758 | 74.3280 +/- 0.4359 | 7.3596 +/- 0.3031 | this update |
| MoTE | VTAB processed 50-class CIL | B0-Inc10 | 1991-1995 | 90.3576 +/- 1.8217 | 82.4040 +/- 1.1545 | 7.2710 +/- 1.6902 | 3f691e2 |

## Detailed result locations

### CIFAR100 B0-Inc10

Result directory:

results/mote/cifar224_b0inc10_5seeds_vast_20260630/mote_5seeds_cifar_20260630

- Average Accuracy CNN: 92.7050 +/- 0.2010
- Last Accuracy CNN: 88.5540 +/- 0.2061
- Forgetting CNN: 4.9800 +/- 0.3175

CIFAR100 is the stable baseline condition. The result is close to the reported MoTE value and confirms that the setup is paper-comparable.

### CUB200 B0-Inc20

Result directory:

results/mote/cub_b0inc20_5seeds_vast_20260702/mote_5seeds_cub_20260702

- Average Accuracy CNN: 91.3962 +/- 0.4214
- Last Accuracy CNN: 87.1720 +/- 0.0879
- Forgetting CNN: 4.6464 +/- 0.5002

CUB is a fine-grained dataset with visually similar classes. MoTE remains strong, supporting the idea that confusability should be an input to the future adaptive selector.

### ImageNet-A B0-Inc20

Result directory:

results/mote/imageneta_b0inc20_5seeds_vast_20260720/mote_5seeds_ina_20260720

- Average Accuracy CNN: 65.1350 +/- 1.0472
- Last Accuracy CNN: 54.7620 +/- 0.6096
- Forgetting CNN: 12.0113 +/- 2.0175
- Reported MoTE Average Accuracy: 67.26 +/- 0.96
- Reported MoTE Forgetting: 11.19 +/- 1.69

The five-seed result is stable but lower than the reported average by 2.1250 points. Forgetting is 0.8213 points higher. The result is retained as a completed paper-level reproduction with a documented numerical gap, not as an exact numerical match.

### ImageNet-R B0-Inc20

Result directory:

results/mote/imagenetr_b0inc20_5seeds_vast_20260720/mote_5seeds_inr_20260720

- Average Accuracy CNN: 80.8752 +/- 0.4758
- Last Accuracy CNN: 74.3280 +/- 0.4359
- Forgetting CNN: 7.3596 +/- 0.3031
- Reported MoTE Average Accuracy: 81.93 +/- 0.53
- Reported MoTE Forgetting: 6.91 +/- 0.41

The five-seed result is stable and 1.0548 average-accuracy points below the reported result. Forgetting is 0.4496 points higher. Both ImageNet runs include raw logs, actual configs, Top-1 and Top-5 curves, compatibility patches, runtime records, parameter counts, and environment evidence.

### VTAB B0-Inc10

Result directory:

results/mote/vtab_b0inc10_5seeds_vast_20260702/mote_5seeds_vtab_20260702

- Average Accuracy CNN: 90.3576 +/- 1.8217
- Last Accuracy CNN: 82.4040 +/- 1.1545
- Forgetting CNN: 7.2710 +/- 1.6902

VTAB is more domain-diverse and shows higher variance and lower final accuracy. It is a useful harder case for studying task diversity and domain shift.

## Comparison with reported MoTE values

| Dataset | Our Avg | Reported MoTE Avg | Difference | Our Last | Reported MoTE Final | Difference |
|---|---:|---:|---:|---:|---:|---:|
| CIFAR100 | 92.7050 | 93.06 | -0.3550 | 88.5540 | 88.98 | -0.4260 |
| CUB | 91.3962 | 91.83 | -0.4338 | 87.1720 | 86.77 | +0.4020 |
| ImageNet-A | 65.1350 | 67.26 | -2.1250 | 54.7620 | - | - |
| ImageNet-R | 80.8752 | 81.93 | -1.0548 | 74.3280 | - | - |
| VTAB | 90.3576 | 93.56 | -3.2024 | 82.4040 | 85.94 | -3.5360 |

For the two newly completed datasets, the paper comparison also reports average forgetting:

| Dataset | Our Forgetting | Reported MoTE Forgetting | Difference |
|---|---:|---:|---:|
| ImageNet-A | 12.0113 | 11.19 | +0.8213 |
| ImageNet-R | 7.3596 | 6.91 | +0.4496 |

## MoTE paper-level completion checklist

| # | Experiment or analysis | Status |
|---:|---|---|
| 1 | CIFAR100 B0-Inc10, five seeds, mean/std | Complete |
| 2 | CUB-200-2011 B0-Inc20, five seeds, mean/std | Complete |
| 3 | ImageNet-A B0-Inc20, five seeds, mean/std | Complete |
| 4 | ImageNet-R B0-Inc20, five seeds, mean/std | Complete |
| 5 | VTAB B0-Inc10, five seeds, mean/std | Complete |
| 6 | Adapter-Limited MoTE evaluation | Pending |
| 7 | Expert-filtering ablation | Pending |
| 8 | Expert-weighting ablation, including C-R and SCS-R | Pending |
| 9 | Global versus adaptive scaling-factor comparison | Pending |
| 10 | Inference-time comparison | Pending |
| 11 | Systematic trainable-parameter and total-parameter accounting | Partial |
| 12 | Adapter growth and memory/storage-cost analysis | Pending |
| 13 | Task Identification Accuracy (TIA) comparison | Pending |

Current total: 5 complete, 1 partial, and 7 pending.

## Immediate next work

1. Reproduce Adapter-Limited MoTE, checklist item 6.
2. Lock the exact paper table, datasets, adapter limits, seed protocol, and metrics before launching jobs.
3. Verify the official `mote_limit` path and run seed 1993 as a smoke test.
4. Record adapter caps, per-task parameter growth, accuracy, forgetting, runtime, and memory evidence.
5. Continue checklist items 7-13 in order.
6. Only then continue the paper chain: paper-exact LDEPrompt, QKD, and official TaE.
7. After all four papers are reproduced, compare their ViT modifications, frozen and trainable parameters, inserted modules, parameter expansion, and inference logic to design the adaptive model-selection algorithm.
