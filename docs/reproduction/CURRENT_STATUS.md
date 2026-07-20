# BUPT_COOP Reproduction Status

Last updated: 2026-07-20

This file is the authoritative project-status record. When another summary conflicts with this file, update the outdated summary rather than creating a second status definition.

## Research objective

Reproduce four Linjie-related PTM-CIL methods, understand how each method modifies and uses the ViT backbone, and then design an adaptive strategy-selection algorithm that chooses the most suitable method for each incoming continual-learning task.

## Authoritative execution order

1. Finish the complete MoTE paper-level reproduction.
2. Finish a paper-exact LDEPrompt reproduction.
3. Reproduce QKD.
4. Reproduce official TaE.
5. Compare the four codebases:
   - ViT modules changed or inserted
   - frozen and trainable parameters
   - parameter expansion across tasks
   - training and inference logic
   - accuracy, forgetting, runtime, and memory trade-offs
6. Define measurable task characteristics and design the adaptive selector among MoTE, LDEPrompt, QKD, and TaE.

C3Box and LAMDA-PILOT are auxiliary environment/testing tools. They are not primary research targets.

## Completed reproduction results

| Method | Dataset and setting | Seeds | Last Top-1 | Average Accuracy | Forgetting | Reproduction type |
|---|---|---|---:|---:|---:|---|
| MoTE | CIFAR100 B0-Inc10 | 1991-1995 | 88.5540 +/- 0.2061 | 92.7050 +/- 0.2010 | 4.9800 +/- 0.3175 | Five-seed |
| MoTE | CUB200 B0-Inc20 | 1991-1995 | 87.1720 +/- 0.0879 | 91.3962 +/- 0.4214 | 4.6464 +/- 0.5002 | Five-seed |
| MoTE | VTAB B0-Inc10 | 1991-1995 | 82.4040 +/- 1.1545 | 90.3576 +/- 1.8217 | 7.2710 +/- 1.6902 | Five-seed |
| LDEPrompt | CIFAR100 B0-Inc10 | 1991, 1993, 1995, 1997, 1999, 2001, 2003, 2005, 2007 | 86.61 +/- 0.46 | 90.96 +/- 1.14 | 7.63 +/- 0.64 | Nine-seed repo-default |

The LDEPrompt result is valid as a repo-default reproduction, but it is not yet a confirmed paper-exact reproduction.

## MoTE 13-item completion checklist

| # | Experiment or analysis | Status | Evidence or next action |
|---:|---|---|---|
| 1 | CIFAR100 B0-Inc10, five seeds | Complete | docs/MoTE_reproduction_summary.md |
| 2 | CUB200 B0-Inc20, five seeds | Complete | docs/MoTE_reproduction_summary.md |
| 3 | ImageNet-A B0-Inc20, five seeds | Pending | Run seeds 1991-1995 |
| 4 | ImageNet-R B0-Inc20, five seeds | Pending | Run seeds 1991-1995 |
| 5 | VTAB B0-Inc10, five seeds | Complete | docs/MoTE_reproduction_summary.md |
| 6 | Adapter-Limited MoTE | Pending | Reproduce adapter-limit evaluation |
| 7 | Expert-filtering ablation | Pending | Reproduce filtering configurations |
| 8 | Expert-weighting ablation | Pending | Reproduce confidence and SCS re-weighting |
| 9 | Global vs adaptive scaling factor | Pending | Reproduce scaling-factor comparison |
| 10 | Inference-time comparison | Pending | Use controlled hardware and batch settings |
| 11 | Trainable and total parameter accounting | Partial | Existing isolated measurements must be standardized |
| 12 | Adapter growth and memory/storage cost | Pending | Record per-task adapter and prototype growth |
| 13 | Task Identification Accuracy (TIA) | Pending | Add paper-consistent TIA evaluation |

Current total: 3 complete, 1 partial, and 9 pending.

## Other method status

| Method | Current status | Action now |
|---|---|---|
| LDEPrompt | CIFAR100 nine-seed repo-default result completed | Preserve; resume after MoTE |
| QKD | Not reproduced | Do not start yet |
| TaE | Official reproduction not started | Do not start yet |
| C3Box | Auxiliary test completed | No further research priority |
| LAMDA-PILOT | Environment/baseline utility | No large-scale analysis |

## Immediate next action

Run MoTE ImageNet-A and ImageNet-R reproductions.

For each dataset:

1. Confirm the dataset structure and class count.
2. Confirm the B0-Inc20 split.
3. Run seed 1993 as a smoke test.
4. Run seeds 1991-1995.
5. Save per-seed metrics and mean/std.
6. Save Top-1 and Top-5 curves.
7. Record runtime, peak GPU memory, trainable parameters, total parameters, adapter count, and prototype memory.
8. Commit configs, parsers, summaries, logs or log manifests, and environment notes.

## Documentation rule

Every completed experiment must update this file in the same commit as its result summary. Do not mark an item complete unless the configuration, seeds, metrics, and result location are recorded.
