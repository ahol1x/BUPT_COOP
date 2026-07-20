# Baseline Comparison: CIFAR100 B0-Inc10

Last updated: 2026-07-20

## Reproduced results

| Method | Run type | Seeds | Last Top-1 | Average Accuracy | Forgetting | Status |
|---|---|---|---:|---:|---:|---|
| **MoTE** | Five-seed reproduction | 1991, 1992, 1993, 1994, 1995 | **88.5540 +/- 0.2061** | **92.7050 +/- 0.2010** | **4.9800 +/- 0.3175** | Completed |
| **LDEPrompt** | Nine-seed repo-default reproduction | 1991, 1993, 1995, 1997, 1999, 2001, 2003, 2005, 2007 | 86.61 +/- 0.46 | 90.96 +/- 1.14 | 7.63 +/- 0.64 | Completed |

## Aggregate difference

The values below are MoTE minus LDEPrompt, using the reproduced aggregate means.

| Comparison point | Difference |
|---|---:|
| Last Top-1 | +1.94 |
| Average Accuracy | +1.75 |
| Forgetting | -2.65 |

MoTE currently has higher reproduced final and average accuracy and lower forgetting on CIFAR100 B0-Inc10.

## Comparison limitations

This is more informative than the earlier single-seed comparison, but it is not yet a fully controlled head-to-head experiment:

- MoTE uses five seeds; LDEPrompt uses nine seeds.
- The two seed sets are different.
- LDEPrompt is a repo-default reproduction, not a confirmed paper-exact reproduction.
- Configuration, optimization, and implementation differences must be checked before making a final method-level claim.

Therefore, the correct current conclusion is that MoTE performs better in the available reproduced aggregates, not that MoTE has been proven universally superior to LDEPrompt.

## Paper/reference comparison

| Method | Source | Last Top-1 | Average Accuracy |
|---|---|---:|---:|
| MoTE | Reported/reference | 88.98 | 93.06 |
| MoTE | Our five-seed reproduction | 88.5540 +/- 0.2061 | 92.7050 +/- 0.2010 |
| LDEPrompt | Paper/reference | 88.13 +/- 0.76 | 91.60 +/- 0.80 |
| LDEPrompt | Our nine-seed repo-default reproduction | 86.61 +/- 0.46 | 90.96 +/- 1.14 |

## Source records

- MoTE summary: docs/MoTE_reproduction_summary.md
- MoTE result directory: results/mote/cifar224_b0inc10_5seeds_vast_20260630/mote_5seeds_cifar_20260630
- LDEPrompt summary: docs/reproduction/ldeprompt_cifar_b0inc10_multiseed_5090_2026-07-12_summary.md

## Current workflow implication

This comparison is a preserved baseline record. It does not change the active execution order. The current priority is to finish the full MoTE paper-level reproduction checklist before continuing LDEPrompt, QKD, or TaE.
