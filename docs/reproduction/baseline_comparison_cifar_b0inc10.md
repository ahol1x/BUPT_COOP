# Baseline Comparison: CIFAR100 B0-Inc10

| Method | Run Type | Seeds | Last Top-1 | Average Accuracy | Forgetting | Status |
|---|---:|---:|---:|---:|---:|---|
| **MoTE** | Single-seed | 1993 | **88.33** | **92.977** | **4.778** | Completed |
| **LDEPrompt** | 9-seed repo-default | 1991, 1993, 1995, 1997, 1999, 2001, 2003, 2005, 2007 | 86.61 ± 0.46 | 90.96 ± 1.14 | 7.63 ± 0.64 | Completed |

## Interpretation

| Comparison Point | Current Observation |
|---|---|
| Best reproduced method so far | **MoTE** |
| Higher final accuracy | **MoTE**, by about `+1.72` Last Top-1 |
| Higher average accuracy | **MoTE**, by about `+2.02` Average Accuracy |
| Lower forgetting | **MoTE**, by about `-2.85` forgetting |
| Stronger statistical evidence | **LDEPrompt**, because it has 9 seeds |
| Fairness warning | MoTE is only single-seed, while LDEPrompt is 9-seed |
| Current conclusion | MoTE looks stronger in our reproduction, but we should run MoTE multi-seed before making a final claim |

## LDEPrompt Paper/Reference Comparison

| Method | Source | Last Top-1 | Average Accuracy |
|---|---|---:|---:|
| LDEPrompt | Paper/reference | 88.13 ± 0.76 | 91.60 ± 0.80 |
| LDEPrompt | Our 9-seed repo-default run | 86.61 ± 0.46 | 90.96 ± 1.14 |
| Difference | Our result minus paper | -1.52 | -0.64 |

## Conclusion

We have completed reproduction of two PTM-CIL baselines on CIFAR100 B0-Inc10. MoTE achieved 88.33 Last Top-1, 92.977 Average Accuracy, and 4.778 Forgetting on seed1993. LDEPrompt was reproduced with 9 seeds, achieving 86.61 ± 0.46 Last Top-1, 90.96 ± 1.14 Average Accuracy, and 7.63 ± 0.64 Forgetting.

Current results suggest MoTE performs better in both accuracy and forgetting, but this comparison is not fully fair yet because MoTE has only been run with one seed while LDEPrompt has been evaluated across nine seeds. The next step should be either MoTE multi-seed reproduction or using these two baselines to motivate an adaptive method-selection strategy.
