# MoTE Reproduction Summary

## Project context

This document summarizes our completed five-seed MoTE reproduction for the BUPT_COOP continual learning project.

Main goal:

First reproduce existing PTM-CIL baselines, then use the results to understand which method works best under which task condition. After that, design an adaptive strategy-selection algorithm that chooses between prompt update, adapter expansion, expert fusion, QKD-style distillation, and TaE-style parameter selection.

## Current reproduced method

- Method: MoTE, Mixture of Task-Specific Experts
- Backbone: ViT-B/16-based pretrained transformer
- Setting: exemplar-free class-incremental learning
- Seeds: 1991, 1992, 1993, 1994, 1995
- Metrics saved: Average Accuracy, Last Accuracy, Forgetting, Top1 curve, Top5 curve
- Extra files saved: configs, logs, patches, scripts, dataset notes, pip freeze

## Completed reproduction results

| Method | Dataset | Split | Seeds | Avg Acc | Last Acc | Forgetting | Commit |
|---|---|---|---|---:|---:|---:|---|
| MoTE | CIFAR100 / cifar224 | B0-Inc10 | 1991-1995 | 92.7050 +/- 0.2010 | 88.5540 +/- 0.2061 | 4.9800 +/- 0.3175 | becebdf |
| MoTE | CUB-200-2011 | B0-Inc20 | 1991-1995 | 91.3962 +/- 0.4214 | 87.1720 +/- 0.0879 | 4.6464 +/- 0.5002 | c3819e1 |
| MoTE | VTAB processed 50-class CIL | B0-Inc10 | 1991-1995 | 90.3576 +/- 1.8217 | 82.4040 +/- 1.1545 | 7.2710 +/- 1.6902 | 3f691e2 |

## Detailed result locations

### CIFAR100 B0-Inc10

Path:

results/mote/cifar224_b0inc10_5seeds_vast_20260630/mote_5seeds_cifar_20260630

Result:

- Average Accuracy CNN: 92.7050 +/- 0.2010
- Last Accuracy CNN: 88.5540 +/- 0.2061
- Forgetting CNN: 4.9800 +/- 0.3175

Notes:

- CIFAR100 reproduction is close to the reported MoTE/QKD-table result.
- This confirms that our MoTE setup, patching, dataset format, and ViT checkpoint are paper-comparable.
- CIFAR100 can be treated as the stable baseline condition.

### CUB B0-Inc20

Path:

results/mote/cub_b0inc20_5seeds_vast_20260702/mote_5seeds_cub_20260702

Result:

- Average Accuracy CNN: 91.3962 +/- 0.4214
- Last Accuracy CNN: 87.1720 +/- 0.0879
- Forgetting CNN: 4.6464 +/- 0.5002

Notes:

- CUB is a fine-grained bird classification dataset.
- Classes are visually similar, so this setting is useful for studying confusability.
- MoTE performs strongly, which suggests expert/adapters are helpful when class boundaries are subtle.

### VTAB B0-Inc10

Path:

results/mote/vtab_b0inc10_5seeds_vast_20260702/mote_5seeds_vtab_20260702

Result:

- Average Accuracy CNN: 90.3576 +/- 1.8217
- Last Accuracy CNN: 82.4040 +/- 1.1545
- Forgetting CNN: 7.2710 +/- 1.6902

Notes:

- VTAB uses processed class folders 10 to 59, remapped by ImageFolder to 0 to 49.
- VTAB is more domain-diverse than CIFAR and CUB.
- Our VTAB result has higher variance and lower final accuracy, so it is a useful harder case for adaptive strategy design.

## Comparison with reported MoTE / QKD table values

| Dataset | Our Avg | Reported MoTE Avg | Difference | Our Last | Reported MoTE Final | Difference |
|---|---:|---:|---:|---:|---:|---:|
| CIFAR100 | 92.7050 | 93.06 | -0.3550 | 88.5540 | 88.98 | -0.4260 |
| CUB | 91.3962 | 91.83 | -0.4338 | 87.1720 | 86.77 | +0.4020 |
| VTAB | 90.3576 | 93.56 | -3.2024 | 82.4040 | 85.94 | -3.5360 |

## Interpretation

### CIFAR100

CIFAR100 is relatively stable. MoTE is strong and close to the reported result.

Meaning for adaptive strategy:

- Full expert/adapters work well.
- But some CIFAR tasks may not need full expert expansion.
- Later, CIFAR can test whether lightweight prompt updates can match MoTE with fewer trainable parameters.

### CUB

CUB is fine-grained and visually confusing. MoTE remains strong.

Meaning for adaptive strategy:

- Confusability should be one controller input.
- When new classes are visually close to old classes, stronger task-specific adaptation or expert filtering may help.
- This supports using a confusability score, not only a novelty score.

### VTAB

VTAB is more diverse and unstable. MoTE has lower final accuracy and higher variance here.

Meaning for adaptive strategy:

- Domain shift and task diversity matter.
- A fixed method is not always equally strong.
- VTAB is useful for testing when the controller should choose stronger adaptation, QKD-style distillation, or different update rules.

## Current conclusion

We have completed a meaningful three-dataset MoTE reproduction:

- CIFAR100: stable and paper-comparable
- CUB: fine-grained and still strong
- VTAB: more diverse, weaker, and higher variance

This supports the main project direction:

Different task types create different learning needs. Instead of using one fixed continual learning method for every incoming task, we should estimate task condition first and then choose a suitable strategy.

## Next work

Recommended next baseline: LDEPrompt.

Why:

- It is a prompt-based method.
- It represents the lightweight prompt-adaptation side of our strategy set.
- Comparing LDEPrompt against MoTE can help identify when prompts are enough and when adapters/experts are needed.

Next experimental plan:

1. Reproduce LDEPrompt on CIFAR100 B0-Inc10.
2. If successful, run CUB B0-Inc20 and VTAB B0-Inc10.
3. Build a master comparison table:
   Method | Dataset | Split | Avg | Last | Forgetting | Trainable Params | Notes
4. Compare MoTE vs LDEPrompt behavior.
5. Add QKD / TaE / prompt baseline.
6. Design the adaptive controller using evidence from these baseline results.
