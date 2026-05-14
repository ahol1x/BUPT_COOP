# Paper 0471 TaE CIL Study

This project keeps `LAMDA-PILOT` untouched and runs a standalone 10-class CIL comparison inspired by paper `0471.pdf`.

## Design

- Dataset: CIFAR-10, all 10 classes.
- Protocol: 5 incremental tasks, 2 new classes per task.
- Studies: 10 seeded runs.
- Case 1: balanced class distribution.
- Case 2: shuffled long-tail class distribution with about 10:1 head-to-tail train counts.
- Baseline: traditional full-update finetuning on the current task.
- Paper-0471 model: TaE-style top-15% sensitive parameter updates plus CEd centroid loss.

The implementation uses the paper ideas without changing the toolbox:

- Accumulate gradients on the incoming task.
- Select the top 15% parameter elements by gradient magnitude.
- Update only selected elements during the task.
- Add a centroid-enhanced loss to pull features toward class centroids and separate centroids.
- Apply class reweighting for the long-tail case.

## Outputs

- `results/tae15_cil_results.csv`: per-study results.
- `results/tae15_cil_summary.json`: aggregate statistics and paired t-tests.
- `results/tae15_cil_visualization.html`: graph visualization.

## Completed Run

Command:

```bash
/Users/ahol1c/BUPT/.venv/bin/python -u paper0471_tae_cil_study/scripts/run_tae15_cil_study.py \
  --studies 10 \
  --epochs 2 \
  --batch-size 512 \
  --case1-train-per-class 5000 \
  --case2-head-train-per-class 5000 \
  --case2-tail-ratio 0.1 \
  --case2-min-train-per-class 500 \
  --test-per-class 1000 \
  --tae-budget 0.15 \
  --mask-batches 20 \
  --centroid-weight 0.1 \
  --resume
```

This run uses all 10 CIFAR-10 classes, 5 CIL tasks with 2 classes per task, and 10 random seeds. Case 1 uses the full CIFAR-10 train set for each class. Case 2 uses a shuffled 10:1 long-tail class distribution with at least 500 training examples per class. Both cases evaluate on 1,000 test images per class.

| Case | Metric | Traditional mean | TaE 15% mean | TaE - Traditional | Paired p-value | Significant |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| case1_balanced | final accuracy | 14.231 | 11.882 | -2.349 | 3.41e-05 | yes |
| case1_balanced | average incremental accuracy | 31.851 | 31.047 | -0.804 | 0.147 | no |
| case2_shuffled_long_tail | final accuracy | 11.075 | 11.146 | 0.071 | 0.860 | no |
| case2_shuffled_long_tail | average incremental accuracy | 27.656 | 26.934 | -0.722 | 8.74e-04 | yes |

The local 10-class CIFAR-10 study does not reproduce a TaE advantage. The significant differences in this run favor traditional full-update finetuning for case 1 final accuracy and case 2 average incremental accuracy. The paper evaluates larger LT-CIL settings with deeper backbones and long training schedules, so these results should be treated as a local stress test rather than a reproduction of the paper tables.
