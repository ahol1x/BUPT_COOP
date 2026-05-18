# TaE Long-tail CIL Study

This folder is a separate experiment package built beside the downloaded `LAMDA-PILOT` baseline. It keeps `LAMDA-PILOT` untouched and adds a runnable comparison between:

- `traditional_full_update`: a standard full-update finetuning baseline.
- `tae_ced_top_p`: a model inspired by paper `0471.pdf`, using task-aware top-p parameter updates, centroid-enhanced feature learning, and long-tail class reweighting.

## What the Paper Fixes

Paper 0471 focuses on long-tailed class-incremental learning. In this setting, new tasks do not have balanced class counts: head classes have many training images and tail classes have few. Traditional finetuning tends to learn the current head classes too strongly, forget old classes, and leave tail-class features poorly separated.

The TaE idea improves this by:

- Selecting only the most gradient-sensitive parameters for each new task.
- Freezing most parameters so old knowledge drifts less.
- Adding class centroids so features are pulled toward their own class and separated from other classes.
- Reweighting classes so tail classes are not overwhelmed by head classes.

The TaE runner also includes long-tail-specific corrections:

- Class-balanced sampling for TaE task batches, so tail classes contribute to both mask selection and training.
- Balanced-softmax loss with bounded effective-number class weights.
- Class-balanced centroid loss, so the centroid objective is not dominated by head-class images.
- Centroid prototype replay for older classes, which preserves an old-class classifier signal without replaying old images.
- Evaluation-time logit adjustment to reduce head-class prior bias on balanced test splits.

## Run

From this folder:

```bash
python main.py
```

If your server does not already have CIFAR-10 under `./data`, run:

```bash
python main.py --download-data
```

Common server options:

```bash
python main.py \
  --device cuda \
  --studies 10 \
  --epochs 20 \
  --batch-size 128 \
  --tae-budget 0.15 \
  --class-weight-beta 0.9999 \
  --max-class-weight 6 \
  --prototype-replay-weight 0.25 \
  --logit-adjustment-tau 0.5 \
  --highlight-cases 1 5
```

For a more detailed live run, add `--verbose`. You can also print running
batch-level stats every N batches:

```bash
python main.py \
  --download-data \
  --device cuda \
  --studies 10 \
  --epochs 20 \
  --batch-size 128 \
  --verbose \
  --log-every-batches 10
```

If you want to compare against the older TaE behavior, disable the new balanced
sampler and remove the extra long-tail correction strength:

```bash
python main.py \
  --download-data \
  --device cuda \
  --studies 10 \
  --epochs 20 \
  --batch-size 128 \
  --no-tae-balanced-sampling \
  --class-weight-beta 0.95 \
  --prototype-replay-weight 0 \
  --logit-adjustment-tau 0
```

## Outputs

Running `main.py` writes readable outputs to `final_results/`:

- `per_case_results.csv`: all 10 long-tailed cases for both models.
- `highlight_case_results.csv`: details for case 1 and case 5 by default.
- `summary.json`: mean, std, min, max, and TaE-minus-baseline differences.
- `comparison.html`: visual comparison curves.
- `paper_method_notes.md`: explanation of what was fixed and why TaE should help.

The default setup uses 10 shuffled long-tailed CIFAR-10 cases. Each case has 5 incremental tasks with 2 classes per task, and the report highlights case 1 and case 5.
