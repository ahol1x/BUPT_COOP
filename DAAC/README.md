# DAAC Prototype

DAAC is a first prototype for testing a controller-based PTM-CIL idea:

- estimate incoming task difficulty in a pre-study pass;
- select one adaptation strategy instead of stacking every mechanism;
- compare adaptive selection against fixed baselines.

The implementation is intentionally compact and offline-friendly. It mirrors the structure found in `LAMDA-PILOT` and `tae_longtail_cil`, but lives in this separate folder to avoid breaking existing methods.

## Fast Debug

```bash
bash scripts/run_daac_fast_debug.sh
```

This uses a synthetic class-incremental dataset and exercises the whole pipeline: pre-study, controller selection, prompt/light updates, TaE-style top-p masks, adapter expansion, fusion, logging, and summaries.

## CIFAR100

```bash
bash scripts/run_daac_cifar100.sh
```

CIFAR100 requires either local files under `DAAC/data/cifar-100-python` or an environment with `torchvision`. Use `--download` with `run_daac.py` if network access is available.

## Outputs

Metrics are saved to:

```text
outputs/daac/{dataset}/{strategy}/{seed}/metrics.csv
outputs/daac/{dataset}/{strategy}/{seed}/summary.json
```

Plots can be generated with:

```bash
python scripts/plot_daac_results.py --dataset synthetic
python scripts/plot_daac_results.py --dataset cifar100
```

## Important Simplifications

- QKD relevance is approximated with cosine similarity to prototypes, not a quantum circuit.
- TaE top-p selection is implemented as element-wise gradient masking, not structural expansion.
- LDEPrompt layer importance uses an activation-change proxy, not mutual information.
- MoTE expert filtering/fusion is approximated with adapter features, class prototypes, confidence, and SCS-style scores.
- Fast debug uses a tiny frozen transformer-like backbone rather than a downloaded ViT-B/16.
