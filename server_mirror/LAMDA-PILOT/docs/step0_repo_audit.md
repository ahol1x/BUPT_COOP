# Step 0 Repo Audit: Finetune CIFAR100 B0-Inc10 Seed 1993

## Scope

This audit was done locally in:

`/Users/ahol1c/BUPT/BUPT_COOP/server_mirror/LAMDA-PILOT`

No full training was started locally. The prepared scripts are intended for the server path:

`/home/kennycao12/Python_Projects/BUPT/Chris/LAMDA-PILOT`

## Repo Structure Summary

- `main.py`: entry point. It only accepts `--config`, loads a JSON file, merges JSON keys into the parsed args dict, then calls `trainer.train(args)`.
- `trainer.py`: orchestration loop over seeds and tasks. It sets random seeds, converts JSON `device` entries into torch devices, creates `DataManager`, builds the model through `utils.factory`, logs parameter counts, trains each task, evaluates CNN/NME metrics, and optionally computes forgetting.
- `exps/`: JSON experiment configs. The CIFAR benchmark configs use `dataset: "cifar224"` for ViT-based CIFAR100 runs.
- `models/`: method implementations. `models/finetune.py` implements the Finetune learner with `IncrementalNet`.
- `utils/`: dataset management, model factory, network wrappers, metrics utilities.
- `backbone/`: ViT variants, prompt/adaptor variants, linears, and ResNet code. The active Finetune config uses timm `vit_base_patch16_224`.
- `resources/`: README images, including the CIFAR B0-Inc10 reproduced-result figure.

## Detected Entry Point and CLI

The detected entry point is:

```bash
python main.py --config=./exps/[MODEL NAME].json
```

`main.py` has no direct CLI flags for dataset, method, seed, GPU, output directory, `init_cls`, or `increment`. Those are supplied by the JSON config.

Relevant config keys:

- method/model: `model_name`
- dataset: `dataset`
- split: `init_cls` and `increment`
- seed: `seed`
- GPU ids: `device`
- log prefix: `prefix`
- backbone: `backbone_type`

Repo log path is hardcoded in `trainer.py` as:

`logs/{model_name}/{dataset}/{init_cls_for_log}/{increment}/{prefix}_{seed}_{backbone_type}.log`

When `init_cls == increment`, `trainer.py` logs the split folder as `0`, so Step 0 writes to:

`logs/finetune/cifar224/0/10/step0_1993_vit_base_patch16_224.log`

## CIFAR100 B0-Inc10 Representation

There is no explicit `B0-Inc10` option. In this repo it is represented as:

```json
"init_cls": 10,
"increment": 10
```

This yields 10 tasks of 10 classes over 100 classes. Because `init_cls == increment`, the trainer uses `0` in the log directory, matching the B0 naming convention.

For the dataset key, the repo has both:

- `cifar100`: downloads torchvision CIFAR100 and uses 32x32-style transforms.
- `cifar224`: downloads torchvision CIFAR100 and applies 224x224 transforms for ViT models.

The existing Finetune CIFAR benchmark config uses `dataset: "cifar224"` with `vit_base_patch16_224`. I therefore prepared Step 0 with `dataset: "cifar224"` while documenting the requested dataset as CIFAR100. Using the literal `cifar100` key with the current ViT-B/16-224 Finetune config is likely incompatible because the backbone expects 224x224 inputs.

## Planned Server-Side Command

After rsyncing this repo to the server, run:

```bash
cd /home/kennycao12/Python_Projects/BUPT/Chris/LAMDA-PILOT
bash scripts/server_check_step0_env.sh
bash scripts/run_step0_finetune_cifar100_b0inc10_seed1993.sh
```

The training command inside the run script is:

```bash
python main.py --config=exps/step0_finetune_cifar100_b0inc10_seed1993.json
```

## Files Created or Edited

- `docs/step0_repo_audit.md`
- `exps/step0_finetune_cifar100_b0inc10_seed1993.json`
- `scripts/run_step0_finetune_cifar100_b0inc10_seed1993.sh`
- `scripts/server_check_step0_env.sh`
- `scripts/extract_step0_metrics.py`
- `trainer.py`

No training loop redesign was made. Metrics extraction is a small post-processing script over the existing logs plus wrapper-generated runtime files. The only source edit in `trainer.py` is a logging-only fix so `Forgetting (NME)` is not printed when no NME matrix exists.

## Metrics Available

Existing trainer logs provide:

- CNN top-1 curve and top-5 curve.
- Average Accuracy (CNN), equivalent to average incremental accuracy over the top-1 curve.
- Final accuracy, available as the last value in the CNN top-1 curve.
- Grouped class-range accuracies per evaluation, used as available per-task/class-block accuracy.
- Total params and trainable params before each task.
- Forgetting (CNN) when `print_forget: true`.

The wrapper adds:

- wall-clock seconds in `outputs/step0_finetune_cifar100_b0inc10_seed1993/wall_clock_seconds.txt`
- GPU memory samples in `outputs/step0_finetune_cifar100_b0inc10_seed1993/gpu_memory_samples.csv`
- parsed summary JSON in `outputs/step0_finetune_cifar100_b0inc10_seed1993/metrics_summary.json`

## Expected Result and Log Files

Expected wrapper log:

`logs/step0_finetune_cifar100_b0inc10_seed1993.log`

Expected trainer log:

`logs/finetune/cifar224/0/10/step0_1993_vit_base_patch16_224.log`

Expected output directory:

`outputs/step0_finetune_cifar100_b0inc10_seed1993/`

Expected files inside the output directory:

- `metrics_summary.json`
- `wall_clock_seconds.txt`
- `gpu_memory_samples.csv`

The run script also creates `logs/`, `results/`, `.cache/torch/`, and `.cache/huggingface/` under the repo on the server.

## Missing Dependencies Discovered Locally

Local command attempted:

```bash
python main.py --help
```

It did not start training, but it failed during imports:

```text
ModuleNotFoundError: No module named 'torchvision'
```

I did not install dependencies locally. The server check script verifies the required imports in the `bupt-cil` env before training.

Core modules checked by `scripts/server_check_step0_env.sh`:

- `torch`
- `torchvision`
- `timm`
- `tqdm`
- `numpy`
- `scipy`
- `PIL`
- `easydict`

## Uncertainty and Notes

- The README says CIFAR100 is automatically downloaded, but the ViT CIFAR benchmark configs use the repo-specific `cifar224` key. This audit treats `cifar224` as the correct CIFAR100 benchmark path for Finetune with ViT-B/16-224.
- There is no config-level output directory. Repo training logs are hardcoded under `logs/`; the wrapper creates the requested Step 0 output directory and writes post-processed metrics there.
- The local mirror's `.git` metadata appears broken: `git status --short` failed because the referenced git module path was missing. This did not affect static inspection or file creation.
