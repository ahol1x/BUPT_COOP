# Step 1 MoTE Repo Audit: CIFAR100 B0-Inc10 Seed 1993

## Scope

This audit was done locally in:

`/Users/ahol1c/BUPT/BUPT_COOP/server_mirror/MoTE`

No full training was started locally. The prepared scripts target the server path:

`/home/kennycao12/Python_Projects/BUPT/Chris/MoTE`

Server environment:

```bash
conda activate bupt-cil
CUDA_VISIBLE_DEVICES=0
```

## Entry Point And Config System

Entry point:

```bash
python main.py --config=exps/step1_mote_cifar100_b0inc10_seed1993.json
```

`main.py` only exposes `--config`. It loads the JSON file, merges keys into the arg dict, and calls `trainer.train(args)`.

`trainer.py` loops over `seed`, converts `device` entries into torch devices, builds `DataManager`, then builds the learner through `utils.factory.get_model(args["model_name"], args)`.

## MoTE Method Name

The current runnable registry in `utils/factory.py` is:

- `model_name: "mote"` -> `models.mote.Learner`
- `model_name: "mote_limit"` -> `models.mote_limit.Learner`

Important repo mismatch:

- The bundled README commands point to `exps/cifar/mote.json` and `exps/cifar/mote_in21k.json`.
- Those bundled JSON files currently use `model_name: "moe"` and backbone names ending in `_moe`.
- Current code does not register `moe`; it registers `mote`.
- Current `utils/inc_net.py` expects `_mote` in the backbone name.

Therefore the prepared Step 1 config uses the runnable actual MoTE values:

```json
"model_name": "mote",
"backbone_type": "vit_base_patch16_224_mote"
```

## Dataset And Split

README says CIFAR100 is automatically downloaded by the code. In this repo the ViT CIFAR configs use:

```json
"dataset": "cifar224"
```

`utils/data.py` maps `cifar224` to torchvision CIFAR100 with 224x224 transforms for ViT. It currently downloads CIFAR100 under `/datasets`.

B0-Inc10 is represented as:

```json
"init_cls": 10,
"increment": 10
```

When `init_cls == increment`, `trainer.py` uses `0` in the log directory, so B0-Inc10 logs under:

`logs/{model_name}/{dataset}/0/{increment}/...`

Seed and device format:

```json
"seed": [1993],
"device": ["0"]
```

## Official CIFAR Hyperparameters Used

The prepared config is based on `exps/cifar/mote.json`, with runnable names corrected from `moe` to `mote`.

Official CIFAR settings retained:

- `init_epochs`: `20`
- `later_epochs`: `20`
- `init_lr`: `0.025`
- `later_lr`: `0.025`
- `batch_size`: `48`
- `optimizer`: `sgd`
- `scheduler`: `cosine`
- `ffn_num`: `64`
- `adapter_num`: `-1`
- `print_forget`: `true`

Step 0 LAMDA-PILOT reference:

- Method: Finetune
- Dataset key: `cifar224`
- Split: B0-Inc10
- Seed: `1993`
- Avg: `77.321`
- Last: `65.18`
- Forgetting: `33.544444444444444`

## Pretrained ViT Handling

`backbone/vit_mote.py` defines:

- `vit_base_patch16_224_mote`
- `vit_base_patch16_224_in21k_mote`

The official CIFAR default in `exps/cifar/mote.json` is the IN1K ViT-B/16 path, so the Step 1 config uses:

```json
"backbone_type": "vit_base_patch16_224_mote"
```

Compatibility patch applied:

- Old `timm.models.layers` imports now fall back to `timm.layers`.
- Old `timm.models.registry` import now falls back to `timm.models`.
- The hardcoded pretrained checkpoint path now checks, in order:
  - `$MOTE_PRETRAIN_DIR`
  - `${PWD}/checkpoints`
  - `/pretrains`
  - `timm.create_model(..., pretrained=True)` fallback
- `models/mote.py` now initializes config-backed attributes used during training/evaluation and no longer unpacks the dict returned by `MoteNet.forward`.

The run script sets:

```bash
export MOTE_PRETRAIN_DIR="${PWD}/checkpoints"
export TORCH_HOME="${PWD}/.cache/torch"
export HF_HOME="${PWD}/.cache/huggingface"
export TRANSFORMERS_CACHE="${PWD}/.cache/huggingface"
export HF_ENDPOINT="https://hf-mirror.com"
export HF_HUB_DISABLE_TELEMETRY=1
```

If the server already has the original `.npz` files under `/pretrains`, those will be used. Otherwise timm will try its pretrained weight path using the configured caches/mirror.

## Required Dependencies

Directly used by this repo:

- `torch`
- `torchvision`
- `timm`
- `numpy`
- `scipy`
- `tqdm`
- `easydict`
- `PIL` / `Pillow`

Optional / environment-dependent:

- `safetensors`: not imported directly by repo code, but may be needed by newer timm/Hugging Face weight downloads.
- `accimage`: only used if torchvision image backend is set to `accimage`.
- `umap`, `matplotlib`: only used by optional t-SNE/visualization code in `models/base.py`.

## Expected Log Path

Wrapper log:

`logs/step1_mote_cifar100_b0inc10_seed1993.log`

Repo trainer log:

`logs/mote/cifar224/0/10/step1_1993_vit_base_patch16_224_mote.log`

Output directory:

`outputs/step1_mote_cifar100_b0inc10_seed1993/`

Expected files:

- `metrics_summary.json`
- `wall_clock_seconds.txt`
- `gpu_memory_samples.csv`

## Files Created Or Edited

Created:

- `docs/step1_mote_repo_audit.md`
- `exps/step1_mote_cifar100_b0inc10_seed1993.json`
- `scripts/extract_mote_metrics.py`
- `scripts/server_check_mote_env.sh`
- `scripts/run_step1_mote_cifar100_b0inc10_seed1993.sh`

Edited:

- `backbone/vit_mote.py`
- `backbone/vit_mote_limit.py`
- `backbone/linears.py`
- `models/mote.py`
- `models/mote_limit.py`

No DAAC code, mixed stream, or training-loop redesign was added.

## Server Commands After Rsync

Run environment check:

```bash
cd /home/kennycao12/Python_Projects/BUPT/Chris/MoTE
bash scripts/server_check_mote_env.sh
```

Run Step 1 MoTE:

```bash
cd /home/kennycao12/Python_Projects/BUPT/Chris/MoTE
bash scripts/run_step1_mote_cifar100_b0inc10_seed1993.sh
```

The training command inside the run script is:

```bash
python main.py --config=exps/step1_mote_cifar100_b0inc10_seed1993.json
```

## Metrics Summary

`scripts/run_step1_mote_cifar100_b0inc10_seed1993.sh` writes:

`outputs/step1_mote_cifar100_b0inc10_seed1993/metrics_summary.json`

The extractor parses:

- top-1 curve
- top-5 curve
- final accuracy / Last
- average incremental accuracy / Avg
- forgetting
- total/trainable params
- wall-clock seconds
- peak GPU memory MiB
- configured/estimated adapter count
- MoTE `multi-expert-num` and `zero-expert-num` logs

It also normalizes `np.float64(...)` values before parsing curves and grouped accuracy dictionaries.

## Risks And Notes

- Bundled official configs appear stale relative to the current code registry: they say `moe`, while the code requires `mote`.
- CIFAR data downloads to `/datasets` in `utils/data.py`. If the server cannot write there or CIFAR100 is already stored elsewhere, update that dataset path before running.
- If neither `${PWD}/checkpoints` nor `/pretrains` has the original `.npz` ViT checkpoint files, timm will attempt to resolve pretrained weights through the configured cache/mirror.
- `scripts/server_check_mote_env.sh` runs `python main.py --help`; this imports the repo but does not train.
