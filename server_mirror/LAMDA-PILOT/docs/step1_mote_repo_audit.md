# Step 1 Repo Audit: MoTE / MOS CIFAR100 B0-Inc10 Seed 1993

## Scope

This audit was done locally in:

`/Users/ahol1c/BUPT/BUPT_COOP/server_mirror/LAMDA-PILOT`

No full training was started locally. The prepared run script targets the server path:

`/home/kennycao12/Python_Projects/BUPT/Chris/LAMDA-PILOT`

Server environment expected by the script:

```bash
conda activate bupt-cil
CUDA_VISIBLE_DEVICES=0
```

## MoTE Availability

I did not find a confirmed MoTE implementation in this local mirror.

Negative checks:

- `find . -iname '*mote*' -o -iname '*expert*'` returned no files.
- `rg -n -i 'mote|expert' .` returned no code or config hits.
- `utils/factory.py` has no `mote` / `MoTE` branch.
- No `exps/*mote*.json`, `models/*mote*.py`, or `backbone/*mote*.py` files exist.

The repo does contain adapter-heavy methods, especially `mos`, `tuna`, and `ease`. The closest "Mo*" method present is `MOS`, not MoTE. Because the requested filename and step label are "mote" but the repo has no `mote` model registry entry, the prepared config uses the runnable repo method `mos` and this remains the main uncertainty.

Searches performed locally:

```bash
rg -n -i 'mote|expert|adapter|model_name|backbone_type|factory' .
rg -n -i '\bmote\b|mixture|expert|task.*expert|adapter_momentum|ensemble|tuna|mos' README.md models backbone utils exps docs scripts
find . -iname '*mote*' -o -iname '*expert*'
rg -n 'model_name":' exps
sed -n '1,220p' utils/factory.py
```

## Discovered `model_name` Values

Registered in `utils/factory.py`:

- `simplecil`
- `aper_finetune`
- `aper_ssf`
- `aper_vpt`
- `aper_adapter`
- `l2p`
- `dualprompt`
- `coda_prompt`
- `finetune`
- `icarl`
- `der`
- `coil`
- `foster`
- `memo`
- `ranpac`
- `ease`
- `slca`
- `lae`
- `fecam`
- `dgr`
- `mos`
- `cofima`
- `duct`
- `tuna`

Additional notes from `exps/`:

- Existing CIFAR MOS config: `exps/mos.json` with `model_name: "mos"` and `backbone_type: "vit_base_patch16_224_mos"`.
- Existing ImageNet-R MOS config: `exps/mos_inr.json` with the same MOS backbone.
- Existing CIFAR TUNA config: `exps/tuna_cifar.json` with `model_name: "tuna"` and `backbone_type: "vit_base_patch16_224_in21k_tuna"`.
- No config uses `model_name: "mote"`.

## Files Related To The Closest Runnable Method

Core runnable MOS files:

- `utils/factory.py`: maps `model_name: "mos"` to `models.mos.Learner`.
- `models/mos.py`: MOS learner, adapter training, classifier alignment, and evaluation.
- `backbone/vit_mos.py`: ViT adapter backbone for MOS.
- `utils/inc_net.py`: `get_backbone` branch for `_mos` backbones and `MOSNet`.
- `exps/mos.json`: CIFAR config used as the base for Step 1.
- `exps/mos_inr.json`: ImageNet-R MOS config.

Related adapter files checked but not treated as MoTE:

- `models/tuna.py`, `backbone/vit_tuna.py`, `exps/tuna_cifar.json`, `exps/tuna_inr.json`
- `models/ease.py`, `backbone/vit_ease.py`, `exps/ease.json`
- `models/aper_adapter.py`, `backbone/vit_adapter.py`

## Model Name And Required Config Keys

No confirmed MoTE `model_name` exists in this repo.

The prepared Step 1 config uses:

```json
"model_name": "mos"
```

Required MOS keys found from `models/mos.py`, `utils/inc_net.py`, and `exps/mos.json`:

- `model_name`
- `backbone_type`
- `dataset`
- `init_cls`
- `increment`
- `seed`
- `device`
- `batch_size`
- `init_lr`
- `weight_decay`
- `min_lr`
- `optimizer`
- `scheduler`
- `tuned_epoch`
- `init_milestones`
- `init_lr_decay`
- `reg`
- `adapter_momentum`
- `ensemble`
- `crct_epochs`
- `ca_lr`
- `ca_storage_efficient_method`
- `n_centroids` if using `multi-centroid`; kept from the base config even though this run uses `covariance`
- `pretrained`
- `drop`
- `drop_path`
- `ffn_num`

The chosen backbone is:

```json
"backbone_type": "vit_base_patch16_224_mos"
```

This matches `exps/mos.json`, the existing CIFAR MOS/PTM config. The ImageNet-R MOS config also uses `vit_base_patch16_224_mos`.

## Config Prepared

Created:

`exps/step1_mote_cifar100_b0inc10_seed1993.json`

Based on:

`exps/mos.json`

Changes from the base MOS CIFAR config:

- `prefix`: `"step1"`
- `dataset`: `"cifar224"`
- `init_cls`: `10`
- `increment`: `10`
- `seed`: `[1993]`
- `device`: `["0"]`
- `print_forget`: `true`

Dataset and split choices follow Step 0:

- Requested dataset: CIFAR100
- Repo dataset key: `cifar224`
- Split: B0-Inc10 represented as `init_cls=10`, `increment=10`
- Seed: `1993`

Step 0 confirmed reference values:

- Method: `finetune`
- Dataset key: `cifar224`
- Avg: `77.321`
- Last: `65.18`
- Forgetting: `33.544444444444444`
- Final trainable params: `85867866`
- Wall clock seconds: `3847`
- Peak GPU memory MiB: `18819`

## Expected Log Path

Wrapper log from the run script:

`logs/step1_mote_cifar100_b0inc10_seed1993.log`

Repo trainer log path, based on `trainer.py`:

`logs/mos/cifar224/0/10/step1_1993_vit_base_patch16_224_mos.log`

Because `init_cls == increment`, `trainer.py` uses `0` as the split folder.

Output directory:

`outputs/step1_mote_cifar100_b0inc10_seed1993/`

Expected files:

- `metrics_summary.json`
- `wall_clock_seconds.txt`
- `gpu_memory_samples.csv`

## Metrics Extraction Fix

Created a general extractor:

`scripts/extract_metrics.py`

It parses numpy scalar reprs in logs, including values like:

```text
CNN top1 curve: [np.float64(98.9), np.float64(83.2)]
```

Extracted fields include:

- `top1_curve`
- `top5_curve`
- `final_accuracy`
- `average_incremental_accuracy`
- `forgetting`
- `params.final_trainable_params`
- `runtime.wall_clock_seconds`
- `runtime.peak_gpu_memory_mib`

Updated:

`scripts/extract_step0_metrics.py`

It is now a compatibility wrapper around `scripts/extract_metrics.py`, so the Step 0 run script can still call the old filename while getting the fixed parser.

## Server Commands After Rsync

Run on the server:

```bash
cd /home/kennycao12/Python_Projects/BUPT/Chris/LAMDA-PILOT
bash scripts/run_step1_mote_cifar100_b0inc10_seed1993.sh
```

The training command inside the script is:

```bash
python main.py --config=exps/step1_mote_cifar100_b0inc10_seed1993.json
```

To re-extract Step 0 metrics with the fixed parser after rsync, without rerunning Step 0 training:

```bash
cd /home/kennycao12/Python_Projects/BUPT/Chris/LAMDA-PILOT
python scripts/extract_metrics.py \
  --run-name step0_finetune_cifar100_b0inc10_seed1993 \
  --config exps/step0_finetune_cifar100_b0inc10_seed1993.json \
  --wrapper-log logs/step0_finetune_cifar100_b0inc10_seed1993.log \
  --repo-log logs/finetune/cifar224/0/10/step0_1993_vit_base_patch16_224.log \
  --gpu-memory-samples outputs/step0_finetune_cifar100_b0inc10_seed1993/gpu_memory_samples.csv \
  --wall-clock-seconds outputs/step0_finetune_cifar100_b0inc10_seed1993/wall_clock_seconds.txt \
  --output outputs/step0_finetune_cifar100_b0inc10_seed1993/metrics_summary.json \
  --dataset-requested CIFAR100 \
  --dataset-config-key cifar224 \
  --split B0-Inc10 \
  --seed 1993
```

## Files Created Or Edited

Created:

- `docs/step1_mote_repo_audit.md`
- `exps/step1_mote_cifar100_b0inc10_seed1993.json`
- `scripts/extract_metrics.py`
- `scripts/run_step1_mote_cifar100_b0inc10_seed1993.sh`

Edited:

- `scripts/extract_step0_metrics.py`

No DAAC code, mixed streams, close/far streams, or training logic changes were made.

## Risks And Uncertainty

- True MoTE appears absent from this local mirror. Running the prepared Step 1 script trains the repo's `mos` method under the requested Step 1 filename, not a verified MoTE implementation.
- If you intended a different MoTE method, the missing upstream files or branch must be added before a real MoTE run can be prepared.
- The run depends on server packages in `bupt-cil`, including `torch`, `torchvision`, `timm`, `numpy`, `tqdm`, `PIL`, and `easydict`.
- The script sets `HF_ENDPOINT=https://hf-mirror.com`, `HF_HOME`, `TRANSFORMERS_CACHE`, and `TORCH_HOME` for mirror/cache compatibility. Timm/Hugging Face weight resolution can still fail if the mirror lacks a required model file.
- The local mirror has broken `.git` metadata: `git status --short` fails because `.git` points to a missing module path. This did not affect static auditing or file creation.
