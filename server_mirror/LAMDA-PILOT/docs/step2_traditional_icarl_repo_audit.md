# Step 2 Repo Audit: Traditional iCaRL CIFAR100 B0-Inc10 Seed 1993

## Decision

This LAMDA-PILOT checkout does **not** currently provide a valid true traditional CNN/ResNet iCaRL run configuration.

The repo does implement `model_name: "icarl"` in `models/icarl.py`, but the official iCaRL CIFAR config uses a pre-trained ViT backbone:

- `model_name`: `icarl`
- `dataset`: `cifar224`
- `backbone_type`: `vit_base_patch16_224`
- backbone family: ViT/PTM, not CNN/ResNet

I did **not** create `exps/step2_icarl_cifar100_b0inc10_seed1993.json` or `scripts/run_step2_icarl_cifar100_b0inc10_seed1993.sh`, because doing so would require inventing a CNN/ResNet iCaRL path that this repo does not actually expose.

## Evidence

### Official iCaRL Config

`exps/icarl.json` is the only official CIFAR iCaRL config inspected. It sets:

```json
{
  "prefix": "reproduce",
  "dataset": "cifar224",
  "memory_size": 2000,
  "memory_per_class": 20,
  "fixed_memory": false,
  "shuffle": true,
  "init_cls": 10,
  "increment": 10,
  "model_name": "icarl",
  "backbone_type": "vit_base_patch16_224",
  "device": ["0"],
  "seed": [1993]
}
```

This is CIFAR100 data resized/transformed for ViT, not a traditional CIFAR100 CNN/ResNet setup.

The other inspected baseline configs requested for comparison also point at ViT/PTM CIFAR settings rather than a traditional CNN backbone: `exps/finetune.json`, `exps/der.json`, `exps/dgr.json`, `exps/foster.json`, and `exps/memo.json` all use `dataset: "cifar224"` with a `vit_base_patch16_224`-family `backbone_type`.

### iCaRL Implementation

`models/icarl.py` constructs:

```python
self._network = IncrementalNet(args, True)
```

It does use standard iCaRL-style components:

- supervised cross-entropy on the current classifier
- knowledge distillation from `_old_network`
- rehearsal memory via `build_rehearsal_memory`
- NME support through exemplar class means inherited from `BaseLearner`

The implementation itself is not MoTE, MOS, or semantic communication. The problem is the backbone selected by the repo config and factory path.

### Backbone Factory

`utils/inc_net.py` is the decisive file. `get_backbone(args, pretrained=False)` handles ViT and ViT adaptation strings such as:

- `vit_base_patch16_224`
- `vit_base_patch16_224_in21k`
- `_memo`
- `_ssf`
- `_vpt`
- `_adapter`
- `_l2p`
- `_dualprompt`
- `_coda_prompt`
- `_ease`
- `_lae`
- `_mos`
- `_tuna`

The fallback is:

```python
raise NotImplementedError("Unknown type {}".format(name))
```

There is no `resnet18`, `resnet32`, `resnet`, or `convnet_type` construction branch in `get_backbone()`.

### ResNet File

`backbone/resnet.py` exists and defines ImageNet-style ResNet variants:

- `resnet18`
- `resnet34`
- `resnet50`
- `resnet101`
- `resnet152`
- `resnext50_32x4d`
- `resnext101_32x8d`
- `wide_resnet50_2`
- `wide_resnet101_2`

However, this file is not imported by `utils/inc_net.py` or selected by any inspected experiment config. A repo-wide search found no live use of `backbone/resnet.py` outside the file itself.

Also note that this is not ResNet32. The CIFAR-specific first-convolution branch is commented out, and the active network uses a 7x7 stride-2 stem plus max-pool. Even wiring this file in would not be the classic CIFAR ResNet32 iCaRL baseline without additional implementation work.

## CIFAR100 Dataset Keys

The repo has two CIFAR100-related dataset keys:

- `cifar100`: uses torchvision CIFAR100 with 32x32 transforms and CIFAR100 normalization.
- `cifar224`: uses torchvision CIFAR100 with 224x224 transforms for ViT/PTM runs.

Both load from project-local `./data` through torchvision:

```python
datasets.cifar.CIFAR100("./data", train=True, download=True)
datasets.cifar.CIFAR100("./data", train=False, download=True)
```

On the server, the given dataset location is compatible with this convention:

```text
/home/kennycao12/Python_Projects/BUPT/Chris/LAMDA-PILOT/data/cifar-100-python
```

The desired traditional baseline should use `dataset: "cifar100"` if a real CNN/ResNet backbone exists. In this repo, the official iCaRL config instead uses `dataset: "cifar224"` because it is a ViT/PTM run.

## B0-Inc10 Representation

The B0-Inc10 CIFAR100 setting is represented by:

- `init_cls: 10`
- `increment: 10`
- `shuffle: true`
- `seed: [1993]`

In `trainer.py`, when `init_cls == increment`, the log directory component is normalized to `0`:

```python
init_cls = 0 if args["init_cls"] == args["increment"] else args["init_cls"]
```

So the official iCaRL config logs under a `0/10` directory even though the config says `init_cls: 10`.

## Memory / Exemplar Setting

For the official CIFAR iCaRL config:

- `memory_size`: `2000`
- `memory_per_class`: `20`
- `fixed_memory`: `false`

Because `fixed_memory` is false, `BaseLearner.samples_per_class` computes:

```python
memory_size // total_classes
```

So the total exemplar budget is 2000, dynamically distributed across seen classes. The `memory_per_class: 20` value is present in the JSON but is not the active memory rule for CIFAR iCaRL while `fixed_memory` is false.

## Expected Log Path

For the official, existing iCaRL config:

```text
logs/icarl/cifar224/0/10/reproduce_1993_vit_base_patch16_224.log
```

For the requested Step 2 artifact, no log path is created because no valid traditional CNN/ResNet iCaRL config was created.

## Required Dependencies

From `README.md`, the repo expects:

- `torch 2.0.1`
- `torchvision 0.15.2`
- `timm 0.6.12`
- `tqdm`
- `numpy`
- `scipy`
- `easydict`

The server environment supplied by the user is:

```bash
conda activate bupt-cil
```

No full training was run locally.

## Files Inspected

- `README.md`
- `exps/icarl.json`
- `exps/icarl_inr.json`
- `exps/finetune.json`
- `exps/der.json`
- `exps/dgr.json`
- `exps/foster.json`
- `exps/memo.json`
- `models/icarl.py`
- `models/finetune.py`
- `models/der.py`
- `models/foster.py`
- `models/memo.py`
- `models/base.py`
- `utils/factory.py`
- `utils/inc_net.py`
- `utils/data.py`
- `utils/data_manager.py`
- `trainer.py`
- `backbone/resnet.py`
- `backbone/`

## Uncertainty

- The repo contains dormant CNN-related checks such as `if 'resnet' in args['backbone_type']`, but there is no live backbone factory branch to instantiate a ResNet.
- `backbone/resnet.py` may be leftover or intended for future use. It is not enough to make a valid traditional iCaRL run without code changes.
- The exact paths `/Users/ahol1c/BUPT/BUPT_COOP/server_mirror/tae` and `/Users/ahol1c/BUPT/BUPT_COOP/server_mirror/tae_longtail_cil` were not present locally during this audit. A similarly named local directory exists at `/Users/ahol1c/BUPT/BUPT_COOP/tae_longtail_cil`, but it appears to be a long-tailed CIFAR-10 TaE study rather than an iCaRL/ResNet32 CIL baseline repo.

## Recommendation

Do not use LAMDA-PILOT for Step 2 traditional iCaRL unless we first implement and verify a real CNN backbone path in `utils/inc_net.py` and add a CIFAR-style ResNet, preferably ResNet32, with matching transforms and hyperparameters.

For the next baseline source, use a PyCIL-style traditional CIL repo/config if available. PyCIL is the most appropriate direction because LAMDA-PILOT itself acknowledges PyCIL and its iCaRL implementation pattern is the traditional CIL style we need: CIFAR100, `init_cls: 10`, `increment: 10`, exemplar memory, and a CNN/ResNet backbone suitable for inserting future semantic channel/noise/compression modules around encoder features, bottleneck features, or classifier inputs.

Among the named local options:

- `/Users/ahol1c/BUPT/BUPT_COOP/server_mirror/tae`: not present locally during this audit.
- `/Users/ahol1c/BUPT/BUPT_COOP/server_mirror/tae_longtail_cil`: not present locally during this audit.
- `/Users/ahol1c/BUPT/BUPT_COOP/tae_longtail_cil`: present, but appears focused on long-tailed CIFAR-10 TaE/CEd studies, not traditional CIFAR100 iCaRL.

Therefore the recommended next step is to locate or clone a PyCIL-style traditional CIL implementation, then prepare a true CIFAR100 B0-Inc10 seed 1993 iCaRL ResNet32 config there.
