# Step 2 PyCIL Audit: Traditional iCaRL CIFAR100 B0-Inc10 Seed 1993

## Decision

PyCIL is suitable for Step 2 traditional iCaRL.

The official PyCIL iCaRL config already uses CIFAR100 with a true CNN backbone:

- method/model name: `icarl`
- Python class: `models.icarl.iCaRL`
- dataset key: `cifar100`
- convnet/backbone name: `resnet32`
- backbone implementation: `convs/cifar_resnet.py`
- model family: CIFAR ResNet/CNN, not ViT/PTM

This is the correct repo path for the requested traditional CNN-based CIL baseline.

## Exact Entry Command

From the server repo:

```bash
cd /home/kennycao12/Python_Projects/BUPT/Chris/PyCIL
conda activate bupt-cil
CUDA_VISIBLE_DEVICES=0 python main.py --config=exps/step2_icarl_cifar100_b0inc10_seed1993.json
```

The prepared wrapper script is:

```bash
bash scripts/run_step2_icarl_cifar100_b0inc10_seed1993.sh
```

## Config System

PyCIL uses JSON experiment configs under `exps/`.

`main.py` parses:

```bash
python main.py --config=./exps/[MODEL NAME].json
```

Then it loads the JSON and merges it into `args` before calling `trainer.train(args)`.

The official reference file for this run is `exps/icarl.json`. The Step 2 config created for this project is:

```text
exps/step2_icarl_cifar100_b0inc10_seed1993.json
```

## Step 2 Settings

```json
{
  "prefix": "step2",
  "dataset": "cifar100",
  "memory_size": 2000,
  "memory_per_class": 20,
  "fixed_memory": false,
  "shuffle": true,
  "init_cls": 10,
  "increment": 10,
  "model_name": "icarl",
  "convnet_type": "resnet32",
  "device": ["0"],
  "seed": [1993]
}
```

These values are copied from the official `exps/icarl.json` except:

- `prefix`: changed from `reproduce` to `step2`
- `device`: changed from `["0","1","2","3"]` to `["0"]` for the single RTX 4090 server GPU

## True CNN / ResNet Evidence

`utils/factory.py` maps:

```python
if name == "icarl":
    from models.icarl import iCaRL
    return iCaRL(args)
```

`models/icarl.py` constructs:

```python
self._network = IncrementalNet(args, False)
```

`utils/inc_net.py` maps:

```python
from convs.cifar_resnet import resnet32

if name == "resnet32":
    return resnet32()
```

`convs/cifar_resnet.py` defines a CIFAR-style ResNet:

- 3x3 input convolution
- CIFAR stages with 16, 32, and 64 channels
- depth 32 through `CifarResNet(ResNetBasicblock, 32)`
- pooled feature dimension: 64

No ViT, timm backbone, prompt module, adapter, MoTE, MOS, or PTM path is used for this config.

## B0-Inc10 Representation

The requested CIFAR100 B0-Inc10 setting is represented by:

- `init_cls: 10`
- `increment: 10`
- `shuffle: true`
- `seed: [1993]`

In `trainer.py`, the log directory uses `0` when `init_cls == increment`:

```python
init_cls = 0 if args["init_cls"] == args["increment"] else args["init_cls"]
```

So the expected PyCIL internal log path is:

```text
logs/icarl/cifar100/0/10/step2_1993_resnet32.log
```

## Memory / Exemplar Setting

The official iCaRL CIFAR100 memory setting is:

- `memory_size`: `2000`
- `memory_per_class`: `20`
- `fixed_memory`: `false`

Because `fixed_memory` is false, `BaseLearner.samples_per_class` uses:

```python
memory_size // total_classes
```

Thus the active rule is a total exemplar budget of 2000, dynamically divided across seen classes.

## Dataset Path

PyCIL's CIFAR100 loader uses torchvision:

```python
datasets.cifar.CIFAR100("./data", train=True, download=True)
datasets.cifar.CIFAR100("./data", train=False, download=True)
```

Torchvision expects CIFAR100 under:

```text
./data/cifar-100-python
```

The server already has CIFAR100 at:

```text
/home/kennycao12/Python_Projects/BUPT/Chris/LAMDA-PILOT/data/cifar-100-python
```

The Step 2 run script creates `./data` and symlinks that existing dataset into PyCIL when `./data/cifar-100-python` is missing. This avoids a repeated download.

## Expected Outputs

Wrapper log:

```text
logs/step2_icarl_cifar100_b0inc10_seed1993.log
```

PyCIL internal log:

```text
logs/icarl/cifar100/0/10/step2_1993_resnet32.log
```

Output directory:

```text
outputs/step2_icarl_cifar100_b0inc10_seed1993/
```

Metrics summary:

```text
outputs/step2_icarl_cifar100_b0inc10_seed1993/metrics_summary.json
```

The metrics extractor parses:

- Average Accuracy (CNN)
- final/last CNN top1 accuracy
- Forgetting (CNN)
- CNN top1 curve
- CNN top5 curve
- final total params, computed from the final 100-class ResNet32 iCaRL head when possible
- final trainable params, computed from the final 100-class ResNet32 iCaRL head when possible
- memory size and exemplar setting from config
- final exemplar size if present in logs
- wall-clock seconds
- peak GPU memory from `nvidia-smi`

## Dependencies

README-listed dependencies:

- Python 3.8
- torch
- torchvision
- tqdm
- numpy
- scipy
- quadprog
- POT

Dependencies used by this iCaRL path:

- torch
- torchvision
- numpy
- scipy
- tqdm
- Pillow / PIL

Additional optional dependencies discovered in the repo:

- `quadprog` for GEM
- `ot` / POT for COIL
- `sklearn` for FeTrIL

The server check script verifies the iCaRL-required packages, CUDA availability, and reports optional package availability.

## Files Inspected

- `README.md`
- `main.py`
- `trainer.py`
- `exps/icarl.json`
- `exps/rmm-icarl.json`
- `utils/factory.py`
- `utils/inc_net.py`
- `utils/data.py`
- `utils/data_manager.py`
- `utils/toolkit.py`
- `models/icarl.py`
- `models/base.py`
- `convs/cifar_resnet.py`
- `convs/resnet.py`
- `convs/conv_cifar.py`
- `convs/ucir_cifar_resnet.py`
- `convs/`

## Uncertainty

- PyCIL's `trainer._set_random()` uses a fixed random seed of `1`; the configured `seed: [1993]` is used for class-order shuffling in `DataManager`, matching the README description.
- iCaRL optimization hyperparameters are module-level constants in `models/icarl.py`, not JSON values. This is PyCIL's official style for iCaRL.
- Torchvision still receives `download=True`, but with the symlinked `./data/cifar-100-python` directory it should detect the existing dataset and avoid downloading.
- No full training was run locally.
