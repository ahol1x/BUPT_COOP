# MoTE Vast Result - 2026-06-29

## Setting

- Method: MoTE
- Dataset: CIFAR100 resized to 224 / cifar224
- Incremental setting: B0-Inc10
- Seed: 1993
- Backbone: ViT-B/16 IN21K
- GPU: RTX 5090
- PyTorch: 2.11.0+cu128
- CUDA: 12.8

## Result

Final CNN Top-1: 88.33

Average Accuracy (CNN): 92.977

Forgetting (CNN): 4.777777777777778

CNN top1 curve:
[98.7, 97.0, 95.87, 94.4, 93.12, 92.4, 91.83, 89.58, 88.54, 88.33]

CNN top5 curve:
[99.9, 99.55, 99.47, 99.38, 99.24, 99.05, 99.0, 98.8, 98.53, 98.39]

## Notes

Compatibility patches were needed for the current Vast/Python/timm environment:

1. Changed config:
   - model_name: moe -> mote
   - backbone_type: vit_base_patch16_224_in21k_moe -> vit_base_patch16_224_in21k_mote
   - device: ["1"] -> ["0"]

2. Patched timm import:
   - backbone/linears.py uses `from timm.layers import trunc_normal_`

3. Patched missing / optional imports:
   - utils/inc_net.py makes `backbone.prompt.CodaPrompt` optional

4. Patched missing Learner attributes in models/mote.py:
   - moni_adam
   - use_init_ptm
   - use_reweight
   - use_old_data
   - alpha
   - beta
   - recalc_sim
   - adapter_num

5. Patched eval output unpacking:
   - `_network.forward(inputs, test=True)` returns one output dict, not a tuple
