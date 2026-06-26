# Vast C3Box Backup - 2026-06-26

## Completed result

C3Box ZS-CLIP on CIFAR100/cifar224 B50-Inc10 seed 1993 completed successfully.

Final Top-1: 71.38
Average Accuracy: 76.48666666666666
Top1 curve: [81.06, 79.68, 78.39, 75.85, 72.56, 71.38]
Top5 curve: [96.5, 95.82, 95.11, 94.26, 93.07, 92.15]

This result was already committed and pushed to GitHub:
commit 497ffd8

## Fixed L2P config

Use:
exps/l2p_cifar224.json

Important fixes:
- dataset: cifar224
- init_cls: 50
- increment: 10
- seed: [1993]
- device: ["0"]
- total_class_num: 100
- nb_classes: 100

## Recreate C3Box env next time

conda create -p /venv/c3box python=3.10 -y
conda activate /venv/c3box
python -m pip install -U pip setuptools wheel
python -m pip install torch torchvision torchaudio
python -m pip install "timm==0.9.12" tqdm numpy scipy easydict "open-clip-torch==2.17.1" ftfy regex matplotlib pillow scikit-learn pandas pyyaml

## Run ZS-CLIP

cd /workspace/BUPT/test/C3Box
CUDA_VISIBLE_DEVICES=0 python -u main.py \
  --config=./exps/zs_clip.json \
  2>&1 | tee /workspace/runs/c3box_logs/zs_clip_cifar_$(date +%F_%H%M).log

## Run L2P

cd /workspace/BUPT/test/C3Box
CUDA_VISIBLE_DEVICES=0 python -u main.py \
  --config=./exps/l2p_cifar224.json \
  2>&1 | tee /workspace/runs/c3box_logs/l2p_cifar224_$(date +%F_%H%M).log
