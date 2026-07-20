# Adapter-Limited MoTE Figure 11 Protocol

Paper target: Figure 11, seed 1993, ViT-B/16-IN21K.

Sweeps:

- CIFAR100 B0-Inc10: adapter limits 1-10
- ImageNet-A B0-Inc20: adapter limits 1-10
- ImageNet-R B0-Inc20: adapter limits 1-10
- VTAB B0-Inc10: adapter limits 1-5

Primary protocol:

Use the repository-intended per-dataset hyperparameters while correcting
broken model names, dataset names, split fields, backbone routing, and
missing learner initialization.

Known paper/code discrepancy:

Section 5.1.4 states SGD, learning rate 0.01, batch size 48, weight decay
0.005, 20 epochs, and cosine scheduling. The released repository uses
dataset-specific learning rates, batch sizes, weight decay, and VTAB
epochs. Figure 11 reproduction first follows the repository-intended
settings because the plotted values were produced by this code path.
A literal-paper hyperparameter sensitivity run may be added separately.

Figure 11 reference Average Accuracy:

- CIFAR100: [92.22, 88.54, 88.27, 89.70, 90.63, 91.12, 91.14, 92.24, 92.84, 93.04]
- ImageNet-A: [60.25, 62.66, 62.35, 64.44, 65.04, 64.82, 64.87, 65.17, 65.41, 68.12]
- ImageNet-R: [75.28, 75.31, 75.58, 76.59, 77.61, 79.44, 80.27, 81.39, 81.62, 81.89]
- VTAB: [90.80, 87.74, 88.16, 90.77, 93.56]
