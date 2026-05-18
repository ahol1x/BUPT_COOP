# What This Study Implements From Paper 0471

Paper 0471 addresses long-tailed class-incremental learning, where each new task can contain head classes with many samples and tail classes with very few samples. Traditional full-update finetuning often overfits the new head classes and overwrites older class representations.

This folder implements three ideas from the paper:

1. Task-aware parameter selection: before each new task after task 1, the runner accumulates gradients on the incoming task and selects the top-p percent most sensitive parameters.
2. Frozen majority update: during that task, only the selected parameters are updated, while the rest of the model is protected from drift.
3. Centroid-enhanced representation: each class has a learnable centroid. The model pulls features toward their own class centroid and pushes different class centroids apart. A class reweighting term also reduces head-class dominance.

Expected benefit over the traditional baseline:

- Less catastrophic forgetting because most older parameters are not changed on every new task.
- Better tail-class feature separation because the centroid loss gives sparse tail classes a stronger geometric target.
- Lower adaptation cost than fully expanding a whole backbone for every task because only a selected parameter subset is trainable per task.

The code here is a compact, server-friendly study runner inspired by the paper and the LAMDA-PILOT baseline folder. It does not modify the downloaded LAMDA-PILOT submodule.
