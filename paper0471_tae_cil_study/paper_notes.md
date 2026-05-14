# Paper Notes

## 0471: TaE

`0471.pdf` proposes Task-aware Expandable Representation for long-tail class-incremental learning. The key implementable idea is to treat the previous task model as the starting point for the next task, run the incoming task data through it, accumulate parameter gradients, rank parameters by sensitivity, and update only the top `p%`. The requested budget is `15%`, so this study uses a top-15% gradient mask for each task after the first.

The paper also introduces CEd, a centroid-enhanced loss. New class centroids are learned during the task, old centroids are retained, features are pulled toward their own centroid, and centroids are pushed apart. The study implements this with cosine feature-to-centroid and centroid-to-centroid losses.

The original paper evaluates CIFAR100-LT and ImageNet100-LT with long-tail training distributions. This study adapts the idea to a faster 10-class CIFAR-10 CIL run so that 10 seeded studies can finish locally while preserving the core comparison.

## KBS-MoTE

MoTE argues for task-specific experts under PTM-based CIL, then filters unreliable experts at inference and weights reliable experts. This study does not implement MoTE directly because the requested model is paper 0471 and the toolbox should remain a clean base. The relevant idea retained here is task-specific parameter isolation: TaE-style masks keep most previous parameters fixed while the current task updates a selected subset.

## LDEPrompt

LDEPrompt uses layer-importance guided prompt placement and dual expandable prompt pools. The relevant idea is to make the expandable capacity task-specific and avoid overwriting old task parameters. In this study, the selected TaE mask is recomputed per task and non-selected parameters are frozen for that task.

## CVPR26 QKD

QKD uses a task-relevance gate for cross-task knowledge distillation and inference routing. This is PTM/adaptor-oriented and is not implemented in the 0471 reproduction. The relevant lesson is that task relation should affect transfer; in this standalone study, task order is shuffled in case 2 to stress the method under distribution shift.
