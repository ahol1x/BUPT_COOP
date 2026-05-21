# DAAC Paper Mapping

DAAC, Difficulty-Aware Adaptive Controller, is a first prototype for feasibility testing in PTM-based class-incremental learning. The point is not to stack QKD, TaE, MoTE, and LDEPrompt. DAAC uses their mechanisms as a menu of actions, then selects one action per incoming task/class after a lightweight pre-study pass.

## Repository Context

The workspace contains two relevant codebases:

- `LAMDA-PILOT`: a PTM-based CIL framework with dataset/task split logic in `utils/data_manager.py`, model dispatch in `utils/factory.py`, base CIL learner utilities in `models/base.py`, prompt methods in `models/l2p.py`, `models/dualprompt.py`, and `backbone/prompt.py`, adapter methods in `models/aper_adapter.py`, adapter expansion in `models/ease.py` and `backbone/vit_ease.py`, and MoE-like adapter behavior in `models/mos.py`.
- `tae_longtail_cil`: a compact TaE study implementation with an explicit long-tailed CIL experiment loop, gradient-mask style TaE selection, result CSV/JSON writing, and plotting/report generation patterns.

This first DAAC prototype is built as a new standalone folder because the local environment does not currently provide `torchvision` or `timm`, which the existing `LAMDA-PILOT` entrypoints require. The implementation keeps the same conceptual boundaries: incremental task datasets, a frozen PTM-like backbone, prompt/light modules, adapter experts, prototype memory, a pre-study estimator, strategy selection, and task-level metrics. CIFAR100 support is optional and will use local CIFAR files or `torchvision` when available; fast debug uses an offline synthetic CIL dataset.

## TaE: Task-Aware Expandable Representation

### Reusable Mechanism

TaE performs a pre-study before training a new task. It passes the current task data through the previous task model, computes cross-entropy gradients, accumulates gradient magnitudes over several loops, sorts parameters by magnitude, and selects the top `p%` most sensitive parameters. Only these selected task-aware parameters are expanded/updated while most parameters remain frozen.

TaE also proposes a Centroid-Enhanced method that maintains class centroids, freezes old centroids, learns new centroids, encourages intra-class compactness, and encourages inter-class separation.

### Difficulty Signal for DAAC

DAAC reuses TaE's gradient sensitivity as one difficulty signal:

- run forward/backward during pre-study without an optimizer step;
- collect gradient norms over candidate trainable parameters;
- normalize the aggregate value into `gradient_sensitivity`;
- use layer/parameter gradient magnitudes to build a top-p update mask when the selected strategy is `tae_top_p`.

### Used in DAAC

- Pre-study gradient pass.
- Top-p sensitive parameter update.
- Difficulty-dependent top-p schedule: 5%, 10%, or 20%.
- Prototype/centroid storage as the lightweight basis for novelty and classifier evaluation.

### Not Implemented in First Prototype

- Full TaE structural parameter expansion in the exact original sense.
- Full long-tailed CEd loss with all min/max/reweighting terms.
- Exemplar or rehearsal variants.

### Approximation

Top-p is implemented as gradient masking over existing trainable tensors. The mask keeps the largest-magnitude gradient elements and zeros the rest during training. This approximates TaE's partial parameter expansion/update without changing backbone tensor shapes.

## LDEPrompt: Layer-Importance Guided Dual Expandable Prompt Pool

### Reusable Mechanism

LDEPrompt performs multiple forward passes before training each task, estimates per-layer information gain, normalizes layer scores with softmax, and selects layers whose importance is above the mean. It uses two prompt pools:

- a frozen global prompt pool containing prompts from old tasks;
- a current task training prompt pool with fixed capacity.

For a new task, similar old prompts are retrieved from the global pool using cosine similarity between intermediate features, then new trainable prompts are added. After training, new prompts are frozen and merged into the global pool.

### Difficulty Signal for DAAC

DAAC reuses the layer-importance idea as `layer_importance_ratio`:

- compute a practical activation-change proxy per transformer block during pre-study;
- mark blocks above mean importance as important;
- set `layer_importance_ratio = important_layers / total_layers`.

This approximates information gain without estimating mutual information directly.

### Used in DAAC

- Prompt reuse/light update for easy tasks.
- Frozen old prompts plus task-local trainable prompts.
- Dynamic prompt expansion when prompt update is selected.
- Layer importance ratio as part of the difficulty score.

### Not Implemented in First Prototype

- Exact mutual information estimation `I(x; h_l) - I(x; h_{l-1})`.
- Full prefix-tuning insertion into pretrained ViT key/value matrices.
- A complete LDEPrompt dual-pool retrieval implementation for every ViT layer.

### Approximation

The prototype includes a compact prompt pool on top of a PTM-like frozen transformer. Layer importance is computed from activation deltas, and prompt reuse/expansion is represented by frozen global prompts plus current trainable prompt vectors. If ported into `LAMDA-PILOT`, this should map naturally to its L2P/DualPrompt/CODA prompt pool code.

## MoTE: Mixture of Task-Specific Experts

### Reusable Mechanism

MoTE treats one lightweight adapter per task as a task-specific expert. The PTM backbone is frozen, each new adapter is trained independently, and class prototypes are computed from task features after training. At inference, all experts can produce features. Expert filtering marks an expert reliable if its predicted class lies within that expert's task label space. If multiple experts are reliable, the sample is considered task-ambiguous and MoTE performs weighted multi-expert inference.

MoTE weights experts by confidence and self-confidence score (SCS), where SCS is the gap between the top logit and the second logit relative to the top logit. Adapter-Limited MoTE motivates not adding a new adapter for every task because adapter count grows linearly with tasks and may be unnecessary for easy or related tasks.

### Difficulty Signal for DAAC

DAAC reuses MoTE's ambiguity idea:

- run available experts on pre-study samples;
- compute expert confidence scores against class prototypes;
- compute entropy over expert scores or use the reliable-expert ratio;
- report this as `expert_ambiguity`.

High ambiguity means multiple old experts look plausible, so fusion or relevance-guided transfer may be more useful than a single update path.

### Used in DAAC

- `new_adapter` strategy for hard and novel tasks.
- `weighted_expert_fusion` strategy for ambiguous tasks.
- Expert filtering based on task label scope.
- Confidence/SCS-style weighting for multi-expert inference.
- Adapter-limited motivation: DAAC only adds adapters when the controller judges the task hard and novel, except fixed baselines that intentionally add one every task.

### Not Implemented in First Prototype

- Exact MoTE architecture from the paper.
- Full adapter-limited prototype synthesis for tasks beyond an adapter cap.
- Separate autoencoder/router alternatives.

### Approximation

The prototype maintains a list of adapter experts and per-class prototypes. Expert fusion uses cosine prototype similarity and confidence/SCS-style weights. This captures MoTE's sparse filtering and reliable joint inference behavior without modifying the existing `LAMDA-PILOT` adapter internals.

## QKD: Quantum-Gated Task-Interaction Knowledge Distillation

### Reusable Mechanism

QKD builds compact task embeddings from task adapters, maps sample features and task embeddings through a learnable quantum-gated task modulation module, measures sample-to-task relevance, normalizes relevance with softmax, and uses those coefficients both for training-time task-interaction knowledge distillation and inference-time adaptive adapter fusion.

### Difficulty Signal for DAAC

DAAC uses QKD-style relevance as a classical signal:

- compare current task features with old class/task prototypes by cosine similarity;
- novelty is `1 - max cosine similarity to old prototypes`;
- expert fusion weights can use cosine relevance to task prototypes or adapter/task embeddings.

### Used in DAAC

- Novelty estimation.
- Expert weighting for ambiguous samples.
- Optional feature/logit distillation weighting from relevant old experts when strategy is `weighted_expert_fusion` or `all_combined`.

### Not Implemented in First Prototype

- No quantum circuit.
- No SVD-derived adapter task states.
- No trainable quantum gate, quantum fidelity, quantum sparsity loss, or full QKD optimization objective.

### Approximation

This prototype explicitly documents relevance as a classical approximation of QKD-style sample-to-task relevance. Cosine similarity between features and task/class prototypes replaces quantum fidelity. This keeps the controller lightweight and testable before adding a more expressive relevance estimator.

## DAAC Controller Summary

DAAC computes:

```text
difficulty_score =
    w_novelty * novelty
  + w_entropy * entropy
  + w_grad * gradient_sensitivity
  + w_layer * layer_importance_ratio
  + w_ambiguity * expert_ambiguity
```

Default weights are:

- `w_novelty = 0.35`
- `w_entropy = 0.20`
- `w_grad = 0.25`
- `w_layer = 0.10`
- `w_ambiguity = 0.10`

Default strategy priority:

1. `base_train` for the first task.
2. `new_adapter` for hard and highly novel tasks.
3. `weighted_expert_fusion` for ambiguous tasks.
4. `tae_top_p` for medium difficulty tasks.
5. `prompt_or_light_update` for easy tasks.

## Prototype Assumptions

- The first task is task id 1 in logs and maps to `base_train`.
- The fast debug path uses a tiny frozen transformer-like backbone rather than a downloaded PTM.
- CIFAR100 full runs require local CIFAR100 files or optional `torchvision`; if neither is present, the script fails with a clear setup message.
- Prompt pools, adapters, and prototypes are implemented in a compact prototype form that is easy to port into `LAMDA-PILOT`.
- Metrics are intended for feasibility and ablation comparisons, not SOTA claims.
