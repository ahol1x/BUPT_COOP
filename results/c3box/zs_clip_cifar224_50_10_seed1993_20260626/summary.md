# C3Box ZS-CLIP CIFAR224 Result

Date: 2026-06-26  
Server: Vast.ai RTX 5090  
Repo: LAMDA-CL/C3Box  
Config: `exps/zs_clip.json`  
Dataset: CIFAR100 resized to 224  
Setting: B50 Inc10  
Seed: 1993  
Backbone: CLIP ViT-B/16, laion400m_e32  
Model: ZS-CLIP  

## Final Results

Final Top-1 Accuracy: 71.38  
Average Accuracy: 76.48666666666666  

Top-1 Curve:

```text
[81.06, 79.68, 78.39, 75.85, 72.56, 71.38]

```

Top-5 Curve:

```text
[96.5, 95.82, 95.11, 94.26, 93.07, 92.15]
```

## Notes

This is a successful zero-shot CLIP baseline reproduction from C3Box.
No training was performed; this result is mostly evaluation-based.
