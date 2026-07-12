# LDEPrompt CIFAR100 B0-Inc10 Multi-Seed Reproduction

Date: 2026-07-12  
Hardware: Vast.ai 2x RTX 5090  
Method: LDEPrompt  
Dataset: CIFAR100  
Setting: B0-Inc10  
Config base: repo default `exps/cifar/ldeprompt.json` with seed changed per run.

## Valid Seeds

| Seed | Last Top-1 | Average Accuracy | Forgetting |
|---:|---:|---:|---:|
| 1991 | 85.73 | 90.939 | 8.678 |
| 1993 | 86.73 | 91.765 | 7.133 |
| 1995 | 87.37 | 91.118 | 7.122 |
| 1997 | 86.45 | 88.757 | 8.244 |
| 1999 | 86.85 | 91.746 | 7.667 |
| 2001 | 86.55 | 90.514 | 7.600 |
| 2003 | 86.96 | 89.701 | 6.544 |
| 2005 | 86.58 | 92.285 | 8.011 |
| 2007 | 86.28 | 91.812 | 7.656 |

## Summary Across 9 Seeds

| Metric | Mean ± Std |
|---|---:|
| Last Top-1 | 86.61 ± 0.46 |
| Average Accuracy | 90.96 ± 1.14 |
| Forgetting | 7.63 ± 0.64 |

## Notes

- Seed 1993 was from the earlier single-seed run.
- Seeds 1991, 1995, 1997, 1999, 2001, 2003, 2005, and 2007 were run on the 2x RTX 5090 Vast instance.
- Backup archive: `server_backups/ldeprompt/ldeprompt_multiseed_5090_backup_2026-07-12.tar.gz`
- Backup SHA256: `11e02dcf2c7e837b42216785adbd5c34b9f8add5b028d9f28b29245425b0d373`
- This is a repo-default reproduction, not a confirmed paper-exact hyperparameter reproduction.
