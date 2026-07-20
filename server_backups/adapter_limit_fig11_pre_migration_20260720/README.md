# Adapter-Limited MoTE pre-migration backup

Created before replacing the Vast.ai instance on 2026-07-20.

## State

- All training and download processes were stopped.
- Both GPUs were idle.
- The official MoTE checkout is recorded in `repository_identity.txt`.
- Five compatibility-modified source files are preserved byte-for-byte.
- The complete compatibility diff is `mote_compatibility.patch.gz`.
- All 35 Figure 11 JSON configurations and `PROTOCOL.md` are included.
- The first CIFAR adapter-limit-1 run did not reach training.
- Two accidental concurrent CIFAR downloads were stopped.
- The resulting partial CIFAR archive is quarantined and must not be reused.
- No experiment result was claimed from the interrupted run.

## Restore

1. Clone the main BUPT repository.
2. Clone the official MoTE repository at the commit in
   `repository_identity.txt` into `/workspace/BUPT/MoTE`.
3. Apply the patch:

       gzip -dc mote_compatibility.patch.gz |
         git -C /workspace/BUPT/MoTE apply -

4. Copy the configuration directory:

       cp -a configs/adapter_limit_fig11          /workspace/BUPT/MoTE/exps/

5. Recreate the dataset and pretrain links and download the assets recorded
   in `asset_manifest.txt`.
6. Validate all 35 configurations.
7. Start exactly one CIFAR adapter-limit-1 seed-1993 smoke process.
8. Do not restore the quarantined partial CIFAR archive.

Environment-variable values were intentionally not archived to avoid saving
credentials or tokens.
