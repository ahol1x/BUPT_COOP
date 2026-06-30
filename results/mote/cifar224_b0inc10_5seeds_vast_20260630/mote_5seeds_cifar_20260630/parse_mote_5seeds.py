import re
import json
import statistics
from pathlib import Path

log_dir = Path("/workspace/runs/mote_5seed_logs")
out_dir = Path("/workspace/export/mote_5seeds_cifar_20260630")
out_dir.mkdir(parents=True, exist_ok=True)

seeds = [1991, 1992, 1993, 1994, 1995]

def floats_from_bracket(text):
    return [float(x) for x in re.findall(r"[-+]?\d*\.\d+|[-+]?\d+", text)]

def parse_log(path):
    text = path.read_text(errors="ignore")

    result = {
        "log": str(path),
        "avg": None,
        "last": None,
        "forgetting": None,
        "top1_curve": None,
        "top5_curve": None,
    }

    # top1/top5 curve
    top1_matches = re.findall(r"top1(?:\s*curve)?[^:\n]*:\s*(\[[^\]]+\])", text, flags=re.I)
    top5_matches = re.findall(r"top5(?:\s*curve)?[^:\n]*:\s*(\[[^\]]+\])", text, flags=re.I)

    if top1_matches:
        result["top1_curve"] = floats_from_bracket(top1_matches[-1])
        if result["top1_curve"]:
            result["avg"] = sum(result["top1_curve"]) / len(result["top1_curve"])
            result["last"] = result["top1_curve"][-1]

    if top5_matches:
        result["top5_curve"] = floats_from_bracket(top5_matches[-1])

    # common MoTE/LAMDA-style lines
    avg_patterns = [
        r"Average Accuracy[^0-9\-+]*([-+]?\d*\.\d+|[-+]?\d+)",
        r"Avg[^0-9\-+]*([-+]?\d*\.\d+|[-+]?\d+)",
        r"average[^0-9\-+]*accuracy[^0-9\-+]*([-+]?\d*\.\d+|[-+]?\d+)",
    ]
    last_patterns = [
        r"Final[^0-9\-+]*accuracy[^0-9\-+]*([-+]?\d*\.\d+|[-+]?\d+)",
        r"Last[^0-9\-+]*([-+]?\d*\.\d+|[-+]?\d+)",
    ]
    forget_patterns = [
        r"Forgetting[^0-9\-+]*([-+]?\d*\.\d+|[-+]?\d+)",
        r"forget[^0-9\-+]*([-+]?\d*\.\d+|[-+]?\d+)",
    ]

    for pat in avg_patterns:
        m = re.findall(pat, text, flags=re.I)
        if m:
            result["avg"] = float(m[-1])
            break

    for pat in last_patterns:
        m = re.findall(pat, text, flags=re.I)
        if m:
            result["last"] = float(m[-1])
            break

    for pat in forget_patterns:
        m = re.findall(pat, text, flags=re.I)
        if m:
            result["forgetting"] = float(m[-1])
            break

    return result

rows = []
for seed in seeds:
    logs = sorted(log_dir.glob(f"mote_cifar_seed{seed}_*.log"))
    if not logs:
        rows.append({"seed": seed, "error": "missing log"})
        continue
    path = logs[-1]
    r = parse_log(path)
    r["seed"] = seed
    rows.append(r)

def mean_std(values):
    values = [v for v in values if v is not None]
    if not values:
        return None, None
    if len(values) == 1:
        return values[0], 0.0
    return statistics.mean(values), statistics.stdev(values)

avg_mean, avg_std = mean_std([r.get("avg") for r in rows])
last_mean, last_std = mean_std([r.get("last") for r in rows])
forget_mean, forget_std = mean_std([r.get("forgetting") for r in rows])

summary = {
    "dataset": "CIFAR100/cifar224",
    "setting": "B0-Inc10",
    "model": "MoTE",
    "seeds": seeds,
    "rows": rows,
    "mean_std": {
        "avg": [avg_mean, avg_std],
        "last": [last_mean, last_std],
        "forgetting": [forget_mean, forget_std],
    },
}

(out_dir / "metrics_summary.json").write_text(json.dumps(summary, indent=2))

lines = []
lines.append("# MoTE CIFAR100 B0-Inc10 five-seed reproduction\n")
lines.append("\n")
lines.append("| Seed | Avg | Last | Forgetting | Log |\n")
lines.append("|---:|---:|---:|---:|---|\n")
for r in rows:
    lines.append(
        f"| {r.get('seed')} | "
        f"{'' if r.get('avg') is None else f'{r.get('avg'):.4f}'} | "
        f"{'' if r.get('last') is None else f'{r.get('last'):.4f}'} | "
        f"{'' if r.get('forgetting') is None else f'{r.get('forgetting'):.4f}'} | "
        f"{Path(r.get('log', '')).name} |\n"
    )

lines.append("\n")
lines.append("## Mean ± std\n\n")
lines.append(f"- Avg: {avg_mean:.4f} ± {avg_std:.4f}\n" if avg_mean is not None else "- Avg: parse failed\n")
lines.append(f"- Last: {last_mean:.4f} ± {last_std:.4f}\n" if last_mean is not None else "- Last: parse failed\n")
lines.append(f"- Forgetting: {forget_mean:.4f} ± {forget_std:.4f}\n" if forget_mean is not None else "- Forgetting: parse failed\n")

(out_dir / "summary.md").write_text("".join(lines))

print(json.dumps(summary, indent=2))
print(f"\nWrote: {out_dir / 'metrics_summary.json'}")
print(f"Wrote: {out_dir / 'summary.md'}")
