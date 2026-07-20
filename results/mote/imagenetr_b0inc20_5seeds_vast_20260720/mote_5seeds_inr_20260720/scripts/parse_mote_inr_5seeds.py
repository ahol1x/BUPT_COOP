import gzip
import re
import json
from pathlib import Path
import numpy as np

log_dir = Path(__file__).resolve().parents[1] / "raw_logs"
out_dir = Path(__file__).resolve().parents[1]
out_dir.mkdir(parents=True, exist_ok=True)

def extract_numbers_from_list(s):
    s = re.sub(r"np\.float64\(([^()]*)\)", r"\1", s)
    return [float(x) for x in re.findall(r"[-+]?\d+\.\d+|[-+]?\d+", s)]

def parse_log(path):
    text = gzip.open(path, "rt", errors="ignore").read()

    seed_match = re.search(r"seed(\d+)", path.name)
    seed = int(seed_match.group(1)) if seed_match else None

    avg_matches = re.findall(r"Average Accuracy \(CNN\):\s*([0-9.]+)", text)
    forget_matches = re.findall(r"Forgetting \(CNN\):\s*([0-9.]+)", text)
    top1_matches = re.findall(r"CNN top1 curve:\s*(\[[^\n]+\])", text)
    top5_matches = re.findall(r"CNN top5 curve:\s*(\[[^\n]+\])", text)

    if not avg_matches or not forget_matches or not top1_matches:
        raise RuntimeError(f"Incomplete log: {path}")

    top1 = extract_numbers_from_list(top1_matches[-1])
    top5 = extract_numbers_from_list(top5_matches[-1]) if top5_matches else []

    return {
        "seed": seed,
        "log": str(path),
        "average_accuracy_cnn": float(avg_matches[-1]),
        "last_accuracy_cnn": float(top1[-1]),
        "forgetting_cnn": float(forget_matches[-1]),
        "top1_curve": top1,
        "top5_curve": top5,
    }

# Pick the latest complete log for each seed
by_seed = {}
for p in sorted(log_dir.glob("seed*.log.gz"), key=lambda x: x.stat().st_mtime):
    m = re.search(r"seed(\d+)", p.name)
    if not m:
        continue
    seed = int(m.group(1))
    try:
        parsed = parse_log(p)
        by_seed[seed] = parsed
    except Exception as e:
        print(f"skip incomplete {p.name}: {e}")

results = [by_seed[s] for s in sorted(by_seed)]
if sorted(by_seed) != [1991, 1992, 1993, 1994, 1995]:
    raise RuntimeError(f"Expected seeds 1991-1995, got {sorted(by_seed)}")

avg = np.array([r["average_accuracy_cnn"] for r in results], dtype=float)
last = np.array([r["last_accuracy_cnn"] for r in results], dtype=float)
forget = np.array([r["forgetting_cnn"] for r in results], dtype=float)

summary = {
    "experiment": "MoTE ImageNet-R B0-Inc20 five-seed reproduction",
    "dataset": "ImageNet-R",
    "setting": "B0-Inc20",
    "seeds": [r["seed"] for r in results],
    "per_seed": results,
    "mean_std": {
        "average_accuracy_cnn": {
            "mean": float(avg.mean()),
            "std": float(avg.std(ddof=1)),
        },
        "last_accuracy_cnn": {
            "mean": float(last.mean()),
            "std": float(last.std(ddof=1)),
        },
        "forgetting_cnn": {
            "mean": float(forget.mean()),
            "std": float(forget.std(ddof=1)),
        },
    },
}

(out_dir / "metrics_summary.json").write_text(json.dumps(summary, indent=2))

lines = []
lines.append("# MoTE ImageNet-R B0-Inc20 Five-Seed Reproduction")
lines.append("")
lines.append("## Per-seed results")
lines.append("")
lines.append("| Seed | Avg Acc CNN | Last Acc CNN | Forgetting CNN |")
lines.append("|---:|---:|---:|---:|")
for r in results:
    lines.append(
        f"| {r['seed']} | {r['average_accuracy_cnn']:.4f} | "
        f"{r['last_accuracy_cnn']:.4f} | {r['forgetting_cnn']:.4f} |"
    )
lines.append("")
lines.append("## Mean +/- std")
lines.append("")
lines.append(
    f"- Average Accuracy CNN: {avg.mean():.4f} +/- {avg.std(ddof=1):.4f}"
)
lines.append(
    f"- Last Accuracy CNN: {last.mean():.4f} +/- {last.std(ddof=1):.4f}"
)
lines.append(
    f"- Forgetting CNN: {forget.mean():.4f} +/- {forget.std(ddof=1):.4f}"
)
lines.append("")
lines.append("## Curves")
for r in results:
    lines.append("")
    lines.append(f"### Seed {r['seed']}")
    lines.append(f"- Top1 curve: {r['top1_curve']}")
    lines.append(f"- Top5 curve: {r['top5_curve']}")

(out_dir / "summary.md").write_text("\n".join(lines))

print((out_dir / "summary.md").read_text())
