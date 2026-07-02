import re
import json
import ast
from pathlib import Path
from statistics import mean, stdev

log_dir = Path("/workspace/runs/mote_vtab_5seed_logs")
out_dir = Path("/workspace/export/mote_5seeds_vtab_20260702")
out_dir.mkdir(parents=True, exist_ok=True)

def parse_curve(s):
    s = re.sub(r"np\.float64\(([^)]+)\)", r"\1", s)
    return [float(x) for x in ast.literal_eval(s)]

def parse_log(path):
    text = path.read_text(errors="ignore")
    seed = int(re.search(r"seed(\d+)", path.name).group(1))

    avg_matches = re.findall(r"Average Accuracy \(CNN\):\s*([0-9.]+)", text)
    forget_matches = re.findall(r"Forgetting \(CNN\):\s*([0-9.]+)", text)
    top1_matches = re.findall(r"CNN top1 curve:\s*(\[[^\n]+\])", text)
    top5_matches = re.findall(r"CNN top5 curve:\s*(\[[^\n]+\])", text)

    if not avg_matches or not forget_matches or not top1_matches or not top5_matches:
        raise RuntimeError(f"Incomplete log: {path}")

    top1 = parse_curve(top1_matches[-1])
    top5 = parse_curve(top5_matches[-1])

    if len(top1) != 5:
        raise RuntimeError(f"Expected 5 VTAB top1 points for {path.name}, got {len(top1)}: {top1}")

    return {
        "seed": seed,
        "log": str(path),
        "average_accuracy_cnn": float(avg_matches[-1]),
        "last_accuracy_cnn": float(top1[-1]),
        "forgetting_cnn": float(forget_matches[-1]),
        "top1_curve": top1,
        "top5_curve": top5,
    }

by_seed = {}
for p in sorted(log_dir.glob("mote_vtab_seed*.log"), key=lambda x: x.stat().st_mtime):
    seed = int(re.search(r"seed(\d+)", p.name).group(1))
    by_seed[seed] = parse_log(p)

expected = [1991, 1992, 1993, 1994, 1995]
if sorted(by_seed) != expected:
    raise RuntimeError(f"Expected seeds {expected}, got {sorted(by_seed)}")

results = [by_seed[s] for s in expected]

avg = [r["average_accuracy_cnn"] for r in results]
last = [r["last_accuracy_cnn"] for r in results]
forget = [r["forgetting_cnn"] for r in results]

summary = {
    "experiment": "MoTE VTAB B0-Inc10 five-seed reproduction",
    "dataset": "VTAB processed 50-class CIL",
    "setting": "B0-Inc10",
    "seeds": expected,
    "per_seed": results,
    "mean_std": {
        "average_accuracy_cnn": {"mean": mean(avg), "std": stdev(avg)},
        "last_accuracy_cnn": {"mean": mean(last), "std": stdev(last)},
        "forgetting_cnn": {"mean": mean(forget), "std": stdev(forget)},
    },
}

(out_dir / "metrics_summary.json").write_text(json.dumps(summary, indent=2))

lines = [
    "# MoTE VTAB B0-Inc10 Five-Seed Reproduction",
    "",
    "## Per-seed results",
    "",
    "| Seed | Avg Acc CNN | Last Acc CNN | Forgetting CNN |",
    "|---:|---:|---:|---:|",
]

for r in results:
    lines.append(f"| {r['seed']} | {r['average_accuracy_cnn']:.4f} | {r['last_accuracy_cnn']:.4f} | {r['forgetting_cnn']:.4f} |")

lines += [
    "",
    "## Mean +/- std",
    "",
    f"- Average Accuracy CNN: {mean(avg):.4f} +/- {stdev(avg):.4f}",
    f"- Last Accuracy CNN: {mean(last):.4f} +/- {stdev(last):.4f}",
    f"- Forgetting CNN: {mean(forget):.4f} +/- {stdev(forget):.4f}",
    "",
    "## Curves",
]

for r in results:
    lines += [
        "",
        f"### Seed {r['seed']}",
        f"- Top1 curve: {r['top1_curve']}",
        f"- Top5 curve: {r['top5_curve']}",
    ]

(out_dir / "summary.md").write_text("\n".join(lines))
print((out_dir / "summary.md").read_text())
