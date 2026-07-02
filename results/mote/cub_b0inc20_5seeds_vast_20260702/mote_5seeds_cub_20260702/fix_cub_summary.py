import json
from pathlib import Path
from statistics import mean, stdev

exp = Path(r"results/mote/cub_b0inc20_5seeds_vast_20260702/mote_5seeds_cub_20260702")
p = exp / "metrics_summary.json"
m = json.loads(p.read_text())

for r in m["per_seed"]:
    r["top1_curve"] = [float(x) for x in r["top1_curve"] if float(x) != 64.0]
    r["top5_curve"] = [float(x) for x in r["top5_curve"] if float(x) != 64.0]
    r["last_accuracy_cnn"] = float(r["top1_curve"][-1])

avg = [r["average_accuracy_cnn"] for r in m["per_seed"]]
last = [r["last_accuracy_cnn"] for r in m["per_seed"]]
forget = [r["forgetting_cnn"] for r in m["per_seed"]]

m["mean_std"] = {
    "average_accuracy_cnn": {"mean": mean(avg), "std": stdev(avg)},
    "last_accuracy_cnn": {"mean": mean(last), "std": stdev(last)},
    "forgetting_cnn": {"mean": mean(forget), "std": stdev(forget)},
}

p.write_text(json.dumps(m, indent=2))

lines = [
    "# MoTE CUB B0-Inc20 Five-Seed Reproduction",
    "",
    "## Per-seed results",
    "",
    "| Seed | Avg Acc CNN | Last Acc CNN | Forgetting CNN |",
    "|---:|---:|---:|---:|",
]

for r in m["per_seed"]:
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

for r in m["per_seed"]:
    lines += [
        "",
        f"### Seed {r['seed']}",
        f"- Top1 curve: {r['top1_curve']}",
        f"- Top5 curve: {r['top5_curve']}",
    ]

(exp / "summary.md").write_text("\n".join(lines))
print((exp / "summary.md").read_text())
