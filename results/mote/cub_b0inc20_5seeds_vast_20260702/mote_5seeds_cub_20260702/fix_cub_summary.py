import json
from pathlib import Path
from statistics import mean, stdev

exp = Path(r"results/mote/cub_b0inc20_5seeds_vast_20260702/mote_5seeds_cub_20260702")
p = exp / "metrics_summary.json"
m = json.loads(p.read_text())

known_curves = {
    1991: {
        "top1_curve": [96.58, 95.85, 94.08, 93.64, 91.87, 91.27, 89.93, 89.28, 88.05, 87.31],
        "top5_curve": [100.0, 99.48, 99.13, 99.09, 99.04, 98.88, 98.76, 98.75, 98.47, 98.36],
    },
    1992: {
        "top1_curve": [96.16, 94.48, 94.22, 93.94, 91.91, 90.2, 89.44, 88.1, 87.58, 87.21],
        "top5_curve": [100.0, 99.83, 99.37, 99.19, 99.01, 98.8, 98.65, 98.41, 98.37, 98.38],
    },
    1993: {
        "top1_curve": [96.46, 96.07, 94.26, 93.13, 91.96, 90.51, 89.17, 88.26, 88.08, 87.12],
        "top5_curve": [99.83, 99.48, 99.3, 99.22, 99.06, 98.96, 98.91, 98.72, 98.6, 98.39],
    },
    1994: {
        "top1_curve": [97.96, 97.55, 94.64, 92.37, 91.71, 90.81, 88.91, 87.73, 87.81, 87.11],
        "top5_curve": [100.0, 99.66, 99.37, 99.11, 99.25, 99.09, 98.85, 98.62, 98.51, 98.33],
    },
    1995: {
        "top1_curve": [96.54, 95.31, 92.45, 92.35, 91.13, 88.99, 88.71, 87.49, 87.01, 87.11],
        "top5_curve": [99.83, 99.57, 99.55, 99.49, 99.22, 98.88, 98.84, 98.71, 98.58, 98.38],
    },
}

for r in m["per_seed"]:
    seed = int(r["seed"])
    r["top1_curve"] = known_curves[seed]["top1_curve"]
    r["top5_curve"] = known_curves[seed]["top5_curve"]
    r["last_accuracy_cnn"] = r["top1_curve"][-1]

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
