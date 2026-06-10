"""ConvNeXt pruning comparison: VBP / Magnitude / Ours (tp_variance) across T/S/B."""
import matplotlib.pyplot as plt

# (MACs_G, top1_acc) per size: tiny, small, base
results = {
    "VBP (reproduced)": [(2.86, 80.96), (4.96, 82.76), (8.90, 83.32)],
    "Magnitude (TP)":   [(2.84, 80.86), (5.04, 82.88), (8.86, 83.17)],
    "Ours (tp_variance)": [(2.79, 82.24), (4.94, 83.99), (8.71, 85.22)],
}
baselines = [(4.47, 82.90), (8.71, 84.57), (15.38, 85.51)]
size_labels = ["T", "S", "B"]

styles = {
    "VBP (reproduced)":   dict(color="#1f77b4", marker="s", linestyle="--"),
    "Magnitude (TP)":     dict(color="#2ca02c", marker="^", linestyle=":"),
    "Ours (tp_variance)": dict(color="#d62728", marker="o", linestyle="-", linewidth=2.2),
}

fig, ax = plt.subplots(figsize=(8, 5.5))

for name, pts in results.items():
    xs = [p[0] for p in pts]
    ys = [p[1] for p in pts]
    ax.plot(xs, ys, label=name, markersize=9, **styles[name])
    for (x, y), s in zip(pts, size_labels):
        ax.annotate(s, (x, y), textcoords="offset points", xytext=(6, 6),
                    fontsize=9, color=styles[name]["color"])

bx = [p[0] for p in baselines]
by = [p[1] for p in baselines]
ax.plot(bx, by, color="black", marker="*", markersize=14, linestyle="-.",
        linewidth=1.2, label="Baseline (unpruned)", alpha=0.7)
for (x, y), s in zip(baselines, size_labels):
    ax.annotate(s, (x, y), textcoords="offset points", xytext=(6, -12),
                fontsize=9, color="black")

ax.set_xlabel("MACs (G)", fontsize=12)
ax.set_ylabel("ImageNet Top-1 Accuracy (%)", fontsize=12)
ax.set_title("ConvNeXt Pruning: ours vs VBP vs Magnitude (T / S / B)", fontsize=13)
ax.grid(True, alpha=0.3)
ax.legend(loc="lower right", fontsize=10, framealpha=0.95)

plt.tight_layout()
out = "convnext_comparison.png"
plt.savefig(out, dpi=160)
print(f"saved {out}")
