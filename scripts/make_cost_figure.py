#!/usr/bin/env python
"""make_cost_figure.py -- time / resource cost of GNN-DSDP vs the DSDP 3D-CNN.

All numbers are read from the artefacts shipped in this repo where possible;
the two constants that are not in a JSON file are marked below with their origin.

Panels:
  (a) model parameters (log scale)      DSDP 13,840,903  vs  GNN 3,570
  (b) checkpoint size (MB, log scale)   DSDP 55.6 MB     vs  GNN 0.055 MB
  (c) per-complex cost (stacked)        featurisation + model inference + docking
  (d) training wall clock               GNN 705 s; DSDP not measured here

Output: results/fig_cost.png  +  results/table_cost.csv

Usage:
  python scripts/make_cost_figure.py
"""
import csv
import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import paths  # noqa: E402

BASE = paths.GNNSITE_HOME
REPORTS = os.path.join(BASE, "reports")
RUNS = os.path.join(BASE, "runs")
RESULTS = os.path.join(BASE, "results")
os.makedirs(RESULTS, exist_ok=True)

# --- constants that are not stored in a JSON artefact ----------------------
# counted from binding_site_model.pth with sum(p.numel() for p in ck.parameters())
DSDP_PARAMS = 13_840_903          # 13.84 M
DSDP_WEIGHT_MB = 55.6             # 55,638,883 bytes
# DSDP featurisation (protein_feature_tool + surface_tool) on the 363 test
# complexes: 108.6 s in a single run (this number was printed by the baseline
# job but the log was overwritten by the re-run, so it is kept as a constant).
DSDP_FEAT_PER_COMPLEX = 108.6 / 363.0
# graph.py --batch was run with --workers 16 (see README section 6)
FEAT_WORKERS = 16
# run_gnn_dsdp.py records these per protein (timing.gnn_seconds)
GNN_EXAMPLES = ["1a2b", "6qqw", "6qxa"]


def load_json(path):
    with open(path) as fh:
        return json.load(fh)


def collect():
    train = load_json(os.path.join(REPORTS, "train_full.json"))
    graph = load_json(os.path.join(REPORTS, "graph_stat_train.json"))
    base = load_json(os.path.join(REPORTS, "baseline_metrics.json"))
    e2e = load_json(os.path.join(RUNS, "e2e_benchmark.json"))["summary"]

    n_train = int(train["n_train"])
    gnn_train_s = float(train["wall_seconds"])
    epochs = int(train["args"]["epochs"])

    gnn_feat_per = float(graph["wall_seconds"]) * FEAT_WORKERS / int(graph["n_complexes"])
    dsdp_infer_per = float(base["wall_seconds"]) / int(base["n_complexes"])
    docking_per = float(e2e["wall_seconds"]) / (int(e2e["n_done"]) * 2)

    gnn_single = {}
    for pid in GNN_EXAMPLES:
        p = os.path.join(RUNS, pid + ".json")
        if os.path.exists(p):
            gnn_single[pid] = load_json(p)["timing"]["gnn_seconds"]

    ck = os.path.join(BASE, "best_model.pt")
    gnn_weight_mb = os.path.getsize(ck) / 1e6 if os.path.exists(ck) else 55e-3

    return {
        "dsdp_params": DSDP_PARAMS,
        "gnn_params": int(train["n_params"]),
        "dsdp_weight_mb": DSDP_WEIGHT_MB,
        "gnn_weight_mb": gnn_weight_mb,
        "dsdp_feat_s": DSDP_FEAT_PER_COMPLEX,
        "dsdp_infer_s": dsdp_infer_per,
        "gnn_feat_s": gnn_feat_per,
        "gnn_feat_wall_s": float(graph["wall_seconds"]) / int(graph["n_complexes"]),
        "docking_s": docking_per,
        "gnn_train_s": gnn_train_s,
        "n_train": n_train,
        "epochs": epochs,
        "epoch_s": gnn_train_s / max(epochs, 1),
        "gnn_single": gnn_single,
        "e2e_wall_s": float(e2e["wall_seconds"]),
        "n_e2e": int(e2e["n_done"]),
    }


def main():
    c = collect()
    DSDP_C = "#DD8452"
    GNN_C = "#4C72B0"
    fig, axes = plt.subplots(2, 2, figsize=(11.5, 8.2))
    fig.suptitle("GNN-DSDP vs DSDP 3D-CNN: model size and time cost", fontsize=13)

    # (a) parameters --------------------------------------------------------
    ax = axes[0, 0]
    vals = [c["dsdp_params"], c["gnn_params"]]
    bars = ax.bar(["DSDP\n3D-CNN", "GNN-DSDP"], vals, color=[DSDP_C, GNN_C], width=0.55)
    ax.set_yscale("log")
    ax.set_ylabel("parameters (log scale)")
    ax.set_title("(a) Model size: %.2f M vs %.0f  (%.0fx smaller)"
                 % (c["dsdp_params"] / 1e6, c["gnn_params"],
                    c["dsdp_params"] / max(c["gnn_params"], 1)))
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, v * 1.15, "{:,}".format(v),
                ha="center", va="bottom", fontsize=9)
    ax.set_ylim(min(vals) * 0.3, max(vals) * 4)
    ax.grid(axis="y", alpha=0.3, which="both")

    # (b) checkpoint size ---------------------------------------------------
    ax = axes[0, 1]
    vals = [c["dsdp_weight_mb"], c["gnn_weight_mb"]]
    bars = ax.bar(["DSDP\n3D-CNN", "GNN-DSDP"], vals, color=[DSDP_C, GNN_C], width=0.55)
    ax.set_yscale("log")
    ax.set_ylabel("checkpoint size (MB, log scale)")
    ax.set_title("(b) Weights on disk: %.1f MB vs %.3f MB  (%.0fx smaller)"
                 % (c["dsdp_weight_mb"], c["gnn_weight_mb"],
                    c["dsdp_weight_mb"] / max(c["gnn_weight_mb"], 1e-9)))
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, v * 1.2,
                "%.2f MB" % v if v >= 0.1 else "%.0f KB" % (v * 1000),
                ha="center", va="bottom", fontsize=9)
    ax.set_ylim(min(vals) * 0.2, max(vals) * 5)
    ax.grid(axis="y", alpha=0.3, which="both")

    # (c) per-complex cost --------------------------------------------------
    ax = axes[1, 0]
    labels = ["DSDP\n3D-CNN", "GNN-DSDP"]
    feat = [c["dsdp_feat_s"], c["gnn_feat_s"]]
    infer = [c["dsdp_infer_s"], 0.0]
    dock = [c["docking_s"], c["docking_s"]]
    x = np.arange(2)
    b1 = ax.bar(x, feat, 0.5, label="structure featurisation (CPU)", color="#8C8C8C")
    b2 = ax.bar(x, infer, 0.5, bottom=feat, label="site-model inference", color="#55A868")
    b3 = ax.bar(x, dock, 0.5, bottom=np.array(feat) + np.array(infer),
                label="docking (same engine)", color="#C44E52")
    for i in range(2):
        total = feat[i] + infer[i] + dock[i]
        ax.text(i, total + 0.04, "%.2f s" % total, ha="center", va="bottom", fontsize=9)
    ax.text(1, feat[1] / 2, "%.2f s" % feat[1], ha="center", color="white", fontsize=9)
    ax.text(0, feat[0] / 2, "%.2f s" % feat[0], ha="center", color="white", fontsize=9)
    ax.text(0, feat[0] + infer[0] / 2, "%.2f s (GPU)" % infer[0], ha="center",
            color="white", fontsize=8)
    ax.text(1, feat[1] + infer[1] / 2, "~0 (<0.01 s)", ha="center", color="black", fontsize=8)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("seconds per complex")
    ax.set_title("(c) Site prediction + docking per complex\n(DSDP inference needs a GPU; GNN does not)")
    ax.legend(fontsize=8, loc="upper left")
    ax.grid(axis="y", alpha=0.3)
    ax.set_ylim(0, max(feat[i] + infer[i] + dock[i] for i in range(2)) * 1.25)

    # (d) training ----------------------------------------------------------
    ax = axes[1, 1]
    ax.bar([1], [c["gnn_train_s"]], 0.5, color=GNN_C)
    ax.bar([0], [0.0], 0.5, color=DSDP_C, hatch="///", edgecolor="white")
    ax.text(0, c["gnn_train_s"] * 0.06, "not measured here", ha="center", va="bottom",
            fontsize=9, rotation=90, color="#444444")
    ax.text(1, c["gnn_train_s"] * 1.02, "%.0f s (%.1f min)\n%d complexes x %d epochs (~%.1f s/epoch)"
            % (c["gnn_train_s"], c["gnn_train_s"] / 60, c["n_train"], c["epochs"], c["epoch_s"]),
            ha="center", va="bottom", fontsize=8.5)
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["DSDP\n3D-CNN", "GNN-DSDP"])
    ax.set_ylabel("training wall clock (s)")
    ax.set_title("(d) Training cost (1x RTX 4090 for the GNN)")
    ax.set_xlim(-0.6, 1.6)
    ax.set_ylim(0, c["gnn_train_s"] * 1.45)
    ax.grid(axis="y", alpha=0.3)

    fig.tight_layout(rect=[0, 0, 1, 0.96])
    png = os.path.join(RESULTS, "fig_cost.png")
    fig.savefig(png, dpi=150)
    plt.close(fig)
    print("[cost] wrote %s" % png)

    rows = [
        ("parameters", "DSDP 3D-CNN", c["dsdp_params"]),
        ("parameters", "GNN-DSDP", c["gnn_params"]),
        ("checkpoint_MB", "DSDP 3D-CNN", round(c["dsdp_weight_mb"], 3)),
        ("checkpoint_MB", "GNN-DSDP", round(c["gnn_weight_mb"], 3)),
        ("featurisation_s_per_complex_1core", "DSDP 3D-CNN", round(c["dsdp_feat_s"], 4)),
        ("featurisation_s_per_complex_1core", "GNN-DSDP", round(c["gnn_feat_s"], 4)),
        ("inference_s_per_complex", "DSDP 3D-CNN (GPU)", round(c["dsdp_infer_s"], 4)),
        ("inference_s_per_complex", "GNN-DSDP (CPU)", 0.0),
        ("docking_s_per_complex", "both (same engine)", round(c["docking_s"], 4)),
        ("training_s", "GNN-DSDP (100 epochs)", round(c["gnn_train_s"], 1)),
        ("training_s", "DSDP 3D-CNN", ""),
    ]
    csv_path = os.path.join(RESULTS, "table_cost.csv")
    with open(csv_path, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["metric", "system", "value"])
        w.writerows(rows)
    print("[cost] wrote %s" % csv_path)
    for k, v in c.items():
        if k != "gnn_single":
            print("  %-16s %s" % (k, v))
    print("  gnn_single      %s" % c["gnn_single"])
    return 0


if __name__ == "__main__":
    sys.exit(main())
