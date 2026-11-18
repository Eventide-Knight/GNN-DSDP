#!/usr/bin/env python
"""c10_figures.py -- final comparison tables + figures for the demo.

Outputs (all under gnn_site/results/):
  table_residue_test363.csv   table 1: residue-level AUCs on test 363
  table_e2e.csv               table 2: end-to-end success rates of the two pipelines
  fig_table1_auc.png          bar chart of ROC/PR-AUC for the four scoring variants
  fig_e2e_rmsd.png            box plot + success-rate bars for GNN vs DSDP
  fig_center_error_hist.png   histogram of GNN box-centre error on test 363
  center_error_test363.csv    per-complex centre error / coverage
  c7b_curve.png               copy of the training curve (appendix)

Usage:
  python c10_figures.py
"""
import csv
import glob
import json
import os
import shutil
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from torch_geometric.loader import DataLoader  # noqa: E402

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

import model as M  # noqa: E402
import residue_to_box as RB  # noqa: E402
import build_dataset as B  # noqa: E402
import paths  # noqa: E402  (同目录：统一路径配置)

BASE = paths.GNNSITE_HOME
RESULTS = os.path.join(BASE, "results")
os.makedirs(RESULTS, exist_ok=True)


def load_json(path):
    with open(path) as fh:
        return json.load(fh)


def table1():
    c2 = load_json(os.path.join(BASE, "reports", "c2_metrics.json"))
    ev = load_json(os.path.join(BASE, "reports", "c7b_eval.json"))
    sm = load_json(os.path.join(BASE, "reports", "c7b_gnn_smoothing.json"))
    rows = [
        {"method": "DSDP 3D-CNN", "variant": "CA voxel", "roc_auc": c2["score_voxel"]["roc_auc"],
         "pr_auc": c2["score_voxel"]["pr_auc"],
         "note": "task-book projection: probability of the voxel containing the CA"},
        {"method": "DSDP 3D-CNN", "variant": "3x3x3 neighbourhood max",
         "roc_auc": c2["score_nbhd3"]["roc_auc"], "pr_auc": c2["score_nbhd3"]["pr_auc"],
         "note": "max over the 27 voxels around the CA (tolerance like DSDP's own search)"},
        {"method": "GNN (full 16,379)", "variant": "raw", "roc_auc": ev["test"]["roc_auc"],
         "pr_auc": ev["test"]["pr_auc"], "note": "per-residue log_softmax head, no post-processing"},
        {"method": "GNN (full 16,379)", "variant": "1-hop graph max smoothing",
         "roc_auc": sm["smooth1hop_max"]["roc_auc"], "pr_auc": sm["smooth1hop_max"]["pr_auc"],
         "note": "max over self + 10 A graph neighbours (fair-tolerance variant)"},
    ]
    path = os.path.join(RESULTS, "table_residue_test363.csv")
    with open(path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print("[table1] %s" % path)
    for r in rows:
        print("  %-20s %-28s ROC=%.4f PR=%.4f" % (
            r["method"], r["variant"], r["roc_auc"], r["pr_auc"]))
    return rows


def table2():
    blob = load_json(os.path.join(BASE, "runs", "c10_e2e.json"))
    s = blob["summary"]
    rows = [
        {"pipeline": "GNN box (thr0.5/36A)", "n": s["gnn"]["n"],
         "success": s["gnn"]["success"], "success_rate": s["gnn"]["success_rate"],
         "rmsd_median": s["gnn"]["rmsd_median"], "rmsd_mean": s["gnn"]["rmsd_mean"],
         "mean_box_coverage": s["gnn"]["mean_coverage"],
         "box_miss": s["gnn"]["failure_classes"].get("box_miss", 0),
         "sampled_but_fail": s["gnn"]["failure_classes"].get("sampled_but_fail", 0)},
        {"pipeline": "DSDP blind docking (site npy)", "n": s["dsdp"]["n"],
         "success": s["dsdp"]["success"], "success_rate": s["dsdp"]["success_rate"],
         "rmsd_median": s["dsdp"]["rmsd_median"], "rmsd_mean": s["dsdp"]["rmsd_mean"],
         "mean_box_coverage": s["dsdp"]["mean_coverage"],
         "box_miss": s["dsdp"]["failure_classes"].get("box_miss", 0),
         "sampled_but_fail": s["dsdp"]["failure_classes"].get("sampled_but_fail", 0)},
    ]
    path = os.path.join(RESULTS, "table_e2e.csv")
    with open(path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print("[table2] %s" % path)
    for r in rows:
        print("  %-34s n=%d success=%d (%.1f%%) rmsd_med=%.2f coverage=%.3f" % (
            r["pipeline"], r["n"], r["success"], 100 * r["success_rate"],
            r["rmsd_median"], r["mean_box_coverage"]))
    return rows, blob


def center_errors():
    """Per-complex GNN box centre error / coverage on test 363."""
    with open(os.path.join(BASE, "structure_index.json")) as fh:
        index = json.load(fh)["index"]
    files = sorted(glob.glob(os.path.join(BASE, "data", "graph", "test", "*.pt")))
    graphs = [torch.load(f, map_location="cpu", weights_only=False) for f in files]
    ck = torch.load(os.path.join(BASE, "best_model.pt"), map_location="cpu", weights_only=False)
    net = M.GCN_SiteModel(hidden_channels=tuple(ck["hidden"]))
    net.load_state_dict(ck["model_state"])
    net.eval()
    probs = []
    with torch.no_grad():
        for b in DataLoader(graphs, batch_size=64, shuffle=False):
            probs.append(net(b.x, b.edge_index, b.edge_weight)[:, 1].exp().numpy())
    p = np.concatenate(probs)
    per, off = [], 0
    for g in graphs:
        n = int(g.num_nodes)
        per.append(p[off:off + n])
        off += n

    rows = []
    for g, pp in zip(graphs, per):
        pid = str(g.pdb_id).lower()
        e = index.get(pid, {})
        lig = None
        if e.get("ligand_mol2"):
            lig = B.parse_ligand_mol2(e["ligand_mol2"])
        elif e.get("ligand_pdbqt"):
            lig = B.parse_ligand_pdbqt(e["ligand_pdbqt"])
        if lig is None or len(lig) < 3:
            continue
        box = RB.compute_box(pp, g.ca.numpy())
        lo = np.asarray(box["box_min"]); hi = np.asarray(box["box_max"])
        c = np.asarray(box["centre"])
        inside = ((lig >= lo) & (lig <= hi)).all(axis=1)
        rows.append({"pdb_id": pid,
                     "centre_error_A": float(np.linalg.norm(c - lig.mean(axis=0))),
                     "coverage": float(inside.mean()),
                     "n_selected": box["n_selected"],
                     "main_cluster": box["main_cluster_size"]})
    path = os.path.join(RESULTS, "center_error_test363.csv")
    with open(path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    err = np.array([r["centre_error_A"] for r in rows])
    cov = np.array([r["coverage"] for r in rows])
    print("[center] n=%d median=%.2fA mean=%.2fA  cov>=0.9=%.1f%%" % (
        len(err), np.median(err), err.mean(), 100 * (cov >= 0.9).mean()))
    return err, cov


def fig_table1(rows1):
    fig, ax = plt.subplots(figsize=(7.6, 4.0))
    labels = ["DSDP\nCA voxel", "DSDP\n3x3x3 nbhd", "GNN\nraw", "GNN\n1-hop max"]
    roc = [r["roc_auc"] for r in rows1]
    pr = [r["pr_auc"] for r in rows1]
    x = np.arange(len(rows1))
    w = 0.38
    b1 = ax.bar(x - w / 2, roc, w, label="ROC-AUC", color="#4C72B0")
    b2 = ax.bar(x + w / 2, pr, w, label="PR-AUC", color="#DD8452")
    for bars in (b1, b2):
        for b in bars:
            ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.012,
                    "%.3f" % b.get_height(), ha="center", va="bottom", fontsize=8)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylabel("score")
    ax.set_ylim(0, 1.05)
    ax.set_title("Residue-level ligand-site prediction on test 363\n(5 A labels, 130,222 residues, 5.60% positive)")
    ax.legend(loc="upper right")
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    p = os.path.join(RESULTS, "fig_table1_auc.png")
    fig.savefig(p, dpi=150)
    plt.close(fig)
    print("[fig] %s" % p)


def fig_e2e(blob):
    gnn = [r["gnn"]["rmsd"] for r in blob["rows"] if r["gnn"]["rmsd"] is not None]
    dsdp = [r["dsdp"]["rmsd"] for r in blob["rows"] if r["dsdp"]["rmsd"] is not None]
    gs = sum(1 for r in blob["rows"] if r["gnn"]["success"])
    ds = sum(1 for r in blob["rows"] if r["dsdp"]["success"])
    n = len(blob["rows"])
    fig, ax = plt.subplots(1, 2, figsize=(11.5, 4.0))
    bp = ax[0].boxplot([gnn, dsdp], labels=["GNN box", "DSDP blind"], patch_artist=True,
                       widths=0.55, showfliers=True)
    for patch, col in zip(bp["boxes"], ["#4C72B0", "#DD8452"]):
        patch.set_facecolor(col)
        patch.set_alpha(0.65)
    ax[0].axhline(2.0, ls="--", color="crimson", lw=1.2, label="success threshold 2 A")
    ax[0].set_ylabel("heavy-atom RMSD (A, no superposition)")
    ax[0].set_title("End-to-end docking RMSD (n=%d test complexes)" % n)
    ax[0].legend()
    ax[0].grid(axis="y", alpha=0.3)

    bars = ax[1].bar([0, 1], [100 * gs / n, 100 * ds / n], width=0.55,
                     color=["#4C72B0", "#DD8452"])
    for b, s in zip(bars, [gs, ds]):
        ax[1].text(b.get_x() + b.get_width() / 2, b.get_height() + 1.0,
                   "%d/%d (%.1f%%)" % (s, n, 100 * s / n), ha="center", fontsize=9)
    ax[1].set_xticks([0, 1])
    ax[1].set_xticklabels(["GNN box", "DSDP blind"])
    ax[1].set_ylim(0, 100)
    ax[1].set_ylabel("success rate RMSD < 2 A (%)")
    ax[1].set_title("Docking success rate (same DSDP engine, 384x40)")
    ax[1].grid(axis="y", alpha=0.3)
    fig.tight_layout()
    p = os.path.join(RESULTS, "fig_e2e_rmsd.png")
    fig.savefig(p, dpi=150)
    plt.close(fig)
    print("[fig] %s" % p)


def fig_center(err, cov):
    fig, ax = plt.subplots(figsize=(7.6, 4.0))
    ax.hist(err, bins=np.arange(0, 40, 1.0), color="#55A868", alpha=0.85, edgecolor="white")
    ax.axvline(np.median(err), color="crimson", ls="--", lw=1.4,
               label="median = %.1f A" % np.median(err))
    ax.set_xlabel("distance between GNN box centre and true ligand centroid (A)")
    ax.set_ylabel("number of complexes")
    ax.set_title("Box-centre error on test 363 (n=%d, median %.1f A, ligand coverage>=90%% in %.0f%%)"
                 % (len(err), np.median(err), 100 * (cov >= 0.9).mean()))
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    p = os.path.join(RESULTS, "fig_center_error_hist.png")
    fig.savefig(p, dpi=150)
    plt.close(fig)
    print("[fig] %s" % p)


def main():
    rows1 = table1()
    rows2, blob = table2()
    err, cov = center_errors()
    fig_table1(rows1)
    fig_e2e(blob)
    fig_center(err, cov)
    src = os.path.join(BASE, "reports", "c7b_curve.png")
    if os.path.exists(src):
        shutil.copyfile(src, os.path.join(RESULTS, "fig_training_curve.png"))
        print("[fig] %s" % os.path.join(RESULTS, "fig_training_curve.png"))
    with open(os.path.join(RESULTS, "table_all.json"), "w") as fh:
        json.dump({"table1_residue": rows1, "table2_e2e": rows2,
                   "center_error": {"n": len(err), "median": float(np.median(err)),
                                    "mean": float(err.mean()),
                                    "frac_cov_ge_0.9": float((cov >= 0.9).mean())}}, fh, indent=1)
    print("[done] results in %s" % RESULTS)
    return 0


if __name__ == "__main__":
    sys.exit(main())
