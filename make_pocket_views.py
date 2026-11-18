#!/usr/bin/env python
"""make_pocket_views.py -- 3D visualisation of representative proteins + PyMOL scripts.

For each requested test complex it draws (matplotlib 3D, no PyMOL in this
environment):
  * protein CA backbone (light grey trace)
  * the crystal ligand (green)
  * the GNN top-20 residues (red spheres = predicted pocket)
  * the docked pose from the GNN box (blue)
and writes a companion .pml script that reproduces the same view in PyMOL
(`pymol -cq <id>.pml`) for anyone who has PyMOL installed.

Usage:
  python make_pocket_views.py --ids 6g2o,6jao,6kqi,5zk7 --outdir results/viz
"""
import argparse
import glob
import json
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from torch_geometric.loader import DataLoader  # noqa: E402

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401,E402

import model as M  # noqa: E402
import build_dataset as B  # noqa: E402
import paths  # noqa: E402  (同目录：统一路径配置)

BASE = paths.GNNSITE_HOME
REPO_TEST = paths.TEST_PDBBIND_DIR
RUNS = os.path.join(BASE, "runs")


def load_graph(pid):
    p = os.path.join(BASE, "data", "graph", "test", pid + ".pt")
    return torch.load(p, map_location="cpu", weights_only=False)


def probs_for(pid, device="cpu"):
    ck = torch.load(os.path.join(BASE, "best_model.pt"), map_location="cpu", weights_only=False)
    net = M.GCN_SiteModel(hidden_channels=tuple(ck.get("hidden", M.HIDDEN_CHANNELS)))
    net.load_state_dict(ck["model_state"])
    net.to(device).eval()
    g = load_graph(pid)
    with torch.no_grad():
        p = net(g.x.to(device), g.edge_index.to(device), g.edge_weight.to(device))[:, 1].exp().cpu().numpy()
    return g, p


def crystal_ligand(pid):
    p = os.path.join(REPO_TEST, pid, pid + "_ligand.pdbqt")
    return B.parse_ligand_pdbqt(p) if os.path.exists(p) else None


def docked_pose(pid):
    for cand in (os.path.join(RUNS, "c10", "%s_gnn_out.pdbqt" % pid),
                 os.path.join(RUNS, "%s_out.pdbqt" % pid)):
        if os.path.exists(cand):
            return B.parse_ligand_pdbqt(cand)
    return None


def rmsd_of(pid):
    p = os.path.join(RUNS, "e2e_benchmark.json")
    if not os.path.exists(p):
        return None
    d = json.load(open(p))
    for r in d["rows"]:
        if r["pdb_id"] == pid:
            return r["gnn"]["rmsd"], r["dsdp"]["rmsd"]
    return None


def resi_list(g, top):
    ch = [str(c) for c in g.chain] if hasattr(g, "chain") else None
    return None


def draw(pid, ca, top_idx, lig, pose, rmsd, outdir):
    fig = plt.figure(figsize=(7.2, 6.4))
    ax = fig.add_subplot(111, projection="3d")
    ax.plot(ca[:, 0], ca[:, 1], ca[:, 2], color="#bbbbbb", lw=0.7, alpha=0.9, label="protein CA trace")
    if lig is not None:
        ax.scatter(lig[:, 0], lig[:, 1], lig[:, 2], c="#2ca02c", s=26, alpha=0.95,
                   label="crystal ligand", depthshade=False)
    if pose is not None:
        ax.scatter(pose[:, 0], pose[:, 1], pose[:, 2], c="#1f77b4", s=20, alpha=0.85,
                   marker="^", label="docked pose (GNN box)", depthshade=False)
    tp = ca[top_idx]
    ax.scatter(tp[:, 0], tp[:, 1], tp[:, 2], c="#d62728", s=90, alpha=0.9,
               marker="o", label="GNN top-20 residues", depthshade=False)
    ax.set_xticks([]); ax.set_yticks([]); ax.set_zticks([])
    ax.set_xlabel("x (A)"); ax.set_ylabel("y (A)"); ax.set_zlabel("z (A)")
    title = "%s  GNN top-20 pocket" % pid
    if rmsd is not None:
        title += "\nGNN-box docking RMSD = %.2f A  (%s)" % (
            rmsd[0], "SUCCESS <2 A" if rmsd[0] is not None and rmsd[0] < 2 else "FAIL")
    ax.set_title(title, fontsize=10)
    ax.legend(loc="upper left", fontsize=8, framealpha=0.85)
    ax.view_init(elev=18, azim=-60)
    fig.tight_layout()
    png = os.path.join(outdir, "%s_pocket.png" % pid)
    fig.savefig(png, dpi=150)
    plt.close(fig)
    return png


def write_pml(pid, g, top_idx, outdir):
    prot = os.path.join(BASE, "pdbbind_v2020")
    with open(os.path.join(BASE, "structure_index.json")) as fh:
        idx = json.load(fh)["index"]
    e = idx[pid]
    prot_pdb = e["protein"]
    lig = os.path.join(REPO_TEST, pid, pid + "_ligand.pdbqt")
    pose = os.path.join(RUNS, "c10", "%s_gnn_out.pdbqt" % pid)
    if not os.path.exists(pose):
        pose = os.path.join(RUNS, "%s_out.pdbqt" % pid)
    z = np.load(os.path.join(BASE, "data", "label", "test", pid + ".npz"))
    chains = [str(c) for c in z["chain"]]
    resseq = [int(r) for r in z["resseq"]]
    sel = []
    for r in sorted(set(int(i) for i in top_idx)):
        sel.append("%d%s" % (resseq[r], chains[r]))
    lines = [
        "# PyMOL script for %s -- run with: pymol -cq %s_pocket.pml" % (pid, pid),
        "load %s, prot" % prot_pdb,
        "load %s, lig" % lig,
        "load %s, pose" % pose,
        "hide everything",
        "show cartoon, prot",
        "set cartoon_transparency, 0.35, prot",
        "show sticks, lig",
        "color green, lig",
        "show sticks, pose",
        "color marine, pose",
        "select pocket, resi %s" % "+".join(sel),
        "show spheres, pocket",
        "color red, pocket",
        "set sphere_scale, 0.6, pocket",
        "distance d_lig_pose, lig, pose",
        "zoom pocket, 8",
        "bg_color white",
        "ray 900,900",
        "png %s_pocket_pymol.png, dpi=150" % pid,
        "quit",
    ]
    p = os.path.join(outdir, "%s_pocket.pml" % pid)
    with open(p, "w") as fh:
        fh.write("\n".join(lines) + "\n")
    return p


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ids", default="6g2o,6jao,6qqw,6kqi,5zk7")
    ap.add_argument("--outdir", default=os.path.join(BASE, "results", "viz"))
    ap.add_argument("--topk", type=int, default=20)
    args = ap.parse_args()
    os.makedirs(args.outdir, exist_ok=True)

    summary = []
    for pid in [x.strip() for x in args.ids.split(",") if x.strip()]:
        try:
            g, p = probs_for(pid)
            ca = g.ca.numpy()
            top_idx = np.argsort(-p)[:args.topk]
            lig = crystal_ligand(pid)
            pose = docked_pose(pid)
            rms = rmsd_of(pid)
            png = draw(pid, ca, top_idx, lig, pose, rms, args.outdir)
            pml = write_pml(pid, g, top_idx, args.outdir)
            n_hit = int(g.y.numpy()[top_idx].sum())
            print("[viz] %-5s top-%d hits=%d/%d  ligand=%s pose=%s -> %s, %s" % (
                pid, args.topk, n_hit, int(g.y.sum()),
                "yes" if lig is not None else "NO",
                "yes" if pose is not None else "NO", os.path.basename(png), os.path.basename(pml)))
            summary.append({"pdb_id": pid, "top20_hits": n_hit,
                            "n_true_pos": int(g.y.sum()),
                            "gnn_rmsd": rms[0] if rms else None,
                            "dsdp_rmsd": rms[1] if rms else None,
                            "png": png, "pml": pml})
        except Exception as e:
            print("[viz] %s FAILED: %r" % (pid, e))
    with open(os.path.join(args.outdir, "viz_summary.json"), "w") as fh:
        json.dump(summary, fh, indent=1)
    print("[viz] wrote %s" % os.path.join(args.outdir, "viz_summary.json"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
