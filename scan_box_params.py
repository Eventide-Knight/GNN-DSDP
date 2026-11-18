#!/usr/bin/env python
"""scan_box_params.py -- choose the residue->box rule on the VALIDATION split.

C8 must not be tuned on test, so this script sweeps, on val 968 only:
  * residue selection: probability threshold 0.3/0.4/0.5/0.6/0.7, or top-k 10/20/30
  * box side length: 30 / 33 / 36 / 43 A
  * box centre rule: plain CA centroid vs probability-weighted CA centroid
and reports, for every combination, how much of the true ligand (5 A label is
derived from it) the box captures:
  ligand coverage = fraction of ligand heavy atoms inside the box.

Selection follows the C8 spec: select residues (fallback to top-20 if fewer
than 5 are picked), single-linkage cluster at 8 A on CA, take the cluster with
the largest (n_residues * sum(prob)), then build the box and clip its centre so
the box stays inside the protein CA bounding box.

Usage:
  python scan_box_params.py --model best_model.pt --split val
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

import model as M  # noqa: E402
import build_dataset as B  # noqa: E402
import paths  # noqa: E402  (同目录：统一路径配置)

BASE = paths.GNNSITE_HOME
THRESHOLDS = (0.3, 0.4, 0.5, 0.6, 0.7)
TOPKS = (10, 20, 30)
SIDES = (30.0, 33.0, 36.0, 43.0)
CENTRES = ("centroid", "weighted")


def load_split(split):
    files = sorted(glob.glob(os.path.join(BASE, "data", "graph", split, "*.pt")))
    return [torch.load(f, map_location="cpu", weights_only=False) for f in files]


@torch.no_grad()
def probs_for(net, graphs, device, batch_size=64):
    loader = DataLoader(graphs, batch_size=batch_size, shuffle=False)
    out = []
    for batch in loader:
        batch = batch.to(device)
        o = net(batch.x, batch.edge_index, batch.edge_weight)
        out.append(o[:, 1].exp().cpu().numpy())
    p = np.concatenate(out)
    per, off = [], 0
    for g in graphs:
        n = int(g.num_nodes)
        per.append(p[off:off + n])
        off += n
    return per


def ligand_xyz(pid, index, split):
    e = index.get(pid)
    if not e:
        return None
    path = e.get("ligand_mol2") or e.get("ligand_pdbqt")
    if not path or not os.path.exists(path):
        return None
    if path.endswith("mol2"):
        return B.parse_ligand_mol2(path)
    return B.parse_ligand_pdbqt(path)


def cluster_ca(ca, eps=8.0):
    """Single-linkage clustering on CA positions (BFS over pairs <= eps)."""
    n = len(ca)
    if n == 0:
        return []
    d = np.linalg.norm(ca[:, None, :] - ca[None, :, :], axis=-1)
    adj = d <= eps
    np.fill_diagonal(adj, False)
    seen = np.zeros(n, dtype=bool)
    clusters = []
    for i in range(n):
        if seen[i]:
            continue
        stack, comp = [i], []
        seen[i] = True
        while stack:
            u = stack.pop()
            comp.append(u)
            for v in np.where(adj[u])[0]:
                if not seen[v]:
                    seen[v] = True
                    stack.append(v)
        clusters.append(np.array(comp))
    return clusters


def choose_cluster(ca_sel, prob_sel, eps=8.0):
    clusters = cluster_ca(ca_sel, eps)
    if not clusters:
        return None
    best, best_score = None, -1.0
    for c in clusters:
        score = len(c) * float(prob_sel[c].sum())
        if score > best_score:
            best, best_score = c, score
    return best


def box_from_centre(centre, side, prot_lo, prot_hi):
    half = side / 2.0
    lo = centre - half
    hi = centre + half
    # keep the box inside the protein bounding box (shift the centre, never leave it hanging)
    for k in range(3):
        span = prot_hi[k] - prot_lo[k]
        if side >= span:
            lo[k] = prot_lo[k] - (side - span) / 2.0
            hi[k] = lo[k] + side
        else:
            lo[k] = min(max(lo[k], prot_lo[k]), prot_hi[k] - side)
            hi[k] = lo[k] + side
    return lo, hi


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=os.path.join(BASE, "best_model.pt"))
    ap.add_argument("--split", default="val")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--eps", type=float, default=8.0)
    ap.add_argument("--json", default=os.path.join(BASE, "reports", "box_params_val.json"))
    args = ap.parse_args()

    device = torch.device(args.device if (args.device == "cpu" or torch.cuda.is_available())
                          else "cpu")
    ck = torch.load(args.model, map_location="cpu", weights_only=False)
    net = M.GCN_SiteModel(hidden_channels=tuple(ck.get("hidden", M.HIDDEN_CHANNELS)))
    net.load_state_dict(ck["model_state"])
    net.to(device).eval()

    graphs = load_split(args.split)
    probs = probs_for(net, graphs, device)
    with open(os.path.join(BASE, "structure_index.json")) as fh:
        index = json.load(fh)["index"]

    rules = [("thr_%.1f" % t, ("thr", t)) for t in THRESHOLDS]
    rules += [("top_%d" % k, ("top", k)) for k in TOPKS]

    # accumulate coverage per (rule, side, centre)
    acc = {(rn, s, c): [] for rn, _ in rules for s in SIDES for c in CENTRES}
    centres_err = {(rn, s, c): [] for rn, _ in rules for s in SIDES for c in CENTRES}
    used = 0
    skipped = 0
    for g, p in zip(graphs, probs):
        pid = str(g.pdb_id).lower()
        lig = ligand_xyz(pid, index, args.split)
        if lig is None or len(lig) < 3:
            skipped += 1
            continue
        used += 1
        ca = g.ca.numpy().astype(np.float64)
        prot_lo, prot_hi = ca.min(axis=0), ca.max(axis=0)
        lig_c = lig.mean(axis=0)
        for rn, (kind, val) in rules:
            if kind == "thr":
                sel = np.where(p >= val)[0]
                if len(sel) < 5:
                    sel = np.argsort(-p)[:20]
            else:
                sel = np.argsort(-p)[:min(val, len(p))]
            ca_sel = ca[sel]
            pr_sel = p[sel].astype(np.float64)
            comp = choose_cluster(ca_sel, pr_sel, args.eps)
            if comp is None:
                comp = np.arange(len(sel))
            ca_c = ca_sel[comp]
            pr_c = pr_sel[comp]
            centres = {
                "centroid": ca_c.mean(axis=0),
                "weighted": (ca_c * pr_c[:, None]).sum(axis=0) / max(pr_c.sum(), 1e-9),
            }
            for side in SIDES:
                for cname in CENTRES:
                    lo, hi = box_from_centre(centres[cname], side, prot_lo, prot_hi)
                    inside = ((lig >= lo) & (lig <= hi)).all(axis=1)
                    acc[(rn, side, cname)].append(float(inside.mean()))
                    centres_err[(rn, side, cname)].append(
                        float(np.linalg.norm(((lo + hi) / 2.0) - lig_c)))
    print("[scan] used=%d skipped=%d" % (used, skipped))

    rows = []
    for rn, _ in rules:
        for side in SIDES:
            for cname in CENTRES:
                cov = np.asarray(acc[(rn, side, cname)])
                err = np.asarray(centres_err[(rn, side, cname)])
                rows.append({"rule": rn, "side": side, "centre": cname,
                             "mean_coverage": float(cov.mean()),
                             "frac_cov_ge_0.9": float((cov >= 0.9).mean()),
                             "frac_cov_ge_0.5": float((cov >= 0.5).mean()),
                             "median_centre_error_A": float(np.median(err))})
    rows.sort(key=lambda r: (-r["frac_cov_ge_0.9"], -r["mean_coverage"], r["side"]))
    print("%-8s %5s %-9s %8s %10s %10s %8s" % (
        "rule", "side", "centre", "cover", "cov>=.9", "cov>=.5", "err_med"))
    for r in rows[:20]:
        print("%-8s %5.0f %-9s %8.3f %9.1f%% %9.1f%% %8.1f" % (
            r["rule"], r["side"], r["centre"], r["mean_coverage"],
            100 * r["frac_cov_ge_0.9"], 100 * r["frac_cov_ge_0.5"],
            r["median_centre_error_A"]))

    # decision rule, fixed BEFORE looking at test:
    #   side <= 36 A, then maximise frac_cov_ge_0.9, tie-break mean coverage
    eligible = [r for r in rows if r["side"] <= 36.0]
    eligible.sort(key=lambda r: (-r["frac_cov_ge_0.9"], -r["mean_coverage"]))
    best = eligible[0]
    print("\n[scan] CHOSEN (val): rule=%s side=%.0f centre=%s  cov>=.9=%.1f%% mean=%.3f err_med=%.1fA" % (
        best["rule"], best["side"], best["centre"], 100 * best["frac_cov_ge_0.9"],
        best["mean_coverage"], best["median_centre_error_A"]))

    out = {"split": args.split, "model": args.model, "epoch": ck.get("epoch"),
           "n_complexes_used": used, "n_skipped": skipped, "eps": args.eps,
           "decision_rule": "side<=36A, maximise frac(coverage>=0.9), then mean coverage",
           "chosen": best, "table": rows}
    with open(args.json, "w") as fh:
        json.dump(out, fh, indent=1)
    print("[scan] wrote %s" % args.json)
    return 0


if __name__ == "__main__":
    sys.exit(main())
