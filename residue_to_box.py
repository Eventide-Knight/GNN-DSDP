#!/usr/bin/env python
"""residue_to_box.py -- GNN residue probabilities -> a docking box.

Flow (task book section 3 "residue -> box"):
  1. per-residue probability from the final GNN (reuses predict.py, no rewrite);
  2. select residues with p >= threshold (default 0.5, chosen on val by
     box_param_scan.py); if fewer than 5 residues are selected, fall back to
     top-20; --topk overrides the threshold rule entirely;
  3. single-linkage clustering on CA at <= eps (default 8 A); the main pocket is
     the cluster with the largest (n_residues * sum(p));
  4. box centre = probability-weighted CA centroid of the main cluster;
     side = --box-size (default 36 A, chosen on val); the centre is shifted so
     the box never hangs outside the protein CA bounding box;
  5. box_min / box_max in the same coordinate frame as the input PDB
     (verified: PDBBind <id>_protein.pdb CA == repo <id>_protein.pdbqt CA,
     max deviation 0.0000 A on 6qqw).

Usage:
  python residue_to_box.py --protein 6qqw
  python residue_to_box.py --protein 6qqw --ligand <id>_ligand.pdbqt --json reports/box_6qqw.json
  python residue_to_box.py --protein 6qqw --threshold 0.6 --box-size 30
"""
import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import predict as P  # noqa: E402
import build_dataset as B  # noqa: E402
import paths  # noqa: E402  (同目录：统一路径配置)

BASE = paths.GNNSITE_HOME
DEFAULT_THRESHOLD = 0.5      # chosen on val 968 (box_param_scan.py)
DEFAULT_BOX_SIZE = 36.0      # chosen on val 968
DEFAULT_EPS = 8.0


def cluster_ca(ca, eps=DEFAULT_EPS):
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


def clip_centre_into_protein(centre, side, prot_lo, prot_hi):
    half = side / 2.0
    lo = centre - half
    hi = centre + half
    for k in range(3):
        span = prot_hi[k] - prot_lo[k]
        if side >= span:
            lo[k] = prot_lo[k] - (side - span) / 2.0
            hi[k] = lo[k] + side
        else:
            lo[k] = min(max(lo[k], prot_lo[k]), prot_hi[k] - side)
            hi[k] = lo[k] + side
    return lo, hi


def compute_box(prob, ca, threshold=DEFAULT_THRESHOLD, box_size=DEFAULT_BOX_SIZE,
                centre_rule="weighted", eps=DEFAULT_EPS, topk=0):
    prob = np.asarray(prob, dtype=np.float64)
    ca = np.asarray(ca, dtype=np.float64)
    selection = "threshold"
    if topk:
        sel = np.argsort(-prob)[:min(topk, len(prob))]
        selection = "topk"
    else:
        sel = np.where(prob >= threshold)[0]
        if len(sel) < 5:
            sel = np.argsort(-prob)[:min(20, len(prob))]
            selection = "fallback_top20"
    ca_sel, pr_sel = ca[sel], prob[sel]
    clusters = cluster_ca(ca_sel, eps)
    if not clusters:
        clusters = [np.arange(len(sel))]
    best, best_score = None, -1.0
    for c in clusters:
        score = len(c) * float(pr_sel[c].sum())
        if score > best_score:
            best, best_score = c, score
    ca_c, pr_c = ca_sel[best], pr_sel[best]
    if centre_rule == "weighted":
        centre = (ca_c * pr_c[:, None]).sum(axis=0) / max(pr_c.sum(), 1e-9)
    else:
        centre = ca_c.mean(axis=0)
    prot_lo, prot_hi = ca.min(axis=0), ca.max(axis=0)
    lo, hi = clip_centre_into_protein(centre, box_size, prot_lo, prot_hi)
    return {
        "box_min": lo.tolist(),
        "box_max": hi.tolist(),
        "box_size": float(box_size),
        "side_lengths": (hi - lo).round(3).tolist(),
        "centre": ((lo + hi) / 2.0).tolist(),
        "centre_rule": centre_rule,
        "threshold": float(threshold),
        "selection": selection,
        "n_selected": int(len(sel)),
        "n_clusters": int(len(clusters)),
        "cluster_sizes": sorted([int(len(c)) for c in clusters], reverse=True)[:10],
        "main_cluster_size": int(len(best)),
        "main_cluster_prob_sum": float(pr_c.sum()),
        "main_cluster_prob_mean": float(pr_c.mean()),
        "selected_index": sel.tolist(),
        "main_cluster_index": sel[best].tolist(),
        "protein_bbox_min": prot_lo.tolist(),
        "protein_bbox_max": prot_hi.tolist(),
        "eps": float(eps),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--protein", required=True, help="4-char PDB id or *_protein.pdb path")
    ap.add_argument("--model", default=P.DEFAULT_MODEL)
    ap.add_argument("--threshold", type=float, default=DEFAULT_THRESHOLD)
    ap.add_argument("--box-size", type=float, default=DEFAULT_BOX_SIZE)
    ap.add_argument("--centre", default="weighted", choices=["weighted", "centroid"])
    ap.add_argument("--eps", type=float, default=DEFAULT_EPS)
    ap.add_argument("--topk", type=int, default=0)
    ap.add_argument("--ligand", default=None,
                    help="optional ligand file for diagnostics (coverage / centre distance)")
    ap.add_argument("--json", default=None)
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()

    pdb = P.resolve_protein(args.protein)
    pid = os.path.basename(os.path.dirname(pdb)) or os.path.basename(pdb).split("_")[0]
    net, ck = P.load_model(args.model, args.device)
    prob, meta = P.predict_one(net, pdb, args.device)
    res = compute_box(prob, meta["ca"], threshold=args.threshold,
                      box_size=args.box_size, centre_rule=args.centre,
                      eps=args.eps, topk=args.topk)
    res.update({"pdb": pdb, "pdb_id": pid, "model": args.model,
                "model_epoch": ck.get("epoch"), "n_residues": int(len(prob)),
                "prob_mean": float(prob.mean()), "prob_max": float(prob.max())})

    if args.ligand:
        if args.ligand.endswith(".mol2"):
            lig = B.parse_ligand_mol2(args.ligand)
        else:
            lig = B.parse_ligand_pdbqt(args.ligand)
        lo = np.asarray(res["box_min"]); hi = np.asarray(res["box_max"])
        inside = ((lig >= lo) & (lig <= hi)).all(axis=1)
        ctr = np.asarray(res["centre"])
        res["ligand"] = {
            "path": args.ligand, "n_heavy": int(len(lig)),
            "coverage": float(inside.mean()),
            "centre_to_ligand_centroid_A": float(np.linalg.norm(ctr - lig.mean(axis=0))),
            "ligand_centroid_inside_box": bool(
                ((lig.mean(axis=0) >= lo) & (lig.mean(axis=0) <= hi)).all()),
        }

    print("[box] %s (%d residues) model epoch=%s" % (pid, res["n_residues"], ck.get("epoch")))
    print("[box] selection=%s n_selected=%d clusters=%d main=%d" % (
        res["selection"], res["n_selected"], res["n_clusters"], res["main_cluster_size"]))
    print("[box] box_size=%.0f centre=%s -> box_min=%s box_max=%s" % (
        res["box_size"], res["centre_rule"],
        [round(v, 3) for v in res["box_min"]], [round(v, 3) for v in res["box_max"]]))
    print("[box] centre=%s side=%s" % (
        [round(v, 3) for v in res["centre"]], res["side_lengths"]))
    if "ligand" in res:
        lg = res["ligand"]
        print("[box] ligand coverage=%.3f  centre->ligand-centroid=%.2f A  centroid_inside=%s" % (
            lg["coverage"], lg["centre_to_ligand_centroid_A"], lg["ligand_centroid_inside_box"]))

    out = args.json or os.path.join(BASE, "reports", "box_%s.json" % pid.lower())
    with open(out, "w") as fh:
        json.dump(res, fh, indent=1)
    print("[box] wrote %s" % out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
