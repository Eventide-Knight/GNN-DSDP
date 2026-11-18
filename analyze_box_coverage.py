#!/usr/bin/env python
"""analyze_box_coverage.py -- which threshold should C8 use to build the docking box?

For every test complex and every selection rule (probability threshold
0.3/0.4/0.5/0.6/0.7, or top-k residues) we build the naive box = bounding box
of the selected residues' CA atoms (+ margin) and ask:

  * ligand coverage  = fraction of true ligand heavy atoms inside the box
                       (the metric that actually matters for docking: if the
                       ligand is not inside the box, docking cannot succeed),
  * box side lengths (A) -- DSDP's default box is 30 A,
  * centre error     = |box centre - ligand centroid| (A).

This is analysis only; it does not build boxes for docking (that is C8).

Usage:
  python analyze_box_coverage.py --model best_model.pt --split test
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


def ligand_xyz(pid, index):
    e = index.get(pid)
    if not e:
        return None
    path = e.get("ligand_pdbqt") or e.get("ligand_mol2")
    if not path or not os.path.exists(path):
        return None
    if path.endswith("mol2"):
        return B.parse_ligand_mol2(path)
    return B.parse_ligand_pdbqt(path)


def box_stats(ca_sel, lig, margin):
    lo = ca_sel.min(axis=0) - margin
    hi = ca_sel.max(axis=0) + margin
    side = hi - lo
    inside = ((lig >= lo) & (lig <= hi)).all(axis=1)
    cover = float(inside.mean()) if len(lig) else 0.0
    centre = (lo + hi) / 2.0
    err = float(np.linalg.norm(centre - lig.mean(axis=0))) if len(lig) else np.nan
    return cover, side, err


def fixed_box_stats(ca_sel, lig, half=15.0):
    """DSDP-style fixed 30 A box centred on the selected-residue centroid."""
    centre = ca_sel.mean(axis=0)
    lo = centre - half
    hi = centre + half
    inside = ((lig >= lo) & (lig <= hi)).all(axis=1)
    cover = float(inside.mean()) if len(lig) else 0.0
    err = float(np.linalg.norm(centre - lig.mean(axis=0))) if len(lig) else np.nan
    return cover, err


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=os.path.join(BASE, "best_model.pt"))
    ap.add_argument("--split", default="test")
    ap.add_argument("--margin", type=float, default=5.0,
                    help="A added on each side of the CA bounding box")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--json", default=os.path.join(BASE, "reports", "box_coverage.json"))
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

    rules = [("thr_%.1f" % t, ("thr", t)) for t in (0.3, 0.4, 0.5, 0.6, 0.7)]
    rules += [("top_%d" % k, ("top", k)) for k in (10, 20, 30)]

    agg = {name: {"cover": [], "side": [], "err": [], "n_sel": [], "empty": 0,
                  "fix_cover": [], "fix_err": []}
           for name, _ in rules}
    used = 0
    for g, p in zip(graphs, probs):
        pid = str(g.pdb_id).lower()
        lig = ligand_xyz(pid, index)
        if lig is None or len(lig) < 3:
            continue
        used += 1
        ca = g.ca.numpy()
        for name, (kind, val) in rules:
            if kind == "thr":
                sel = np.where(p >= val)[0]
                if len(sel) == 0:
                    agg[name]["empty"] += 1
                    sel = np.array([int(np.argmax(p))])   # fall back to the peak residue
            else:
                sel = np.argsort(-p)[:min(val, len(p))]
            cover, side, err = box_stats(ca[sel], lig, args.margin)
            fcover, ferr = fixed_box_stats(ca[sel], lig, half=15.0)
            agg[name]["cover"].append(cover)
            agg[name]["side"].append(side)
            agg[name]["err"].append(err)
            agg[name]["n_sel"].append(len(sel))
            agg[name]["fix_cover"].append(fcover)
            agg[name]["fix_err"].append(ferr)

    summary = {"split": args.split, "n_complexes_used": used,
               "margin_A": args.margin,
               "model": args.model, "epoch": ck.get("epoch"), "rules": {}}
    print("%-8s %8s %9s %10s %10s %9s %9s %9s | %10s %11s" % (
        "rule", "n_sel", "cover", "cover>=.9", "cover>=.5", "med_err", "side_x", "side_y",
        "fix30_cov", "fix30>=.9"))
    for name, _ in rules:
        a = agg[name]
        cov = np.asarray(a["cover"])
        side = np.asarray(a["side"])
        err = np.asarray(a["err"])
        rec = {
            "n_complexes": int(len(cov)),
            "empty_selection": int(a["empty"]),
            "mean_n_selected": float(np.mean(a["n_sel"])),
            "mean_ligand_coverage": float(cov.mean()),
            "frac_coverage_ge_0.9": float((cov >= 0.9).mean()),
            "frac_coverage_ge_0.5": float((cov >= 0.5).mean()),
            "median_centre_error_A": float(np.median(err)),
            "mean_centre_error_A": float(err.mean()),
            "mean_side_A": side.mean(axis=0).round(2).tolist(),
            "median_side_A": np.median(side, axis=0).round(2).tolist(),
            "frac_all_sides_le_40A": float(((side <= 40).all(axis=1)).mean()),
            "fixed30_mean_coverage": float(np.mean(a["fix_cover"])),
            "fixed30_frac_coverage_ge_0.9": float((np.asarray(a["fix_cover"]) >= 0.9).mean()),
            "fixed30_median_centre_error_A": float(np.median(a["fix_err"])),
        }
        summary["rules"][name] = rec
        print("%-8s %8.1f %8.3f %9.1f%% %9.1f%% %8.1f %8.1f %8.1f | %10.3f %10.1f%%" % (
            name, rec["mean_n_selected"], rec["mean_ligand_coverage"],
            100 * rec["frac_coverage_ge_0.9"], 100 * rec["frac_coverage_ge_0.5"],
            rec["median_centre_error_A"], rec["mean_side_A"][0], rec["mean_side_A"][1],
            rec["fixed30_mean_coverage"], 100 * rec["fixed30_frac_coverage_ge_0.9"]))

    with open(args.json, "w") as fh:
        json.dump(summary, fh, indent=1)
    print("[boxcheck] wrote %s" % args.json)
    return 0


if __name__ == "__main__":
    sys.exit(main())
