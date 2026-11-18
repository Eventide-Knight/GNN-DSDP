#!/usr/bin/env python
"""evaluate_dude.py -- optional bonus column: GNN residue-level AUC on DUD-E 102.

DUD-E complexes only ship pdbqt (no PDB ids), so each <id>_protein.pdbqt is
cleaned into a standard PDB (element column restored, AD4 types dropped), then
the normal pipeline runs: features.py -> graph edges -> best_model -> per-residue
probability, with 5 A labels from <id>_ligand.pdbqt.

Smoke gate: aa2ar + abl1 must featurise and align before the full 102 run.

Output: gnn_site/reports/eval_dude.json + gnn_site/results/table_dude102.csv

Usage:
  python scripts/evaluate_dude.py --workers 8
  python scripts/evaluate_dude.py --ids aa2ar,abl1 --no-write     # smoke
"""
import argparse
import csv
import json
import os
import sys
import tempfile
import traceback

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from sklearn.metrics import average_precision_score, roc_auc_score  # noqa: E402

import build_dataset as B  # noqa: E402
import features as F  # noqa: E402
import graph as G  # noqa: E402
import model as M  # noqa: E402
import run_gnn_dsdp as RD  # noqa: E402
import paths  # noqa: E402  (同目录：统一路径配置)

BASE = paths.GNNSITE_HOME
DUD = paths.DUD_E_DIR
_MODEL = None


def _load_model():
    ck = torch.load(os.path.join(BASE, "best_model.pt"), map_location="cpu", weights_only=False)
    net = M.GCN_SiteModel(hidden_channels=tuple(ck.get("hidden", M.HIDDEN_CHANNELS)))
    net.load_state_dict(ck["model_state"])
    net.eval()
    return net


def _work(tid):
    try:
        net = _MODEL
        if net is None:
            net = _load_model()
        src = os.path.join(DUD, tid, tid + "_protein.pdbqt")
        lig_path = os.path.join(DUD, tid, tid + "_ligand.pdbqt")
        tmpdir = tempfile.mkdtemp(prefix="dude_")
        pdb = RD.pdbqt_to_pdb(src, os.path.join(tmpdir, tid + ".pdb"))
        x, meta = F.residue_features(pdb)
        data = G.build_graph(x, meta["ca"], y=None)
        with torch.no_grad():
            p = net(data.x, data.edge_index, data.edge_weight)[:, 1].exp().numpy()
        prot = B.parse_protein_pdb(pdb)
        lig = B.parse_ligand_pdbqt(lig_path)
        if len(lig) < 3:
            return tid, None, "ligand_heavy<%d" % len(lig)
        y, _ = B.residue_labels(prot, lig, 5.0)
        if len(y) != len(p):
            return tid, None, "align %d vs %d" % (len(y), len(p))
        return tid, (y.astype(np.int8), p.astype(np.float32)), None
    except Exception as e:
        return tid, None, "%s: %s" % (type(e).__name__, e)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--ids", default=None)
    ap.add_argument("--no-write", action="store_true")
    ap.add_argument("--json", default=os.path.join(BASE, "reports", "eval_dude.json"))
    ap.add_argument("--csv", default=os.path.join(BASE, "results", "table_dude102.csv"))
    args = ap.parse_args()

    global _MODEL
    _MODEL = _load_model()

    ids = ([x.strip() for x in args.ids.split(",") if x.strip()] if args.ids
           else sorted(d for d in os.listdir(DUD) if os.path.isdir(os.path.join(DUD, d))))
    print("[dude] %d targets, workers=%d" % (len(ids), args.workers))

    import multiprocessing as mp
    ctx = mp.get_context("fork")
    results, skipped = {}, []
    if args.workers <= 1:
        for tid in ids:
            tid, res, err = _work(tid)
            (results.__setitem__(tid, res) if res is not None else skipped.append("%s\t%s" % (tid, err)))
    else:
        with ctx.Pool(args.workers) as pool:
            for tid, res, err in pool.imap_unordered(_work, ids, chunksize=2):
                if res is None:
                    skipped.append("%s\t%s" % (tid, err))
                    print("[dude] skip %s: %s" % (tid, err))
                else:
                    results[tid] = res
    ys = np.concatenate([v[0] for v in results.values()]) if results else np.array([])
    ps = np.concatenate([v[1] for v in results.values()]) if results else np.array([])
    per = []
    for tid, (y, p) in sorted(results.items()):
        rec = {"target": tid, "n_residues": int(len(y)), "n_pos": int(y.sum())}
        if y.sum() > 0 and y.sum() < len(y):
            rec["roc_auc"] = float(roc_auc_score(y, p))
            rec["pr_auc"] = float(average_precision_score(y, p))
        per.append(rec)
    summary = {"n_targets": len(results), "n_skipped": len(skipped),
               "n_residues": int(len(ys)), "n_positive": int(ys.sum()) if len(ys) else 0,
               "roc_auc": float(roc_auc_score(ys, ps)) if len(ys) and 0 < ys.sum() < len(ys) else None,
               "pr_auc": float(average_precision_score(ys, ps)) if len(ys) and 0 < ys.sum() < len(ys) else None,
               "skipped": skipped, "per_target": per}
    print(json.dumps({k: v for k, v in summary.items() if k not in ("per_target", "skipped")}, indent=1))
    if not args.no_write and len(ys):
        with open(args.json, "w") as fh:
            json.dump(summary, fh, indent=1)
        with open(args.csv, "w", newline="") as fh:
            w = csv.writer(fh)
            w.writerow(["scope", "n", "roc_auc", "pr_auc"])
            w.writerow(["DUD-E 102 (all residues)", summary["n_targets"],
                        summary["roc_auc"], summary["pr_auc"]])
        print("[dude] wrote %s and %s" % (args.json, args.csv))
    return 0


if __name__ == "__main__":
    sys.exit(main())
