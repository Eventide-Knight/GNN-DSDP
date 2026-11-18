#!/usr/bin/env python
"""predict.py -- per-residue ligand-binding probability for one protein.

Uses the final full-data model (gnn_site/best_model.pt by default), the same
46-dim features (features.py) and the same 10 A CA graph (graph.py) used in
training.

Usage:
  python predict.py --protein 6qqw
  python predict.py --protein 6qqw --out reports/predict_6qqw.csv
  python predict.py --protein /abs/path/x_protein.pdb --model best_model.pt
  python predict.py --protein 6qqw --topk 15 --json reports/predict_6qqw.json

Output CSV columns:
  rank,chain,resseq,icode,resname,ca_x,ca_y,ca_z,prob
(sorted by probability, descending). `prob` is P(ligand-binding residue).
"""
import argparse
import csv
import json
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import features as F  # noqa: E402
import graph as G  # noqa: E402
import model as M  # noqa: E402

BASE = "/share/home/huangyy/DSDP/gnn_site"
DEFAULT_MODEL = os.path.join(BASE, "best_model.pt")


def load_model(path=DEFAULT_MODEL, device="cpu"):
    ck = torch.load(path, map_location="cpu", weights_only=False)
    hidden = tuple(ck.get("hidden", M.HIDDEN_CHANNELS))
    net = M.GCN_SiteModel(in_channels=ck.get("in_channels", M.IN_CHANNELS),
                          hidden_channels=hidden, head="logsoftmax")
    net.load_state_dict(ck["model_state"])
    net.to(device).eval()
    return net, ck


def predict_one(net, pdb_path, device="cpu", cutoff=G.DEFAULT_CUTOFF):
    x, meta = F.residue_features(pdb_path, cutoff=cutoff)
    data = G.build_graph(x, meta["ca"], y=None, cutoff=cutoff,
                         pdb_id=os.path.basename(os.path.dirname(pdb_path)), split="")
    with torch.no_grad():
        out = net(data.x.to(device), data.edge_index.to(device),
                  data.edge_weight.to(device))
        prob = out[:, 1].exp().cpu().numpy().astype(np.float64)
    return prob, meta


def resolve_protein(spec):
    if os.path.exists(spec):
        return spec
    if len(spec) == 4 and spec.isalnum():
        with open(os.path.join(BASE, "structure_index.json")) as fh:
            idx = json.load(fh)["index"]
        e = idx.get(spec.lower())
        if e is None:
            raise SystemExit("unknown PDB id %r" % spec)
        return e["protein"]
    raise SystemExit("not a file and not a 4-char PDB id: %r" % spec)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--protein", required=True, help="4-char PDB id or path to *_protein.pdb")
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--out", default=None, help="output CSV (default: reports/predict_<id>.csv)")
    ap.add_argument("--json", default=None)
    ap.add_argument("--topk", type=int, default=10, help="print top-K residues")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--cutoff", type=float, default=G.DEFAULT_CUTOFF)
    args = ap.parse_args()

    device = torch.device(args.device if (args.device == "cpu" or torch.cuda.is_available())
                          else "cpu")
    pdb = resolve_protein(args.protein)
    pid = os.path.basename(os.path.dirname(pdb)) or os.path.basename(pdb).split("_")[0]
    print("[predict] pdb=%s  model=%s  device=%s" % (pdb, args.model, device))

    net, ck = load_model(args.model, device)
    print("[predict] model epoch=%s val PR-AUC=%.4f ROC-AUC=%.4f params=%d" % (
        ck.get("epoch"), ck.get("val_pr_auc", -1), ck.get("val_roc_auc", -1),
        sum(p.numel() for p in net.parameters())))

    prob, meta = predict_one(net, pdb, device, args.cutoff)
    order = np.argsort(-prob)
    chains = meta["chains"]
    resseq = meta["resseq"]
    icode = meta["icode"]
    resname = meta["resname"]
    ca = meta["ca"]

    out = args.out or os.path.join(BASE, "reports", "predict_%s.csv" % pid.lower())
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["rank", "chain", "resseq", "icode", "resname", "ca_x", "ca_y", "ca_z", "prob"])
        for rank, r in enumerate(order, 1):
            w.writerow([rank, chains[r], int(resseq[r]), str(icode[r]).strip(), resname[r],
                        "%.3f" % ca[r][0], "%.3f" % ca[r][1], "%.3f" % ca[r][2],
                        "%.6f" % prob[r]])
    print("[predict] %d residues -> %s" % (len(prob), out))
    print("[predict] prob: min=%.4f max=%.4f mean=%.4f  n>0.5=%d" % (
        prob.min(), prob.max(), prob.mean(), int((prob > 0.5).sum())))
    print("[predict] top-%d residues:" % args.topk)
    for rank, r in enumerate(order[:args.topk], 1):
        print("  %2d. %s%d%s %-3s  prob=%.4f" % (
            rank, chains[r], int(resseq[r]), str(icode[r]).strip(), resname[r], prob[r]))

    summary = {"pdb": pdb, "pdb_id": pid, "n_residues": int(len(prob)),
               "prob_min": float(prob.min()), "prob_max": float(prob.max()),
               "prob_mean": float(prob.mean()),
               "n_prob_gt_0.5": int((prob > 0.5).sum()),
               "model_epoch": ck.get("epoch"),
               "csv": out,
               "top": [{"chain": str(chains[r]), "resseq": int(resseq[r]),
                        "icode": str(icode[r]).strip(), "resname": str(resname[r]),
                        "prob": float(prob[r])} for r in order[:args.topk]]}
    if args.json:
        with open(args.json, "w") as fh:
            json.dump(summary, fh, indent=1)
        print("[predict] wrote %s" % args.json)
    return 0


if __name__ == "__main__":
    sys.exit(main())
