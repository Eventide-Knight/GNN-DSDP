#!/usr/bin/env python
"""evaluate_model.py -- final evaluation of the full-data model.

  * val 968 and test 363: NLL loss, PR-AUC, ROC-AUC (test is evaluated ONCE,
    never used for model selection),
  * threshold sweep on val (0.3/0.4/0.5/0.6/0.7) -> precision / recall / F1,
  * top-k residue sweep on val (k = 5/10/20/30/50) -> precision / recall,
  * bonus: on test, distance between the predicted site centre (top-k CA
    centroid) and the true ligand heavy-atom centroid.

Usage:
  python evaluate_model.py --model gnn_site/best_model.pt --device cuda
"""
import argparse
import csv
import glob
import json
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import average_precision_score, roc_auc_score

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from torch_geometric.loader import DataLoader  # noqa: E402

import model as M  # noqa: E402
import paths  # noqa: E402  (同目录：统一路径配置)

BASE = paths.GNNSITE_HOME


def load_graphs(split, limit=0):
    files = sorted(glob.glob(os.path.join(BASE, "data", "graph", split, "*.pt")))
    if limit:
        files = files[:limit]
    graphs = [torch.load(f, map_location="cpu", weights_only=False) for f in files]
    return graphs


def predict_all(net, graphs, device, batch_size=64):
    """Return per-graph probability arrays and labels."""
    net.eval()
    loader = DataLoader(graphs, batch_size=batch_size, shuffle=False)
    probs, labels, losses = [], [], 0.0
    n_nodes = 0
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            out = net(batch.x, batch.edge_index, batch.edge_weight)
            losses += float(F.nll_loss(out, batch.y)) * int(batch.num_nodes)
            n_nodes += int(batch.num_nodes)
            probs.append(out[:, 1].exp().cpu().numpy())
            labels.append(batch.y.cpu().numpy())
    p = np.concatenate(probs)
    y = np.concatenate(labels)
    # split per graph using graph node counts
    per_graph, off = [], 0
    for g in graphs:
        n = int(g.num_nodes)
        per_graph.append(p[off:off + n])
        off += n
    return p, y, per_graph, losses / max(n_nodes, 1)


def metrics(y, p):
    pred = (p >= 0.5).astype(np.int64)
    tp = int(((pred == 1) & (y == 1)).sum())
    fp = int(((pred == 1) & (y == 0)).sum())
    fn = int(((pred == 0) & (y == 1)).sum())
    return {
        "pr_auc": float(average_precision_score(y, p)),
        "roc_auc": float(roc_auc_score(y, p)),
        "precision@0.5": tp / max(tp + fp, 1),
        "recall@0.5": tp / max(tp + fn, 1),
        "n_nodes": int(len(y)), "n_pos": int(y.sum()),
    }


def threshold_table(y, p, thresholds=(0.3, 0.4, 0.5, 0.6, 0.7)):
    rows = []
    for t in thresholds:
        pred = p >= t
        tp = int((pred & (y == 1)).sum())
        fp = int((pred & (y == 0)).sum())
        fn = int(((~pred) & (y == 1)).sum())
        prec = tp / max(tp + fp, 1)
        rec = tp / max(tp + fn, 1)
        f1 = 2 * prec * rec / max(prec + rec, 1e-9)
        rows.append({"threshold": t, "precision": prec, "recall": rec, "f1": f1,
                     "n_pred": int(pred.sum()), "tp": tp, "fp": fp, "fn": fn})
    return rows


def topk_table(graphs, per_graph_probs, ks=(5, 10, 20, 30, 50)):
    rows = []
    for k in ks:
        precs, recs, micro_tp, micro_n, micro_pos = [], [], 0, 0, 0
        for g, p in zip(graphs, per_graph_probs):
            y = g.y.numpy()
            n_pos = int(y.sum())
            kk = min(k, len(p))
            top = np.argsort(-p)[:kk]
            tp = int(y[top].sum())
            precs.append(tp / kk)
            recs.append(tp / n_pos if n_pos else 0.0)
            micro_tp += tp
            micro_n += kk
            micro_pos += n_pos
        rows.append({"k": k,
                     "macro_precision": float(np.mean(precs)),
                     "macro_recall": float(np.mean(recs)),
                     "micro_precision": micro_tp / max(micro_n, 1),
                     "micro_recall": micro_tp / max(micro_pos, 1)})
    return rows


def ligand_centroid(entry):
    import build_dataset as B
    xyz = B.parse_ligand_pdbqt(entry["ligand_pdbqt"])
    if len(xyz) == 0:
        return None
    return xyz.mean(axis=0)


def center_distances(graphs, per_graph_probs, ks=(10, 20, 30)):
    with open(os.path.join(BASE, "structure_index.json")) as fh:
        index = json.load(fh)["index"]
    out = {}
    for k in ks:
        ds = []
        for g, p in zip(graphs, per_graph_probs):
            pid = str(g.pdb_id).lower()
            if pid not in index:
                continue
            true_c = ligand_centroid(index[pid])
            if true_c is None:
                continue
            ca = g.ca.numpy()
            top = np.argsort(-p)[:min(k, len(p))]
            pred_c = ca[top].mean(axis=0)
            ds.append(float(np.linalg.norm(pred_c - true_c)))
        ds = np.asarray(ds)
        if len(ds):
            out[str(k)] = {"n": int(len(ds)), "mean": float(ds.mean()),
                           "median": float(np.median(ds)),
                           "p25": float(np.percentile(ds, 25)),
                           "p75": float(np.percentile(ds, 75)),
                           "min": float(ds.min()), "max": float(ds.max()),
                           "frac_lt_5A": float((ds < 5).mean()),
                           "frac_lt_10A": float((ds < 10).mean())}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=os.path.join(BASE, "best_model.pt"))
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--json", default=os.path.join(BASE, "reports", "eval_model.json"))
    ap.add_argument("--csv-threshold",
                    default=os.path.join(BASE, "reports", "threshold_val.csv"))
    ap.add_argument("--csv-topk", default=os.path.join(BASE, "reports", "topk_val.csv"))
    args = ap.parse_args()

    device = torch.device(args.device if (args.device == "cpu" or torch.cuda.is_available())
                          else "cpu")
    ck = torch.load(args.model, map_location="cpu", weights_only=False)
    net = M.GCN_SiteModel(hidden_channels=tuple(ck.get("hidden", M.HIDDEN_CHANNELS)),
                          head="logsoftmax")
    net.load_state_dict(ck["model_state"])
    net.to(device).eval()
    print("[eval] model=%s epoch=%s params=%d device=%s" % (
        args.model, ck.get("epoch"), sum(p.numel() for p in net.parameters()), device))

    report = {"model": args.model, "epoch": ck.get("epoch"),
              "ckpt_val_pr_auc": ck.get("val_pr_auc"),
              "ckpt_val_roc_auc": ck.get("val_roc_auc"),
              "n_train_used": ck.get("args", {}).get("train_limit") or "full"}

    val_graphs = load_graphs("val")
    pv, yv, pv_g, loss_v = predict_all(net, val_graphs, device)
    mv = metrics(yv, pv)
    mv["loss_unweighted"] = loss_v
    report["val"] = mv
    print("[eval] val  n=%d nodes PR-AUC=%.4f ROC-AUC=%.4f loss=%.4f" % (
        len(val_graphs), mv["pr_auc"], mv["roc_auc"], loss_v))

    test_graphs = load_graphs("test")
    pt, yt, pt_g, loss_t = predict_all(net, test_graphs, device)
    mt = metrics(yt, pt)
    mt["loss_unweighted"] = loss_t
    report["test"] = mt
    print("[eval] test n=%d nodes PR-AUC=%.4f ROC-AUC=%.4f loss=%.4f" % (
        len(test_graphs), mt["pr_auc"], mt["roc_auc"], loss_t))

    thr = threshold_table(yv, pv)
    report["val_threshold_sweep"] = thr
    with open(args.csv_threshold, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(thr[0].keys()))
        w.writeheader()
        w.writerows(thr)
    print("[eval] val threshold sweep:")
    for r in thr:
        print("   t=%.1f  P=%.4f  R=%.4f  F1=%.4f  n_pred=%d" % (
            r["threshold"], r["precision"], r["recall"], r["f1"], r["n_pred"]))

    topk = topk_table(val_graphs, pv_g)
    report["val_topk_sweep"] = topk
    with open(args.csv_topk, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(topk[0].keys()))
        w.writeheader()
        w.writerows(topk)
    print("[eval] val top-k sweep:")
    for r in topk:
        print("   k=%2d  macroP=%.4f  macroR=%.4f  microP=%.4f  microR=%.4f" % (
            r["k"], r["macro_precision"], r["macro_recall"],
            r["micro_precision"], r["micro_recall"]))

    try:
        cd = center_distances(test_graphs, pt_g, ks=(10, 20, 30))
        report["test_center_distance_A"] = cd
        print("[eval] test predicted-centre vs ligand-centroid distance (A):")
        for k, v in cd.items():
            print("   top-%-2s  mean=%.2f  median=%.2f  p25=%.2f p75=%.2f  <5A=%.1f%%  <10A=%.1f%%" % (
                k, v["mean"], v["median"], v["p25"], v["p75"],
                100 * v["frac_lt_5A"], 100 * v["frac_lt_10A"]))
    except Exception as e:
        report["test_center_distance_A"] = {"error": repr(e)}
        print("[eval] center distance skipped: %r" % (e,))

    os.makedirs(os.path.dirname(args.json), exist_ok=True)
    with open(args.json, "w") as fh:
        json.dump(report, fh, indent=1)
    print("[eval] wrote %s" % args.json)
    return 0


if __name__ == "__main__":
    sys.exit(main())
