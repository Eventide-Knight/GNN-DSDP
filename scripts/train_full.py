#!/usr/bin/env python
"""train_full.py -- production training loop for the residue GCN (C7a).

Differences from the C6 smoke script (train_smoke.py):
  * trains on the whole 3,000-complex train split and evaluates on the whole
    968-complex val split (both loaded from data/graph/),
  * saves the best model by validation PR-AUC (weights + optimizer state +
    epoch + metrics) to <tag>_best.pt,
  * writes a per-epoch log to reports/<tag>_train.json and a curve to
    reports/<tag>_curve.png,
  * auto class weight from the actual training split (neg/pos),
  * configurable device / epochs / batch size / patience / seed,
  * one printed line per epoch (no spam).

Test split is NEVER touched here (C7b uses it).

Usage (GPU):
  python scripts/train_full.py --device cuda --epochs 100 --batch-size 32 \
      --patience 15 --seed 0 --tag c7a
Usage (CPU fallback):
  python scripts/train_full.py --device cpu --epochs 60 --batch-size 16 \
      --patience 12 --seed 0 --tag c7a_cpu
"""
import argparse
import glob
import json
import math
import os
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import average_precision_score, roc_auc_score

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from torch_geometric.loader import DataLoader  # noqa: E402

import model as M  # noqa: E402
import paths  # noqa: E402  (同目录：统一路径配置)

BASE = paths.GNNSITE_HOME


def load_graphs(split, limit=0, offset=0, verbose=True):
    files = sorted(glob.glob(os.path.join(BASE, "data", "graph", split, "*.pt")))
    files = files[offset:]
    if limit:
        files = files[:limit]
    if not files:
        raise SystemExit("no .pt for split=%s" % split)
    graphs = []
    for f in files:
        g = torch.load(f, map_location="cpu", weights_only=False)
        if getattr(g, "y", None) is None:
            raise SystemExit("%s has no labels" % f)
        graphs.append(g)
    if verbose:
        n_res = int(sum(g.num_nodes for g in graphs))
        n_pos = int(sum(int(g.y.sum()) for g in graphs))
        print("[data] %-5s graphs=%d residues=%d positives=%d (%.2f%%)" % (
            split, len(graphs), n_res, n_pos, 100.0 * n_pos / max(n_res, 1)))
    return graphs


def class_weight(graphs, mode="linear"):
    n_pos = int(sum(int(g.y.sum()) for g in graphs))
    n_tot = int(sum(int(g.y.numel()) for g in graphs))
    ratio = (n_tot - n_pos) / max(n_pos, 1)
    if mode == "linear":
        w_pos = ratio
    elif mode == "logarithmic":
        w_pos = math.log(max(ratio, 1.0001))
    elif mode == "smoothed_linear":
        w_pos = math.sqrt(ratio)
    elif mode == "none":
        w_pos = 1.0
    else:
        raise ValueError(mode)
    return torch.tensor([1.0, w_pos], dtype=torch.float32), {
        "n_pos": n_pos, "n_tot": n_tot, "neg_pos_ratio": ratio,
        "mode": mode, "w_pos": w_pos}


def evaluate(model, graphs, device, batch_size=64, with_loss=True, weight=None):
    model.eval()
    loader = DataLoader(graphs, batch_size=batch_size, shuffle=False)
    probs, labels, total_loss, total_nodes = [], [], 0.0, 0
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            out = model(batch.x, batch.edge_index, batch.edge_weight)
            if with_loss and weight is not None:
                total_loss += float(F.nll_loss(out, batch.y, weight=weight.to(device))) * int(batch.num_nodes)
                total_nodes += int(batch.num_nodes)
            p = out[:, 1].exp()
            probs.append(p.detach().cpu().numpy())
            labels.append(batch.y.detach().cpu().numpy())
    p = np.concatenate(probs).astype(np.float64)
    y = np.concatenate(labels).astype(np.int64)
    return {
        "loss": (total_loss / max(total_nodes, 1)) if total_nodes else None,
        "pr_auc": float(average_precision_score(y, p)),
        "roc_auc": float(roc_auc_score(y, p)),
        "n_nodes": int(len(y)),
        "n_pos": int(y.sum()),
        "p_pos_mean": float(p.mean()),
    }


def train_epoch(model, loader, optimizer, weight, device):
    model.train()
    total_loss, total_nodes = 0.0, 0
    for batch in loader:
        batch = batch.to(device)
        optimizer.zero_grad(set_to_none=True)
        out = model(batch.x, batch.edge_index, batch.edge_weight)
        loss = F.nll_loss(out, batch.y, weight=weight.to(device))
        if not torch.isfinite(loss):
            raise RuntimeError("non-finite loss")
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 10.0)
        optimizer.step()
        total_loss += float(loss) * int(batch.num_nodes)
        total_nodes += int(batch.num_nodes)
    return total_loss / max(total_nodes, 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train-split", default="train")
    ap.add_argument("--val-split", default="val")
    ap.add_argument("--train-limit", type=int, default=0)
    ap.add_argument("--val-limit", type=int, default=0)
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--lr", type=float, default=0.01)
    ap.add_argument("--weight-decay", type=float, default=0.0)
    ap.add_argument("--patience", type=int, default=15)
    ap.add_argument("--hidden", default="38,28,18,8")
    ap.add_argument("--dropout", type=float, default=0.0)
    ap.add_argument("--weight-mode", default="linear",
                    choices=["linear", "logarithmic", "smoothed_linear", "none"])
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--num-workers", type=int, default=0)
    ap.add_argument("--tag", default="c7a")
    ap.add_argument("--ckpt-name", default=None,
                    help="checkpoint filename inside --out-dir (default <tag>_best.pt)")
    ap.add_argument("--out-dir", default=BASE)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    hidden = tuple(int(x) for x in args.hidden.split(","))
    if len(hidden) != 4:
        raise SystemExit("--hidden must have 4 values, e.g. 38,28,18,8")

    if args.device == "cuda" and not torch.cuda.is_available():
        print("[warn] cuda requested but not available -> cpu")
        args.device = "cpu"
    device = torch.device(args.device)
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True
        print("[dev] cuda '%s' (%d device(s))" % (
            torch.cuda.get_device_name(0), torch.cuda.device_count()))
    else:
        print("[dev] cpu (%d threads)" % torch.get_num_threads())

    train_graphs = load_graphs(args.train_split, args.train_limit)
    val_graphs = load_graphs(args.val_split, args.val_limit)
    weight, wstats = class_weight(train_graphs, args.weight_mode)
    print("[data] class weight w_pos=%.3f (mode=%s, neg/pos=%.2f)" % (
        wstats["w_pos"], args.weight_mode, wstats["neg_pos_ratio"]))

    model = M.GCN_SiteModel(hidden_channels=hidden, dropout=args.dropout, head="logsoftmax")
    n_params = model.n_parameters()
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr,
                                 weight_decay=args.weight_decay)
    print("[model] params=%d hidden=%s dropout=%.2f lr=%g" % (
        n_params, list(hidden), args.dropout, args.lr))
    print("[run] epochs=%d batch=%d patience=%d seed=%d tag=%s" % (
        args.epochs, args.batch_size, args.patience, args.seed, args.tag))

    train_loader = DataLoader(train_graphs, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.num_workers,
                              pin_memory=(device.type == "cuda"))

    os.makedirs(os.path.join(args.out_dir, "reports"), exist_ok=True)
    ckpt_name = args.ckpt_name or ("%s_best.pt" % args.tag)
    ckpt_path = os.path.join(args.out_dir, ckpt_name)
    json_path = os.path.join(args.out_dir, "reports", "%s_train.json" % args.tag)
    png_path = os.path.join(args.out_dir, "reports", "%s_curve.png" % args.tag)

    best = {"pr_auc": -1.0, "epoch": -1, "roc_auc": 0.0, "val_loss": None}
    bad = 0
    history = []
    t_start = time.time()
    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        tr_loss = train_epoch(model, train_loader, optimizer, weight, device)
        val = evaluate(model, val_graphs, device, weight=weight)
        dt = time.time() - t0
        rec = {"epoch": epoch, "train_loss": tr_loss, "val_loss": val["loss"],
               "val_pr_auc": val["pr_auc"], "val_roc_auc": val["roc_auc"],
               "seconds": round(dt, 1)}
        history.append(rec)
        improved = val["pr_auc"] > best["pr_auc"]
        flag = ""
        if improved:
            best = {"pr_auc": val["pr_auc"], "epoch": epoch,
                    "roc_auc": val["roc_auc"], "val_loss": val["loss"]}
            bad = 0
            torch.save({
                "model_state": model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "epoch": epoch,
                "val_pr_auc": val["pr_auc"],
                "val_roc_auc": val["roc_auc"],
                "val_loss": val["loss"],
                "train_loss": tr_loss,
                "args": vars(args),
                "hidden": list(hidden),
                "in_channels": M.IN_CHANNELS,
                "n_params": n_params,
            }, ckpt_path)
            flag = " *best"
        else:
            bad += 1
        print("epoch %3d | train_loss %.4f | val_loss %.4f | PR-AUC %.4f | "
              "ROC-AUC %.4f | %5.1fs%s" % (
                  epoch, tr_loss, val["loss"] if val["loss"] is not None else -1,
                  val["pr_auc"], val["roc_auc"], dt, flag))
        if bad >= args.patience:
            print("[early-stop] no PR-AUC improvement for %d epochs" % args.patience)
            break

    total = time.time() - t_start
    print("[done] best epoch=%d val PR-AUC=%.4f ROC-AUC=%.4f -> %s" % (
        best["epoch"], best["pr_auc"], best["roc_auc"], ckpt_path))

    # final evaluation with the best checkpoint
    ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    model.load_state_dict(ck["model_state"])
    model.to(device)
    final = evaluate(model, val_graphs, device, weight=weight)
    print("[final] val PR-AUC=%.4f ROC-AUC=%.4f (epoch %d)" % (
        final["pr_auc"], final["roc_auc"], ck["epoch"]))

    report = {
        "tag": args.tag,
        "device": str(device),
        "device_name": (torch.cuda.get_device_name(0) if device.type == "cuda" else "cpu"),
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "n_train": len(train_graphs),
        "n_val": len(val_graphs),
        "n_params": n_params,
        "hidden": list(hidden),
        "args": vars(args),
        "class_weight": wstats,
        "history": history,
        "best": best,
        "final_val": final,
        "checkpoint": ckpt_path,
        "wall_seconds": round(total, 1),
        "early_stopped": len(history) < args.epochs,
    }
    with open(json_path, "w") as fh:
        json.dump(report, fh, indent=1)
    print("[write] %s" % json_path)

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(1, 2, figsize=(11, 3.6))
        ep = [h["epoch"] for h in history]
        ax[0].plot(ep, [h["train_loss"] for h in history], label="train")
        ax[0].plot(ep, [h["val_loss"] for h in history], label="val")
        ax[0].set_xlabel("epoch"); ax[0].set_ylabel("NLL loss"); ax[0].legend(); ax[0].grid(alpha=0.3)
        ax[1].plot(ep, [h["val_pr_auc"] for h in history], label="val PR-AUC")
        ax[1].plot(ep, [h["val_roc_auc"] for h in history], label="val ROC-AUC")
        ax[1].axvline(best["epoch"], ls="--", color="grey", lw=1)
        ax[1].set_xlabel("epoch"); ax[1].set_ylabel("score"); ax[1].legend(); ax[1].grid(alpha=0.3)
        fig.tight_layout(); fig.savefig(png_path, dpi=130)
        print("[write] %s" % png_path)
    except Exception as e:
        print("[plot] skipped: %r" % (e,))

    return 0 if best["roc_auc"] > 0.8 else 3


if __name__ == "__main__":
    sys.exit(main())
