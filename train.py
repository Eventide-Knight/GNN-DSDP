#!/usr/bin/env python
"""train.py -- minimal training loop for the residue GCN (C6 smoke only).

This is the pipeline proof, NOT the real training run:
  * a handful of train graphs (20-50) and the 10 val graphs,
  * 1-2 epochs, Adam lr=0.01, NLLLoss with class weights (~4% positives),
  * early stopping on validation PR-AUC,
  * no checkpoint is written to disk (best state is kept in memory only).

It verifies, with assertions and printed evidence:
  1. graphs are read from .pt and collated into mini-batches, and the PyG
     `batch` vector is consistent (num_graphs, node counts, batch.max+1);
  2. every loss is finite and the loss trend goes down (per-step trace plus a
     controlled overfit stage on a few graphs);
  3. backward + optimizer.step() actually changes the weights (no silent
     no-op, no NaN gradients);
  4. PR-AUC / ROC-AUC can be computed on the val graphs.

Loss head: 8 -> 2 logits, NLLLoss(weight=[1, neg/pos]).
Modelled on /share/home/huangyy/Nodecoder/src/NodeCoder/gcn/NodeCoder_train.py
(`weighted_loss` variants: Linear / Logarithmic / Smoothed_Linear / none).

Usage:
  python train.py --train-limit 30 --val-limit 10 --epochs 2 --batch-size 8 \
      --lr 0.01 --weight-mode linear --device cpu
"""
import argparse
import glob
import json
import math
import os
import sys
import time
import traceback

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import average_precision_score, roc_auc_score

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from torch_geometric.loader import DataLoader  # noqa: E402

import model as M  # noqa: E402

BASE = "/share/home/huangyy/DSDP/gnn_site"


def load_graphs(split, limit, offset=0):
    files = sorted(glob.glob(os.path.join(BASE, "data", "graph", split, "*.pt")))
    files = files[offset:]
    if limit:
        files = files[:limit]
    if not files:
        raise SystemExit("no .pt found for split=%s" % split)
    graphs = []
    for f in files:
        g = torch.load(f, map_location="cpu", weights_only=False)
        if not hasattr(g, "y") or g.y is None:
            raise SystemExit("%s has no labels" % f)
        graphs.append(g)
    return graphs, files


def class_weights(graphs, mode="linear"):
    n_pos = int(sum(int(g.y.sum()) for g in graphs))
    n_tot = int(sum(int(g.y.numel()) for g in graphs))
    ratio = (n_tot - n_pos) / max(n_pos, 1)
    if mode == "linear":
        w = [1.0, ratio]
    elif mode == "logarithmic":
        w = [1.0, math.log(ratio)]
    elif mode == "smoothed_linear":
        w = [1.0, math.sqrt(ratio)]
    elif mode == "none":
        w = [1.0, 1.0]
    else:
        raise ValueError(mode)
    return torch.tensor(w, dtype=torch.float32), {"n_pos": n_pos, "n_tot": n_tot,
                                                 "neg_pos_ratio": ratio, "mode": mode}


def batch_sanity(batch, verbose=True):
    """Verify PyG collation and the `batch` vector."""
    nb = int(batch.num_graphs)
    n_nodes = int(batch.x.size(0))
    n_edges = int(batch.edge_index.size(1))
    batch_max = int(batch.batch.max()) + 1
    ok = (batch.batch.numel() == n_nodes and batch_max == nb
          and batch.y.numel() == n_nodes)
    if verbose:
        print("  [batch] graphs=%d nodes=%d edges=%d | batch vector %s max+1=%d ok=%s"
              % (nb, n_nodes, n_edges, tuple(batch.batch.shape), batch_max, ok))
    return {"num_graphs": nb, "nodes": n_nodes, "edges": n_edges,
            "batch_ok": bool(ok)}


def run_epoch(model, loader, optimizer, weight, device, epoch, log_every=1):
    model.train()
    total_loss = 0.0
    total_nodes = 0
    steps = []
    for step, batch in enumerate(loader, 1):
        batch = batch.to(device)
        optimizer.zero_grad()
        out = model(batch.x, batch.edge_index, batch.edge_weight)
        loss = F.nll_loss(out, batch.y, weight=weight.to(device))
        if not torch.isfinite(loss):
            raise RuntimeError("non-finite loss at epoch %d step %d" % (epoch, step))
        loss.backward()
        gnorm = torch.nn.utils.clip_grad_norm_(model.parameters(), 10.0)
        optimizer.step()
        total_loss += float(loss) * int(batch.num_nodes)
        total_nodes += int(batch.num_nodes)
        steps.append(float(loss))
        if log_every and (step == 1 or step % log_every == 0):
            print("  epoch %d step %2d/%d  loss=%.6f  grad_norm=%.4f  graphs=%d"
                  % (epoch, step, len(loader), float(loss), float(gnorm),
                     int(batch.num_graphs)))
    return total_loss / max(total_nodes, 1), steps


@torch.no_grad()
def evaluate(model, graphs, device, batch_size=4):
    model.eval()
    loader = DataLoader(graphs, batch_size=batch_size, shuffle=False)
    probs, labels = [], []
    for batch in loader:
        batch = batch.to(device)
        p = model.positive_probability(batch.x, batch.edge_index, batch.edge_weight)
        probs.append(p.detach().cpu().numpy())
        labels.append(batch.y.detach().cpu().numpy())
    p = np.concatenate(probs).astype(np.float64)
    y = np.concatenate(labels).astype(np.int64)
    pr = float(average_precision_score(y, p))
    roc = float(roc_auc_score(y, p))
    pred = (p >= 0.5).astype(np.int64)
    tp = int(((pred == 1) & (y == 1)).sum())
    fp = int(((pred == 1) & (y == 0)).sum())
    fn = int(((pred == 0) & (y == 1)).sum())
    prec = tp / max(tp + fp, 1)
    rec = tp / max(tp + fn, 1)
    return {"pr_auc": pr, "roc_auc": roc, "precision@0.5": float(prec),
            "recall@0.5": float(rec), "n_nodes": int(len(y)),
            "n_pos": int(y.sum())}


def overfit_check(model, graphs, device, steps, lr, weight, seed=0):
    """Few graphs, many steps: prove the loss really goes down."""
    torch.manual_seed(seed)
    if len(graphs) > 8:
        graphs = graphs[:8]
    loader = DataLoader(graphs, batch_size=len(graphs), shuffle=True)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    model.train()
    hist = []
    for i in range(1, steps + 1):
        batch = next(iter(loader))
        batch = batch.to(device)
        opt.zero_grad()
        out = model(batch.x, batch.edge_index, batch.edge_weight)
        loss = F.nll_loss(out, batch.y, weight=weight.to(device))
        loss.backward()
        opt.step()
        hist.append(float(loss))
        if i == 1 or i % 10 == 0 or i == steps:
            print("  [overfit] step %3d/%d  loss=%.6f" % (i, steps, float(loss)))
    return hist


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train-limit", type=int, default=30)
    ap.add_argument("--train-offset", type=int, default=0)
    ap.add_argument("--val-limit", type=int, default=10)
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--lr", type=float, default=0.01)
    ap.add_argument("--weight-mode", default="linear",
                    choices=["linear", "logarithmic", "smoothed_linear", "none"])
    ap.add_argument("--patience", type=int, default=5)
    ap.add_argument("--dropout", type=float, default=0.0)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--overfit-steps", type=int, default=60)
    ap.add_argument("--json", default=os.path.join(BASE, "reports", "train_smoke.json"))
    ap.add_argument("--plot", default=os.path.join(BASE, "reports", "train_smoke_curve.png"))
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device(args.device if args.device == "cpu" or torch.cuda.is_available()
                          else "cpu")
    print("[train] device=%s  torch=%s" % (device, torch.__version__))

    train_graphs, train_files = load_graphs("train", args.train_limit, args.train_offset)
    val_graphs, val_files = load_graphs("val", args.val_limit)
    print("[train] train graphs=%d  val graphs=%d" % (len(train_graphs), len(val_graphs)))

    weight, wstats = class_weights(train_graphs, args.weight_mode)
    print("[train] class weights %s  (mode=%s, pos=%d/%d, neg/pos=%.2f)" % (
        weight.tolist(), args.weight_mode, wstats["n_pos"], wstats["n_tot"],
        wstats["neg_pos_ratio"]))

    model = M.GCN_SiteModel(dropout=args.dropout, head="logsoftmax")
    n_params = model.n_parameters()
    print("[train] model params=%d" % n_params)
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    loader = DataLoader(train_graphs, batch_size=args.batch_size, shuffle=True)
    first_batch = next(iter(loader))
    batch_rep = batch_sanity(first_batch)
    print("[train] first mini-batch example: graphs=%d nodes=%d edges=%d" % (
        batch_rep["num_graphs"], batch_rep["nodes"], batch_rep["edges"]))

    w_before = model.enc[0].lin.weight.detach().clone()

    history = []
    best = {"pr_auc": -1.0}
    bad_epochs = 0
    t0 = time.time()
    for epoch in range(1, args.epochs + 1):
        tr_loss, steps = run_epoch(model, loader, optimizer, weight, device, epoch)
        val = evaluate(model, val_graphs, device)
        print("[epoch %d] train_loss=%.6f  val PR-AUC=%.4f  ROC-AUC=%.4f  "
              "P@0.5=%.3f R@0.5=%.3f" % (epoch, tr_loss, val["pr_auc"], val["roc_auc"],
                                         val["precision@0.5"], val["recall@0.5"]))
        history.append({"epoch": epoch, "train_loss": tr_loss, "val": val,
                        "step_losses": steps})
        if val["pr_auc"] > best["pr_auc"]:
            best = {"epoch": epoch, "pr_auc": val["pr_auc"], "roc_auc": val["roc_auc"],
                    "state": {k: v.detach().clone() for k, v in model.state_dict().items()}}
            bad_epochs = 0
        else:
            bad_epochs += 1
            if bad_epochs >= args.patience:
                print("[early-stop] no PR-AUC gain for %d epochs" % args.patience)
                break

    w_after = model.enc[0].lin.weight.detach()
    delta = float((w_after - w_before).abs().max())
    print("[train] weight delta after training: max|dw|=%.6e" % delta)

    # controlled overfit: the clearest evidence that loss actually decreases
    hist_over = []
    if args.overfit_steps > 0:
        print("[train] controlled overfit on %d graphs for %d steps" % (
            min(8, len(train_graphs)), args.overfit_steps))
        hist_over = overfit_check(model, train_graphs, device, args.overfit_steps,
                                  args.lr, weight, args.seed)
        drop = 1.0 - hist_over[-1] / max(hist_over[0], 1e-9)
        print("[train] overfit loss %.6f -> %.6f  (drop %.1f%%)" % (
            hist_over[0], hist_over[-1], 100 * drop))

    # restore best weights (in memory, nothing written to disk)
    if best.get("state") is not None:
        model.load_state_dict(best["state"])
    final = evaluate(model, val_graphs, device)
    print("[train] best epoch=%d  val PR-AUC=%.4f  ROC-AUC=%.4f" % (
        best.get("epoch", -1), final["pr_auc"], final["roc_auc"]))

    # ---- verdict ----
    losses = [s for h in history for s in h["step_losses"]]
    finite = all(np.isfinite(l) for l in losses)
    decreased = (len(history) > 1 and history[-1]["train_loss"] < history[0]["train_loss"])
    overfit_ok = bool(hist_over) and hist_over[-1] < hist_over[0]
    metrics_ok = (0.0 <= final["pr_auc"] <= 1.0 and 0.0 <= final["roc_auc"] <= 1.0
                  and final["roc_auc"] > 0.0)
    checks = {
        "loss_all_finite": bool(finite),
        "n_steps": len(losses),
        "epoch_loss_decreased": bool(decreased),
        "epoch_losses": [h["train_loss"] for h in history],
        "overfit_loss_decreased": bool(overfit_ok),
        "overfit_first_last": [hist_over[0], hist_over[-1]] if hist_over else None,
        "weights_changed": bool(delta > 0),
        "weight_max_delta": delta,
        "batch_vector_ok": batch_rep["batch_ok"],
        "metrics_ok": bool(metrics_ok),
        "no_oom": True,
        "device": str(device),
        "params": n_params,
    }
    print(json.dumps(checks, indent=1))
    ok = all([finite, overfit_ok, delta > 0, batch_rep["batch_ok"], metrics_ok])

    report = {"args": vars(args), "class_weights": wstats,
              "history": [{k: v for k, v in h.items() if k != "step_losses"}
                          for h in history],
              "final": final, "checks": checks,
              "overfit_curve": hist_over,
              "step_losses": losses,
              "wall_seconds": round(time.time() - t0, 1)}
    os.makedirs(os.path.dirname(args.json), exist_ok=True)
    with open(args.json, "w") as fh:
        json.dump(report, fh, indent=1)
    print("[train] wrote %s" % args.json)

    if args.plot:
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            fig, ax = plt.subplots(1, 2, figsize=(11, 3.6))
            ax[0].plot(range(1, len(losses) + 1), losses, marker="o", ms=3, lw=1)
            ax[0].set_xlabel("step (epoch-major)")
            ax[0].set_ylabel("NLL loss")
            ax[0].set_title("train loss per step")
            ax[0].grid(alpha=0.3)
            if hist_over:
                ax[1].plot(range(1, len(hist_over) + 1), hist_over, lw=1.2, color="tab:red")
                ax[1].set_xlabel("overfit step")
                ax[1].set_ylabel("NLL loss")
                ax[1].set_title("controlled overfit (8 graphs)")
                ax[1].grid(alpha=0.3)
            fig.tight_layout()
            fig.savefig(args.plot, dpi=130)
            print("[train] wrote %s" % args.plot)
        except Exception as e:
            print("[train] plot skipped: %r" % (e,))

    print("[train] RESULT: %s" % ("PASS" if ok else "FAIL"))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
