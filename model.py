#!/usr/bin/env python
"""model.py -- 4-layer GCN for residue-level ligand-site prediction.

Architecture (task book section 3, modelled on
`$NODECODER_DIR/src/NodeCoder/gcn/NodeCoder.py`):

    x [N,46]
      -> GCNConv(46,38) -> ELU          h1
      -> GCNConv(38,28) -> ELU          h2
      -> GCNConv(28,18) -> ELU          h3
      -> GCNConv(18, 8) -> ELU          h4
      -> GCNConv( 8, 2)                 logits
      -> log_softmax(dim=1)             out [N,2]

Edge weights (1/d^2 from graph.py) are passed to GCNConv as `edge_weight`.

Why 8 -> 2 + log_softmax instead of 8 -> 1 + sigmoid
-----------------------------------------------------
1. It is literally what NodeCoder does (`log_softmax` head + NLLLoss), and the
   task book says to copy NodeCoder first and optimise later.
2. NLLLoss accepts a per-class `weight` tensor, which is the natural way to
   fight the ~4% positive imbalance (NodeCoder's `weighted_loss` variants).
3. Two logits keep the background class explicit, so `exp(out)` gives a proper
   (residue, class) probability table and PR-AUC is read off column 1.
4. `--head sigmoid` is still available for 8 -> 1 + BCEWithLogits training.

C5 only does a structural / dimensional self-check: forward pass on real graphs,
per-layer shapes, parameter count, finiteness. Nothing is trained and no
checkpoint is written.

Usage:
  python model.py --self-check
  python model.py --pt $GNNSITE_HOME/data/graph/train/13gs.pt --self-check
  python model.py --head sigmoid --self-check
"""
import argparse
import os
import sys

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from torch_geometric.nn import GCNConv  # noqa: E402
import paths  # noqa: E402  (同目录：统一路径配置)

BASE = paths.GNNSITE_HOME
DEFAULT_PT = [
    os.path.join(BASE, "data", "graph", "train", "13gs.pt"),
    os.path.join(BASE, "reports", "graph_6qqw.pt"),
]
IN_CHANNELS = 46
HIDDEN_CHANNELS = (38, 28, 18, 8)
OUT_CHANNELS = 2


class GCN_SiteModel(nn.Module):
    """Stacked GCN: 4 hidden layers (ELU) + linear GCN head."""

    def __init__(self, in_channels=IN_CHANNELS, hidden_channels=HIDDEN_CHANNELS,
                 out_channels=OUT_CHANNELS, dropout=0.0, head="logsoftmax"):
        super().__init__()
        self.in_channels = in_channels
        self.hidden_channels = tuple(hidden_channels)
        self.out_channels = out_channels
        self.dropout = dropout
        self.head = head
        self.enc = nn.ModuleList()
        prev = in_channels
        for h in self.hidden_channels:
            self.enc.append(GCNConv(prev, h))
            prev = h
        self.out = GCNConv(prev, out_channels)

    def forward(self, x, edge_index, edge_weight=None, return_hidden=False):
        h = x
        hiddens = []
        for layer in self.enc:
            h = layer(h, edge_index, edge_weight)
            h = F.elu(h)
            hiddens.append(h)
        logits = self.out(h, edge_index, edge_weight)
        if self.dropout:
            logits = F.dropout(logits, p=self.dropout, training=self.training)
        if self.head == "logsoftmax":
            out = F.log_softmax(logits, dim=1)
        elif self.head == "sigmoid":
            out = torch.sigmoid(logits[:, :1])
        else:
            out = logits
        if return_hidden:
            return out, hiddens, logits
        return out

    # convenience for metrics
    def positive_probability(self, x, edge_index, edge_weight=None):
        out = self.forward(x, edge_index, edge_weight)
        if self.head == "logsoftmax":
            return out[:, 1].exp()
        return out.view(-1)

    def n_parameters(self):
        return sum(p.numel() for p in self.parameters())


def layer_table(model):
    rows = []
    prev = model.in_channels
    for i, lay in enumerate(model.enc):
        rows.append((i + 1, "GCNConv(%d->%d)+ELU" % (prev, lay.out_channels),
                     sum(p.numel() for p in lay.parameters())))
        prev = lay.out_channels
    out = model.out
    rows.append((len(model.enc) + 1, "GCNConv(%d->%d)" % (prev, out.out_channels),
                 sum(p.numel() for p in out.parameters())))
    return rows


def model_self_check(model, pt_path, verbose=True):
    """Forward a real graph and verify shapes / finiteness / head normalisation."""
    data = torch.load(pt_path, map_location="cpu", weights_only=False)
    x, ei = data.x, data.edge_index
    ew = getattr(data, "edge_weight", None)
    model.eval()
    with torch.no_grad():
        out, hiddens, logits = model(x, ei, ew, return_hidden=True)

    rep = {"pt": os.path.basename(pt_path), "pdb_id": getattr(data, "pdb_id", "?"),
           "split": getattr(data, "split", "?"), "n_nodes": int(x.shape[0]),
           "n_edges": int(ei.shape[1]), "in_shape": list(x.shape),
           "out_shape": list(out.shape), "hidden_shapes": [list(h.shape) for h in hiddens],
           "finite_out": bool(torch.isfinite(out).all()),
           "hidden_finite": [bool(torch.isfinite(h).all()) for h in hiddens],
           "logits_finite": bool(torch.isfinite(logits).all())}
    if model.head == "logsoftmax":
        rowsum = out.exp().sum(dim=1)
        rep["prob_rowsum_min"] = float(rowsum.min())
        rep["prob_rowsum_max"] = float(rowsum.max())
        rep["prob_rowsum_ok"] = bool(torch.allclose(rowsum, torch.ones_like(rowsum), atol=1e-5))
        rep["p_pos_min"] = float(out[:, 1].exp().min())
        rep["p_pos_max"] = float(out[:, 1].exp().max())
        rep["p_pos_mean"] = float(out[:, 1].exp().mean())
    else:
        rep["p_pos_min"] = float(out.min())
        rep["p_pos_max"] = float(out.max())
        rep["p_pos_mean"] = float(out.mean())
        rep["prob_rowsum_ok"] = None

    if verbose:
        print("=" * 78)
        print("graph: %s  (id=%s, split=%s)  nodes=%d edges=%d" % (
            rep["pt"], rep["pdb_id"], rep["split"], rep["n_nodes"], rep["n_edges"]))
        print("  input  x            %s" % (rep["in_shape"],))
        names = ["h1 (GCNConv 46->38 + ELU)", "h2 (GCNConv 38->28 + ELU)",
                 "h3 (GCNConv 28->18 + ELU)", "h4 (GCNConv 18->8  + ELU)"]
        for nm, sh, fin in zip(names, rep["hidden_shapes"], rep["hidden_finite"]):
            print("  %-30s %s finite=%s" % (nm, sh, fin))
        print("  logits (GCNConv 8->2)        %s finite=%s" % (
            list(logits.shape), rep["logits_finite"]))
        print("  output %-22s %s finite=%s" % (
            "log_softmax" if model.head == "logsoftmax" else model.head,
            rep["out_shape"], rep["finite_out"]))
        if model.head == "logsoftmax":
            print("  prob row-sum: [%.6f, %.6f] ok=%s | p(pos) min/max/mean = %.4f / %.4f / %.4f" % (
                rep["prob_rowsum_min"], rep["prob_rowsum_max"], rep["prob_rowsum_ok"],
                rep["p_pos_min"], rep["p_pos_max"], rep["p_pos_mean"]))
        else:
            print("  p(pos) min/max/mean = %.4f / %.4f / %.4f" % (
                rep["p_pos_min"], rep["p_pos_max"], rep["p_pos_mean"]))
    return rep


def build_default_model(head="logsoftmax", dropout=0.0):
    return GCN_SiteModel(head=head, dropout=dropout)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pt", action="append", default=None,
                    help="graph .pt to forward (repeatable); default: 13gs(train) + 6qqw(test)")
    ap.add_argument("--head", default="logsoftmax", choices=["logsoftmax", "sigmoid"])
    ap.add_argument("--dropout", type=float, default=0.0)
    ap.add_argument("--self-check", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    model = build_default_model(head=args.head, dropout=args.dropout)
    n_params = model.n_parameters()

    print("model: GCN_SiteModel(head=%s, dropout=%.2f)" % (args.head, args.dropout))
    print("%-5s %-32s %10s" % ("idx", "layer", "params"))
    for idx, name, n in layer_table(model):
        print("%-5d %-32s %10d" % (idx, name, n))
    print("%-5s %-32s %10d" % ("", "TOTAL", n_params))

    pts = args.pt or [p for p in DEFAULT_PT if os.path.exists(p)]
    if not pts:
        ap.error("no graph .pt found")
    reports = []
    for p in pts:
        reports.append(model_self_check(model, p, verbose=True))

    ok = all(r["finite_out"] and r["logits_finite"]
             and all(r["hidden_finite"]) and r["out_shape"][1] == (2 if args.head == "logsoftmax" else 1)
             and r["hidden_shapes"][-1][1] == 8
             and (r["prob_rowsum_ok"] in (True, None)) for r in reports)
    print("=" * 78)
    print("RESULT: %s  (params=%d, head=%s, graphs=%d)" % (
        "PASS" if ok else "FAIL", n_params, args.head, len(reports)))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
