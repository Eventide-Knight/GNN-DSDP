#!/usr/bin/env python
"""graph.py -- residue graph (CA distance <= 10 A, weight 1/d^2) -> PyG Data.

Node  = residue
Edge  = two residues whose CA atoms are <= 10.0 A apart (both directions are
        emitted so the graph is symmetric; GCNConv adds self loops itself)
Weight = 1 / d^2  (d = CA-CA euclidean distance, stored in `edge_attr`; the
        1-D view is also exposed in `edge_weight` for GCNConv)

Data fields:
  x            (N, 46) float32   node features from features.py
  edge_index   (2, 2E) int64     directed edges (i->j and j->i)
  edge_attr    (2E, 1) float32   1/d^2
  edge_weight  (2E,)  float32    same, 1-D convenience view
  y            (N,) int64        residue label (1 = ligand within 5 A)
  ca           (N, 3) float32    CA coordinates
  pdb_id, split, num_nodes

Usage:
  python graph.py --id 6qqw --split test --self-check
  python graph.py --protein /path/x_protein.pdb --labels /path/x.npz
  python graph.py --batch --split train --limit 3000 --workers 16
"""
import argparse
import glob
import json
import os
import sys
import time
import traceback

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import features as F  # noqa: E402

from torch_geometric.data import Data  # noqa: E402

BASE = "/share/home/huangyy/DSDP/gnn_site"
INDEX_JSON = os.path.join(BASE, "structure_index.json")
DEFAULT_CUTOFF = 10.0


# --------------------------------------------------------------------------- #
# core graph construction
# --------------------------------------------------------------------------- #
def ca_edges(ca, cutoff=DEFAULT_CUTOFF):
    """Return edge_index / edge_attr / distance matrix for a CA point cloud."""
    ca = np.asarray(ca, dtype=np.float64)
    n = len(ca)
    diff = ca[:, None, :] - ca[None, :, :]
    dist = np.sqrt(np.maximum((diff * diff).sum(-1), 0.0))
    mask = (dist <= cutoff)
    np.fill_diagonal(mask, False)
    rows, cols = np.nonzero(mask)
    d = dist[rows, cols]
    edge_index = torch.from_numpy(np.stack([rows, cols], axis=0)).long()
    w = 1.0 / np.maximum(d, 1e-6) ** 2
    edge_attr = torch.from_numpy(w.astype(np.float32)).view(-1, 1)
    return edge_index, edge_attr, dist


def build_graph(x, ca, y=None, cutoff=DEFAULT_CUTOFF, pdb_id="", split=""):
    """Build a PyG Data object for one protein."""
    edge_index, edge_attr, _ = ca_edges(ca, cutoff)
    data = Data(
        x=torch.as_tensor(np.asarray(x), dtype=torch.float32),
        edge_index=edge_index,
        edge_attr=edge_attr,
        ca=torch.as_tensor(np.asarray(ca), dtype=torch.float32),
    )
    data.edge_weight = edge_attr.view(-1)
    if y is not None:
        data.y = torch.as_tensor(np.asarray(y), dtype=torch.int64)
    data.pdb_id = str(pdb_id)
    data.split = str(split)
    data.cutoff = float(cutoff)
    return data


def graph_self_check(data, cutoff=DEFAULT_CUTOFF, verbose=True):
    """Shapes, symmetry, weights == 1/d^2, degree consistency, label stats."""
    ei = data.edge_index.cpu().numpy()
    ea = data.edge_attr.cpu().numpy().reshape(-1)
    x = data.x.cpu().numpy()
    ca = data.ca.cpu().numpy()
    n = data.num_nodes
    d = np.linalg.norm(ca[ei[0]] - ca[ei[1]], axis=1)
    weights_ok = bool(np.allclose(ea, 1.0 / np.maximum(d, 1e-6) ** 2, rtol=1e-4))
    directed = {(int(a), int(b)) for a, b in ei.T}
    sym = all((b, a) in directed for a, b in directed)
    deg = np.bincount(ei[0], minlength=n)
    rep = {
        "num_nodes": int(n),
        "num_edges": int(ei.shape[1]),
        "mean_degree": float(deg.mean()),
        "min_degree": int(deg.min()),
        "max_degree": int(deg.max()),
        "isolated_nodes": int((deg == 0).sum()),
        "edge_dist_min": float(d.min()) if len(d) else 0.0,
        "edge_dist_max": float(d.max()) if len(d) else 0.0,
        "edge_weight_min": float(ea.min()) if len(ea) else 0.0,
        "edge_weight_max": float(ea.max()) if len(ea) else 0.0,
        "weights_are_1_over_d2": weights_ok,
        "symmetric": bool(sym),
        "x_shape": list(x.shape),
        "x_finite": bool(np.isfinite(x).all()),
        "feature_dim": int(x.shape[1]),
        "cutoff": cutoff,
    }
    if hasattr(data, "y") and data.y is not None:
        y = data.y.cpu().numpy()
        rep["positives"] = int(y.sum())
        rep["positive_ratio"] = float(y.mean()) if len(y) else 0.0
        # degree of positive vs negative residues (sanity: sites are buried)
        rep["mean_degree_pos"] = float(deg[y == 1].mean()) if (y == 1).any() else 0.0
        rep["mean_degree_neg"] = float(deg[y == 0].mean()) if (y == 0).any() else 0.0
    if verbose:
        for k, v in rep.items():
            print("  %-24s %s" % (k, v))
    return rep


# --------------------------------------------------------------------------- #
# batch
# --------------------------------------------------------------------------- #
def _load_labels(path):
    with np.load(path) as z:
        return (z["label"].astype(np.int64), z["chain"], z["resseq"], z["icode"])


def process_one(pid, split, index, label_dir, out_dir, cutoff, overwrite=False):
    """Feature + graph one complex; returns (pid, stat|None, error|None)."""
    out_path = os.path.join(out_dir, pid + ".pt")
    if os.path.exists(out_path) and not overwrite:
        with np.load(os.path.join(label_dir, pid + ".npz")) as z:
            return pid, {"cached": True, "n_res": int(len(z["label"]))}, None
    try:
        entry = index[pid]
        x, meta = F.residue_features(entry["protein"], cutoff=cutoff)
        y, chain, resseq, icode = _load_labels(os.path.join(label_dir, pid + ".npz"))
        if not (list(meta["chains"]) == list(chain)
                and list(meta["resseq"]) == list(resseq)
                and [i.strip() for i in meta["icode"]] == [i.strip() for i in icode]):
            raise RuntimeError("features/labels residue misalignment")
        if len(y) != x.shape[0]:
            raise RuntimeError("label rows %d != feature rows %d" % (len(y), x.shape[0]))
        data = build_graph(x, meta["ca"], y=y, cutoff=cutoff, pdb_id=pid, split=split)
        data.n_missing_dssp = int(meta["n_missing_dssp"])
        torch.save(data, out_path)
        return pid, {"n_res": int(x.shape[0]),
                     "n_edges": int(data.num_edges),
                     "pos": int(y.sum()),
                     "missing_dssp": int(meta["n_missing_dssp"])}, None
    except Exception as e:
        return pid, None, "%s\t%s\t%s" % (split, pid, repr(e))


def scan_out_dir(out_dir, ids):
    """Accurate summary over the on-disk .pt files (cached runs included)."""
    n_c = n_res = n_pos = n_edges = n_md = 0
    for pid in ids:
        p = os.path.join(out_dir, pid + ".pt")
        if not os.path.exists(p):
            continue
        d = torch.load(p, map_location="cpu", weights_only=False)
        n_c += 1
        n_res += int(d.num_nodes)
        n_edges += int(d.num_edges)
        n_md += int(getattr(d, "n_missing_dssp", 0))
        if getattr(d, "y", None) is not None:
            n_pos += int(d.y.sum())
    return {
        "n_complexes": n_c,
        "n_residues": n_res,
        "n_positives": n_pos,
        "positive_ratio": (n_pos / n_res) if n_res else 0.0,
        "mean_residues": (n_res / n_c) if n_c else 0.0,
        "mean_edges": (n_edges / n_c) if n_c else 0.0,
        "mean_missing_dssp": (n_md / n_c) if n_c else 0.0,
    }


def run_batch(args):
    with open(args.index) as fh:
        index = json.load(fh)["index"]
    label_dir = os.path.join(BASE, "data", "label", args.split)
    out_dir = args.out_dir or os.path.join(BASE, "data", "graph", args.split)
    os.makedirs(out_dir, exist_ok=True)
    files = sorted(glob.glob(os.path.join(label_dir, "*.npz")))
    ids = [os.path.basename(f)[:-4] for f in files]
    ids = ids[args.offset:]
    if args.limit:
        ids = ids[:args.limit]
    print("[graph] split=%s ids=%d workers=%d out=%s cutoff=%.1f" % (
        args.split, len(ids), args.workers, out_dir, args.cutoff))

    t0 = time.time()
    stats, skipped = {}, []
    tasks = [(pid, args.split, index, label_dir, out_dir, args.cutoff, args.overwrite)
             for pid in ids]
    if args.workers <= 1:
        it = map(lambda t: process_one(*t), tasks)
        for i, (pid, st, err) in enumerate(it, 1):
            if err:
                skipped.append(err)
            else:
                stats[pid] = st
            if i % 50 == 0 or i == len(ids):
                print("[graph] %d/%d %.0fs" % (i, len(ids), time.time() - t0))
    else:
        import multiprocessing as mp
        ctx = mp.get_context("fork")
        with ctx.Pool(args.workers, maxtasksperchild=200) as pool:
            for i, (pid, st, err) in enumerate(
                    pool.imap_unordered(_worker, tasks, chunksize=4), 1):
                if err:
                    skipped.append(err)
                else:
                    stats[pid] = st
                if i % 100 == 0 or i == len(ids):
                    print("[graph] %d/%d %.0fs" % (i, len(ids), time.time() - t0))

    scan = scan_out_dir(out_dir, ids)
    summary = {
        "split": args.split,
        "n_skipped": len(skipped),
        "wall_seconds": round(time.time() - t0, 1),
        "out_dir": out_dir,
        "n_missing_dssp_last_pass": sum(v.get("missing_dssp", 0)
                                        for v in stats.values() if not v.get("cached")),
    }
    summary.update(scan)
    with open(os.path.join(BASE, "graph_stat_%s.json" % args.split), "w") as fh:
        json.dump(summary, fh, indent=1)
    if skipped:
        with open(os.path.join(BASE, "skipped_graph.txt"), "a") as fh:
            fh.write("\n".join(skipped) + "\n")
    print(json.dumps(summary, indent=1))
    return 0 if not skipped else 2


def _worker(task):
    return process_one(*task)


# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--id", default=None)
    ap.add_argument("--split", default="test", choices=["train", "val", "test"])
    ap.add_argument("--protein", default=None)
    ap.add_argument("--labels", default=None, help="label npz for --protein")
    ap.add_argument("--index", default=INDEX_JSON)
    ap.add_argument("--cutoff", type=float, default=DEFAULT_CUTOFF)
    ap.add_argument("--out", default=None, help="write the Data object here (.pt)")
    ap.add_argument("--self-check", action="store_true")
    ap.add_argument("--batch", action="store_true")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--offset", type=int, default=0)
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--out-dir", default=None)
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--json", default=None)
    args = ap.parse_args()

    if args.batch:
        return run_batch(args)

    if args.protein:
        pdb, label_npz = args.protein, args.labels
        pid = os.path.basename(pdb).split("_")[0]
    elif args.id:
        pdb = F.protein_path_from_id(args.id, args.split, args.index)
        label_npz = os.path.join(BASE, "data", "label", args.split, args.id.lower() + ".npz")
        pid = args.id.lower()
    else:
        ap.error("need --id/--protein (or --batch)")
    print("[graph] %s" % pdb)

    t0 = time.time()
    x, meta = F.residue_features(pdb, cutoff=args.cutoff)
    y = None
    if label_npz and os.path.exists(label_npz):
        y, chain, resseq, icode = _load_labels(label_npz)
        aligned = (list(meta["chains"]) == list(chain)
                   and list(meta["resseq"]) == list(resseq)
                   and [i.strip() for i in meta["icode"]] == [i.strip() for i in icode])
        print("[graph] labels=%s aligned=%s pos=%d" % (label_npz, aligned, int(y.sum())))
        if not aligned:
            raise RuntimeError("features/labels residue misalignment")
    data = build_graph(x, meta["ca"], y=y, cutoff=args.cutoff, pdb_id=pid, split=args.split)
    print("[graph] Data(x=%s, edge_index=%s, edge_attr=%s%s) built in %.2fs" % (
        tuple(data.x.shape), tuple(data.edge_index.shape), tuple(data.edge_attr.shape),
        ", y=%s" % (tuple(data.y.shape),) if data.y is not None else "",
        time.time() - t0))
    print(data)
    rep = graph_self_check(data, args.cutoff, verbose=args.self_check)
    if args.out:
        torch.save(data, args.out)
        print("[graph] wrote %s" % args.out)
    if args.json:
        with open(args.json, "w") as fh:
            json.dump(rep, fh, indent=1)
        print("[graph] wrote %s" % args.json)
    ok = (rep["weights_are_1_over_d2"] and rep["symmetric"] and rep["x_finite"]
          and rep["num_edges"] > 0)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
