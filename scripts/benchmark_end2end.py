#!/usr/bin/env python
"""benchmark_end2end.py -- end-to-end success rate: GNN box vs original DSDP blind docking.

For a stratified sample of test-363 complexes (strata by GNN top-20 hit rate)
two pipelines are run with identical DSDP settings (exhaustiveness 384,
search_depth 40, top_n 1):

  A) GNN: best_model -> residue probabilities -> residue_to_box (val-chosen
     rule: threshold 0.5, 36 A, weighted centroid, single-linkage 8 A) ->
     ./DSDP_redocking/DSDP --box_min/--box_max
  B) DSDP baseline: the original pipeline, i.e. the 36^3 probability grid from
     C2 (binding_site_model.pth / DrugSiteMind) fed to
     ./DSDP_blind_docking/DSDP --site_npy, which does DSDP's own clustering and
     box placement.

Success = heavy-atom RMSD (no superposition, same coordinate frame) < 2 A.
Failure classes: box_miss (ligand coverage < 0.5), sampled_but_fail
(coverage >= 0.5 but RMSD >= 2), error.

Usage (GPU node):
  python scripts/benchmark_end2end.py --n 40 --json runs/e2e_benchmark.json
  python scripts/benchmark_end2end.py --ids 6qqw,1a2b --json runs/e2e_smoke.json
"""
import argparse
import csv
import glob
import json
import os
import subprocess
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from torch_geometric.loader import DataLoader  # noqa: E402

import model as M  # noqa: E402
import predict as P  # noqa: E402
import residue_to_box as RB  # noqa: E402
import build_dataset as B  # noqa: E402
import run_gnn_dsdp as R  # noqa: E402
import paths  # noqa: E402  (同目录：统一路径配置)

BASE = paths.GNNSITE_HOME
REPO_TEST = paths.TEST_PDBBIND_DIR
BLIND_BIN = paths.BLIND_BIN
REDOCK_BIN = paths.REDOCK_BIN
NPY_DIR = os.path.join(BASE, "c2_site", "npy")
EXH = 384
DEPTH = 40
TOP_N = 1


def load_test_graphs(device):
    files = sorted(glob.glob(os.path.join(BASE, "data", "graph", "test", "*.pt")))
    graphs = [torch.load(f, map_location="cpu", weights_only=False) for f in files]
    ck = torch.load(os.path.join(BASE, "best_model.pt"), map_location="cpu", weights_only=False)
    net = M.GCN_SiteModel(hidden_channels=tuple(ck.get("hidden", M.HIDDEN_CHANNELS)))
    net.load_state_dict(ck["model_state"])
    net.to(device).eval()
    probs = []
    loader = DataLoader(graphs, batch_size=64, shuffle=False)
    with torch.no_grad():
        for b in loader:
            b = b.to(device)
            probs.append(net(b.x, b.edge_index, b.edge_weight)[:, 1].exp().cpu().numpy())
    p = np.concatenate(probs)
    per, off = [], 0
    for g in graphs:
        n = int(g.num_nodes)
        per.append(p[off:off + n])
        off += n
    return graphs, per, ck


def top20_hits(g, p):
    y = g.y.numpy()
    top = np.argsort(-p)[:20]
    return int(y[top].sum()), int(y.sum())


def stratified_sample(graphs, probs, n_total):
    rows = []
    for g, p in zip(graphs, probs):
        h, npos = top20_hits(g, p)
        rows.append((str(g.pdb_id).lower(), h, npos, int(g.num_nodes)))
    high = sorted([r for r in rows if r[1] >= 12], key=lambda r: (-r[1], r[0]))
    mid = sorted([r for r in rows if 5 <= r[1] <= 11], key=lambda r: (-r[1], r[0]))
    low = sorted([r for r in rows if r[1] <= 4], key=lambda r: (r[1], r[0]))
    print("[sample] strata sizes: high=%d mid=%d low=%d" % (len(high), len(mid), len(low)))
    n_high = n_mid = n_low = n_total // 3
    rest = n_total - (n_high + n_mid + n_low)
    n_high += rest

    def take(lst, k):
        if k >= len(lst):
            return lst
        idx = np.linspace(0, len(lst) - 1, k).round().astype(int)
        return [lst[i] for i in sorted(set(idx.tolist()))]

    sel = take(high, n_high) + take(mid, n_mid) + take(low, n_low)
    print("[sample] selected %d: high=%d mid=%d low=%d" % (
        len(sel), len(take(high, n_high)), len(take(mid, n_mid)), len(take(low, n_low))))
    return sel


def ligand_for(pid):
    p = os.path.join(REPO_TEST, pid, pid + "_ligand.pdbqt")
    return p if os.path.exists(p) else None


def protein_pdbqt(pid):
    p = os.path.join(REPO_TEST, pid, pid + "_protein.pdbqt")
    return p if os.path.exists(p) else None


def coverage(lig, box_min, box_max):
    lo = np.asarray(box_min); hi = np.asarray(box_max)
    return float(((lig >= lo) & (lig <= hi)).all(axis=1).mean())


def run_docking(binary, args, timeout):
    t0 = time.time()
    try:
        proc = subprocess.run([binary] + args, capture_output=True, text=True, timeout=timeout)
        return proc.returncode, proc.stderr[-300:], time.time() - t0
    except subprocess.TimeoutExpired:
        return -999, "timeout after %ss" % timeout, time.time() - t0


def classify(cov, rmsd, ok):
    if ok:
        return "success"
    if rmsd is None:
        return "error"
    if cov is not None and cov < 0.5:
        return "box_miss"
    return "sampled_but_fail"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=40)
    ap.add_argument("--ids", default=None, help="comma separated override")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--json", default=os.path.join(BASE, "runs", "e2e_benchmark.json"))
    ap.add_argument("--csv", default=os.path.join(BASE, "runs", "e2e_benchmark.csv"))
    ap.add_argument("--timeout", type=int, default=180)
    ap.add_argument("--outdir", default=os.path.join(BASE, "runs", "c10"))
    args = ap.parse_args()
    os.makedirs(args.outdir, exist_ok=True)
    os.makedirs(os.path.dirname(args.json), exist_ok=True)

    device = torch.device(args.device if (args.device == "cpu" or torch.cuda.is_available())
                          else "cpu")
    graphs, probs, ck = load_test_graphs(device)
    gmap = {str(g.pdb_id).lower(): (g, p) for g, p in zip(graphs, probs)}

    if args.ids:
        sel = [(i, -1, -1, 0) for i in args.ids.split(",")]
    else:
        sel = stratified_sample(graphs, probs, args.n)
    print("[e2e] running %d complexes (device=%s)" % (len(sel), device))

    rows = []
    t_start = time.time()
    for k, (pid, hits, npos, nres) in enumerate(sel, 1):
        if pid not in gmap:
            print("[e2e] skip %s (not in test graphs)" % pid)
            continue
        g, p = gmap[pid]
        lig_path = ligand_for(pid)
        prot = protein_pdbqt(pid)
        if lig_path is None or prot is None:
            print("[e2e] skip %s (missing repo pdbqt)" % pid)
            continue
        lig = B.parse_ligand_pdbqt(lig_path)
        h, np_ = top20_hits(g, p)

        # ---- A) GNN box ----
        box = RB.compute_box(p, g.ca.numpy(), threshold=RB.DEFAULT_THRESHOLD,
                             box_size=RB.DEFAULT_BOX_SIZE, centre_rule="weighted",
                             eps=RB.DEFAULT_EPS)
        out_a = os.path.join(args.outdir, "%s_gnn_out.pdbqt" % pid)
        log_a = os.path.join(args.outdir, "%s_gnn_out.log" % pid)
        rc_a, err_a, t_a = run_docking(REDOCK_BIN, [
            "--ligand", lig_path, "--protein", prot,
            "--box_min"] + ["%.3f" % v for v in box["box_min"]] + \
            ["--box_max"] + ["%.3f" % v for v in box["box_max"]] + \
            ["--exhaustiveness", str(EXH), "--search_depth", str(DEPTH),
             "--top_n", str(TOP_N), "--out", out_a, "--log", log_a], args.timeout)
        rmsd_a = None
        if rc_a == 0 and os.path.exists(out_a) and os.path.getsize(out_a) > 0:
            rmsd_a = R.rmsd_no_alignment(lig_path, out_a).get("rmsd")
        cov_a = coverage(lig, box["box_min"], box["box_max"])

        # ---- B) DSDP baseline (its own site npy) ----
        npy = os.path.join(NPY_DIR, "%s.npy" % pid)
        rmsd_b, cov_b, rc_b, err_b, t_b = None, None, -1, "no npy", 0.0
        if os.path.exists(npy):
            out_b = os.path.join(args.outdir, "%s_dsdp_out.pdbqt" % pid)
            log_b = os.path.join(args.outdir, "%s_dsdp_out.log" % pid)
            rc_b, err_b, t_b = run_docking(BLIND_BIN, [
                "--ligand", lig_path, "--protein", prot, "--site_npy", npy,
                "--out", out_b, "--log", log_b,
                "--exhaustiveness", str(EXH), "--search_depth", str(DEPTH),
                "--top_n", str(TOP_N)], args.timeout)
            if rc_b == 0 and os.path.exists(out_b) and os.path.getsize(out_b) > 0:
                rmsd_b = R.rmsd_no_alignment(lig_path, out_b).get("rmsd")
            dens = np.load(npy).reshape(36, 36, 36)
            ca = g.ca.numpy()
            cen = ca.mean(axis=0)
            idx = np.rint((ca - cen + 35.0) / 2.0).astype(int).clip(0, 35)
            # DSDP's own box: bounding box of high-probability voxels (as the
            # binary does) -- approximate by the voxel cloud above 0.5
            ii, jj, kk = np.where(dens > 0.5)
            if len(ii) > 0:
                lo = cen + (np.array([ii.min(), jj.min(), kk.min()]) * 2.0 - 35)
                hi = cen + (np.array([ii.max(), jj.max(), kk.max()]) * 2.0 - 35)
                cov_b = coverage(lig, lo, hi)
        ok_a = rmsd_a is not None and rmsd_a < 2.0
        ok_b = rmsd_b is not None and rmsd_b < 2.0
        row = {"pdb_id": pid, "stratum": "high" if h >= 12 else ("mid" if h >= 5 else "low"),
               "gnn_top20_hits": h, "n_true_pos": np_, "n_residues": int(g.num_nodes),
               "gnn": {"rmsd": rmsd_a, "success": bool(ok_a), "coverage": cov_a,
                       "rc": rc_a, "seconds": round(t_a, 2),
                       "failure_class": classify(cov_a, rmsd_a, ok_a),
                       "n_selected": box["n_selected"], "box": [box["box_min"], box["box_max"]]},
               "dsdp": {"rmsd": rmsd_b, "success": bool(ok_b), "coverage": cov_b,
                        "rc": rc_b, "seconds": round(t_b, 2),
                        "failure_class": classify(cov_b, rmsd_b, ok_b),
                        "err": None if rc_b == 0 else err_b}}
        rows.append(row)
        print("[e2e] %2d/%d %-5s hits=%2d | GNN rmsd=%s cov=%.2f (%.1fs) | DSDP rmsd=%s cov=%s (%.1fs)" % (
            k, len(sel), pid, h,
            ("%.2f" % rmsd_a) if rmsd_a is not None else "FAIL",
            cov_a, t_a,
            ("%.2f" % rmsd_b) if rmsd_b is not None else "FAIL",
            ("%.2f" % cov_b) if cov_b is not None else "n/a", t_b))

    def summ(key):
        d = [r[key] for r in rows]
        rms = [x["rmsd"] for x in d if x["rmsd"] is not None]
        succ = sum(1 for x in d if x["success"])
        covs = [x["coverage"] for x in d if x["coverage"] is not None]
        classes = {}
        for x in d:
            classes[x["failure_class"]] = classes.get(x["failure_class"], 0) + 1
        return {"n": len(d), "success": succ, "success_rate": succ / max(len(d), 1),
                "rmsd_median": float(np.median(rms)) if rms else None,
                "rmsd_mean": float(np.mean(rms)) if rms else None,
                "success_rmsds": sorted([round(x, 2) for x in rms]),
                "mean_coverage": float(np.mean(covs)) if covs else None,
                "failure_classes": classes}

    summary = {"n_requested": len(sel), "n_done": len(rows),
               "gnn": summ("gnn"), "dsdp": summ("dsdp"),
               "wall_seconds": round(time.time() - t_start, 1),
               "settings": {"exhaustiveness": EXH, "search_depth": DEPTH,
                            "top_n": TOP_N, "box_rule": "thr0.5/side36/weighted/eps8"}}
    print(json.dumps(summary, indent=1))
    with open(args.json, "w") as fh:
        json.dump({"summary": summary, "rows": rows}, fh, indent=1)
    with open(args.csv, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["pdb_id", "stratum", "gnn_top20_hits", "n_true_pos", "n_residues",
                    "gnn_rmsd", "gnn_success", "gnn_coverage", "gnn_failure_class",
                    "dsdp_rmsd", "dsdp_success", "dsdp_coverage", "dsdp_failure_class"])
        for r in rows:
            w.writerow([r["pdb_id"], r["stratum"], r["gnn_top20_hits"], r["n_true_pos"],
                        r["n_residues"], r["gnn"]["rmsd"], r["gnn"]["success"],
                        round(r["gnn"]["coverage"], 4), r["gnn"]["failure_class"],
                        r["dsdp"]["rmsd"], r["dsdp"]["success"],
                        round(r["dsdp"]["coverage"], 4) if r["dsdp"]["coverage"] is not None else "",
                        r["dsdp"]["failure_class"]])
    print("[e2e] wrote %s and %s" % (args.json, args.csv))
    return 0


if __name__ == "__main__":
    sys.exit(main())
