#!/usr/bin/env python
"""c10_dsdp_coverage.py -- faithful DSDP-box coverage for the C10 e2e table.

The DSDP blind-docking binary never prints its box, so we reproduce it from
source (DSDP_blind_docking/main_SingleRun.cu + Site_Information.cu + CLUSTER.cpp):

  * voxel (i,j,k) sits at  2*voxel - 35 + protein_center   (scale=2 A, half=35 A)
  * protein_center = mean of ALL atoms of the docking pdbqt (Rigid_Protein.cu)
  * the top 200 voxels by probability are clustered with a 27-neighbourhood
    flood fill; every cluster gets box = cluster_centroid +- 15 A (box_length=30)
  * the binary docks each cluster, so a complex is "covered" if ANY cluster box
    contains the ligand.

This script recomputes dsdp.coverage / dsdp.failure_class for the rows already
produced by c10_e2e.py (no docking involved) and refreshes the summaries.

Usage:
  python c10_dsdp_coverage.py --json runs/c10_e2e.json --csv runs/c10_e2e.csv
"""
import argparse
import csv
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import build_dataset as B  # noqa: E402
import paths  # noqa: E402  (同目录：统一路径配置)

BASE = paths.GNNSITE_HOME
REPO_TEST = paths.TEST_PDBBIND_DIR
NPY_DIR = os.path.join(BASE, "c2_site", "npy")
TOP_POINTS = 200
SCALE = 2.0
HALF = 35.0
BOX_LENGTH = 30.0


def protein_center(pdbqt):
    xyz = []
    for line in open(pdbqt):
        if line.startswith(("ATOM", "HETATM")):
            xyz.append((float(line[30:38]), float(line[38:46]), float(line[46:54])))
    return np.asarray(xyz, dtype=np.float64).mean(axis=0) if xyz else np.zeros(3)


def clusters_of_top_voxels(dens, k=TOP_POINTS):
    flat = dens.reshape(-1)
    idx = np.argsort(-flat)[:k]
    coords = np.stack(np.unravel_index(idx, dens.shape), axis=1)  # (k,3) i,j,k
    pts = {tuple(c) for c in coords.tolist()}
    seen, comps = set(), []
    for p in pts:
        if p in seen:
            continue
        stack, comp = [p], []
        seen.add(p)
        while stack:
            q = stack.pop()
            comp.append(q)
            for dx in (-1, 0, 1):
                for dy in (-1, 0, 1):
                    for dz in (-1, 0, 1):
                        if dx == dy == dz == 0:
                            continue
                        r = (q[0] + dx, q[1] + dy, q[2] + dz)
                        if r in pts and r not in seen:
                            seen.add(r)
                            stack.append(r)
        comps.append(np.asarray(comp, dtype=np.float64))
    return comps


def dsdp_box_coverage(pid, lig):
    npy = os.path.join(NPY_DIR, "%s.npy" % pid)
    prot = os.path.join(REPO_TEST, pid, pid + "_protein.pdbqt")
    if not (os.path.exists(npy) and os.path.exists(prot)):
        return None, None, 0
    dens = np.load(npy).reshape(36, 36, 36)
    cen = protein_center(prot)
    comps = clusters_of_top_voxels(dens)
    best_cov, best_box = -1.0, None
    for comp in comps:
        real = SCALE * comp - HALF + cen            # (n,3)
        center = real.mean(axis=0)
        lo = center - BOX_LENGTH / 2.0
        hi = center + BOX_LENGTH / 2.0
        cov = float(((lig >= lo) & (lig <= hi)).all(axis=1).mean())
        if cov > best_cov:
            best_cov, best_box = cov, (lo.tolist(), hi.tolist())
    return round(best_cov, 4), best_box, len(comps)


def classify(cov, rmsd, ok):
    if ok:
        return "success"
    if rmsd is None:
        return "error"
    if cov is not None and cov < 0.5:
        return "box_miss"
    return "sampled_but_fail"


def summarize(rows, key):
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", default=os.path.join(BASE, "runs", "c10_e2e.json"))
    ap.add_argument("--csv", default=os.path.join(BASE, "runs", "c10_e2e.csv"))
    args = ap.parse_args()

    blob = json.load(open(args.json))
    rows = blob["rows"]
    for r in rows:
        pid = r["pdb_id"]
        lig = B.parse_ligand_pdbqt(os.path.join(REPO_TEST, pid, pid + "_ligand.pdbqt"))
        cov, box, ncl = dsdp_box_coverage(pid, lig)
        r["dsdp"]["coverage"] = cov
        r["dsdp"]["box"] = box
        r["dsdp"]["n_clusters"] = ncl
        r["dsdp"]["coverage_method"] = "reproduced from DSDP source (top200 voxels, 27-neigh clusters, 30A boxes)"
        r["dsdp"]["failure_class"] = classify(cov, r["dsdp"]["rmsd"], r["dsdp"]["success"])
        print("%-5s dsdp cov=%-6s clusters=%-2d rmsd=%-6s class=%s" % (
            pid, cov, ncl,
            ("%.2f" % r["dsdp"]["rmsd"]) if r["dsdp"]["rmsd"] is not None else "FAIL",
            r["dsdp"]["failure_class"]))

    blob["summary"]["gnn"] = summarize(rows, "gnn")
    blob["summary"]["dsdp"] = summarize(rows, "dsdp")
    blob["summary"]["dsdp"]["coverage_method"] = (
        "reproduced DSDP boxes: top-200 voxels, 27-neighbour clusters, 30 A cube per cluster")
    with open(args.json, "w") as fh:
        json.dump(blob, fh, indent=1)
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
                        r["dsdp"]["coverage"], r["dsdp"]["failure_class"]])
    print(json.dumps({k: {kk: vv for kk, vv in v.items() if kk != "success_rmsds"}
                      for k, v in (("gnn", blob["summary"]["gnn"]),
                                   ("dsdp", blob["summary"]["dsdp"]))}, indent=1))
    print("[cov] updated %s and %s" % (args.json, args.csv))


if __name__ == "__main__":
    main()
