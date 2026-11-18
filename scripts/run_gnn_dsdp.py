#!/usr/bin/env python
"""run_gnn_dsdp.py -- end-to-end: GNN -> residue box -> DSDP docking -> RMSD.

Steps:
  1. per-residue probabilities (predict.py) -> box (residue_to_box.py logic:
     threshold 0.5, 36 A, weighted centroid, single-linkage 8 A);
  2. call the existing ./DSDP_redocking/DSDP with --box_min/--box_max
     (exhaustiveness 384, search_depth 40, top_n 1) -- must run on a GPU node;
  3. parse the docked pose and the reference (crystal) ligand, both heavy atoms,
     same coordinate frame -> RMSD without superposition; RMSD < 2 A = success.

The reference ligand defaults to the input ligand file (for test complexes and
1a2b the shipped *_ligand.pdbqt IS the crystal pose used to build the 5 A
labels). A separate --rmsd-ref can override it.

Protein handling:
  * 4-char id        -> GNN uses PDBBind <id>_protein.pdb, docking uses the
                        repo test_dataset/PDBBind_time_split_dataset pdbqt;
  * *_protein.pdb    -> docking protein = sibling *_protein.pdbqt or --docking-protein;
  * *_protein.pdbqt  -> converted to a temporary PDB for the GNN (same frame,
                        CA coordinates identical).

Usage (on a GPU node):
  python scripts/run_gnn_dsdp.py --protein 6qqw \
      --ligand .../6qqw_ligand.pdbqt --out runs/6qqw_out.pdbqt \
      --log runs/6qqw_out.log --json runs/6qqw.json
"""
import argparse
import json
import os
import re
import subprocess
import sys
import tempfile
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import predict as P  # noqa: E402
import residue_to_box as R  # noqa: E402
import paths  # noqa: E402  (同目录：统一路径配置)

BASE = paths.GNNSITE_HOME
DSDP_BIN = paths.REDOCK_BIN
REPO_TEST = paths.TEST_PDBBIND_DIR
REPO_DSDP = paths.DSDP_DATASET_DIR

AD4_TO_ELEMENT = {
    "A": "C", "C": "C", "N": "N", "NA": "N", "NS": "N", "OA": "O", "O": "O",
    "SA": "S", "S": "S", "H": "H", "HD": "H", "HS": "H", "F": "F", "CL": "Cl",
    "BR": "Br", "I": "I", "P": "P", "MG": "Mg", "ZN": "Zn", "CA": "Ca",
    "MN": "Mn", "FE": "Fe", "K": "K", "SE": "Se",
}


def pdbqt_to_pdb(src, dst):
    """Minimal PDBQT -> PDB conversion (keep coordinates/naming, add element)."""
    out = []
    for line in open(src):
        if line.startswith(("ATOM", "HETATM")):
            typ = line[77:79].strip().upper()
            el = AD4_TO_ELEMENT.get(typ, typ[:1].upper() or "C")
            new = line[:66].ljust(66)
            new = new[:76].ljust(76) + ("%-2s" % el) + line[78:].rstrip("\n")[0:0]
            out.append(new.rstrip())
        elif line.startswith("TER"):
            out.append("TER")
    out.append("END")
    with open(dst, "w") as fh:
        fh.write("\n".join(out) + "\n")
    return dst


def resolve_paths(protein, ligand, docking_protein=None, gnn_pdb=None):
    """Return (gnn_pdb, docking_pdbqt, ligand)."""
    if os.path.exists(protein):
        p = protein
    elif re.fullmatch(r"[0-9a-zA-Z]{4}", protein):
        with open(os.path.join(BASE, "structure_index.json")) as fh:
            idx = json.load(fh)["index"]
        e = idx.get(protein.lower())
        if e is None:
            raise SystemExit("unknown PDB id %r" % protein)
        gnn_pdb = gnn_pdb or e["protein"]
        for cand in (os.path.join(REPO_TEST, protein.lower(), protein.lower() + "_protein.pdbqt"),
                     os.path.join(REPO_DSDP, protein.lower(), protein.lower() + "_protein.pdbqt")):
            if os.path.exists(cand):
                return gnn_pdb, cand, ligand
        raise SystemExit("no docking pdbqt found for %r (pass --docking-protein)" % protein)
    else:
        raise SystemExit("protein not found: %r" % protein)

    if p.endswith(".pdbqt"):
        docking = p
        if gnn_pdb:
            g = gnn_pdb
        else:
            tmp = tempfile.mkdtemp(prefix="gnn_pdb_")
            base = os.path.basename(p).replace(".pdbqt", "")
            g = pdbqt_to_pdb(p, os.path.join(tmp, base + ".pdb"))
        return g, docking, ligand
    # .pdb
    gnn = gnn_pdb or p
    docking = docking_protein
    if docking is None:
        guess = p.replace("_protein.pdb", "_protein.pdbqt")
        if os.path.exists(guess):
            docking = guess
        else:
            raise SystemExit("cannot guess docking pdbqt for %r, pass --docking-protein" % p)
    return gnn, docking, ligand


def parse_pdbqt_heavy(path, model=0):
    atoms, cur, seen_model = [], 0, False
    for line in open(path):
        if line.startswith("MODEL"):
            cur += 1
            seen_model = True
            continue
        if line.startswith("ENDMDL"):
            continue
        if not line.startswith(("ATOM", "HETATM")):
            continue
        idx = cur - 1 if seen_model else 0
        if idx != model:
            continue
        typ = line[77:79].strip().upper() or line[12:16].strip()[:1].upper()
        if typ.startswith("H"):
            continue
        xyz = (float(line[30:38]), float(line[38:46]), float(line[46:54]))
        atoms.append((typ, xyz))
    return atoms


def rmsd_no_alignment(ref_path, pose_path, model=0):
    ref = parse_pdbqt_heavy(ref_path, model=0)
    pose = parse_pdbqt_heavy(pose_path, model=model)
    rep = {"n_ref_heavy": len(ref), "n_pose_heavy": len(pose)}
    if len(ref) == 0 or len(pose) == 0:
        rep.update({"rmsd": None, "error": "empty atom list"})
        return rep
    if len(ref) != len(pose):
        rep.update({"rmsd": None,
                    "error": "heavy-atom count mismatch (%d vs %d), no 1:1 correspondence"
                             % (len(ref), len(pose))})
        return rep
    a = np.asarray([x[1] for x in ref])
    b = np.asarray([x[1] for x in pose])
    d = np.sqrt(((a - b) ** 2).sum(axis=1))
    rep.update({"rmsd": float(np.sqrt((d ** 2).mean())),
                "max_atom_dev": float(d.max()),
                "elements_match": [x[0] for x in ref] == [x[0] for x in pose]})
    return rep


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--protein", required=True)
    ap.add_argument("--ligand", required=True)
    ap.add_argument("--docking-protein", default=None)
    ap.add_argument("--gnn-pdb", default=None)
    ap.add_argument("--rmsd-ref", default=None)
    ap.add_argument("--out", required=True)
    ap.add_argument("--log", default=None)
    ap.add_argument("--box-size", type=float, default=R.DEFAULT_BOX_SIZE)
    ap.add_argument("--threshold", type=float, default=R.DEFAULT_THRESHOLD)
    ap.add_argument("--topk", type=int, default=0)
    ap.add_argument("--eps", type=float, default=R.DEFAULT_EPS)
    ap.add_argument("--centre", default="weighted", choices=["weighted", "centroid"])
    ap.add_argument("--exhaustiveness", type=int, default=384)
    ap.add_argument("--search-depth", type=int, default=40)
    ap.add_argument("--top-n", type=int, default=1)
    ap.add_argument("--model", default=P.DEFAULT_MODEL)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--json", default=None)
    ap.add_argument("--timeout", type=int, default=1800)
    args = ap.parse_args()

    gnn_pdb, docking_pdbqt, ligand = resolve_paths(
        args.protein, args.ligand, args.docking_protein, args.gnn_pdb)
    os.makedirs(os.path.dirname(os.path.abspath(args.out)) or ".", exist_ok=True)
    log = args.log or (os.path.splitext(args.out)[0] + ".log")
    json_path = args.json or (os.path.splitext(args.out)[0] + ".json")

    print("[run] gnn_pdb=%s" % gnn_pdb)
    print("[run] docking_pdbqt=%s" % docking_pdbqt)
    print("[run] ligand=%s" % ligand)

    net, ck = P.load_model(args.model, args.device)
    t0 = time.time()
    prob, meta = P.predict_one(net, gnn_pdb, args.device)
    box = R.compute_box(prob, meta["ca"], threshold=args.threshold, box_size=args.box_size,
                        centre_rule=args.centre, eps=args.eps, topk=args.topk)
    print("[run] GNN box: min=%s max=%s (selected=%d, main_cluster=%d) in %.1fs" % (
        [round(v, 3) for v in box["box_min"]], [round(v, 3) for v in box["box_max"]],
        box["n_selected"], box["main_cluster_size"], time.time() - t0))

    cmd = [DSDP_BIN,
           "--ligand", ligand,
           "--protein", docking_pdbqt,
           "--box_min"] + ["%.3f" % v for v in box["box_min"]] + \
          ["--box_max"] + ["%.3f" % v for v in box["box_max"]] + \
          ["--exhaustiveness", str(args.exhaustiveness),
           "--search_depth", str(args.search_depth),
           "--top_n", str(args.top_n),
           "--out", args.out, "--log", log]
    print("[run] " + " ".join(cmd))
    t1 = time.time()
    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=args.timeout)
    dock_s = time.time() - t1
    print("[run] docking rc=%d in %.1fs  out_exists=%s" % (
        proc.returncode, dock_s, os.path.exists(args.out)))
    if proc.stdout.strip():
        print("[run] stdout tail: %s" % proc.stdout.strip().splitlines()[-1][:200])
    if proc.returncode != 0:
        print("[run] stderr tail: %s" % proc.stderr.strip()[-400:])

    rep = {"protein_input": args.protein, "gnn_pdb": gnn_pdb, "docking_pdbqt": docking_pdbqt,
           "ligand": ligand, "model": args.model, "model_epoch": ck.get("epoch"),
           "box": {k: box[k] for k in ("box_min", "box_max", "box_size", "centre",
                                       "n_selected", "n_clusters", "main_cluster_size",
                                       "threshold", "centre_rule", "selection")},
           "docking": {"cmd": cmd, "returncode": proc.returncode, "seconds": round(dock_s, 1),
                       "out": args.out, "log": log},
           "timing": {"gnn_seconds": round(t1 - t0, 1), "docking_seconds": round(dock_s, 1),
                      "total_seconds": round(time.time() - t0, 1)}}

    if os.path.exists(args.out) and os.path.getsize(args.out) > 0:
        ref = args.rmsd_ref or ligand
        rep["rmsd"] = rmsd_no_alignment(ref, args.out)
        rep["rmsd"]["reference"] = ref
        rep["rmsd"]["aligned"] = False
        rep["success_rmsd_lt_2A"] = (rep["rmsd"].get("rmsd") is not None
                                     and rep["rmsd"]["rmsd"] < 2.0)
        print("[run] RMSD(no superposition) = %s A  success(<2A)=%s" % (
            "%.3f" % rep["rmsd"]["rmsd"] if rep["rmsd"].get("rmsd") is not None
            else "FAILED(%s)" % rep["rmsd"].get("error"),
            rep["success_rmsd_lt_2A"]))
    else:
        rep["rmsd"] = {"error": "docking output missing or empty"}
        rep["success_rmsd_lt_2A"] = False

    with open(json_path, "w") as fh:
        json.dump(rep, fh, indent=1)
    print("[run] wrote %s" % json_path)
    return 0 if rep["success_rmsd_lt_2A"] or proc.returncode == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
