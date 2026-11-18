#!/usr/bin/env python
"""features.py -- 46-dim residue-level node features (task book section 3).

Layout (46 dims, pLDDT dropped because crystal structures have no pLDDT;
pass --with-plddt to append a constant 1.0 as dim 46 -> 47 dims):

  0-19   amino-acid one-hot          (order "ACDEFGHIKLMNPQRSTVWY")
  20     iPlus   = 1 if the residue has a sequence neighbour before it
  21     iMinus  = 1 if the residue has a sequence neighbour after it
  22-28  DSSP secondary structure one-hot (H B E G I T S)
  29-36  DSSP backbone H-bond features (8):
           NH-->O_1_relidx, NH-->O_1_energy, O-->HN_1_relidx, O-->HN_1_energy,
           NH-->O_2_relidx, NH-->O_2_energy, O-->HN_2_relidx, O-->HN_2_energy
  37-40  dihedrals phi, psi, tau, theta   (radians; undefined -> 0.0)
  41     BBSASA  (backbone N+CA+C+O SASA in A^2)
  42     SCSASA  (side-chain SASA in A^2)
  43     distance of CA to the protein CA centroid (A)
  44     cosine distance of the CA position vector to the centroid vector
  45     node degree in the 10 A CA graph
  46     pLDDT (only with --with-plddt; constant 1.0 for crystal structures)

Input is always the PDBBind `<id>_protein.pdb` (standard PDB).
DSSP goes through `mkdssp` (4.x legacy output), SASA through freesasa.
Residue ordering / filtering is identical to build_dataset.py so features and
labels line up row by row.

Usage:
  python features.py --id 6qqw --split test
  python features.py --protein /path/to/x_protein.pdb --out-npz /tmp/x.npz
  python features.py --id 6qqw --split test --self-check
"""
import argparse
import json
import math
import os
import re
import subprocess
import sys
import tempfile

import numpy as np
import paths  # noqa: E402  (同目录：统一路径配置)

MKDSSP = os.environ.get("MKDSSP", paths.MKDSSP)
INDEX_JSON = paths.INDEX_JSON

_WARNED = set()


def _warn_once(tag, message):
    """只打印一次的警告；GNNSITE_STRICT=1 时直接抛错（防止静默产出坏特征）。"""
    if os.environ.get("GNNSITE_STRICT"):
        raise RuntimeError(message)
    if tag in _WARNED:
        return
    _WARNED.add(tag)
    sys.stderr.write(message + "\n")
    sys.stderr.flush()

AA_ORDER = "ACDEFGHIKLMNPQRSTVWY"
AA3 = {
    "ALA": "A", "ARG": "R", "ASN": "N", "ASP": "D", "CYS": "C", "GLN": "Q",
    "GLU": "E", "GLY": "G", "HIS": "H", "ILE": "I", "LEU": "L", "LYS": "K",
    "MET": "M", "PHE": "F", "PRO": "P", "SER": "S", "THR": "T", "TRP": "W",
    "TYR": "Y", "VAL": "V",
}
AA3_ALT = {"MSE": "MET", "HSD": "HIS", "HSE": "HIS", "HSP": "HIS",
           "CYX": "CYS", "CYM": "CYS", "HID": "HIS", "HIE": "HIS"}
SS_ORDER = "HBEGITS"          # 7 classes from the task book
BACKBONE_ATOMS = ("N", "CA", "C", "O")
HYDROGEN_NAMES = {"H", "D"}

FEATURE_NAMES = (
    ["aa_%s" % c for c in AA_ORDER]
    + ["iPlus", "iMinus"]
    + ["ss_%s" % c for c in SS_ORDER]
    + ["hbond_%d" % i for i in range(8)]
    + ["phi", "psi", "tau", "theta", "bb_sasa", "sc_sasa",
       "centroid_dist", "centroid_cos", "degree"]
)
N_FEATURES = len(FEATURE_NAMES)          # 46
N_FEATURES_PLDDT = N_FEATURES + 1        # 47
ONEHOT_SLICES = [(0, 20), (22, 29)]      # aa one-hot, ss one-hot


# --------------------------------------------------------------------------- #
# protein parsing (same residue filter/order as build_dataset.py)
# --------------------------------------------------------------------------- #
def _element(line):
    el = line[76:78].strip()
    if el:
        return el.upper()
    for ch in line[12:16].strip():
        if ch.isalpha():
            return ch.upper()
    return "C"


def parse_protein(path):
    """Return residues in file order with their heavy-atom coordinates."""
    res = []            # list of dicts
    key2idx = {}
    in_model = True
    with open(path) as fh:
        for line in fh:
            rec = line[:6]
            if rec.startswith("ENDMDL"):
                in_model = False
                continue
            if rec.startswith("MODEL"):
                in_model = True
                continue
            if not line.startswith("ATOM") or not in_model:
                continue
            if line[16] not in (" ", "A"):
                continue
            rn = line[17:20].strip()
            rn = AA3_ALT.get(rn, rn)
            if rn not in AA3:
                continue
            el = _element(line)
            if el in HYDROGEN_NAMES:
                continue
            key = (line[21], int(line[22:26]), line[26])
            i = key2idx.get(key)
            if i is None:
                i = len(res)
                key2idx[key] = i
                res.append({"chain": line[21], "resseq": int(line[22:26]),
                            "icode": line[26], "resname": rn,
                            "atoms": {}, "heavy": []})
            name = line[12:16].strip()
            xyz = (float(line[30:38]), float(line[38:46]), float(line[46:54]))
            res[i]["heavy"].append(xyz)
            if name not in res[i]["atoms"]:
                res[i]["atoms"][name] = np.asarray(xyz, dtype=np.float64)
    if not res:
        raise ValueError("no residues parsed from %s" % path)
    return res


def residues_arrays(res):
    """Residue key arrays + CA coordinates.

    A handful of PDBBind structures have residues with a missing CA atom; for
    those the heavy-atom centroid is used instead (same fallback as
    build_dataset.py, so features stay row-aligned with the labels).
    """
    ca = []
    n_missing_ca = 0
    for r in res:
        c = r["atoms"].get("CA")
        if c is None:
            n_missing_ca += 1
            c = np.asarray(r["heavy"], dtype=np.float64).mean(axis=0)
        ca.append(c)
    return {
        "chain": np.array([r["chain"] for r in res]),
        "resseq": np.array([r["resseq"] for r in res], dtype=np.int32),
        "icode": np.array([r["icode"] for r in res]),
        "resname": np.array([r["resname"] for r in res]),
        "ca": np.asarray(ca, dtype=np.float64),
        "n_missing_ca": n_missing_ca,
    }


# --------------------------------------------------------------------------- #
# DSSP (mkdssp 4.x legacy output; parsed by fixed columns, see env_check.md)
# --------------------------------------------------------------------------- #
_HDR_AA = 13      # column of the AA letter (0-based)
_HDR_SS = 16      # column of the secondary-structure letter


def run_mkdssp(pdb_path, dssp_path=None):
    """Run mkdssp; returns the dssp file path (caller may delete it)."""
    tmpdir = None
    if dssp_path is None:
        tmpdir = tempfile.mkdtemp(prefix="dssp_")
        dssp_path = os.path.join(tmpdir, "out.dssp")
    proc = subprocess.run([MKDSSP, "--quiet", "--output-format", "dssp",
                           pdb_path, dssp_path],
                          stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                          universal_newlines=True)
    if proc.returncode != 0 or not os.path.exists(dssp_path):
        raise RuntimeError("mkdssp failed rc=%d: %s" % (
            proc.returncode, proc.stderr.strip().splitlines()[:3]))
    return dssp_path


def parse_dssp(dssp_path):
    """key=(chain,resseq,icode) -> {'ss': char, 'hb': [8 floats]}.

    DSSP legacy layout (verified against the header, the raw columns and a
    Biopython cross-check of all H-bond values):
        cols  1-5   DSSP sequential residue index (global, 1..N)
        cols  6-10  author residue number (the PDB numbering)
        col   11    insertion code
        col   12    chain id
        col   14    amino acid
        col   17    secondary structure
    DSSP 4 renumbers sequentially in cols 1-5, so the author number in cols
    6-10 is what has to be used as the residue key. Values may be glued
    ("360.0-158.4"), so fixed columns are used and the four H-bond pairs are
    read by regex (they are the only comma-separated fields on the line).
    """
    out = {}
    hb_re = re.compile(r"(-?\d+),\s*(-?\d+\.\d+)")
    header_checked = False
    with open(dssp_path) as fh:
        for line in fh:
            if not header_checked and line.lstrip().startswith("#"):
                header_checked = True
                if line.find("AA") != _HDR_AA or line.find("STRUCTURE") != _HDR_SS:
                    raise RuntimeError(
                        "unexpected DSSP header layout: AA at %d, STRUCTURE at %d"
                        % (line.find("AA"), line.find("STRUCTURE")))
                continue
            if len(line) < 120 or line[_HDR_AA] not in AA_ORDER:
                continue
            if line[_HDR_SS] not in SS_ORDER + "P ":
                continue
            if not line[0:5].strip().isdigit() or not line[5:10].strip().isdigit():
                continue
            pairs = hb_re.findall(line)
            if len(pairs) != 4:
                continue
            key = (line[11], int(line[5:10]), line[10])
            hb = []
            for a, b in pairs:
                hb.extend((float(a), float(b)))
            out[key] = {"ss": line[_HDR_SS], "hb": hb}
    return out


# --------------------------------------------------------------------------- #
# geometry
# --------------------------------------------------------------------------- #
def _dihedral(p0, p1, p2, p3):
    b0 = p1 - p0
    b1 = p2 - p1
    b2 = p3 - p2
    n1 = np.cross(b0, b1)
    n2 = np.cross(b1, b2)
    d1 = np.linalg.norm(n1)
    d2 = np.linalg.norm(n2)
    if d1 < 1e-8 or d2 < 1e-8:
        return None
    m = np.cross(n1 / d1, n2 / d2)
    x = np.dot(n1 / d1, n2 / d2)
    y = np.dot(m, b1 / (np.linalg.norm(b1) + 1e-12))
    return float(math.atan2(y, x))


def _angle(p0, p1, p2):
    v1 = p0 - p1
    v2 = p2 - p1
    n1 = np.linalg.norm(v1)
    n2 = np.linalg.norm(v2)
    if n1 < 1e-8 or n2 < 1e-8:
        return None
    c = float(np.clip(np.dot(v1, v2) / (n1 * n2), -1.0, 1.0))
    return float(math.acos(c))


def dihedral_features(res):
    """phi, psi, tau, theta in radians (None/undefined -> 0.0)."""
    n = len(res)
    out = np.zeros((n, 4), dtype=np.float32)

    def get(r, name):
        return r["atoms"].get(name)

    for i, r in enumerate(res):
        N, CA, C = get(r, "N"), get(r, "CA"), get(r, "C")
        # tau = N-CA-C
        if N is not None and CA is not None and C is not None:
            out[i, 2] = _angle(N, CA, C) or 0.0
        # theta = CA-C-N(next)
        j = i + 1
        if j < n and res[j]["chain"] == r["chain"]:
            Nn = get(res[j], "N")
            if CA is not None and C is not None and Nn is not None:
                out[i, 3] = _angle(CA, C, Nn) or 0.0
        # psi = N-CA-C-N(next)
        if j < n and res[j]["chain"] == r["chain"]:
            Nn = get(res[j], "N")
            if N is not None and CA is not None and C is not None and Nn is not None:
                v = _dihedral(N, CA, C, Nn)
                out[i, 1] = v or 0.0
        # phi = C(prev)-N-CA-C
        k = i - 1
        if k >= 0 and res[k]["chain"] == r["chain"]:
            Cp = get(res[k], "C")
            if Cp is not None and N is not None and CA is not None and C is not None:
                v = _dihedral(Cp, N, CA, C)
                out[i, 0] = v or 0.0
    return out


def sasa_features(res):
    """Per-residue backbone SASA (N+CA+C+O) and side-chain SASA, in A^2."""
    import freesasa
    freesasa.setVerbosity(freesasa.silent)
    st = freesasa.Structure()
    flat = []           # (residue index, atom name, element)
    for i, r in enumerate(res):
        for name, xyz in r["atoms"].items():
            el = "".join(ch for ch in name if ch.isalpha())[:2]
            el = el[0].upper() + el[1:].lower() if len(el) > 1 else el.upper()
            if el.upper() in ("H", "D"):
                continue
            st.addAtom(el, r["resname"], "%d%s" % (r["resseq"], r["icode"].strip()),
                       r["chain"], float(xyz[0]), float(xyz[1]), float(xyz[2]))
            flat.append((i, name))
    bb = np.zeros(len(res), dtype=np.float32)
    tot = np.zeros(len(res), dtype=np.float32)
    if st.nAtoms() == 0:
        return bb, tot
    result = freesasa.calc(st)
    for k, (i, name) in enumerate(flat):
        a = float(result.atomArea(k))
        tot[i] += a
        if name in BACKBONE_ATOMS:
            bb[i] += a
    return bb, tot - bb


def centroid_features(ca):
    """Euclidean and cosine distance of each CA to the CA centroid."""
    n = len(ca)
    out = np.zeros((n, 2), dtype=np.float32)
    if n == 0:
        return out
    c = np.nanmean(ca, axis=0)
    vec = ca - c
    dist = np.linalg.norm(vec, axis=1)
    nvec = np.linalg.norm(ca, axis=1)
    nc = float(np.linalg.norm(c))
    if nc < 1e-9:
        cosd = np.zeros(n)
    else:
        cosd = 1.0 - (ca @ c) / (np.maximum(nvec, 1e-9) * nc)
    out[:, 0] = dist
    out[:, 1] = cosd
    return out


def ca_distance_matrix(ca):
    d = ca[:, None, :] - ca[None, :, :]
    return np.sqrt(np.maximum((d * d).sum(-1), 0.0))


def degree_features(ca, cutoff=10.0):
    d = ca_distance_matrix(ca)
    adj = (d <= cutoff) & (d > 0)
    return adj.sum(axis=1).astype(np.float32)


# --------------------------------------------------------------------------- #
# main entry
# --------------------------------------------------------------------------- #
def residue_features(pdb_path, with_plddt=False, cutoff=10.0, degree=None):
    """Compute the (N, 46) / (N, 47) feature matrix for one protein."""
    res = parse_protein(pdb_path)
    arrays = residues_arrays(res)
    n = len(res)
    if not np.isfinite(arrays["ca"]).all():
        # never expected (missing CA falls back to the heavy-atom centroid)
        arrays["ca"] = np.nan_to_num(arrays["ca"], nan=0.0)

    # 1) one-hot amino acid
    X = np.zeros((n, N_FEATURES_PLDDT if with_plddt else N_FEATURES), dtype=np.float32)
    aa_idx = {c: i for i, c in enumerate(AA_ORDER)}
    for i, r in enumerate(res):
        X[i, aa_idx[AA3[r["resname"]]]] = 1.0

    # 2) iPlus / iMinus (sequence neighbour in the same chain)
    for i, r in enumerate(res):
        if i > 0 and res[i - 1]["chain"] == r["chain"]:
            X[i, 20] = 1.0
        if i + 1 < n and res[i + 1]["chain"] == r["chain"]:
            X[i, 21] = 1.0

    # 3) DSSP: SS one-hot + 8 backbone H-bonds
    dssp_path = None
    try:
        dssp_path = run_mkdssp(pdb_path)
        dssp = parse_dssp(dssp_path)
    except Exception as exc:  # noqa: BLE001
        _warn_once("dssp", "[features] WARNING: mkdssp unavailable or failed (%r).\n"
                           "  -> DSSP features (15 dims) are ALL ZERO, predictions are degraded.\n"
                           "  -> fix: conda install -c conda-forge dssp   (or export MKDSSP=/path/to/mkdssp)\n"
                           "  -> set GNNSITE_STRICT=1 to turn this into a hard error." % (exc,))
        dssp = {}
    n_missing_dssp = 0
    for i, r in enumerate(res):
        entry = dssp.get((r["chain"], r["resseq"], r["icode"]))
        if entry is None:
            n_missing_dssp += 1
            continue
        ss = entry["ss"]
        if ss in SS_ORDER:
            X[i, 22 + SS_ORDER.index(ss)] = 1.0
        X[i, 29:37] = entry["hb"]

    # 4) dihedrals
    X[:, 37:41] = dihedral_features(res)

    # 5) SASA
    try:
        bb, sc = sasa_features(res)
    except Exception as exc:  # noqa: BLE001
        _warn_once("sasa", "[features] WARNING: freesasa failed (%r).\n"
                           "  -> SASA features (2 dims) are ALL ZERO, predictions are degraded.\n"
                           "  -> fix: pip install freesasa  (or conda install -c conda-forge freesasa)"
                           % (exc,))
        bb = sc = np.zeros(n, dtype=np.float32)
    # freesasa returns NaN for atoms it cannot classify
    X[:, 41] = np.nan_to_num(bb, nan=0.0, posinf=0.0, neginf=0.0)
    X[:, 42] = np.nan_to_num(sc, nan=0.0, posinf=0.0, neginf=0.0)

    # 6) centroid
    X[:, 43:45] = np.nan_to_num(centroid_features(arrays["ca"]))

    # 7) graph degree
    if degree is None:
        degree = degree_features(arrays["ca"], cutoff)
    X[:, 45] = degree

    # 8) optional pLDDT (constant 1.0: crystal structures have no pLDDT)
    if with_plddt:
        X[:, 46] = 1.0

    meta = {
        "id": os.path.basename(os.path.dirname(pdb_path)) or os.path.basename(pdb_path),
        "protein": pdb_path,
        "n_residues": int(n),
        "n_chains": int(len(set(arrays["chain"]))),
        "n_missing_dssp": int(n_missing_dssp),
        "n_missing_ca": int(arrays["n_missing_ca"]),
        "n_ss_assigned": int(X[:, 22:29].sum()),
        "with_plddt": bool(with_plddt),
        "n_features": int(X.shape[1]),
        "chains": arrays["chain"],
        "resseq": arrays["resseq"],
        "icode": arrays["icode"],
        "resname": arrays["resname"],
        "ca": arrays["ca"],
    }
    if dssp_path:
        try:
            os.unlink(dssp_path)
            os.rmdir(os.path.dirname(dssp_path))
        except OSError:
            pass
    return X, meta


def self_check(X, names=None, verbose=True):
    """Per-dimension min/max/mean + one-hot and finiteness checks."""
    names = names or FEATURE_NAMES[:X.shape[1]]
    rep = []
    for j in range(X.shape[1]):
        col = X[:, j]
        rep.append({
            "dim": j, "name": names[j],
            "min": float(col.min()), "max": float(col.max()),
            "mean": float(col.mean()), "nan": int(np.isnan(col).sum()),
            "inf": int(np.isinf(col).sum()),
        })
    checks = {
        "n_rows": int(X.shape[0]),
        "n_cols": int(X.shape[1]),
        "nan_total": int(np.isnan(X).sum()),
        "inf_total": int(np.isinf(X).sum()),
        "all_finite": bool(np.isfinite(X).all()),
        "per_dim": rep,
    }
    onehot = []
    for lo, hi in ONEHOT_SLICES:
        if hi > X.shape[1]:
            continue
        block = X[:, lo:hi]
        ok_vals = bool(np.isin(block, (0.0, 1.0)).all())
        rowsum = block.sum(axis=1)
        onehot.append({
            "slice": [lo, hi],
            "values_are_0_or_1": ok_vals,
            "row_sum_min": float(rowsum.min()),
            "row_sum_max": float(rowsum.max()),
            "rows_sum_1": int((rowsum == 1).sum()),
            "rows_sum_0": int((rowsum == 0).sum()),
        })
    checks["onehot"] = onehot
    if verbose:
        print("%-16s %8s %8s %8s %5s %5s" % ("dim/name", "min", "max", "mean", "nan", "inf"))
        for r in rep:
            print("%2d %-13s %8.3f %8.3f %8.3f %5d %5d" % (
                r["dim"], r["name"], r["min"], r["max"], r["mean"], r["nan"], r["inf"]))
        print("shape=%s finite=%s nan=%d inf=%d" % (
            X.shape, checks["all_finite"], checks["nan_total"], checks["inf_total"]))
        for o in onehot:
            print("one-hot %s values_0_or_1=%s rowsum=[%g,%g] rows_sum1=%d rows_sum0=%d" % (
                o["slice"], o["values_are_0_or_1"], o["row_sum_min"],
                o["row_sum_max"], o["rows_sum_1"], o["rows_sum_0"]))
    return checks


def protein_path_from_id(pid, split, index_json=INDEX_JSON):
    with open(index_json) as fh:
        idx = json.load(fh)["index"]
    e = idx[pid.lower()]
    return e["protein"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--id", default=None, help="PDB ID, needs --split")
    ap.add_argument("--split", default="test", choices=["train", "val", "test"])
    ap.add_argument("--protein", default=None, help="explicit <id>_protein.pdb")
    ap.add_argument("--index", default=INDEX_JSON)
    ap.add_argument("--out-npz", default=None)
    ap.add_argument("--self-check", action="store_true")
    ap.add_argument("--json", default=None, help="write the self-check report here")
    ap.add_argument("--with-plddt", action="store_true")
    args = ap.parse_args()

    if args.protein:
        pdb = args.protein
    elif args.id:
        pdb = protein_path_from_id(args.id, args.split, args.index)
    else:
        ap.error("need --protein or --id")
    print("[features] %s" % pdb)

    X, meta = residue_features(pdb, with_plddt=args.with_plddt)
    print("[features] residues=%d chains=%d dim=%d missing_dssp=%d ss_assigned=%d" % (
        meta["n_residues"], meta["n_chains"], X.shape[1],
        meta["n_missing_dssp"], meta["n_ss_assigned"]))
    checks = self_check(X, verbose=args.self_check)
    checks["meta"] = {k: v for k, v in meta.items()
                      if k not in ("chains", "resseq", "icode", "resname", "ca")}
    if args.out_npz:
        np.savez_compressed(args.out_npz, x=X,
                            chain=meta["chains"], resseq=meta["resseq"],
                            icode=meta["icode"], resname=meta["resname"],
                            ca=meta["ca"], feature_names=np.array(FEATURE_NAMES[:X.shape[1]]))
        print("[features] wrote %s" % args.out_npz)
    if args.json:
        with open(args.json, "w") as fh:
            json.dump(checks, fh, indent=1)
        print("[features] wrote %s" % args.json)
    return 0 if checks["all_finite"] else 1


if __name__ == "__main__":
    sys.exit(main())
