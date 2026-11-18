#!/usr/bin/env python
"""build_dataset.py -- residue-level ligand-binding labels (5 A rule).

For every complex of a given time split:
  * protein  = PDBBind v2020 <id>_protein.pdb   (ATOM records, standard residues,
               HETATM waters/ions ignored)
  * ligand   = <id>_ligand.mol2 for train/val,
               the DSDP evaluation conformer <id>_ligand.pdbqt for test
  * label(r) = 1 if min heavy-atom distance(residue r, ligand) <= 5.0 A

Output: one .npz per complex in data/label/<split>/<id>.npz holding
  id, chain, resseq, icode, resname, ca (N,3), label (N,), mindist (N,)
plus an aggregate reports/label_stat.txt and reports/skipped.txt.

Usage:
  python build_dataset.py --split test --limit 5          # smoke
  python build_dataset.py --split test                    # all 363
  python build_dataset.py --split val  --limit 50
  python build_dataset.py --split train --limit 3000
"""
import argparse
import json
import os
import sys
import time
import traceback
from collections import Counter

import numpy as np
from scipy.spatial import cKDTree
import paths  # noqa: E402  (同目录：统一路径配置)

AA3 = {
    "ALA", "ARG", "ASN", "ASP", "CYS", "GLN", "GLU", "GLY", "HIS", "ILE",
    "LEU", "LYS", "MET", "PHE", "PRO", "SER", "THR", "TRP", "TYR", "VAL",
}
# occasional PDBBind variants mapped onto the canonical 20
AA3_ALT = {"MSE": "MET", "HSD": "HIS", "HSE": "HIS", "HSP": "HIS",
           "CYX": "CYS", "CYM": "CYS", "HID": "HIS", "HIE": "HIS"}
HYDROGEN_NAMES = {"H", "D"}


def _element_from_pdb(line):
    el = line[76:78].strip()
    if el:
        return el.upper()
    name = line[12:16].strip()
    for ch in name:
        if ch.isalpha():
            return ch.upper()
    return "C"


def parse_protein_pdb(path):
    """Return dict of residue info + heavy atom coordinates from ATOM records."""
    chain, resseq, icode, resname = [], [], [], []
    coords, atom_res, ca_list = [], [], []
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
            alt = line[16]
            if alt not in (" ", "A"):
                continue
            rn = line[17:20].strip()
            rn = AA3_ALT.get(rn, rn)
            if rn not in AA3:
                continue
            ch = line[21]
            rs = int(line[22:26])
            ic = line[26]
            el = _element_from_pdb(line)
            if el in HYDROGEN_NAMES:
                continue
            key = (ch, rs, ic)
            idx = key2idx.get(key)
            if idx is None:
                idx = len(chain)
                key2idx[key] = idx
                chain.append(ch)
                resseq.append(rs)
                icode.append(ic)
                resname.append(rn)
                ca_list.append((float("nan"),) * 3)
            aname = line[12:16].strip().upper()
            xyz = (float(line[30:38]), float(line[38:46]), float(line[46:54]))
            if aname == "CA" and ca_list[idx][0] != ca_list[idx][0]:
                ca_list[idx] = xyz
            coords.append(xyz)
            atom_res.append(idx)
    if not chain:
        raise ValueError("no protein residues parsed from %s" % path)
    ca = np.asarray(ca_list, dtype=np.float32)
    if np.isnan(ca).any():
        # residues without a CA record: use the residue heavy-atom centroid
        bad = np.unique(np.where(np.isnan(ca))[0])
        cnt = np.bincount(atom_res, minlength=len(chain)).astype(np.float32)
        for ax in range(3):
            cent = np.bincount(atom_res, weights=np.asarray(coords)[:, ax],
                               minlength=len(chain)) / np.maximum(cnt, 1)
            ca[bad, ax] = cent[bad]
    return {
        "chain": np.array(chain),
        "resseq": np.array(resseq, dtype=np.int32),
        "icode": np.array(icode),
        "resname": np.array(resname),
        "xyz": np.asarray(coords, dtype=np.float32),
        "atom_res": np.asarray(atom_res, dtype=np.int32),
        "ca": ca,
    }


def parse_ligand_mol2(path):
    atoms, section = [], None
    with open(path) as fh:
        for line in fh:
            if line.startswith("@<TRIPOS>"):
                section = line.strip()
                continue
            if section != "@<TRIPOS>ATOM":
                continue
            parts = line.split()
            if len(parts) < 6:
                continue
            typ = parts[5]
            el = typ[:2] if len(typ) > 1 and typ[1].islower() else typ[:1]
            if el.upper() in HYDROGEN_NAMES or typ.upper().startswith("H"):
                continue
            atoms.append((float(parts[2]), float(parts[3]), float(parts[4])))
    return np.asarray(atoms, dtype=np.float32)


def parse_ligand_pdbqt(path):
    atoms, in_model = [], True
    with open(path) as fh:
        for line in fh:
            if line.startswith("ENDMDL"):
                in_model = False
                continue
            if line.startswith("MODEL"):
                in_model = True
                continue
            if not line.startswith(("ATOM", "HETATM")) or not in_model:
                continue
            typ = line[77:79].strip().upper()
            if typ.startswith("H"):
                continue
            atoms.append((float(line[30:38]), float(line[38:46]), float(line[46:54])))
    return np.asarray(atoms, dtype=np.float32)


def residue_labels(prot, lig_xyz, cutoff):
    """min heavy-atom distance per residue -> labels / distances."""
    d = np.linalg.norm(lig_xyz[:, None, :] - prot["xyz"][None, :, :], axis=-1)
    per_atom = d.min(axis=0)
    nres = len(prot["chain"])
    mindist = np.full(nres, np.inf, dtype=np.float32)
    np.minimum.at(mindist, prot["atom_res"], per_atom.astype(np.float32))
    return (mindist <= cutoff), mindist


def ca_coords(prot):
    """CA position of each residue (already resolved in parse_protein_pdb)."""
    return prot["ca"], np.ones(len(prot["chain"]), dtype=bool)


def process_one(pid, entry, split, out_dir, cutoff):
    prot_path = entry["protein"]
    if split == "test" and "ligand_pdbqt" in entry:
        lig_path, lig_kind = entry["ligand_pdbqt"], "pdbqt"
    elif "ligand_mol2" in entry:
        lig_path, lig_kind = entry["ligand_mol2"], "mol2"
    elif "ligand_pdbqt" in entry:
        lig_path, lig_kind = entry["ligand_pdbqt"], "pdbqt"
    else:
        raise ValueError("no ligand file in index entry")

    prot = parse_protein_pdb(prot_path)
    if lig_kind == "mol2":
        lig = parse_ligand_mol2(lig_path)
    else:
        lig = parse_ligand_pdbqt(lig_path)
    if len(lig) < 3:
        raise ValueError("ligand heavy atoms=%d (<3)" % len(lig))

    labels, mindist = residue_labels(prot, lig, cutoff)
    ca, _ = ca_coords(prot)

    os.makedirs(out_dir, exist_ok=True)
    np.savez_compressed(
        os.path.join(out_dir, pid + ".npz"),
        id=pid,
        chain=prot["chain"],
        resseq=prot["resseq"],
        icode=prot["icode"],
        resname=prot["resname"],
        ca=ca,
        label=labels.astype(np.int8),
        mindist=mindist,
        ligand_kind=lig_kind,
        n_ligand_heavy=np.int32(len(lig)),
        cutoff=np.float32(cutoff),
    )
    return {
        "n_res": int(len(labels)),
        "n_pos": int(labels.sum()),
        "n_lig": int(len(lig)),
        "lig_kind": lig_kind,
    }


def write_stat(path, split, args, per_complex, skipped, elapsed):
    n_res = sum(v["n_res"] for v in per_complex.values())
    n_pos = sum(v["n_pos"] for v in per_complex.values())
    ligs = Counter(v["lig_kind"] for v in per_complex.values())
    lines = [
        "# reports/label_stat.txt -- %s" % time.strftime("%Y-%m-%d %H:%M:%S"),
        "split                 : %s" % split,
        "cutoff_angstrom       : %.2f" % args.cutoff,
        "complexes_done        : %d" % len(per_complex),
        "complexes_skipped     : %d" % len(skipped),
        "residues_total        : %d" % n_res,
        "residues_positive     : %d" % n_pos,
        "residues_negative     : %d" % (n_res - n_pos),
        "positive_ratio        : %.4f" % (n_pos / n_res if n_res else 0.0),
        "ligand_source         : %s" % dict(ligs),
        "mean_residues_per_cplx: %.1f" % (n_res / len(per_complex) if per_complex else 0),
        "mean_pos_per_cplx     : %.2f" % (n_pos / len(per_complex) if per_complex else 0),
        "wall_seconds          : %.1f" % elapsed,
        "out_dir               : %s" % args.out_dir,
    ]
    if per_complex:
        dims = [v["n_res"] for v in per_complex.values()]
        lines.append("residues_min/max      : %d / %d" % (min(dims), max(dims)))
        zero = [k for k, v in per_complex.items() if v["n_pos"] == 0]
        lines.append("complexes_with_0_pos  : %d %s" % (len(zero), zero[:10]))
    with open(path, "w") as fh:
        fh.write("\n".join(lines) + "\n")
    print("\n".join(lines))


def write_summary(base, out_path):
    """Aggregate every data/label/<split>/*.npz into one reports/label_stat.txt."""
    import glob
    lines = ["# reports/label_stat.txt (aggregate over data/label/<split>/*.npz) -- %s"
             % time.strftime("%Y-%m-%d %H:%M:%S"),
             "%-6s %10s %10s %10s %10s %8s" % (
                 "split", "proteins", "residues", "positive", "negative", "pos%")]
    tot_c = tot_r = tot_p = 0
    for split in ("train", "val", "test"):
        files = sorted(glob.glob(os.path.join(base, "data", "label", split, "*.npz")))
        nc = nr = np_pos = 0
        for f in files:
            with np.load(f) as z:
                nc += 1
                n = int(len(z["label"]))
                p = int(z["label"].sum())
                nr += n
                np_pos += p
        if nc:
            lines.append("%-6s %10d %10d %10d %10d %7.2f%%" % (
                split, nc, nr, np_pos, nr - np_pos, 100.0 * np_pos / max(nr, 1)))
        tot_c += nc
        tot_r += nr
        tot_p += np_pos
    lines.append("%-6s %10d %10d %10d %10d %7.2f%%" % (
        "ALL", tot_c, tot_r, tot_p, tot_r - tot_p, 100.0 * tot_p / max(tot_r, 1)))
    lines.append("# pos = residue whose min heavy-atom distance to ligand <= 5.0 A")
    with open(out_path, "w") as fh:
        fh.write("\n".join(lines) + "\n")
    print("\n".join(lines))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="test", choices=["train", "val", "test"])
    ap.add_argument("--index", default=paths.INDEX_JSON)
    ap.add_argument("--out-dir", default=None,
                    help="default gnn_site/data/label/<split>")
    ap.add_argument("--cutoff", type=float, default=5.0)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--offset", type=int, default=0)
    ap.add_argument("--stat-file", default=None)
    ap.add_argument("--skipped", default=paths.SKIPPED)
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--summary", action="store_true",
                    help="only aggregate existing data/label/*/*.npz into reports/label_stat.txt")
    args = ap.parse_args()

    base = paths.GNNSITE_HOME
    if args.summary:
        write_summary(base, os.path.join(base, "reports/label_stat.txt"))
        return 0
    if args.out_dir is None:
        args.out_dir = os.path.join(base, "data", "label", args.split)
    if args.stat_file is None:
        args.stat_file = os.path.join(base, "reports", "label_stat_%s.txt" % args.split)

    with open(args.index) as fh:
        blob = json.load(fh)
    index = blob["index"]
    cov = blob["coverage"][args.split]
    split_file = cov["file"]
    ids = [ln.strip().lower() for ln in open(split_file) if ln.strip()]
    ids = ids[args.offset:]
    if args.limit:
        ids = ids[:args.limit]
    print("[build] split=%s ids=%d out=%s cutoff=%.2f" % (
        args.split, len(ids), args.out_dir, args.cutoff))

    os.makedirs(args.out_dir, exist_ok=True)
    t0 = time.time()
    per_complex, skipped = {}, []
    for i, pid in enumerate(ids, 1):
        out_npz = os.path.join(args.out_dir, pid + ".npz")
        if os.path.exists(out_npz) and not args.overwrite:
            with np.load(out_npz) as z:
                per_complex[pid] = {"n_res": int(len(z["label"])),
                                    "n_pos": int(z["label"].sum()),
                                    "n_lig": int(z["n_ligand_heavy"]),
                                    "lig_kind": str(z["ligand_kind"])}
            continue
        entry = index.get(pid)
        if entry is None:
            skipped.append("%s\t%s\tnot_in_index" % (args.split, pid))
            continue
        try:
            per_complex[pid] = process_one(pid, entry, args.split, args.out_dir, args.cutoff)
        except Exception as e:
            skipped.append("%s\t%s\t%s" % (args.split, pid, repr(e)))
            print("[skip] %s: %r" % (pid, e))
            traceback.print_exc(limit=1)
        if i % 25 == 0 or i == len(ids):
            n_res = sum(v["n_res"] for v in per_complex.values())
            n_pos = sum(v["n_pos"] for v in per_complex.values())
            print("[build] %d/%d done, skipped=%d, pos_ratio=%.4f, %.0fs" % (
                i, len(ids), len(skipped), n_pos / max(n_res, 1), time.time() - t0))

    write_stat(args.stat_file, args.split, args, per_complex, skipped, time.time() - t0)
    if skipped:
        mode = "a" if os.path.exists(args.skipped) else "w"
        with open(args.skipped, mode) as fh:
            fh.write("\n".join(skipped) + "\n")
    print("[write] %s" % args.stat_file)
    return 0 if not skipped else 2


if __name__ == "__main__":
    sys.exit(main())
