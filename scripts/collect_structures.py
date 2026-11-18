#!/usr/bin/env python
"""collect_structures.py -- build an id -> structure index for PDBBind v2020.

Scans the unpacked PDBBind v2020 tree (refined-set/ + v2020-other-PL/) and writes
a JSON index mapping lowercase PDB ID -> absolute paths of
  <id>_protein.pdb, <id>_ligand.mol2, <id>_ligand.sdf, <id>_pocket.pdb

Optionally verifies that every ID of the official time split
(train/val/test) is present, writing gnn_site/reports/skipped.txt for any miss.

Test-split note (task book 2.5): for test IDs the ligand used for 5 A labels is
the DSDP evaluation conformer from the repo
  test_dataset/PDBBind_time_split_dataset/<id>/<id>_ligand.pdbqt
so the index records that path too when it exists.

Usage:
  python scripts/collect_structures.py \
      --root $GNNSITE_HOME/pdbbind_v2020 \
      --out  $GNNSITE_HOME/structure_index.json \
      --split-dir $SPLIT_DIR \
      --repo-test $DSDP_HOME/test_dataset/PDBBind_time_split_dataset
"""
import argparse
import json
import os
import re
import sys
import paths  # noqa: E402  (同目录：统一路径配置)

ID_RE = re.compile(r"^[0-9a-z]{4}$")
SUBSETS = ["refined-set", "v2020-other-PL"]
SPLIT_FILES = {
    "train": "timesplit_no_lig_overlap_train",
    "val": "timesplit_no_lig_overlap_val",
    "test": "timesplit_test",  # task book calls it timesplit_no_lig_overlap_test
}


def index_subset(root, subset):
    base = os.path.join(root, subset)
    out = {}
    if not os.path.isdir(base):
        return out
    for name in os.listdir(base):
        if not ID_RE.match(name.lower()):
            continue
        d = os.path.join(base, name)
        if not os.path.isdir(d):
            continue
        pid = name.lower()
        entry = {"subset": subset, "dir": d}
        for key, suffix in (
            ("protein", "_protein.pdb"),
            ("ligand_mol2", "_ligand.mol2"),
            ("ligand_sdf", "_ligand.sdf"),
            ("pocket", "_pocket.pdb"),
        ):
            p = os.path.join(d, pid + suffix)
            if os.path.exists(p):
                entry[key] = p
        out[pid] = entry
    return out


def read_split(split_dir, split):
    fn = SPLIT_FILES[split]
    path = os.path.join(split_dir, fn)
    if not os.path.exists(path):
        return None, path
    ids = [ln.strip().lower() for ln in open(path) if ln.strip()]
    return ids, path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=paths.GNNSITE_HOME + "/pdbbind_v2020")
    ap.add_argument("--out", default=paths.INDEX_JSON)
    ap.add_argument("--split-dir", default=paths.SPLIT_DIR)
    ap.add_argument("--repo-test", default=paths.TEST_PDBBIND_DIR)
    ap.add_argument("--skipped", default=paths.SKIPPED)
    args = ap.parse_args()

    index = {}
    for subset in SUBSETS:
        part = index_subset(args.root, subset)
        dup = set(part) & set(index)
        if dup:
            print("[warn] %d IDs present in both subsets, keeping %s: %s" % (
                len(dup), subset, sorted(dup)[:5]))
        for pid, e in part.items():
            index.setdefault(pid, e)
        print("[scan] %-18s %6d ids" % (subset, len(part)))
    print("[scan] total unique ids: %d" % len(index))

    # attach DSDP test conformers (ligand used for labels on the test split)
    if os.path.isdir(args.repo_test):
        n = 0
        for name in os.listdir(args.repo_test):
            d = os.path.join(args.repo_test, name)
            pid = name.lower()
            if not os.path.isdir(d) or pid not in index:
                continue
            lig = os.path.join(d, pid + "_ligand.pdbqt")
            prot = os.path.join(d, pid + "_protein.pdbqt")
            if os.path.exists(lig):
                index[pid]["ligand_pdbqt"] = lig
                n += 1
            if os.path.exists(prot):
                index[pid]["repo_protein_pdbqt"] = prot
        print("[scan] attached repo test conformers: %d" % n)

    # split coverage
    skipped = []
    coverage = {}
    for split in ("train", "val", "test"):
        ids, path = read_split(args.split_dir, split)
        if ids is None:
            coverage[split] = {"file": path, "error": "split file not found"}
            print("[split] %-5s file missing: %s" % (split, path))
            continue
        missing = [i for i in ids if i not in index]
        for pid in missing:
            skipped.append("%s\t%s\tmissing_in_pdbbind_index" % (split, pid))
        coverage[split] = {
            "file": path,
            "n_ids": len(ids),
            "n_unique": len(set(ids)),
            "n_present": len(ids) - len(missing),
            "n_missing": len(missing),
            "missing_examples": missing[:10],
        }
        print("[split] %-5s n=%d present=%d missing=%d" % (
            split, len(ids), len(ids) - len(missing), len(missing)))

    # files that exist in the index but lack a protein / ligand
    bad = [pid for pid, e in index.items()
           if "protein" not in e or ("ligand_mol2" not in e and "ligand_pdbqt" not in e)]
    for pid in bad:
        skipped.append("index\t%s\tmissing_protein_or_ligand" % pid)
    print("[check] entries missing protein/ligand: %d" % len(bad))

    with open(args.out, "w") as f:
        json.dump({"root": args.root, "index": index, "coverage": coverage}, f, indent=1)
    with open(args.skipped, "w") as f:
        f.write("\n".join(skipped) + ("\n" if skipped else ""))
    print("[write] %s (%d ids)" % (args.out, len(index)))
    print("[write] %s (%d lines)" % (args.skipped, len(skipped)))

    ok = all(c.get("n_missing", 1) == 0 for c in coverage.values()) and len(coverage) == 3
    print("[result] split coverage 100%%: %s" % ok)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
