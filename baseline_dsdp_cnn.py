#!/usr/bin/env python
"""baseline_dsdp_cnn.py -- DSDP 3D-CNN (DrugSiteMind) site prediction projected to residues.

Pipeline (task book section 4, Step 2):
  1. run the original DSDP site predictor on the 363 test complexes of
     test_dataset/PDBBind_time_split_dataset (repo <id>_protein.pdbqt),
     producing the 36^3 probability grid per complex;
  2. project each residue's CA onto that grid: voxel index
        i = round((ca - centroid + 35) / 2),  i in [0, 35]
     (grid.py of DSDP: grid_resolution = 1/resolution_scale = 2 A,
      max_distance = 35 A, box_size = 36). Score = probability of the CA voxel
     and the max over its 3x3x3 neighbourhood;
  3. join with the same 5 A residue labels used by the GNN
     (data/label/test/<id>.npz, aligned by residue order) and compute
     ROC-AUC / PR-AUC.

Outputs:
  gnn_site/c2_site/npy/<id>.npy        36^3 prediction grids (saved exactly as
                                       DSDP's get_pockets_segmentation does)
  gnn_site/baseline_dsdp_residue.csv   pdb_id,res_idx,chain,resseq,icode,resname,
                                       label,score_voxel,score_nbhd
  gnn_site/reports/baseline_metrics.json     metrics + timings

The DrugSiteData / DrugSiteMind classes are imported from the original
DSDP_blind_docking.py (source executed up to the `if __name__` guard), so the
features/model are exactly the shipped ones. Must run on a GPU node (CUDA).

Usage:
  python baseline_dsdp_cnn.py --limit 0 --device cuda
"""
import argparse
import json
import os
import sys
import time
import traceback

import numpy as np
import torch
from sklearn.metrics import average_precision_score, roc_auc_score
import paths  # noqa: E402  (同目录：统一路径配置)

DSDP_DIR = paths.DSDP_HOME
DSDP_SCRIPT = os.path.join(DSDP_DIR, "DSDP_blind_docking.py")
BASE = paths.GNNSITE_HOME


def load_dsdp_machinery():
    src = open(DSDP_SCRIPT).read()
    src = src.split("if __name__")[0]
    ns = {"__name__": "dsdp_machinery", "__file__": DSDP_SCRIPT}
    old = sys.argv
    sys.argv = [DSDP_SCRIPT]
    try:
        exec(compile(src, DSDP_SCRIPT, "exec"), ns)
    finally:
        sys.argv = old
    return ns


def voxel_xyz(ca, centroid, res_scale=0.5, max_dist=35.0):
    step = 1.0 / res_scale
    idx = np.rint((ca - centroid + max_dist) / step).astype(int)
    return np.clip(idx, 0, 35)


def project_density(dens, ca, centroid):
    """dens (36,36,36) -> per-residue voxel score and 3x3x3 neighbourhood max."""
    idx = voxel_xyz(np.asarray(ca, dtype=np.float64), np.asarray(centroid, dtype=np.float64))
    n = len(idx)
    s_vox = np.zeros(n, dtype=np.float32)
    s_nb = np.zeros(n, dtype=np.float32)
    for r, (i, j, k) in enumerate(idx):
        s_vox[r] = dens[i, j, k]
        i0, i1 = max(i - 1, 0), min(i + 2, 36)
        j0, j1 = max(j - 1, 0), min(j + 2, 36)
        k0, k1 = max(k - 1, 0), min(k + 2, 36)
        s_nb[r] = dens[i0:i1, j0:j1, k0:k1].max()
    return s_vox, s_nb, idx


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset-path",
                    default=paths.TEST_PDBBIND_DIR)
    ap.add_argument("--dataset-name", default="PDBBind_time_split_dataset")
    ap.add_argument("--site-path", default=os.path.join(BASE, "c2_site"))
    ap.add_argument("--model", default=os.path.join(DSDP_DIR, "binding_site_model.pth"))
    ap.add_argument("--labels-dir", default=os.path.join(BASE, "data", "label", "test"))
    ap.add_argument("--csv", default=os.path.join(BASE, "baseline_dsdp_residue.csv"))
    ap.add_argument("--json", default=os.path.join(BASE, "reports", "baseline_metrics.json"))
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--batch-size", type=int, default=1)
    args = ap.parse_args()

    os.makedirs(args.site_path, exist_ok=True)
    os.makedirs(os.path.join(args.site_path, "npy"), exist_ok=True)
    os.makedirs(os.path.dirname(args.json), exist_ok=True)

    device = torch.device(args.device if (args.device == "cpu" or torch.cuda.is_available())
                          else "cpu")
    print("[c2] device=%s" % device)
    if device.type == "cuda":
        print("[c2] gpu=%s" % torch.cuda.get_device_name(0))

    t0 = time.time()
    ns = load_dsdp_machinery()
    DrugSiteData = ns["DrugSiteData"]
    # the shipped binding_site_model.pth was pickled from DSDP_blind_docking.py
    # executed as __main__, so its class has to be reachable there for unpickling
    import __main__
    __main__.DrugSiteMind = ns["DrugSiteMind"]
    __main__.DrugSiteData = DrugSiteData
    print("[c2] DSDP machinery loaded in %.1fs" % (time.time() - t0))

    t_cache = time.time()
    data = DrugSiteData(
        dataset_path=args.dataset_path,
        site_path=args.site_path + "/",
        cache_file=os.path.join(args.site_path, args.dataset_name + ".npz"),
        use_cache=True,
    )
    print("[c2] dataset prepared: %d complexes in %.1fs" % (len(data), time.time() - t_cache))

    names_all = [data.cache[i][3] for i in range(len(data))]
    if args.limit:
        names_all = names_all[:args.limit]

    loader = torch.utils.data.DataLoader(
        data, batch_size=args.batch_size,
        collate_fn=DrugSiteData.collate, shuffle=0, num_workers=0,
        drop_last=False)

    net = torch.load(args.model, weights_only=False).to(device)
    net.eval()
    print("[c2] model loaded: %s" % args.model)

    rows = []
    labels_all, vox_all, nb_all = [], [], []
    skipped = []
    t_inf = time.time()
    done = 0
    with torch.no_grad():
        for batch in loader:
            x, names, centroids = batch
            if args.limit and done >= args.limit:
                break
            x = torch.Tensor(x).to(device).permute(0, 4, 1, 2, 3)
            out = net(x)
            p = out.permute(0, 2, 3, 4, 1)
            densities = p.detach().cpu().numpy()          # (B,36,36,36,1)
            for b, name in enumerate(names):
                if args.limit and done >= args.limit:
                    break
                done += 1
                dens = densities[b, :, :, :, 0]
                np.save(os.path.join(args.site_path, "npy", "%s.npy" % name), densities[b])
                lp = os.path.join(args.labels_dir, "%s.npz" % name.lower())
                if not os.path.exists(lp):
                    skipped.append("%s\tno_label_npz" % name)
                    continue
                z = np.load(lp)
                ca = z["ca"]
                centroid = np.asarray(centroids[b], dtype=np.float64)
                s_vox, s_nb, idx = project_density(dens, ca, centroid)
                y = z["label"].astype(np.int8)
                if len(y) != len(s_vox):
                    skipped.append("%s\tlabel_len_mismatch(%d vs %d)" % (name, len(y), len(s_vox)))
                    continue
                labels_all.append(y)
                vox_all.append(s_vox)
                nb_all.append(s_nb)
                for r in range(len(y)):
                    rows.append((name, r, str(z["chain"][r]), int(z["resseq"][r]),
                                 str(z["icode"][r]).strip(), str(z["resname"][r]),
                                 int(y[r]), float(s_vox[r]), float(s_nb[r])))
                if done % 25 == 0:
                    print("[c2] %d/%d complexes, %.0fs" % (done, len(names_all), time.time() - t_inf))

    print("[c2] inference done: %d complexes in %.1fs" % (done, time.time() - t_inf))

    y = np.concatenate(labels_all).astype(np.int64)
    sv = np.concatenate(vox_all).astype(np.float64)
    sn = np.concatenate(nb_all).astype(np.float64)
    metrics = {
        "n_complexes": done,
        "n_residues": int(len(y)),
        "n_positive": int(y.sum()),
        "positive_ratio": float(y.mean()),
        "score_voxel": {
            "roc_auc": float(roc_auc_score(y, sv)),
            "pr_auc": float(average_precision_score(y, sv)),
            "mean": float(sv.mean()), "max": float(sv.max()),
        },
        "score_nbhd3": {
            "roc_auc": float(roc_auc_score(y, sn)),
            "pr_auc": float(average_precision_score(y, sn)),
            "mean": float(sn.mean()), "max": float(sn.max()),
        },
        "skipped": skipped,
        "wall_seconds": round(time.time() - t0, 1),
        "device": str(device),
    }
    print("[c2] voxel  score: ROC-AUC=%.4f PR-AUC=%.4f" % (
        metrics["score_voxel"]["roc_auc"], metrics["score_voxel"]["pr_auc"]))
    print("[c2] nbhd3  score: ROC-AUC=%.4f PR-AUC=%.4f" % (
        metrics["score_nbhd3"]["roc_auc"], metrics["score_nbhd3"]["pr_auc"]))

    with open(args.csv, "w") as fh:
        fh.write("pdb_id,res_idx,chain,resseq,icode,resname,label,score_voxel,score_nbhd\n")
        for r in rows:
            fh.write("%s,%d,%s,%d,%s,%s,%d,%.6f,%.6f\n" % r)
    print("[c2] wrote %s (%d rows)" % (args.csv, len(rows)))

    with open(args.json, "w") as fh:
        json.dump(metrics, fh, indent=1)
    print("[c2] wrote %s" % args.json)
    return 0


if __name__ == "__main__":
    sys.exit(main())
