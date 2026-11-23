#!/usr/bin/env python3
"""Scores of the demo run: the paper's metrics on the simulated benchmark, with baselines and the known answer.

Reads, under the demo data root (config/config.yaml, demo.data_root):
  derived/xpert_sdst/v1/{sdst.npz, truth.npz, manifest.json}          scripts/demo/simulate_benchmark.py
  results/sdst_hybrid/cache_v1/fold_<f>/{cache.npz, summary.json}     scripts/cache_undetect.py
  results/sdst_hybrid/stage2_v2/fold_<f>/{pred_test.npz, summary.json} scripts/stage2_context.py --v2
Writes to the results folder (demo.results):
  scores.csv    one row per fold and arm: PCC_DEG on all test rows, PCC_plate on test rows with a training plate-mate
  weights.csv   one row per fold and view: UnDetect's view weight (kernel-target alignment) and the simulated weight
  summary.json  five-fold means and standard deviations, data sizes and settings, for the report

Metrics as in finalize_sdst.py: PCC_DEG = mean over test samples of the Pearson correlation of true and predicted
response across genes; PCC_plate = the same after subtracting the mean training response of the sample's plate
(plates are identified by their control profile), on samples whose plate has training samples.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[2]
ARMS = {
    "cell_line_mean": "Cell-line mean",
    "plate_baseline": "Plate response baseline",
    "mlp": "Drug-free backbone",
    "pool_kta": "UnDetect, first stage",
    "ctx_pc": "UnDetect",
    "noise_ceiling": "Noise ceiling",
}


def pcc(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    a = a - a.mean(1, keepdims=True)
    b = b - b.mean(1, keepdims=True)
    return (a * b).sum(1) / np.sqrt((a * a).sum(1) * (b * b).sum(1) + 1e-12)


def group_mean(key: np.ndarray, Y: np.ndarray, rows: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Mean of Y[rows] per key value, and the number of rows per key value."""
    n = np.bincount(key[rows], minlength=key.max() + 1)
    S = np.zeros((key.max() + 1, Y.shape[1]))
    np.add.at(S, key[rows], Y[rows])
    return S / np.maximum(n, 1)[:, None], n


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--config", type=Path, default=ROOT / "config" / "config.yaml")
    args = p.parse_args()
    cfg = yaml.safe_load(args.config.read_text())["demo"]
    root = ROOT / cfg["data_root"]
    out = ROOT / cfg["results"]
    D = root / "derived" / "xpert_sdst" / "v1"
    R = root / "results" / "sdst_hybrid"
    z = np.load(D / "sdst.npz", allow_pickle=False)
    truth = np.load(D / "truth.npz", allow_pickle=False)
    manifest = json.loads((D / "manifest.json").read_text())
    ctl, Y, g, cell = z["ctl"], z["delta"].astype(np.float64), z["group"], z["cell_idx"]
    clean = truth["context"].astype(np.float64) + truth["compound"]
    key = np.unique(np.array([hashlib.md5(r.tobytes()).digest() for r in ctl]), return_inverse=True)[1].ravel()
    views = [str(v) for v in truth["views"]]
    planted = dict(zip(views, map(float, truth["view_weights"])))

    rows, wrows, choices = [], [], []
    for f in cfg["undetect"]["folds"]:
        te = g == f
        tr = ~te & (g != (f + 1) % 5)
        tri = np.flatnonzero(tr)
        line_mean = group_mean(cell, Y, tri)[0][cell[te]]
        plate_mean, n_plate = group_mean(key, Y, tri)
        shared = n_plate[key[te]] > 0
        m = plate_mean[key[te]]
        cache = np.load(R / "cache_v1" / f"fold_{f}" / "cache.npz")
        s1 = json.loads((R / "cache_v1" / f"fold_{f}" / "summary.json").read_text())
        s2 = json.loads((R / "stage2_v2" / f"fold_{f}" / "summary.json").read_text())
        P2 = np.load(R / "stage2_v2" / f"fold_{f}" / "pred_test.npz", allow_pickle=False)
        assert np.allclose(P2["y"].astype(np.float64), Y[te], atol=1e-2), "stage-2 rows do not match the data"
        preds = {
            "cell_line_mean": line_mean,
            "plate_baseline": np.where(shared[:, None], m, line_mean),  # the line mean where a plate has no mates
            "mlp": cache["B"][te].astype(np.float64),
            "pool_kta": P2["pool_kta"].astype(np.float64),
            "ctx_pc": P2["ctx_pc"].astype(np.float64),
            "noise_ceiling": clean[te],
        }
        Yt = Y[te]
        for arm, P in preds.items():
            rows.append({"fold": f, "arm": arm, "label": ARMS[arm], "deg": float(pcc(Yt, P).mean()),
                         "plate": float(pcc(Yt[shared] - m[shared], P[shared] - m[shared]).mean()),
                         "n_test": int(te.sum()), "n_shared": int(shared.sum())})
        for v in views:
            wrows.append({"fold": f, "view": v, "learned": s1["weights_kta"][v], "simulated": planted[v]})
        choices.append({"fold": f, "backbone_epoch": s1["mlp_epoch"], "lambda": s1["lam"], "g": s1["g"],
                        "h": s1["h"], "ctx_pc_c": s2["arms"]["ctx_pc"]["c"], "ctx_pc_K": s2["arms"]["ctx_pc"]["K"]})

    scores, weights = pd.DataFrame(rows), pd.DataFrame(wrows)
    out.mkdir(parents=True, exist_ok=True)
    scores.to_csv(out / "scores.csv", index=False)
    weights.to_csv(out / "weights.csv", index=False)
    agg = scores.groupby("arm", sort=False).agg(label=("label", "first"), deg_mean=("deg", "mean"),
                                                deg_sd=("deg", "std"), plate_mean=("plate", "mean"),
                                                plate_sd=("plate", "std"))
    summary = {
        "arms": agg.to_dict(orient="index"),
        "view_weights": {v: {"simulated": planted[v],
                             "learned_mean": float(weights[weights.view == v].learned.mean()),
                             "learned_sd": float(weights[weights.view == v].learned.std())} for v in views},
        "choices": choices,
        "data": {k: manifest[k] for k in ("n", "drugs", "cells", "plates", "genes", "moa_classes",
                                          "with_cell_painting")},
        "settings": {"simulation": cfg["simulation"], "undetect": cfg["undetect"], "seed": cfg["seed"]},
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(agg.round(4).to_string())


if __name__ == "__main__":
    main()
