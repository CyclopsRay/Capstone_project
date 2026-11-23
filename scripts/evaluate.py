#!/usr/bin/env python3
"""Collect the five folds: how well each stage of UnDetect predicts training, validation and test compounds.

Reads results/folds/fold_*/ (scripts/train.py) and data/processed/ (scripts/simulate_data.py). Writes to results/:
  scores.csv        one row per fold, split and stage: PCC_DEG, the mean over the split's samples. Stage "ceiling"
                    is the simulated response without its noise: the best any model could do.
  view_weights.csv  one row per fold and view: the weight UnDetect learned and the weight used in the simulation
  summary.json      means and standard deviations over the folds, data sizes and settings, for the report

On training compounds the compound terms are leave-one-compound-out values, so the "train" scores of the first
stage and of UnDetect are not in-sample fits.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from undetect.metrics import pcc_deg

ROOT = Path(__file__).resolve().parents[1]
STAGES = ("backbone", "first_stage", "final", "ceiling")
SPLITS = ("train", "valid", "test")


def main() -> None:
    cfg = yaml.safe_load((ROOT / "config" / "config.yaml").read_text())
    data = ROOT / cfg["data"]["processed"]
    results = ROOT / cfg["results"]
    z = np.load(data / "responses.npz")
    truth = np.load(data / "truth.npz")
    Y = z["delta"].astype(np.float64)
    ceiling = pcc_deg(Y, truth["context"].astype(np.float64) + truth["compound_effect"])
    simulated = dict(zip(map(str, truth["views"]), map(float, truth["view_weights"])))

    rows, weights, settings = [], [], []
    for f in cfg["model"]["folds"]:
        fold = results / "folds" / f"fold_{f}"
        s = pd.read_csv(fold / "scores.csv")
        s["ceiling"] = ceiling[s["sample"]]
        for split in SPLITS:
            part = s[s.split == split]
            rows += [{"fold": f, "split": split, "stage": st, "pcc_deg": float(part[st].mean()), "samples": len(part)}
                     for st in STAGES]
        st = json.loads((fold / "settings.json").read_text())
        settings.append(st)
        weights += [{"fold": f, "view": v, "learned": st["view_weights"][v], "simulated": simulated[v]}
                    for v in simulated]
    scores, weights = pd.DataFrame(rows), pd.DataFrame(weights)
    results.mkdir(parents=True, exist_ok=True)
    scores.to_csv(results / "scores.csv", index=False, float_format="%.6f")
    weights.to_csv(results / "view_weights.csv", index=False, float_format="%.6f")

    agg = scores.groupby(["stage", "split"]).pcc_deg.agg(["mean", "std"])
    summary = {
        "pcc_deg": {st: {sp: {"mean": float(agg.loc[(st, sp), "mean"]), "sd": float(agg.loc[(st, sp), "std"])}
                         for sp in SPLITS} for st in STAGES},
        "view_weights": {v: {"simulated": simulated[v],
                             "learned_mean": float(weights[weights.view == v].learned.mean()),
                             "learned_sd": float(weights[weights.view == v].learned.std())} for v in simulated},
        "folds": settings,
        "data": json.loads((data / "manifest.json").read_text()),
        "model": cfg["model"],
    }
    (results / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    table = scores.groupby(["stage", "split"]).pcc_deg.mean().unstack()[list(SPLITS)].loc[list(STAGES)]
    print("PCC_DEG, mean over folds:\n" + table.round(4).to_string())


if __name__ == "__main__":
    main()
