#!/usr/bin/env python3
"""Train and test UnDetect on one fold of the simulated data.

Fold f tests the compounds of group f, validates on group (f + 1) mod 5 and trains on the other three groups, so
every test compound is new to the model. Reads data/processed/ (scripts/simulate_data.py) and writes
results/folds/fold_<f>/:
  scores.csv            PCC_DEG of every sample at each stage of the model, and the sample's split
  settings.json         what was chosen on the validation compounds: view weights, ridge, term weights, gate
  test_predictions.npz  measured and predicted responses of the test samples

Usage: python scripts/train.py --fold 0
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from undetect.kernels import compound_kernels
from undetect.metrics import pcc_deg
from undetect.model import fit_fold, split

ROOT = Path(__file__).resolve().parents[1]
STAGES = ("backbone", "first_stage", "final")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--fold", type=int, required=True)
    args = parser.parse_args()
    cfg = yaml.safe_load((ROOT / "config" / "config.yaml").read_text())
    m = cfg["model"]
    data = ROOT / cfg["data"]["processed"]
    z = np.load(data / "responses.npz")
    compounds = pd.read_csv(data / "compounds.tsv", sep="\t", keep_default_na=False)
    cp = np.load(data / "cell_painting.npz")

    t0 = time.time()
    K = compound_kernels(compounds, cp["inchikey"], cp["profiles"])
    fit = fit_fold(z["ctl"], z["delta"], z["compound"], z["line"], z["group"], K, args.fold,
                   seed=m["seed"], epochs=m["epochs"], patience=m["patience"])
    Y = z["delta"].astype(np.float64)
    train, valid, test = split(z["group"], args.fold)
    scores = pd.DataFrame({"sample": np.arange(len(Y)),
                           "split": np.where(train, "train", np.where(valid, "valid", "test")),
                           **{s: pcc_deg(Y, getattr(fit, s)) for s in STAGES}})

    out = ROOT / cfg["results"] / "folds" / f"fold_{args.fold}"
    out.mkdir(parents=True, exist_ok=True)
    scores.to_csv(out / "scores.csv", index=False, float_format="%.6f")
    settings = {k: (None if isinstance(v, float) and not np.isfinite(v) else v) for k, v in fit.settings.items()}
    settings["seconds"] = round(time.time() - t0, 1)
    (out / "settings.json").write_text(json.dumps(settings, indent=2) + "\n")
    np.savez(out / "test_predictions.npz", sample=np.flatnonzero(test), measured=Y[test].astype(np.float32),
             final=fit.final[test].astype(np.float32), backbone=fit.backbone[test].astype(np.float32))
    means = scores.groupby("split")[list(STAGES)].mean().loc[["train", "valid", "test"]]
    print(f"fold {args.fold}: PCC_DEG by split and stage ({settings['seconds']} s)\n{means.round(4).to_string()}")


if __name__ == "__main__":
    main()
