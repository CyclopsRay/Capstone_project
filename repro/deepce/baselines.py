#!/usr/bin/env python3
"""Hard baselines DeepCE did not report, on the identical filtered upstream splits (repro/deepce).

mean_global     training-set mean signature
mean_cell       training mean per cell line
mean_cell_dose  training mean per (cell line, dose); falls back to mean_cell
knn_morgan      mean_cell_dose + alpha * softmax-weighted (Tanimoto/tau) residuals of the top-k training
                drugs measured in the same (cell, dose); k, tau, alpha chosen on dev Pearson
Metric functions are upstream's (mean per-signature Pearson over 978 genes, RMSE, P@100).
"""
from __future__ import annotations
import os as _os  # noqa: E402
UNDETECT_DATA = _os.environ.get("UNDETECT_DATA", "data")  # data root (README)
UNDETECT_ROOT = _os.environ.get("UNDETECT_ROOT", ".")  # repository root (README)

import json
import sys
from itertools import product
from pathlib import Path

import numpy as np
from rdkit import Chem, DataStructs
from rdkit.Chem import rdFingerprintGenerator

UP = Path(__file__).resolve().parent / "upstream/DeepCE"
sys.path.insert(0, str(UP))
from utils import correlation, precision_k, rmse  # noqa: E402
from utils.data_utils import read_data  # noqa: E402

from run_deepce import FILTER  # noqa: E402

OUT = Path((UNDETECT_DATA + "/repro/deepce/results/baselines"))


def scores(y, p):
    pear, per_sig = correlation(y, p, "pearson")
    neg, pos = precision_k(y, p, 100)
    return {"pearson": float(pear), "rmse": float(rmse(y, p)), "p100_pos": float(pos), "p100_neg": float(neg)}, per_sig


def main() -> None:
    d = UP / "data"
    smiles = dict(line.strip().split(",", 1) for line in open(d / "drugs_smiles.csv"))
    X, Y = {}, {}
    for s in ("train", "dev", "test"):
        X[s], Y[s] = read_data(str(d / f"signature_{s}.csv"), FILTER)
    gen = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)
    fps = {k: gen.GetFingerprint(Chem.MolFromSmiles(v)) for k, v in smiles.items()}
    tr, ytr = X["train"], Y["train"]
    m_all = ytr.mean(0)
    m_cell = {c: ytr[tr[:, 2] == c].mean(0) for c in set(tr[:, 2])}
    m_cd = {(c, s): ytr[(tr[:, 2] == c) & (tr[:, 3] == s)].mean(0) for c, s in set(map(tuple, tr[:, 2:4]))}

    def mean_cd(row):
        return m_cd.get((row[2], row[3]), m_cell[row[2]])

    def knn_parts(split):
        """Per row: base prediction, Tanimoto to same-(cell, dose) training rows, their residuals."""
        parts = []
        for row in X[split]:
            same = np.where((tr[:, 2] == row[2]) & (tr[:, 3] == row[3]))[0]
            base = mean_cd(row)
            sim = np.array(DataStructs.BulkTanimotoSimilarity(fps[row[0]], [fps[tr[j, 0]] for j in same]))
            parts.append((base, sim, ytr[same] - base))
        return parts

    def knn_predict(parts, k, tau, alpha):
        out = []
        for base, sim, res in parts:
            if len(sim) == 0:
                out.append(base); continue
            top = np.argsort(-sim)[:k]
            w = np.ones(len(top)) if np.isinf(tau) else np.exp((sim[top] - sim[top].max()) / tau)
            out.append(base + alpha * (w / w.sum()) @ res[top])
        return np.array(out)

    dev_parts, test_parts = knn_parts("dev"), knn_parts("test")
    grid = list(product((1, 3, 10, 30, 100), (0.03, 0.1, 0.3, np.inf), (0.25, 0.5, 0.75, 1.0, 1.5)))
    dev_scores = {g: correlation(Y["dev"], knn_predict(dev_parts, *g), "pearson")[0] for g in grid}
    best = max(dev_scores, key=dev_scores.get)
    preds = {
        "mean_global": np.tile(m_all, (len(X["test"]), 1)),
        "mean_cell": np.array([m_cell[r[2]] for r in X["test"]]),
        "mean_cell_dose": np.array([mean_cd(r) for r in X["test"]]),
        "knn_morgan": knn_predict(test_parts, *best),
    }
    OUT.mkdir(parents=True, exist_ok=True)
    summary = {"n_train": len(tr), "n_dev": len(X["dev"]), "n_test": len(X["test"]),
               "knn_selected": {"k": best[0], "tau": str(best[1]), "alpha": best[2], "dev_pearson": float(dev_scores[best])}}
    for name, p in preds.items():
        summary[name], per_sig = scores(Y["test"], p)
        np.savez(OUT / f"{name}.npz", pred=p, label=Y["test"], per_sig_pearson=np.asarray(per_sig),
                 pert_id=X["test"][:, 0].astype(str), cell_id=X["test"][:, 2].astype(str),
                 pert_idose=X["test"][:, 3].astype(str))
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
