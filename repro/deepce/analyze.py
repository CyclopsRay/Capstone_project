#!/usr/bin/env python3
"""Summarize the DeepCE gene-feature ablation (repro/deepce): test Pearson at the best-dev epoch per
variant and seed, seed-averaged paired contrasts with a bootstrap over test drugs (pert_id), and the
upstream-split baselines on the same test rows."""
from __future__ import annotations
import os as _os  # noqa: E402
UNDETECT_DATA = _os.environ.get("UNDETECT_DATA", "data")  # data root (README)
UNDETECT_ROOT = _os.environ.get("UNDETECT_ROOT", ".")  # repository root (README)

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

R = Path((UNDETECT_DATA + "/repro/deepce/results"))
VARIANTS = ("ppi", "shuffled", "random", "onehot")
SEEDS = (0, 1, 2)
N_BOOT = 2000


def main() -> None:
    rows, per_sig, keys = [], {}, None
    for v in VARIANTS:
        for s in SEEDS:
            e = pd.read_csv(R / f"{v}_seed{s}" / "epochs.csv")
            b = int(e.dev_pearson.idxmax())
            z = np.load(R / f"{v}_seed{s}" / "test_pred_bestdev.npz")
            k = np.stack([z["pert_id"], z["cell_id"], z["pert_idose"]], 1)
            keys = k if keys is None else keys
            assert (k == keys).all()
            per_sig[(v, s)] = z["per_sig_pearson"]
            rows.append({"variant": v, "seed": s, "epochs_done": len(e), "best_dev_epoch": b + 1,
                         "dev": e.dev_pearson[b], "test_at_best_dev": e.test_pearson[b],
                         "test_best_epoch_optimistic": e.test_pearson.max(), "test_last": e.test_pearson.iloc[-1]})
    frame = pd.DataFrame(rows)
    print(frame.round(4).to_string(index=False))
    print("\nseed mean / sd (test at best dev):")
    print(frame.groupby("variant").test_at_best_dev.agg(["mean", "std", "min", "max"]).round(4).loc[list(VARIANTS)].to_string())
    base = {}
    for name in ("mean_cell", "knn_morgan"):
        z = np.load(R / "baselines" / f"{name}.npz")
        assert (np.stack([z["pert_id"], z["cell_id"], z["pert_idose"]], 1) == keys).all()
        base[name] = z["per_sig_pearson"]
    avg = {v: np.mean([per_sig[(v, s)] for s in SEEDS], 0) for v in VARIANTS}
    drugs = np.unique(keys[:, 0]); idx = {d: np.where(keys[:, 0] == d)[0] for d in drugs}
    rng = np.random.default_rng(0)
    boots = [np.concatenate([idx[d] for d in rng.choice(drugs, len(drugs))]) for _ in range(N_BOOT)]

    def contrast(a, b):
        diff = a - b
        reps = [diff[i].mean() for i in boots]
        return diff.mean(), np.quantile(reps, 0.025), np.quantile(reps, 0.975)

    print(f"\nseed-averaged per-signature contrasts, bootstrap over {len(drugs)} test drugs:")
    out = []
    pairs = [("ppi", "shuffled"), ("ppi", "random"), ("ppi", "onehot"), ("onehot", "random")]
    pairs += [(v, "knn_morgan") for v in VARIANTS] + [(v, "mean_cell") for v in VARIANTS]
    series = {**avg, **base}
    for a, b in pairs:
        m, lo, hi = contrast(series[a], series[b])
        out.append({"contrast": f"{a} - {b}", "diff": m, "lo": lo, "hi": hi})
        print(f"  {a:9s} - {b:10s} {m:+.4f} [{lo:+.4f}, {hi:+.4f}]")
    print("\nabsolute (seed-averaged per-signature mean):", {k: round(float(v.mean()), 4) for k, v in series.items()})
    frame.to_csv(R / "ablation_runs.csv", index=False)
    pd.DataFrame(out).to_csv(R / "ablation_contrasts.csv", index=False)
    (R / "ablation_absolute.json").write_text(json.dumps({k: float(v.mean()) for k, v in series.items()}, indent=2) + "\n")


if __name__ == "__main__":
    sys.exit(main())
