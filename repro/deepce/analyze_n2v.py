#!/usr/bin/env python3
"""E3 summary: DeepCE with node2vec gene vectors from STRING functional >= 400 and its nulls
(real / rewired / labels / er; make_n2v_genes.py), next to the earlier gene-variant ablation
(ppi / shuffled / random / onehot). Test Pearson at the best-dev epoch; seed-averaged per-signature
contrasts with a bootstrap over the 92 test drugs (as analyze.py)."""
import os as _os  # noqa: E402
UNDETECT_DATA = _os.environ.get("UNDETECT_DATA", "data")  # data root (README)
UNDETECT_ROOT = _os.environ.get("UNDETECT_ROOT", ".")  # repository root (README)
import json
from pathlib import Path
import numpy as np, pandas as pd
R = Path((UNDETECT_DATA + "/repro/deepce/results"))
LAB = ["n2v_real", "n2v_rewired", "n2v_labels", "n2v_er", "ppi", "shuffled", "random", "onehot"]
rows, per, keys = [], {}, None
for v in LAB:
    for s in (0, 1, 2):
        e = pd.read_csv(R / f"{v}_seed{s}" / "epochs.csv"); b = int(e.dev_pearson.idxmax())
        z = np.load(R / f"{v}_seed{s}" / "test_pred_bestdev.npz")
        k = np.stack([z["pert_id"], z["cell_id"], z["pert_idose"]], 1); keys = k if keys is None else keys
        assert (k == keys).all()
        per[(v, s)] = z["per_sig_pearson"]
        rows.append({"variant": v, "seed": s, "epochs": len(e), "best_dev_epoch": b + 1, "test": e.test_pearson[b]})
f = pd.DataFrame(rows)
print(f.pivot(index="variant", columns="seed", values="test").loc[LAB].round(4).assign(
    mean=lambda x: x.mean(1).round(4), sd=lambda x: x[[0, 1, 2]].std(1).round(4)).to_string())
print("epochs run:", sorted(set(f.epochs)))
avg = {v: np.mean([per[(v, s)] for s in (0, 1, 2)], 0) for v in LAB}
drugs = np.unique(keys[:, 0]); idx = {d: np.where(keys[:, 0] == d)[0] for d in drugs}
rng = np.random.default_rng(0); boots = [np.concatenate([idx[d] for d in rng.choice(drugs, len(drugs))]) for _ in range(2000)]
out = []
for a, b in [("n2v_real", "n2v_rewired"), ("n2v_real", "n2v_labels"), ("n2v_real", "n2v_er"), ("n2v_rewired", "n2v_labels"),
             ("n2v_real", "onehot"), ("n2v_real", "random"), ("n2v_real", "ppi"), ("ppi", "onehot")]:
    d = avg[a] - avg[b]; reps = [d[i].mean() for i in boots]
    lo, hi = np.quantile(reps, [0.025, 0.975]); out.append({"contrast": f"{a} - {b}", "diff": d.mean(), "lo": lo, "hi": hi})
    print(f"  {a:12s} - {b:12s} {d.mean():+.4f} [{lo:+.4f}, {hi:+.4f}]")
f.to_csv(R / "n2v_ladder_runs.csv", index=False); pd.DataFrame(out).to_csv(R / "n2v_ladder_contrasts.csv", index=False)
