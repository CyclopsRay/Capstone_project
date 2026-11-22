#!/usr/bin/env python3
"""Same-input neural control for xline_kernel.py on DeepCE's split (exploratory, 2026-10-28).

MLP on [drug views | cell one-hot | log10 dose] -> 978-gene residual from the training mean per
(cell, dose), i.e. the kernel model's inputs and target with a neural map instead of the kernel.
Drug views: chem (Morgan-2, 2048 bits) or multi (+ target multi-hot, MOA multi-hot, z-scored Cell
Painting, and one availability flag per optional view; missing views are zero). Hyper-parameters
(hidden width, dropout, weight decay) and the epoch are chosen on upstream dev Pearson; three seeds
of the selected configuration are scored on test and averaged per signature.
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
import pandas as pd
import torch
from rdkit import Chem, DataStructs
from rdkit.Chem import rdFingerprintGenerator

UP = Path(__file__).resolve().parent / "upstream/DeepCE"
sys.path.insert(0, str(UP))
from utils import correlation  # noqa: E402
from utils.data_utils import read_data  # noqa: E402

from run_deepce import FILTER  # noqa: E402

D = Path((UNDETECT_DATA + "/derived/l1000_unseen_drug"))
OUT = Path((UNDETECT_DATA + "/repro/deepce/results/mlp_multimodal"))
GRID = list(product((256, 1024), (0.1, 0.3), (1e-5, 1e-3)))
EPOCHS = 200


def drug_features(drugs: list[str], smiles: dict) -> dict:
    gen = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)
    n = len(drugs); pos = {k: i for i, k in enumerate(drugs)}
    chem = np.zeros((n, 2048), np.float32)
    for k in drugs:
        DataStructs.ConvertToNumpyArray(gen.GetFingerprint(Chem.MolFromSmiles(smiles[k])), chem[pos[k]])
    table = pd.read_csv(OUT.parent.parent / "drug_table_l1000.tsv", sep="\t", keep_default_na=False)
    table = table[table.source_id.isin(drugs)]
    tg = {k: set() for k in drugs}; mo = {k: set() for k in drugs}; ik = {}
    for r in table.itertuples():
        if r.targets:
            tg[r.source_id].update(r.targets.split(";"))
        if r.moa:
            mo[r.source_id].update(r.moa.split(";"))
        ik.setdefault(r.source_id, r.inchikey)
    hots = []
    for sets in (tg, mo):
        vocab = sorted({x for v in sets.values() for x in v}); vp = {x: i for i, x in enumerate(vocab)}
        M = np.zeros((n, len(vocab)), np.float32)
        for k, v in sets.items():
            M[pos[k], [vp[x] for x in v]] = 1
        hots.append(M)
    cp = np.load(D / "cell_painting_features.npz", allow_pickle=False)
    cpos = pd.Series(np.arange(len(cp["inchikey"])), index=cp["inchikey"].astype(str))
    has_m = np.array([ik.get(k, "") in cpos.index for k in drugs])
    Fm = np.zeros((n, cp["morph10"].shape[1]), np.float32)
    Fm[has_m] = cp["morph10"][cpos[[ik[k] for k in np.array(drugs)[has_m]]].to_numpy()]
    Fm[has_m] = (Fm[has_m] - Fm[has_m].mean(0)) / np.maximum(Fm[has_m].std(0), 1e-9)
    flags = np.stack([hots[0].any(1), hots[1].any(1), has_m], 1).astype(np.float32)
    return {"chem": chem, "multi": np.concatenate([chem, hots[0], hots[1], Fm, flags], 1)}


def main() -> None:
    torch.set_num_threads(8)
    d = UP / "data"
    smiles = dict(line.strip().split(",", 1) for line in open(d / "drugs_smiles.csv"))
    X, Y = {}, {}
    for s in ("train", "dev", "test"):
        X[s], Y[s] = read_data(str(d / f"signature_{s}.csv"), FILTER)
    drugs = sorted(set(np.concatenate([X[s][:, 0] for s in X]))); dpos = {k: i for i, k in enumerate(drugs)}
    feats = drug_features(drugs, smiles)
    cells = sorted(set(X["train"][:, 2]))
    tr = X["train"]; ytr = Y["train"]
    m_cell = {c: ytr[tr[:, 2] == c].mean(0) for c in cells}
    m_cd = {(c, s): ytr[(tr[:, 2] == c) & (tr[:, 3] == s)].mean(0) for c, s in set(map(tuple, tr[:, 2:4]))}
    base = {s: np.array([m_cd.get((r[2], r[3]), m_cell[r[2]]) for r in X[s]]) for s in X}

    def inputs(view, s):
        cell = np.array([[r[2] == c for c in cells] for r in X[s]], np.float32)
        dose = np.log10(np.array([float(x.split()[0]) for x in X[s][:, 3]], np.float32))[:, None]
        return torch.from_numpy(np.concatenate([feats[view][[dpos[k] for k in X[s][:, 0]]], cell, dose], 1))

    def train(view, hidden, drop, wd, seed):
        torch.manual_seed(seed); np.random.seed(seed)
        xi = {s: inputs(view, s) for s in X}
        yt = torch.from_numpy((Y["train"] - base["train"]).astype(np.float32))
        net = torch.nn.Sequential(torch.nn.Linear(xi["train"].shape[1], hidden), torch.nn.ReLU(), torch.nn.Dropout(drop),
                                  torch.nn.Linear(hidden, hidden), torch.nn.ReLU(), torch.nn.Dropout(drop),
                                  torch.nn.Linear(hidden, 978))
        opt = torch.optim.Adam(net.parameters(), lr=1e-3, weight_decay=wd)
        best, best_ep, best_test = -1.0, 0, None
        for ep in range(EPOCHS):
            net.train()
            for b in torch.randperm(len(yt)).split(64):
                opt.zero_grad(); loss = ((net(xi["train"][b]) - yt[b]) ** 2).mean(); loss.backward(); opt.step()
            net.eval()
            with torch.no_grad():
                dev = correlation(Y["dev"], base["dev"] + net(xi["dev"]).numpy(), "pearson")[0]
                if dev > best:
                    best, best_ep, best_test = dev, ep + 1, base["test"] + net(xi["test"]).numpy()
        return best, best_ep, best_test

    OUT.mkdir(parents=True, exist_ok=True)
    summary = {}
    for view in ("chem", "multi"):
        grid = {g: train(view, *g, seed=0)[:2] for g in GRID}
        g = max(grid, key=lambda k: grid[k][0])
        print(f"{view}: dev grid {{{', '.join(f'{k}: {v[0]:.4f}@{v[1]}' for k, v in grid.items())}}}", flush=True)
        runs = [train(view, *g, seed=s) for s in range(3)]
        per_sig = np.mean([correlation(Y["test"], r[2], "pearson")[1] for r in runs], 0)
        tests = [float(correlation(Y["test"], r[2], "pearson")[0]) for r in runs]
        summary[f"mlp_{view}"] = {"hidden": g[0], "dropout": g[1], "wd": g[2], "dev": [r[0] for r in runs],
                                  "epoch": [r[1] for r in runs], "test": tests, "test_seedavg_per_sig": float(per_sig.mean())}
        np.savez(OUT / f"mlp_{view}.npz", per_sig_pearson=per_sig, pert_id=X["test"][:, 0].astype(str),
                 cell_id=X["test"][:, 2].astype(str), pert_idose=X["test"][:, 3].astype(str))
        print(f"mlp_{view}: {summary[f'mlp_{view}']}", flush=True)
    ker = Path((UNDETECT_DATA + "/repro/deepce/results/xline_kernel/deepce_split"))
    ref = np.load(ker / "krr_multi_xline.npz")
    drugs_t, gi = np.unique(ref["pert_id"], return_inverse=True)
    rng = np.random.default_rng(0); W = rng.multinomial(len(drugs_t), np.full(len(drugs_t), 1 / len(drugs_t)), size=2000)
    n = np.bincount(gi).astype(float)
    for a, b in (("krr_multi_xline", "mlp_multi"), ("krr_chem_perline", "mlp_chem"), ("mlp_multi", "mlp_chem")):
        za = np.load(ker / f"{a}.npz") if a.startswith("krr") else np.load(OUT / f"{a}.npz")
        zb = np.load(OUT / f"{b}.npz")
        assert (za["pert_id"] == zb["pert_id"]).all() and (za["cell_id"] == zb["cell_id"]).all()
        s = np.bincount(gi, weights=za["per_sig_pearson"] - zb["per_sig_pearson"])
        boot = (W @ s) / (W @ n)
        summary[f"{a}-{b}"] = [float((za["per_sig_pearson"] - zb["per_sig_pearson"]).mean()),
                               float(np.quantile(boot, 0.025)), float(np.quantile(boot, 0.975)), int((s / n > 0).sum())]
        print(f"{a} - {b}: {summary[f'{a}-{b}']}", flush=True)
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")


if __name__ == "__main__":
    main()
