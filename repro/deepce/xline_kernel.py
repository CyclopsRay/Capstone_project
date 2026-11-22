#!/usr/bin/env python3
"""Cross-line multimodal kernel model on DeepCE's own split (exploratory sanity check, 2026-10-28).

Rows are (drug, cell line, dose) signatures, as upstream read_data returns them. One joint KRR:
  K[(d,c,s), (d',c',s')] = K_drug(d, d') * k_dose(s, s') * (rho + [c == c'])
  pred = mean_cell_dose(row) + K_te,tr (K_tr + lam I)^-1 (Y_tr - mean_cell_dose)
K_drug: chem (Morgan-2 Tanimoto) alone, or a weighted sum of chem, targets, MOA and Cell Painting
kernels (zero where a drug lacks the view; weights = centred alignment with the drug-level mean
residual of the training drugs). k_dose: RBF on log10 dose with width sigma (sigma = 0: same dose
only). rho = 0 is per-line pooling only; rho > 0 lets every training drug inform every line.
(view set, sigma, rho, lambda) chosen on upstream dev Pearson; test is scored once per arm.
"""
from __future__ import annotations
import os as _os  # noqa: E402
UNDETECT_DATA = _os.environ.get("UNDETECT_DATA", "data")  # data root (README)
UNDETECT_ROOT = _os.environ.get("UNDETECT_ROOT", ".")  # repository root (README)

import argparse
import hashlib
import json
import sys
from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd
from rdkit import Chem, DataStructs
from rdkit.Chem import rdFingerprintGenerator

UP = Path(__file__).resolve().parent / "upstream/DeepCE"
sys.path.insert(0, str(UP))
from utils import correlation  # noqa: E402
from utils.data_utils import read_data  # noqa: E402

from run_deepce import FILTER  # noqa: E402

D = Path((UNDETECT_DATA + "/derived/l1000_unseen_drug"))
OUT = Path((UNDETECT_DATA + "/repro/deepce/results/xline_kernel"))
SIGMAS = (0.0, 0.25, 0.5, 1.0, 2.0, 4.0, np.inf)
RHOS = (0.0, 0.1, 0.25, 0.5, 1.0, 2.0, 4.0)
LAMS = (0.01, 0.03, 0.1, 0.3, 1.0, 3.0, 10.0, 30.0)
VIEWSETS = {"chem": ["chem"], "chem_kn": ["chem", "tgt", "moa"], "multi": ["chem", "tgt", "moa", "morph"],
            # added 05:45: availability normalisation K_ij / sqrt(K_ii K_jj) of the combined drug kernel (PSD-preserving)
            "chem_kn_norm": ["chem", "tgt", "moa"], "multi_norm": ["chem", "tgt", "moa", "morph"]}


def tanimoto_bits(F: np.ndarray) -> np.ndarray:
    F = F.astype(np.float64); inter = F @ F.T; n = F.sum(1)
    union = n[:, None] + n[None, :] - inter
    return np.where(union > 0, inter / np.maximum(union, 1e-9), 0.0)


def masked(K: np.ndarray, has: np.ndarray) -> np.ndarray:
    K = K.copy(); K[~has] = 0; K[:, ~has] = 0
    return K


def alignment(K: np.ndarray, R: np.ndarray) -> float:
    Kc = K - K.mean(0)[None] - K.mean(1)[:, None] + K.mean(); Rc = R - R.mean(0)
    den = np.linalg.norm(Kc) * np.linalg.norm(Rc.T @ Rc)
    return float(np.einsum("ij,ij->", Rc.T @ Kc, Rc.T) / den) if den > 0 else 0.0


def main() -> None:
    d = UP / "data"
    smiles = dict(line.strip().split(",", 1) for line in open(d / "drugs_smiles.csv"))
    X, Y = {}, {}
    for s in ("train", "dev", "test"):
        X[s], Y[s] = read_data(str(d / f"signature_{s}.csv"), FILTER)
    drugs = sorted(set(np.concatenate([X[s][:, 0] for s in X])))
    dpos = {k: i for i, k in enumerate(drugs)}; n_d = len(drugs)
    gen = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)
    F = np.zeros((n_d, 2048), np.uint8)
    for k in drugs:
        DataStructs.ConvertToNumpyArray(gen.GetFingerprint(Chem.MolFromSmiles(smiles[k])), F[dpos[k]])
    Kd = {"chem": tanimoto_bits(F)}
    table = pd.read_csv(OUT.parent.parent / "drug_table_l1000.tsv", sep="\t", keep_default_na=False)  # exported from drug_table.parquet (no pyarrow here)
    table = table[table.source.eq("L1000") & table.source_id.isin(drugs)]
    tg = {k: set() for k in drugs}; mo = {k: set() for k in drugs}; ik = {}
    for r in table.itertuples():
        if r.targets:
            tg[r.source_id].update(r.targets.split(";"))
        if r.moa:
            mo[r.source_id].update(r.moa.split(";"))
        ik.setdefault(r.source_id, r.inchikey)
    for name, sets in (("tgt", tg), ("moa", mo)):
        vocab = sorted({x for v in sets.values() for x in v}); vp = {x: i for i, x in enumerate(vocab)}
        M = np.zeros((n_d, max(1, len(vocab))), np.uint8)
        for k, v in sets.items():
            M[dpos[k], [vp[x] for x in v]] = 1
        Kd[name] = masked(tanimoto_bits(M), M.any(1))
    cp = np.load(D / "cell_painting_features.npz", allow_pickle=False)
    cpos = pd.Series(np.arange(len(cp["inchikey"])), index=cp["inchikey"].astype(str))
    has_m = np.array([ik.get(k, "") in cpos.index for k in drugs])
    Fm = np.zeros((n_d, cp["morph10"].shape[1]))
    Fm[has_m] = cp["morph10"][cpos[[ik[k] for k in np.array(drugs)[has_m]]].to_numpy()]
    Fm[has_m] = (Fm[has_m] - Fm[has_m].mean(0)) / np.maximum(Fm[has_m].std(0), 1e-9)
    nrm = np.maximum(np.linalg.norm(Fm, axis=1, keepdims=True), 1e-12)
    Kd["morph"] = masked((Fm / nrm) @ (Fm / nrm).T, has_m)
    test_drugs = set(X["test"][:, 0])
    cover = {v: int(sum(Kd[v][dpos[k], dpos[k]] > 0 for k in test_drugs)) for v in Kd}
    print(f"drugs {n_d} (test {len(test_drugs)}); test-drug view coverage {cover}", flush=True)

    if ARGS.cv:
        allX = np.concatenate([X[s] for s in ("train", "dev", "test")]); allY = np.concatenate([Y[s] for s in ("train", "dev", "test")])
        fold = np.array([int(hashlib.md5(f"cv{ARGS.seed}|{k}".encode()).hexdigest(), 16) % ARGS.cv for k in allX[:, 0]])
        inner = np.array([int(hashlib.md5(f"dev{ARGS.seed}|{k}".encode()).hexdigest(), 16) % 5 for k in allX[:, 0]])
        per_fold = []
        for f in range(ARGS.cv):
            te = fold == f; dv = ~te & (inner == 0); trm = ~te & ~dv
            Xf = {"train": allX[trm], "dev": allX[dv], "test": allX[te]}; Yf = {"train": allY[trm], "dev": allY[dv], "test": allY[te]}
            per_fold.append(run_split(Xf, Yf, Kd, dpos, OUT / f"cv{ARGS.cv}_seed{ARGS.seed}" / f"fold{f}"))
        rows = [{"fold": f, "arm": a, **v} for f, arms in enumerate(per_fold) for a, v in arms.items()]
        df = pd.DataFrame(rows); df.to_csv(OUT / f"cv{ARGS.cv}_seed{ARGS.seed}" / "folds.csv", index=False)
        print(df.groupby("arm")[["dev", "test"]].agg(["mean", "std"]).round(4).to_string(), flush=True)
        print("folds won vs krr_chem_perline:", {a: int((df[df.arm == a].test.to_numpy() > df[df.arm == "krr_chem_perline"].test.to_numpy()).sum())
                                                 for a in df.arm.unique()}, flush=True)
    else:
        arms = run_split(X, Y, Kd, dpos, OUT / "deepce_split")
        compare(OUT / "deepce_split")


def run_split(X, Y, Kd, dpos, out: Path) -> dict:
    tr = X["train"]; ytr = Y["train"]
    m_cell = {c: ytr[tr[:, 2] == c].mean(0) for c in set(tr[:, 2])}
    m_cd = {(c, s): ytr[(tr[:, 2] == c) & (tr[:, 3] == s)].mean(0) for c, s in set(map(tuple, tr[:, 2:4]))}
    base = {s: np.array([m_cd.get((r[2], r[3]), m_cell[r[2]]) for r in X[s]]) for s in X}
    R = ytr - base["train"]
    di = {s: np.array([dpos[k] for k in X[s][:, 0]]) for s in X}
    ldose = {s: np.log10(np.array([float(x.split()[0]) for x in X[s][:, 3]])) for s in X}
    tdr = sorted(set(di["train"]))
    Rd = np.stack([R[di["train"] == j].mean(0) for j in tdr])
    align = {v: alignment(Kd[v][np.ix_(tdr, tdr)], Rd) for v in Kd}
    print(f"{out.name}: train/dev/test {len(tr)}/{len(X['dev'])}/{len(X['test'])}; alignment",
          {v: round(a, 4) for v, a in align.items()}, flush=True)

    def row_kernel(views, sigma, rho, a, b, norm=False):
        w = np.array([max(align[v], 0) for v in views]); w = w / w.sum()
        Kfull = sum(wi * Kd[v] for wi, v in zip(w, views))
        if norm:
            dg = np.sqrt(np.clip(np.diag(Kfull), 1e-12, None)); Kfull = Kfull / dg[:, None] / dg[None, :]
        K = Kfull[np.ix_(di[a], di[b])]
        dd = ldose[a][:, None] - ldose[b][None, :]
        if sigma == 0:
            K = K * (dd == 0)
        elif np.isfinite(sigma):
            K = K * np.exp(-dd ** 2 / (2 * sigma ** 2))
        same = X[a][:, 2][:, None] == X[b][:, 2][None, :]
        return K * (rho + same)

    results = []
    for vs, sigma, rho in product(VIEWSETS, SIGMAS, RHOS):
        Ktr = row_kernel(VIEWSETS[vs], sigma, rho, "train", "train", vs.endswith("_norm"))
        vals, vecs = np.linalg.eigh(Ktr); vals = np.clip(vals, 0, None); proj = vecs.T @ R
        Kdev = row_kernel(VIEWSETS[vs], sigma, rho, "dev", "train", vs.endswith("_norm"))
        for lam in LAMS:
            alpha = vecs @ (proj / (vals + lam)[:, None])
            dev = correlation(Y["dev"], base["dev"] + Kdev @ alpha, "pearson")[0]
            results.append({"views": vs, "sigma": sigma, "rho": rho, "lam": lam, "dev": float(dev)})
    res = pd.DataFrame(results)
    out.mkdir(parents=True, exist_ok=True)
    res.to_csv(out / "dev_grid.csv", index=False)
    # arms: best per view set with rho = 0 (per-line) and rho free (cross-line); test scored once each
    summary = {"n_train": len(tr), "n_dev": len(X["dev"]), "n_test": len(X["test"]), "alignment": align, "arms": {}}
    pear, _ = correlation(Y["test"], base["test"], "pearson")
    summary["arms"]["mean_cell_dose"] = {"dev": float(correlation(Y["dev"], base["dev"], "pearson")[0]), "test": float(pear)}
    for vs in VIEWSETS:
        for tag, sub in (("perline", res[(res.views == vs) & (res.rho == 0)]), ("xline", res[res.views == vs])):
            r = sub.loc[sub.dev.idxmax()]
            nm = vs.endswith("_norm")
            Ktr = row_kernel(VIEWSETS[vs], r.sigma, r.rho, "train", "train", nm)
            alpha = np.linalg.solve(Ktr + r.lam * np.eye(len(Ktr)), R)
            P = base["test"] + row_kernel(VIEWSETS[vs], r.sigma, r.rho, "test", "train", nm) @ alpha
            pear, per_sig = correlation(Y["test"], P, "pearson")
            arm = f"krr_{vs}_{tag}"
            summary["arms"][arm] = {"sigma": float(r.sigma), "rho": float(r.rho), "lam": float(r.lam),
                                    "dev": float(r.dev), "test": float(pear)}
            np.savez(out / f"{arm}.npz", pred=P, label=Y["test"], per_sig_pearson=np.asarray(per_sig),
                     pert_id=X["test"][:, 0].astype(str), cell_id=X["test"][:, 2].astype(str),
                     pert_idose=X["test"][:, 3].astype(str))
            print(f"  {arm}: sigma {r.sigma} rho {r.rho} lam {r.lam} dev {r.dev:.4f} test {pear:.4f}", flush=True)
    (out / "summary.json").write_text(json.dumps(summary, indent=2, default=str) + "\n")
    return summary["arms"]


def compare(out: Path) -> None:
    """Paired per-signature differences, bootstrap over test drugs (as analyze.py), vs baselines and DeepCE runs."""
    res = Path((UNDETECT_DATA + "/repro/deepce/results"))
    arms = {p.stem: np.load(p)["per_sig_pearson"] for p in sorted(out.glob("krr_*.npz"))}
    ref = np.load(out / "krr_multi_xline.npz")
    for b in ("knn_morgan", "mean_cell_dose"):
        z = np.load(res / "baselines" / f"{b}.npz"); assert (z["pert_id"] == ref["pert_id"]).all(); arms[b] = z["per_sig_pearson"]
    for v in ("ppi", "onehot"):
        zs = [np.load(res / f"{v}_seed{s}" / "test_pred_bestdev.npz") for s in range(3)]
        assert all((z["pert_id"] == ref["pert_id"]).all() and (z["cell_id"] == ref["cell_id"]).all() for z in zs)
        arms[f"deepce_{v}_3seed"] = np.mean([z["per_sig_pearson"] for z in zs], 0)
    drugs, g = np.unique(ref["pert_id"], return_inverse=True)
    rng = np.random.default_rng(0); W = rng.multinomial(len(drugs), np.full(len(drugs), 1 / len(drugs)), size=2000)
    n = np.bincount(g, minlength=len(drugs)).astype(float)
    rows = []
    for a, b in [("krr_multi_xline", "deepce_ppi_3seed"), ("krr_multi_xline", "deepce_onehot_3seed"),
                 ("krr_multi_xline", "knn_morgan"), ("krr_chem_xline", "deepce_ppi_3seed"),
                 ("krr_chem_perline", "deepce_ppi_3seed"), ("krr_chem_perline", "knn_morgan"),
                 ("krr_chem_xline", "krr_chem_perline"), ("krr_chem_kn_xline", "krr_chem_kn_perline"),
                 ("krr_multi_xline", "krr_multi_perline"), ("krr_chem_kn_xline", "krr_chem_xline"),
                 ("krr_multi_xline", "krr_chem_kn_xline"), ("krr_multi_xline", "krr_chem_xline"),
                 ("krr_multi_norm_xline", "krr_multi_xline"), ("krr_chem_kn_norm_xline", "krr_chem_kn_xline"),
                 ("krr_multi_norm_xline", "deepce_ppi_3seed"), ("krr_multi_norm_xline", "krr_chem_xline")]:
        dlt = arms[a] - arms[b]; s = np.bincount(g, weights=dlt, minlength=len(drugs))
        boot = (W @ s) / (W @ n)
        rows.append({"a": a, "b": b, "diff": float(dlt.mean()), "lo": float(np.quantile(boot, 0.025)),
                     "hi": float(np.quantile(boot, 0.975)), "drugs_a_better": int((s / n > 0).sum())})
    df = pd.DataFrame(rows); df.to_csv(out / "contrasts.csv", index=False)
    print({k: round(float(v.mean()), 4) for k, v in arms.items()}, flush=True)
    print(df.round(4).to_string(), flush=True)

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--cv", type=int, default=0, help="0 = DeepCE's own split; k = k-fold drug-disjoint resplit of all rows")
    ap.add_argument("--seed", type=int, default=0)
    ARGS = ap.parse_args()
    main()
