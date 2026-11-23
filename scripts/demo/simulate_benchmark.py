#!/usr/bin/env python3
"""Simulated drug-blind benchmark of the same shape as the paper's data, for the demo run.

The paper's benchmark (XPert L1000 SDST: 78,453 control/treated pairs, 8,276 compounds, 164 cell lines) is a 1.6 GB
download and its runs take hours on a cluster. This script writes a small benchmark in exactly the files and formats
that the paper's scripts read (the outputs of `prep_xpert_sdst.py` and `build_cell_painting_features.py`), so that
`cache_undetect.py` and `stage2_context.py` run on it unchanged, in minutes on a laptop:

  <root>/derived/xpert_sdst/v1/sdst.npz       ctl, delta (float32, samples x genes), pert_idx, cell_idx, cell, group,
                                              genes; group = 5 compound-disjoint folds by the benchmark's own rule
  <root>/derived/xpert_sdst/v1/drugs.tsv      pert_idx, pert_id, cmap_name, target, moa, canonical_smiles, inchi_key
  <root>/derived/xpert_sdst/v1/manifest.json  inputs, settings and sizes
  <root>/derived/xpert_sdst/v1/truth.npz      the parts of every simulated response; used for scoring only, never by
                                              UnDetect
  <root>/derived/l1000_unseen_drug/cell_painting_features.npz   inchikey, morph10, feature

The compounds are real: structure, targets and mechanism of action (MOA) of 240 compounds from 24 MOA classes of the
CLUE LINCS2020 compound table. Their Cell Painting profiles and all responses are simulated, so the answer is known:

  delta_i = context(ctl_i) + plate_i + s_line * (compound_d + line compound_d,line) * gate(ctl_i) + noise_i

  context   a fixed nonlinear function of the control profile ctl_i, the same for every compound
  plate     shared by the treated samples of one assay plate
  compound  sum over the four views (structure, targets, MOA, Cell Painting) of Gaussian-process draws whose
            covariance is that view's compound kernel, weighted by `view_weights`, in a few gene programs; the
            line-specific term is drawn the same way, independently for each line
  s_line    each line's overall sensitivity
  gate      per-gene rescaling by the top principal components of the control profile

Settings: config/config.yaml (demo.simulation). Usage: python scripts/demo/simulate_benchmark.py [--config PATH]
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from rdkit import Chem, DataStructs, RDLogger
from rdkit.Chem import rdFingerprintGenerator

RDLogger.DisableLog("rdApp.*")
ROOT = Path(__file__).resolve().parents[2]
VIEWS = ("chem", "tgt", "moa", "morph")  # the order of run_sdst_hybrid.VIEWS


def select_compounds(path: Path, n_classes: int, per_class: int, rng: np.random.Generator) -> pd.DataFrame:
    """`per_class` compounds from each of the `n_classes` most common MOA classes, one row per compound."""
    t = pd.read_csv(path, sep="\t", keep_default_na=False, dtype=str)
    t = t[t.canonical_smiles.ne("") & t.inchi_key.ne("") & t.target.ne("") & t.moa.ne("")]
    union = lambda s: ";".join(sorted(set(s)))  # noqa: E731  (CLUE has one row per compound, target and MOA)
    c = (t.groupby("pert_id")
         .agg(cmap_name=("cmap_name", "first"), target=("target", union), moa=("moa", union),
              canonical_smiles=("canonical_smiles", "first"), inchi_key=("inchi_key", "first"))
         .reset_index())
    c = c[~c.moa.str.contains(";")]                      # one MOA per compound, so classes are well defined
    c = c[~c.inchi_key.str[:14].duplicated(keep=False)]  # one compound per structure (the Cell Painting join key)
    c = c[c.canonical_smiles.map(lambda s: Chem.MolFromSmiles(s) is not None)]
    counts = c.moa.value_counts()
    classes = sorted((m for m, n in counts.items() if n >= per_class), key=lambda m: (-counts[m], m))[:n_classes]
    if len(classes) < n_classes:
        raise SystemExit(f"only {len(classes)} MOA classes have {per_class} compounds; lower the settings")
    pick = np.concatenate([rng.choice(c.index[c.moa == m].to_numpy(), per_class, replace=False) for m in classes])
    out = c.loc[pick].sort_values("pert_id").reset_index(drop=True)
    out.insert(0, "pert_idx", np.arange(len(out)))
    return out


def tanimoto(F: np.ndarray) -> np.ndarray:
    F = F.astype(np.float64)
    inter = F @ F.T
    n = F.sum(1)
    union = n[:, None] + n[None, :] - inter
    return np.divide(inter, union, out=np.zeros_like(inter), where=union > 0)


def view_kernels(table: pd.DataFrame, morph: np.ndarray, has_morph: np.ndarray) -> dict[str, np.ndarray]:
    """The four compound kernels, defined as in run_sdst_hybrid.drug_kernels (zero where a view is missing)."""
    n = len(table)
    gen = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)
    F = np.zeros((n, 2048), np.uint8)
    for i, s in enumerate(table.canonical_smiles):
        DataStructs.ConvertToNumpyArray(gen.GetFingerprint(Chem.MolFromSmiles(s)), F[i])
    K = {"chem": tanimoto(F)}
    for v, col in (("tgt", "target"), ("moa", "moa")):
        sets = [set(x.split(";")) - {""} for x in table[col]]
        vocab = {x: j for j, x in enumerate(sorted(set().union(*sets)))}
        M = np.zeros((n, len(vocab)), np.uint8)
        for i, s in enumerate(sets):
            M[i, [vocab[x] for x in s]] = 1
        K[v] = tanimoto(M)
    Fm = np.zeros_like(morph, dtype=np.float64)
    z = morph[has_morph]
    z = (z - z.mean(0)) / np.maximum(z.std(0), 1e-9)
    Fm[has_morph] = z / np.maximum(np.linalg.norm(z, axis=1, keepdims=True), 1e-12)
    K["morph"] = Fm @ Fm.T
    return K


def psd_sqrt(K: np.ndarray) -> np.ndarray:
    w, V = np.linalg.eigh((K + K.T) / 2)
    return (V * np.sqrt(np.clip(w, 0, None))) @ V.T


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--config", type=Path, default=ROOT / "config" / "config.yaml")
    args = p.parse_args()
    cfg = yaml.safe_load(args.config.read_text())["demo"]
    s = cfg["simulation"]
    rng = np.random.default_rng(cfg["seed"])
    root = ROOT / cfg["data_root"]
    out = root / "derived" / "xpert_sdst" / "v1"
    out_cp = root / "derived" / "l1000_unseen_drug"
    G, L, P = s["genes"], s["cell_lines"], s["plates_per_line"]
    w = {v: float(s["view_weights"][v]) for v in VIEWS}
    assert abs(sum(w.values()) - 1) < 1e-9, "view_weights must sum to 1"

    # Compounds (real) and their simulated Cell Painting profiles: a class centroid plus noise
    source = ROOT / s["compound_table"]
    table = select_compounds(source, s["moa_classes"], s["compounds_per_class"], rng)
    n = len(table)
    cls = pd.factorize(table.moa, sort=True)[0]
    nf = s["morph_features"]
    centroid = rng.normal(0, 1, (cls.max() + 1, nf))
    morph = (centroid[cls] + rng.normal(0, 0.8, (n, nf))).astype(np.float32)
    has_morph = rng.random(n) < s["cell_painting_share"]
    K = view_kernels(table, morph, has_morph)
    root_K = {v: psd_sqrt(K[v]) for v in VIEWS}

    # Compound effects: Gaussian-process draws over compounds, mapped to genes through shared gene programs
    k = s["gene_programs"]
    A = rng.normal(0, 1 / np.sqrt(k), (k, G))

    def compound_draw() -> np.ndarray:
        return sum(np.sqrt(w[v]) * root_K[v] @ rng.normal(0, 1, (n, k)) for v in VIEWS) @ A

    E = s["compound_sd"] * compound_draw()                              # n x G, all lines
    E_line = s["line_compound_sd"] * np.stack([compound_draw() for _ in range(L)])  # L x n x G

    # Cell lines and plates: control profiles, the context response, plate effects and the context gate
    gene_mean = rng.normal(8, 2, G)
    line_dev = rng.normal(0, 1, (L, 8)) @ rng.normal(0, 1 / np.sqrt(8), (8, G))
    ctl_plate = gene_mean + line_dev[:, None, :] + rng.normal(0, 0.25, (L, P, G))   # L x P x G
    X = (ctl_plate - gene_mean).reshape(L * P, G)
    ctx = np.tanh(X @ rng.normal(0, 1 / np.sqrt(G), (G, 16))) @ rng.normal(0, 1, (16, G))
    ctx = (s["context_sd"] * ctx / ctx.std()).reshape(L, P, G)
    plate = rng.normal(0, s["plate_sd"], (L, P, G))
    Xc = X - X.mean(0)
    Z = Xc @ np.linalg.svd(Xc, full_matrices=False)[2][:3].T
    gate = (1 + s["gate_strength"] * np.tanh((Z / Z.std(0)) @ rng.normal(0, 1, (3, G)))).reshape(L, P, G)
    sens = np.exp(rng.normal(0, s["line_sensitivity_sd"], L))

    # Samples: each compound in a few lines, on one plate of each line
    lo, hi = s["lines_per_compound"]
    rows = [(d, c, rng.integers(P)) for d in range(n) for c in rng.choice(L, rng.integers(lo, hi + 1), replace=False)]
    rows = np.array(rows)[rng.permutation(len(rows))]
    d, c, pl = rows.T
    context = ctx[c, pl] + plate[c, pl]
    compound = sens[c, None] * (E[d] + E_line[c, d]) * gate[c, pl]
    noise = rng.normal(0, s["noise_sd"], (len(rows), G))
    delta = context + compound + noise
    ctl = ctl_plate[c, pl]

    # Folds: the benchmark's rule (prep_xpert_sdst.py): compounds in order of first appearance, shuffled, 5 groups
    drugs = pd.unique(d)
    np.random.RandomState(s["split_seed"]).shuffle(drugs)
    group_of = {x: g for g, grp in enumerate(np.array_split(drugs, 5)) for x in grp}
    group = np.array([group_of[x] for x in d], np.int8)

    out.mkdir(parents=True, exist_ok=True)
    out_cp.mkdir(parents=True, exist_ok=True)
    genes = np.array([f"gene_{j + 1:03d}" for j in range(G)])
    cells = np.array([f"line_{x + 1}" for x in range(L)])
    np.savez(out / "sdst.npz", ctl=ctl.astype(np.float32), delta=delta.astype(np.float32), pert_idx=d,
             cell_idx=c, cell=cells[c], group=group, genes=genes)
    table.to_csv(out / "drugs.tsv", sep="\t", index=False)
    np.savez(out / "truth.npz", context=context.astype(np.float32), compound=compound.astype(np.float32),
             noise=noise.astype(np.float32), plate_idx=c * P + pl, line_sensitivity=sens,
             view_weights=np.array([w[v] for v in VIEWS]), views=np.array(VIEWS))
    np.savez(out_cp / "cell_painting_features.npz", inchikey=table.inchi_key.to_numpy()[has_morph].astype(str),
             morph10=morph[has_morph], feature=np.array([f"feature_{j + 1:02d}" for j in range(nf)]))
    manifest = {
        "simulated": True, "description": "simulated benchmark of the same shape as XPert L1000 SDST (demo run)",
        "compound_source": str(source.relative_to(ROOT)),
        "compound_source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        "n": int(len(d)), "drugs": int(n), "cells": int(L), "plates": int(L * P), "genes": int(G),
        "moa_classes": int(cls.max() + 1), "with_cell_painting": int(has_morph.sum()),
        "seed": cfg["seed"], "split_rule": "prep_xpert_sdst.py: first appearance, RandomState(split_seed), 5 groups",
        "settings": s,
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    var = {k_: float(v.var()) for k_, v in (("context", context), ("compound", compound), ("noise", noise))}
    print(f"simulated {len(d)} samples x {G} genes: {n} compounds in {cls.max() + 1} MOA classes, {L} lines, "
          f"{L * P} plates, {int(has_morph.sum())} compounds with Cell Painting; variance {var}", flush=True)


if __name__ == "__main__":
    main()
