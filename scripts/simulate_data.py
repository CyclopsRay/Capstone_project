#!/usr/bin/env python3
"""Simulate the demo's data: real compounds, simulated responses with a known answer.

The compounds are real: 240 compounds from 24 mechanism-of-action (MOA) classes of the CLUE compound table, with
their structures, targets and MOA. Their Cell Painting profiles and all responses are simulated. A response is the
change in expression of 978 genes when a compound treats a cell line on an assay plate:

  delta_i = context(ctl_i) + plate_i + s_line * (compound_d + line compound_d,line) * gate(ctl_i) + noise_i

  context   a fixed nonlinear function of the plate's control profile ctl_i, the same for every compound
  plate     an effect shared by the treated samples of one plate
  compound  drawn from Gaussian processes whose covariances are the four compound kernels of the model
            (structure, targets, MOA, Cell Painting), mixed with `view_weights`; the line-specific term is drawn the
            same way for each line; s_line is the line's sensitivity
  gate      a per-gene rescaling by the control profile's top principal components

The model is therefore well specified: the demo checks the pipeline and the code, not the model's assumptions.
Compounds are split into 5 groups: fold f tests group f.

Writes to data/processed/ (config/config.yaml):
  responses.npz    ctl, delta (float32, samples x genes), compound, line, plate, group, genes, lines
  compounds.tsv    one row per compound: compound, pert_id, cmap_name, target, moa, canonical_smiles, inchi_key
  cell_painting.npz  inchikey, profiles, features: the compounds with a Cell Painting profile
  truth.npz        the parts of every response (context, compound_effect, noise) and the view weights; for
                   evaluation only, never seen by the model
  manifest.json    sizes, inputs and settings
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from rdkit import Chem

from undetect.kernels import VIEWS, compound_kernels

ROOT = Path(__file__).resolve().parents[1]


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
    c = c[~c.inchi_key.str[:14].duplicated(keep=False)]  # one compound per structure
    c = c[c.canonical_smiles.map(lambda s: Chem.MolFromSmiles(s) is not None)]
    counts = c.moa.value_counts()
    classes = sorted((m for m, n in counts.items() if n >= per_class), key=lambda m: (-counts[m], m))[:n_classes]
    if len(classes) < n_classes:
        raise SystemExit(f"only {len(classes)} MOA classes have {per_class} compounds; lower the settings")
    pick = np.concatenate([rng.choice(c.index[c.moa == m].to_numpy(), per_class, replace=False) for m in classes])
    out = c.loc[pick].sort_values("pert_id").reset_index(drop=True)
    out.insert(0, "compound", np.arange(len(out)))
    return out


def psd_sqrt(K: np.ndarray) -> np.ndarray:
    w, V = np.linalg.eigh((K + K.T) / 2)
    return (V * np.sqrt(np.clip(w, 0, None))) @ V.T


def main() -> None:
    cfg = yaml.safe_load((ROOT / "config" / "config.yaml").read_text())
    s = cfg["simulation"]
    rng = np.random.default_rng(s["seed"])
    out = ROOT / cfg["data"]["processed"]
    G, L, P = s["genes"], s["cell_lines"], s["plates_per_line"]
    w = {v: float(s["view_weights"][v]) for v in VIEWS}
    assert abs(sum(w.values()) - 1) < 1e-9, "view_weights must sum to 1"

    # Compounds (real), their simulated Cell Painting profiles (MOA class centroid plus noise) and kernels
    source = ROOT / cfg["data"]["compound_table"]
    compounds = select_compounds(source, s["moa_classes"], s["compounds_per_class"], rng)
    n = len(compounds)
    cls = pd.factorize(compounds.moa, sort=True)[0]
    nf = s["morph_features"]
    centroid = rng.normal(0, 1, (cls.max() + 1, nf))
    morph = (centroid[cls] + rng.normal(0, 0.8, (n, nf))).astype(np.float32)
    has_morph = rng.random(n) < s["cell_painting_share"]
    keys = compounds.inchi_key.to_numpy()[has_morph].astype(str)
    K = compound_kernels(compounds, keys, morph[has_morph])
    root_K = {v: psd_sqrt(K[v]) for v in VIEWS}

    # Compound effects: Gaussian-process draws over compounds, mapped to genes through shared gene programs
    k = s["gene_programs"]
    A = rng.normal(0, 1 / np.sqrt(k), (k, G))

    def compound_draw() -> np.ndarray:
        return sum(np.sqrt(w[v]) * root_K[v] @ rng.normal(0, 1, (n, k)) for v in VIEWS) @ A

    E = s["compound_sd"] * compound_draw()                                           # n x G, all lines
    E_line = s["line_compound_sd"] * np.stack([compound_draw() for _ in range(L)])  # L x n x G

    # Cell lines and plates: control profiles, context response, plate effects, context gate, line sensitivity
    gene_mean = rng.normal(8, 2, G)
    line_dev = rng.normal(0, 1, (L, 8)) @ rng.normal(0, 1 / np.sqrt(8), (8, G))
    ctl_plate = gene_mean + line_dev[:, None, :] + rng.normal(0, 0.25, (L, P, G))  # L x P x G
    X = (ctl_plate - gene_mean).reshape(L * P, G)
    ctx = np.tanh(X @ rng.normal(0, 1 / np.sqrt(G), (G, 16))) @ rng.normal(0, 1, (16, G))
    ctx = (s["context_sd"] * ctx / ctx.std()).reshape(L, P, G)
    plate_effect = rng.normal(0, s["plate_sd"], (L, P, G))
    Xc = X - X.mean(0)
    Z = Xc @ np.linalg.svd(Xc, full_matrices=False)[2][:3].T
    gate = (1 + s["gate_strength"] * np.tanh((Z / Z.std(0)) @ rng.normal(0, 1, (3, G)))).reshape(L, P, G)
    sens = np.exp(rng.normal(0, s["line_sensitivity_sd"], L))

    # Samples: each compound in a few lines, on one plate of each line
    lo, hi = s["lines_per_compound"]
    rows = [(d, c, rng.integers(P)) for d in range(n) for c in rng.choice(L, rng.integers(lo, hi + 1), replace=False)]
    rows = np.array(rows)[rng.permutation(len(rows))]
    d, c, pl = rows.T
    context = ctx[c, pl] + plate_effect[c, pl]
    compound_effect = sens[c, None] * (E[d] + E_line[c, d]) * gate[c, pl]
    noise = rng.normal(0, s["noise_sd"], (len(rows), G))
    delta = context + compound_effect + noise

    # Five compound groups: compounds in order of first appearance, shuffled, split into 5
    drugs = pd.unique(d)
    np.random.RandomState(s["split_seed"]).shuffle(drugs)
    group_of = {x: g for g, grp in enumerate(np.array_split(drugs, 5)) for x in grp}
    group = np.array([group_of[x] for x in d], np.int8)

    out.mkdir(parents=True, exist_ok=True)
    np.savez(out / "responses.npz", ctl=ctl_plate[c, pl].astype(np.float32), delta=delta.astype(np.float32),
             compound=d, line=c, plate=c * P + pl, group=group,
             genes=np.array([f"gene_{j + 1:03d}" for j in range(G)]), lines=np.array([f"line_{x + 1}" for x in range(L)]))
    compounds.to_csv(out / "compounds.tsv", sep="\t", index=False)
    np.savez(out / "cell_painting.npz", inchikey=keys, profiles=morph[has_morph],
             features=np.array([f"feature_{j + 1:02d}" for j in range(nf)]))
    np.savez(out / "truth.npz", context=context.astype(np.float32), compound_effect=compound_effect.astype(np.float32),
             noise=noise.astype(np.float32), line_sensitivity=sens, views=np.array(VIEWS),
             view_weights=np.array([w[v] for v in VIEWS]))
    manifest = {
        "samples": int(len(d)), "compounds": int(n), "moa_classes": int(cls.max() + 1), "cell_lines": int(L),
        "plates": int(L * P), "genes": int(G), "with_cell_painting": int(has_morph.sum()),
        "compound_table": cfg["data"]["compound_table"],
        "compound_table_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        "settings": s,
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"simulated {len(d)} samples x {G} genes: {n} compounds in {cls.max() + 1} MOA classes, {L} cell lines, "
          f"{L * P} plates, {int(has_morph.sum())} compounds with Cell Painting", flush=True)


if __name__ == "__main__":
    main()
