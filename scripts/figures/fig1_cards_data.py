"""Inputs for the compound-card Fig. 1: view kernels, fused + normalised kernel and 2D structures of three compounds.

Compounds: daunorubicin (BRD-A37630846, the entry with the most profiled samples; four other BRD entries exist),
mitoxantrone, vorinostat. Kernels exactly as `run_sdst_hybrid.drug_kernels` (Morgan r2/2048 Tanimoto; Tanimoto of
target and MoA sets; cosine of z-scored Cell Painting profiles, z-scored over every benchmark compound that has one;
zero where a view is missing). Fusion weights = mean of the five `cache_v1` fold KTA weights.
Writes <outdir>/fig1_pipeline_source.json (kernels, weights, annotations, Morgan on-bits, atoms and bonds with 2D
coordinates) and
<outdir>/fig1_pipeline_source.csv (every plotted kernel value).
Run: $UNDETECT_DATA/envs/undetect/bin/python -I scripts/figures/fig1_cards_data.py [outdir]
"""
import os as _os  # noqa: E402
UNDETECT_DATA = _os.environ.get("UNDETECT_DATA", "data")  # data root (README)
UNDETECT_ROOT = _os.environ.get("UNDETECT_ROOT", ".")  # repository root (README)
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from rdkit import Chem, DataStructs
from rdkit.Chem import rdDepictor, rdFingerprintGenerator

D = Path((UNDETECT_DATA + "/derived/xpert_sdst/v1"))
CP = Path((UNDETECT_DATA + "/derived/l1000_unseen_drug/cell_painting_features.npz"))
CACHE = Path((UNDETECT_DATA + "/results/sdst_hybrid/cache_v1"))
PERT_IDS = {"daunorubicin": "BRD-A37630846", "mitoxantrone": "BRD-K21680192", "vorinostat": "BRD-K81418486"}
VIEWS = ("chem", "tgt", "moa", "morph")


def structure(smiles: str) -> dict:
    """Heavy-atom 2D layout (CoordGen), kekulised, centred and scaled to a bond length of 1."""
    m = Chem.MolFromSmiles(smiles)
    Chem.Kekulize(m, clearAromaticFlags=True)
    rdDepictor.SetPreferCoordGen(True)
    rdDepictor.Compute2DCoords(m)
    xy = m.GetConformer().GetPositions()[:, :2]
    bl = np.median([np.linalg.norm(xy[b.GetBeginAtomIdx()] - xy[b.GetEndAtomIdx()]) for b in m.GetBonds()])
    xy = (xy - xy.mean(0)) / bl
    atoms = [{"el": a.GetSymbol(), "h": a.GetTotalNumHs(), "x": float(x), "y": float(y)}
             for a, (x, y) in zip(m.GetAtoms(), xy)]
    bonds = [{"i": b.GetBeginAtomIdx(), "j": b.GetEndAtomIdx(), "order": int(b.GetBondTypeAsDouble()),
              "ring": b.IsInRing()} for b in m.GetBonds()]
    return {"atoms": atoms, "bonds": bonds}


def main() -> None:
    out = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).resolve().parents[2] / "figures"
    t = pd.read_csv(D / "drugs.tsv", sep="\t", keep_default_na=False)
    cp = np.load(CP, allow_pickle=False)
    pos = {k[:14]: i for i, k in enumerate(cp["inchikey"].astype(str))}
    cpi = np.array([pos.get(k[:14], -1) if k else -1 for k in t.inchi_key])
    F = cp["morph10"][cpi[cpi >= 0]].astype(float)
    mu, sd = F.mean(0), np.maximum(F.std(0), 1e-9)
    names = list(PERT_IDS)
    rows = [int(np.flatnonzero(t.pert_id.values == PERT_IDS[n])[0]) for n in names]
    gen = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)
    fps = [gen.GetFingerprint(Chem.MolFromSmiles(t.canonical_smiles[r])) for r in rows]
    sets = {v: [set(t[col][r].split(";")) - {""} for r in rows] for v, col in (("tgt", "target"), ("moa", "moa"))}
    prof = []
    for r in rows:
        if cpi[r] < 0:
            prof.append(None)
        else:
            z = (cp["morph10"][cpi[r]] - mu) / sd
            prof.append(z / np.linalg.norm(z))
    K = {"chem": [[DataStructs.TanimotoSimilarity(a, b) for b in fps] for a in fps]}
    for v in ("tgt", "moa"):
        K[v] = [[len(a & b) / len(a | b) if a and b else 0.0 for b in sets[v]] for a in sets[v]]
    K["morph"] = [[float(a @ b) if a is not None and b is not None else 0.0 for b in prof] for a in prof]
    K = {v: np.array(K[v]) for v in VIEWS}
    w = {v: float(np.mean([json.loads((CACHE / f"fold_{f}" / "summary.json").read_text())["weights_kta"][v]
                           for f in range(5)])) for v in VIEWS}
    Kf = sum(w[v] * K[v] for v in VIEWS)
    dg = np.sqrt(np.diag(Kf))
    Kn = Kf / np.outer(dg, dg)
    res = {
        "names": names, "pert_ids": [PERT_IDS[n] for n in names], "weights": w,
        "has": {v: [bool(K[v][i, i] > 0) for i in range(3)] for v in VIEWS},
        "target": [sorted(s) for s in sets["tgt"]], "moa": [sorted(s) for s in sets["moa"]],
        "kernels": {**{v: K[v].round(4).tolist() for v in VIEWS}, "fused": Kf.round(4).tolist(),
                    "norm": Kn.round(4).tolist()},
        "structures": [structure(t.canonical_smiles[r]) for r in rows],
        "fp_bits": [sorted(int(b) for b in fp.GetOnBits()) for fp in fps],
    }
    out.mkdir(parents=True, exist_ok=True)
    (out / "fig1_pipeline_source.json").write_text(json.dumps(res, indent=1) + "\n")
    pd.DataFrame([{"kernel": k, "row": names[a], "col": names[b], "value": res["kernels"][k][a][b]}
                  for k in res["kernels"] for a in range(3) for b in range(3)]).to_csv(
        out / "fig1_pipeline_source.csv", index=False)
    print("weights", {k: round(x, 3) for k, x in w.items()})
    for k in ("chem", "tgt", "moa", "morph", "norm"):
        print(k, np.array(res["kernels"][k]).round(3).tolist())
    print("wrote", out / "fig1_pipeline_source.json", out / "fig1_pipeline_source.csv")


if __name__ == "__main__":
    main()
