"""Compound kernels: one similarity matrix per view of a compound, weighted and fused into one kernel."""
from __future__ import annotations

import numpy as np
import pandas as pd
from rdkit import Chem, DataStructs, RDLogger
from rdkit.Chem import rdFingerprintGenerator

RDLogger.DisableLog("rdApp.*")
VIEWS = ("chem", "tgt", "moa", "morph")
VIEW_NAMES = {"chem": "Structure", "tgt": "Targets", "moa": "Mechanism of action", "morph": "Cell Painting"}


def tanimoto(F: np.ndarray) -> np.ndarray:
    """Tanimoto similarity of binary feature rows; 0 for a row without features."""
    F = F.astype(np.float64)
    inter = F @ F.T
    n = F.sum(1)
    union = n[:, None] + n[None, :] - inter
    return np.divide(inter, union, out=np.zeros_like(inter), where=union > 0)


def morgan_fingerprints(smiles: pd.Series, radius: int = 2, bits: int = 2048) -> np.ndarray:
    gen = rdFingerprintGenerator.GetMorganGenerator(radius=radius, fpSize=bits)
    F = np.zeros((len(smiles), bits), np.uint8)
    for i, s in enumerate(smiles):
        mol = Chem.MolFromSmiles(s) if s else None
        if mol is not None:
            DataStructs.ConvertToNumpyArray(gen.GetFingerprint(mol), F[i])
    return F


def multi_hot(column: pd.Series) -> np.ndarray:
    """One column per distinct item of ';'-separated sets (targets, mechanisms)."""
    sets = [set(x.split(";")) - {""} for x in column]
    vocab = {x: j for j, x in enumerate(sorted(set().union(*sets)))}
    M = np.zeros((len(sets), len(vocab)), np.uint8)
    for i, s in enumerate(sets):
        M[i, [vocab[x] for x in s]] = 1
    return M


def compound_kernels(compounds: pd.DataFrame, morph_keys: np.ndarray, morph: np.ndarray) -> dict[str, np.ndarray]:
    """The four view kernels over the rows of `compounds` (canonical_smiles, target, moa, inchi_key).

    chem: Morgan fingerprints (radius 2, 2048 bits), Tanimoto. tgt, moa: sets of targets and of mechanisms of
    action, Tanimoto. morph: Cell Painting profiles (`morph`, one row per InChIKey in `morph_keys`), z-scored over the
    compounds that have one, cosine similarity. A compound without a view has zeros in that view's row and column.
    """
    K = {"chem": tanimoto(morgan_fingerprints(compounds.canonical_smiles)),
         "tgt": tanimoto(multi_hot(compounds.target)),
         "moa": tanimoto(multi_hot(compounds.moa))}
    row = {str(k)[:14]: i for i, k in enumerate(morph_keys)}  # match on the InChIKey skeleton
    idx = np.array([row.get(k[:14], -1) if k else -1 for k in compounds.inchi_key])
    has = idx >= 0
    Z = np.zeros((len(compounds), morph.shape[1]))
    Z[has] = morph[idx[has]]
    Z[has] = (Z[has] - Z[has].mean(0)) / np.maximum(Z[has].std(0), 1e-9)
    Z /= np.maximum(np.linalg.norm(Z, axis=1, keepdims=True), 1e-12)
    K["morph"] = Z @ Z.T
    return K


def alignment(K: np.ndarray, R: np.ndarray) -> float:
    """Centred kernel-target alignment of a compound kernel K with the compounds' responses R (compounds x genes)."""
    Kc = K - K.mean(0)[None] - K.mean(1)[:, None] + K.mean()
    Rc = R - R.mean(0)
    den = np.linalg.norm(Kc) * np.linalg.norm(Rc.T @ Rc)
    return float(np.einsum("ij,ij->", Rc.T @ Kc, Rc.T) / den) if den > 0 else 0.0


def view_weights(K: dict[str, np.ndarray], R: np.ndarray, compounds: np.ndarray) -> dict[str, float]:
    """Weight of each view: its alignment with the residual responses R of `compounds`, negatives set to 0, sum 1."""
    a = np.array([max(alignment(K[v][np.ix_(compounds, compounds)], R), 0.0) for v in VIEWS])
    w = a / a.sum() if a.sum() > 0 else np.full(len(VIEWS), 1 / len(VIEWS))
    return dict(zip(VIEWS, map(float, w)))


def fuse(K: dict[str, np.ndarray], w: dict[str, float]) -> np.ndarray:
    """Weighted sum of the view kernels, scaled to a unit diagonal so that compounds with missing views keep unit
    self-similarity."""
    Kf = sum(w[v] * K[v] for v in VIEWS)
    d = np.sqrt(np.clip(np.diag(Kf), 1e-12, None))
    return Kf / d[:, None] / d[None, :]
