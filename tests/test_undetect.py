"""Unit tests of the UnDetect package (scripts/undetect): metric, kernels, closed forms and one small fold."""
import numpy as np
import pandas as pd
import pytest

from undetect.kernels import VIEWS, alignment, compound_kernels, fuse, tanimoto
from undetect.metrics import pcc_deg
from undetect.model import KernelRidge, fit_fold, split


def test_pcc_deg_is_a_per_sample_correlation():
    rng = np.random.default_rng(0)
    y, p = rng.normal(size=(5, 50)), rng.normal(size=(5, 50))
    assert np.allclose(pcc_deg(y, p), [np.corrcoef(a, b)[0, 1] for a, b in zip(y, p)])
    assert np.allclose(pcc_deg(y, y), 1)
    assert np.allclose(pcc_deg(y, -y), -1)
    assert np.allclose(pcc_deg(y, 3 * y + 2), 1)  # scale and shift do not matter


def test_tanimoto():
    F = np.array([[1, 1, 0], [1, 0, 0], [0, 0, 0]])
    K = tanimoto(F)
    assert np.allclose(K, K.T)
    assert K[0, 0] == 1 and K[0, 1] == 0.5
    assert not K[2].any()  # a compound without features has no similarity


def test_compound_kernels_and_fusion():
    compounds = pd.DataFrame({
        "canonical_smiles": ["CCO", "CCN", "c1ccccc1O"],
        "target": ["A;B", "B", "C"],
        "moa": ["x", "x", "y"],
        "inchi_key": ["AAAAAAAAAAAAAA-X", "BBBBBBBBBBBBBB-X", "CCCCCCCCCCCCCC-X"]})
    K = compound_kernels(compounds, np.array(["AAAAAAAAAAAAAA-Y", "CCCCCCCCCCCCCC-Y"]), np.array([[1.0, 2.0], [3.0, 1.0]]))
    assert set(K) == set(VIEWS)
    assert K["moa"][0, 1] == 1 and K["moa"][0, 2] == 0
    assert not K["morph"][1].any()  # no Cell Painting profile for the second compound
    Kf = fuse(K, dict.fromkeys(VIEWS, 0.25))
    assert np.allclose(np.diag(Kf), 1)  # missing views do not shrink a compound's self-similarity


def test_alignment_prefers_the_kernel_of_the_responses():
    rng = np.random.default_rng(1)
    R = rng.normal(size=(30, 10))
    assert alignment(R @ R.T, R) > 0.99
    assert abs(alignment(np.eye(30), R)) < alignment(R @ R.T, R)


def test_leave_one_out_matches_refitting():
    rng = np.random.default_rng(2)
    X = rng.normal(size=(12, 4))
    K, C, lam = X @ X.T + 0.1 * np.eye(12), rng.normal(size=(12, 3)), 0.5
    loo = KernelRidge(K, C).leave_one_out(lam)
    for i in range(12):
        keep = np.arange(12) != i
        coef = np.linalg.solve(K[np.ix_(keep, keep)] + lam * np.eye(11), C[keep])
        assert np.allclose(loo[i], K[i, keep] @ coef)


def test_split_holds_out_whole_groups():
    group = np.repeat(np.arange(5), 4)
    for f in range(5):
        train, valid, test = split(group, f)
        assert not (train & valid).any() and not (train & test).any() and not (valid & test).any()
        assert set(group[test]) == {f} and set(group[valid]) == {(f + 1) % 5}


@pytest.mark.parametrize("fold", [0])
def test_fit_fold_runs_on_a_small_problem(fold):
    rng = np.random.default_rng(3)
    n_compounds, genes = 25, 20
    compounds = pd.DataFrame({
        "canonical_smiles": ["C" * (i % 7 + 1) + "O" * (i % 3) for i in range(n_compounds)],
        "target": [f"T{i % 4}" for i in range(n_compounds)],
        "moa": [f"M{i % 5}" for i in range(n_compounds)],
        "inchi_key": [f"{i:014d}-X" for i in range(n_compounds)]})
    K = compound_kernels(compounds, compounds.inchi_key.to_numpy()[:10], rng.normal(size=(10, 6)))
    compound = np.repeat(np.arange(n_compounds), 3)
    line = np.tile(np.arange(3), n_compounds)
    group = (compound % 5).astype(np.int8)
    ctl = rng.normal(size=(len(compound), genes)).astype(np.float32)
    Y = rng.normal(size=(len(compound), genes)).astype(np.float32)
    fit = fit_fold(ctl, Y, compound, line, group, K, fold, seed=0, epochs=2, patience=1)
    for P in (fit.backbone, fit.first_stage, fit.final):
        assert P.shape == Y.shape and np.isfinite(P).all()
    assert abs(sum(fit.settings["view_weights"].values()) - 1) < 1e-9
