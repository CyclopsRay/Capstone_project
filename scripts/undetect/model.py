"""UnDetect on one fold: the drug-free backbone, the compound terms of the multi-view kernel, and the context gate.

Every setting after the backbone is chosen on the validation compounds by PCC_DEG; the test compounds are scored
once. On training rows the compound terms are leave-one-compound-out values: a compound's own responses never enter
its own prediction.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from itertools import product

import numpy as np

from .backbone import train_backbone
from .kernels import fuse, view_weights
from .metrics import score

LAMBDAS = (0.1, 0.3, 1.0, 3.0, 10.0, 30.0, 100.0)  # ridge strengths of the kernel regressions
SCALES = (0.0, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 2.0)  # weights of the compound terms
GATE_RIDGE = (0.001, 0.01, 0.1, 1.0, 10.0, np.inf)  # ridge of the context gate (relative); inf: no gate
GATE_PCS = (5, 10, 20)  # principal components of the control profile used by the gate


@dataclass
class FoldFit:
    """Predictions for every sample, by stage of the model, and the settings chosen on the validation compounds."""
    backbone: np.ndarray     # drug-free backbone
    first_stage: np.ndarray  # backbone + shared and per-line compound terms
    final: np.ndarray        # first stage with the context gate: UnDetect
    settings: dict = field(default_factory=dict)


def split(group: np.ndarray, fold: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Fold f tests compound group f, validates on group (f + 1) mod 5 and trains on the other three."""
    test = group == fold
    valid = group == (fold + 1) % 5
    return ~test & ~valid, valid, test


def compound_means(compound: np.ndarray, R: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """The distinct compounds, each row's index among them, and each compound's mean row of R."""
    ids, inv = np.unique(compound, return_inverse=True)
    M = np.zeros((len(ids), R.shape[1]))
    np.add.at(M, inv, R)
    return ids, inv, M / np.bincount(inv)[:, None]


class KernelRidge:
    """Kernel ridge regression of C (compounds x genes) on kernel K, through one eigendecomposition for all ridges."""

    def __init__(self, K: np.ndarray, C: np.ndarray):
        vals, self.vecs = np.linalg.eigh(K)
        self.vals = np.clip(vals, 0, None)
        self.C, self.proj = C, self.vecs.T @ C

    def coef(self, lam: float) -> np.ndarray:
        return self.vecs @ (self.proj / (self.vals + lam)[:, None])

    def leave_one_out(self, lam: float) -> np.ndarray:
        """Each compound's prediction from the other compounds only (closed form)."""
        shrink = self.vals / (self.vals + lam)
        lev = (self.vecs ** 2) @ shrink
        fit = self.vecs @ (shrink[:, None] * self.proj)
        return (fit - lev[:, None] * self.C) / (1 - lev)[:, None]


def context_gate(ctl, R, U, P, Y, train, valid):
    """Second stage: per gene j, the compound term u_ij is rescaled by a ridge model on the control's top principal
    components z_i, r_ij - u_ij ~ u_ij (d_j + g_j z_i), fitted on training rows. Ridge and number of components are
    chosen on valid; the largest ridge (inf) leaves the first stage unchanged."""
    mu = ctl[train].mean(0)
    Xc = ctl[train] - mu
    plates = np.unique(Xc.round(5), axis=0, return_index=True)[1]  # one row per distinct control (assay plate)
    Z = (ctl - mu) @ np.linalg.svd(Xc[plates], full_matrices=False)[2][:max(GATE_PCS)].T
    Z /= Z[train].std(0) + 1e-9
    U_tr, T = U[train], R[train] - U[train]
    scale = float((U_tr ** 2).sum(0).mean())
    if scale == 0:  # no compound term was chosen: nothing to gate
        return P, {"gate_ridge": np.inf, "gate_pcs": None, "gate_valid": score(Y[valid], P[valid])}
    best = (-2.0, None, None, None)
    for c, k in [(c, k) for c in GATE_RIDGE if np.isfinite(c) for k in GATE_PCS] + [(np.inf, None)]:
        if k is None:
            P2 = P
        else:
            F = np.hstack([np.ones((len(ctl), 1)), Z[:, :k]])
            Ft = F[train]
            p = F.shape[1]
            A = ((U_tr * U_tr).T @ (Ft[:, :, None] * Ft[:, None, :]).reshape(len(Ft), p * p)).reshape(-1, p, p)
            W = np.linalg.solve(A + c * scale * np.eye(p)[None], ((U_tr * T).T @ Ft)[:, :, None])[:, :, 0].T
            P2 = P + U * (F @ W)
        v = score(Y[valid], P2[valid])
        if v > best[0]:
            best = (v, c, k, P2)
    return best[3], {"gate_ridge": best[1], "gate_pcs": best[2], "gate_valid": best[0]}


def fit_fold(ctl, Y, compound, line, group, K, fold, seed=1729, epochs=500, patience=50) -> FoldFit:
    """Fit UnDetect on the training compounds of one fold; predict every sample.

    ctl, Y: control profile and response (samples x genes); compound, line, group: each sample's compound (row of
    the kernels), cell line and compound group (0-4); K: the view kernels (kernels.compound_kernels).
    """
    train, valid, test = split(group, fold)
    Y = Y.astype(np.float64)

    # 1. Context: the drug-free backbone
    predict, backbone_valid, epoch = train_backbone(ctl[train], Y[train], ctl[valid], Y[valid], seed, epochs,
                                                    patience)
    B = predict(ctl).astype(np.float64)
    R = Y - B  # what the backbone leaves to the compounds

    # 2. View weights from the training compounds' mean residuals, and the fused kernel
    drugs, inv, C = compound_means(compound[train], R[train])
    w = view_weights(K, C, drugs)
    Kf = fuse(K, w)

    # 3. Shared compound term G: KRR over the training compounds; ridge and amplitude chosen on valid
    shared = KernelRidge(Kf[np.ix_(drugs, drugs)], C)

    def predict_shared(rows, lam):
        ids, i = np.unique(compound[rows], return_inverse=True)
        return (Kf[np.ix_(ids, drugs)] @ shared.coef(lam))[i]

    G_valid = {lam: predict_shared(valid, lam) for lam in LAMBDAS}
    _, lam, a = max((score(Y[valid], B[valid] + a * G_valid[lam]), lam, a) for lam in LAMBDAS for a in SCALES)
    G = np.zeros_like(Y)
    G[valid], G[test] = G_valid[lam], predict_shared(test, lam)
    G[train] = shared.leave_one_out(lam)[inv]

    # 4. Per-line compound term H: the same regression within each cell line
    H = np.zeros_like(Y)
    for c in np.unique(line[train]):
        rows = np.flatnonzero(train & (line == c))
        ids, i, C_line = compound_means(compound[rows], R[rows])
        ridge = KernelRidge(Kf[np.ix_(ids, ids)], C_line)
        H[rows] = ridge.leave_one_out(lam)[i]
        other = np.flatnonzero((valid | test) & (line == c))
        H[other] = Kf[np.ix_(compound[other], ids)] @ ridge.coef(lam)

    # 5. Weights of the two compound terms, chosen on valid
    best = (-2.0, 0.0, 0.0)
    for g, h in product(SCALES, SCALES):
        v = score(Y[valid], B[valid] + g * G[valid] + h * H[valid])
        if v > best[0]:
            best = (v, g, h)
    first_valid, g, h = best
    U = g * G + h * H
    P1 = B + U

    # 6. Context gate
    P2, gate = context_gate(ctl.astype(np.float64), R, U, P1, Y, train, valid)
    settings = {"fold": fold, "backbone_epoch": epoch, "backbone_valid": backbone_valid, "view_weights": w,
                "lambda": lam, "shared_scale": a, "g": g, "h": h, "first_stage_valid": first_valid, **gate}
    return FoldFit(backbone=B, first_stage=P1, final=P2, settings=settings)
