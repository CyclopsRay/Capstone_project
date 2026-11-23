"""The drug-free backbone: a network that predicts a sample's response from its paired control profile alone."""
from __future__ import annotations

from typing import Callable

import numpy as np
import torch


def pcc_rows(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    a = a - a.mean(1, keepdim=True)
    b = b - b.mean(1, keepdim=True)
    return (a * b).sum(1) / torch.sqrt((a * a).sum(1) * (b * b).sum(1) + 1e-12)


def train_backbone(X_train: np.ndarray, Y_train: np.ndarray, X_valid: np.ndarray, Y_valid: np.ndarray, seed: int,
                   max_epochs: int, patience: int, hidden: int = 2048, layers: int = 3, dropout: float = 0.1,
                   lr: float = 1e-4, batch: int = 512) -> tuple[Callable[[np.ndarray], np.ndarray], float, int]:
    """Train the backbone (control profile -> response) with early stopping on the validation PCC_DEG.

    The drug-blind benchmark's MLP recipe: `layers` hidden layers of `hidden` units with ReLU and dropout, loss
    0.5 MSE + 0.5 (1 - PCC_DEG), Adam, gradients clipped to norm 1. Returns the prediction function of the best
    epoch, its validation PCC_DEG and the epoch.
    """
    torch.manual_seed(seed)
    np.random.seed(seed)
    dims = [X_train.shape[1]] + [hidden] * layers
    net = torch.nn.Sequential(
        *[m for i in range(layers) for m in (torch.nn.Linear(dims[i], dims[i + 1]), torch.nn.ReLU(),
                                             torch.nn.Dropout(dropout))],
        torch.nn.Linear(hidden, Y_train.shape[1]))
    opt = torch.optim.Adam(net.parameters(), lr=lr)
    Xtr, Ytr, Xva, Yva = (torch.from_numpy(a).float() for a in (X_train, Y_train, X_valid, Y_valid))
    best, best_epoch, best_state, wait = -2.0, 0, None, 0
    for epoch in range(1, max_epochs + 1):
        net.train()
        for b in torch.randperm(len(Ytr)).split(batch):
            p = net(Xtr[b])
            loss = 0.5 * ((p - Ytr[b]) ** 2).mean() + 0.5 * (1 - pcc_rows(p, Ytr[b]).mean())
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
            opt.step()
        net.eval()
        with torch.no_grad():
            v = float(torch.cat([pcc_rows(net(Xva[b]), Yva[b]) for b in torch.arange(len(Yva)).split(4096)]).mean())
        if v > best:
            best, best_epoch, wait = v, epoch, 0
            best_state = {k: t.detach().clone() for k, t in net.state_dict().items()}
        else:
            wait += 1
            if wait >= patience:
                break
    net.load_state_dict(best_state)
    net.eval()

    def predict(X: np.ndarray) -> np.ndarray:
        X = torch.from_numpy(X).float()
        with torch.no_grad():
            return torch.cat([net(X[b]) for b in torch.arange(len(X)).split(4096)]).numpy()

    return predict, best, best_epoch
