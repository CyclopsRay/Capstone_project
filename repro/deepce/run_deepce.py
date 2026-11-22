#!/usr/bin/env python3
"""DeepCE reproduction runner (repro/deepce, gitignored; upstream pth1993/DeepCE @ 9b0d04f).

Mirrors upstream main_deepce.py: Adam 2e-4, batch 16, dropout 0.1, point-wise MSE, float64, 24 h,
7 lines x 6 doses, same model hyperparameters, metric = mean per-signature Pearson over 978 genes.
Adds: --seed, --gene-variant, per-epoch metrics, test predictions at the best-dev epoch.

Gene variants (built with a fixed seed 0, independent of the training seed):
  ppi       upstream gene_vector.csv (STRING-derived, 128-d)
  shuffled  the same 978 vectors permuted across genes (PPI geometry kept, gene assignment broken)
  random    Gaussian with the upstream per-dimension mean and sd (no PPI content)
  onehot    978-d identity (free per-gene embedding through the first linear layer)
  file      vectors read from --gene-file (DeepCE format), named --label in the output directory;
            used for the node2vec ladder (make_n2v_genes.py, RECOMB study E3)
"""
from __future__ import annotations
import os as _os  # noqa: E402
UNDETECT_DATA = _os.environ.get("UNDETECT_DATA", "data")  # data root (README)
UNDETECT_ROOT = _os.environ.get("UNDETECT_ROOT", ".")  # repository root (README)

import argparse
import json
import platform
import random
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

UP = Path(__file__).resolve().parent / "upstream/DeepCE"
sys.path.insert(0, str(UP))
from models import DeepCE  # noqa: E402
from utils import DataReader, correlation, precision_k, rmse  # noqa: E402
from utils.data_utils import read_data  # noqa: E402

FILTER = {"time": "24H", "pert_id": ["BRD-U41416256", "BRD-U60236422"], "pert_type": ["trt_cp"],
          "cell_id": ["A375", "HA1E", "HELA", "HT29", "MCF7", "PC3", "YAPC"],
          "pert_idose": ["0.04 um", "0.12 um", "0.37 um", "1.11 um", "3.33 um", "10.0 um"]}


def gene_features(path: Path, variant: str, header_genes: list[str]) -> np.ndarray:
    if variant == "file":
        path = Path(ARGS.gene_file)
    names = np.loadtxt(path, delimiter=",", usecols=0, dtype=str)
    raw = np.loadtxt(path, delimiter=",", usecols=range(1, 129))
    assert list(names) == header_genes, "gene_vector rows are not in signature column order"
    rng = np.random.default_rng(0)
    if variant in ("ppi", "file"):
        return raw
    if variant == "shuffled":
        return raw[rng.permutation(len(raw))]
    if variant == "random":
        return rng.normal(raw.mean(0), raw.std(0), size=raw.shape)
    if variant == "onehot":
        return np.eye(len(raw))
    raise ValueError(variant)


def evaluate(model, data, split, batch_size):
    model.eval()
    labels, preds = [], []
    with torch.no_grad():
        for ft, lb in data.get_batch_data(dataset=split, batch_size=batch_size, shuffle=False):
            out = model(ft["drug"], data.gene, ft["mask"], ft.get("pert_type"), ft.get("cell_id"), ft.get("pert_idose"))
            labels.append(lb.cpu().numpy()); preds.append(out.cpu().numpy())
    y, p = np.concatenate(labels), np.concatenate(preds)
    pear, per_sig = correlation(y, p, "pearson")
    spear, _ = correlation(y, p, "spearman")
    neg, pos = precision_k(y, p, 100)
    return {"pearson": float(pear), "spearman": float(spear), "rmse": float(rmse(y, p)), "p100_pos": float(pos),
            "p100_neg": float(neg)}, p, y, np.asarray(per_sig)


def main(args: argparse.Namespace) -> None:
    started = time.time()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"host {platform.node()} python {sys.version.split()[0]} torch {torch.__version__} cuda {torch.version.cuda} "
          f"device {device} numpy {np.__version__} seed {args.seed} gene_variant {args.gene_variant} "
          f"epochs {args.epochs}", flush=True)
    d = UP / "data"
    data = DataReader(str(d / "drugs_smiles.csv"), str(d / "gene_vector.csv"), str(d / "signature_train.csv"),
                      str(d / "signature_dev.csv"), str(d / "signature_test.csv"), FILTER, device)
    header_genes = open(d / "signature_train.csv").readline().strip().split(",")[5:]
    G = gene_features(d / "gene_vector.csv", args.gene_variant, header_genes)
    data.gene = torch.from_numpy(G.astype(np.float64)).to(device)
    test_keys, _ = read_data(str(d / "signature_test.csv"), FILTER)
    print(f"train {len(data.train_feature['drug'])} dev {len(data.dev_feature['drug'])} "
          f"test {len(data.test_feature['drug'])} gene features {tuple(G.shape)}", flush=True)
    # reseed after DataReader (its import seeds 343) so model init and batch order follow --seed
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    model = DeepCE(drug_input_dim={"atom": 62, "bond": 6}, drug_emb_dim=128, conv_size=[16, 16],
                   degree=[0, 1, 2, 3, 4, 5], gene_input_dim=G.shape[1], gene_emb_dim=128, num_gene=G.shape[0],
                   hid_dim=128, dropout=0.1, loss_type="point_wise_mse", device=device,
                   initializer=torch.nn.init.xavier_uniform_, pert_type_input_dim=len(FILTER["pert_type"]),
                   cell_id_input_dim=len(FILTER["cell_id"]), pert_idose_input_dim=len(FILTER["pert_idose"]),
                   pert_type_emb_dim=4, cell_id_emb_dim=4, pert_idose_emb_dim=4, use_pert_type=data.use_pert_type,
                   use_cell_id=data.use_cell_id, use_pert_idose=data.use_pert_idose).to(device).double()
    opt = torch.optim.Adam(model.parameters(), lr=2e-4)
    out = Path(args.out) / f"{args.label or args.gene_variant}_seed{args.seed}"
    out.mkdir(parents=True, exist_ok=True)
    rows, best_dev = [], -np.inf
    for epoch in range(args.epochs):
        t0 = time.time()
        model.train()
        total, n = 0.0, 0
        for ft, lb in data.get_batch_data(dataset="train", batch_size=args.batch_size, shuffle=True):
            opt.zero_grad()
            pred = model(ft["drug"], data.gene, ft["mask"], ft.get("pert_type"), ft.get("cell_id"), ft.get("pert_idose"))
            loss = model.loss(lb, pred)
            loss.backward(); opt.step()
            total += loss.item(); n += 1
        dev, _, _, _ = evaluate(model, data, "dev", args.batch_size)
        test, p_test, y_test, per_sig = evaluate(model, data, "test", args.batch_size)
        rows.append({"epoch": epoch + 1, "train_loss": total / n, **{f"dev_{k}": v for k, v in dev.items()},
                     **{f"test_{k}": v for k, v in test.items()}, "seconds": time.time() - t0})
        if dev["pearson"] > best_dev:
            best_dev = dev["pearson"]
            np.savez(out / "test_pred_bestdev.npz", pred=p_test, label=y_test, per_sig_pearson=per_sig,
                     pert_id=test_keys[:, 0].astype(str), cell_id=test_keys[:, 2].astype(str),
                     pert_idose=test_keys[:, 3].astype(str), epoch=epoch + 1)
        pd.DataFrame(rows).to_csv(out / "epochs.csv", index=False)  # written every epoch
        print(f"epoch {epoch + 1} loss {total / n:.4f} dev {dev['pearson']:.4f} test {test['pearson']:.4f} "
              f"({time.time() - t0:.0f}s)", flush=True)
    frame = pd.DataFrame(rows)
    b = int(frame.dev_pearson.idxmax()); t = int(frame.test_pearson.idxmax())
    summary = {"gene_variant": args.gene_variant, "label": args.label, "gene_file": args.gene_file, "seed": args.seed, "epochs": args.epochs,
               "best_dev_epoch": b + 1, "test_at_best_dev": frame.iloc[b][[c for c in frame if c.startswith("test_")]].to_dict(),
               "best_test_epoch_optimistic": t + 1, "test_pearson_best_epoch_optimistic": float(frame.test_pearson.max()),
               "host": platform.node(), "torch": torch.__version__, "cuda": torch.version.cuda,
               "upstream_commit": "9b0d04fa920ef73df578b070cfcc3982effdde42", "seconds": round(time.time() - started, 1)}
    (out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2), flush=True)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--gene-variant", default="ppi", choices=["ppi", "shuffled", "random", "onehot", "file"])
    p.add_argument("--gene-file", default=None)
    p.add_argument("--label", default=None)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--out", default=(UNDETECT_DATA + "/repro/deepce/results"))
    return p.parse_args()


if __name__ == "__main__":
    ARGS = parse_args()
    main(ARGS)
