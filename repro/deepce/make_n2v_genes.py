#!/usr/bin/env python3
"""node2vec gene vectors for the DeepCE ladder (RECOMB study E3; repro/deepce, gitignored).

DeepCE's gene_vector.csv is a STRING embedding of unknown version and settings. To put DeepCE's PPI
input through the control ladder we regenerate it ourselves: node2vec (p = q = 1, 128-d, 10 walks of
length 80, window 10; pecanpy SparseOTF) on STRING v12.0 functional (score >= 400) and on its
rewired / labels / er nulls (scripts/netnull.py, seed 0). Landmarks absent from STRING get the same
fixed Gaussian vector in every variant. Output in DeepCE format (symbol, 128 values), rows in the
signature-column order.
"""
from __future__ import annotations
import os as _os  # noqa: E402
UNDETECT_DATA = _os.environ.get("UNDETECT_DATA", "data")  # data root (README)
UNDETECT_ROOT = _os.environ.get("UNDETECT_ROOT", ".")  # repository root (README)

import json
import sys
import time
from pathlib import Path

import numpy as np
from pecanpy import pecanpy as node2vec

sys.path.insert(0, (UNDETECT_ROOT + "/scripts"))
import netnull  # noqa: E402

OUT = Path((UNDETECT_DATA + "/repro/deepce/gene_variants"))
UP = Path(__file__).resolve().parent / "upstream/DeepCE/data"


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    landmarks = open(UP / "signature_train.csv").readline().strip().split(",")[5:]
    hm = netnull.hgnc_map()
    lm_h = [hm.get(g, g) for g in landmarks]
    base = netnull._string("functional", 400, harmonize=True)
    print(netnull.summary(base), flush=True)
    rng = np.random.default_rng(0)
    fallback = rng.normal(0, 0.5, (len(landmarks), 128))
    report = {}
    for kind in ("real", "rewired", "labels", "er"):
        t0 = time.time()
        net = netnull.null(base, kind, 0)
        edg = OUT / f"_string_func400_{kind}.edg"
        np.savetxt(edg, net.edges, fmt="%d", delimiter="\t")
        g = node2vec.SparseOTF(p=1, q=1, workers=8, verbose=False, random_state=0)
        g.read_edg(str(edg), weighted=False, directed=False)
        emb = g.embed(dim=128, num_walks=10, walk_length=80, window_size=10, epochs=1)
        pos = {int(n): i for i, n in enumerate(g.nodes)}
        gidx = {s: i for i, s in enumerate(net.genes)}
        V = fallback.copy(); found = 0
        for j, s in enumerate(lm_h):
            i = gidx.get(s)
            if i is not None and i in pos:
                V[j] = emb[pos[i]]; found += 1
        with open(OUT / f"n2v_{kind}.csv", "w") as f:
            for s, v in zip(landmarks, V):
                f.write(s + "," + ",".join(f"{x:.6g}" for x in v) + "\n")
        edg.unlink()
        report[kind] = {"landmarks_embedded": found, "seconds": round(time.time() - t0, 1), "edges": int(len(net.edges))}
        print(kind, report[kind], flush=True)
    (OUT / "n2v_manifest.json").write_text(json.dumps({"network": "STRING v12.0 functional >= 400, HGNC-harmonised",
        "node2vec": {"p": 1, "q": 1, "dim": 128, "num_walks": 10, "walk_length": 80, "window": 10, "epochs": 1,
                     "random_state": 0}, "null_seed": 0, "variants": report}, indent=2) + "\n")


if __name__ == "__main__":
    main()
