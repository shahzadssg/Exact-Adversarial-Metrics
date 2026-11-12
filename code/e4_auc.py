"""Recompute E4 AUCs with a numerical tolerance: values below 1e-9 count as zero.

Exact metrics show triangle defects and embedding residuals of order 1e-15 from
floating-point rounding. An auditor tests against a tolerance, so ties at zero
must not be broken by rounding noise.
"""
import json, os
import numpy as np
from sklearn.metrics import roc_auc_score

RES = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "results")
TOL = 1e-9
d = json.load(open(os.path.join(RES, "e4.json")))
rows = d["rows"]
clip = lambda v, st: (0.0 if (st in ("axiom", "embed") and v < TOL) else v)
variants = sorted({r["variant"] for r in rows if r["kind"] == "attack"})
auc = {}
for st in ("axiom", "embed", "distort"):
    h0 = [clip(r[st], st) for r in rows if r["kind"] == "benign"]
    auc[st] = {}
    for v in variants:
        h1 = [clip(r[st], st) for r in rows if r["variant"] == v]
        auc[st][v] = float(roc_auc_score([0] * len(h0) + [1] * len(h1), h0 + h1))
succ = {v: float(np.mean([r["success"] for r in rows if r["variant"] == v])) for v in variants}
d["auc"] = auc
d["tolerance"] = TOL
d["success"] = succ
json.dump(d, open(os.path.join(RES, "e4.json"), "w"), indent=1)
print(json.dumps(auc, indent=1)); print(succ)
