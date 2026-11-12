"""E1: cost of the exact construction versus m (two-level targets, random labels)."""
import json, os, time
import numpy as np
from sklearn.cluster import DBSCAN
from sklearn.metrics import adjusted_rand_score as ARI
from exactmetric import TwoLevelExtension

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "results")
rng = np.random.default_rng(11)
rows = []
for m in [1_000, 3_000, 10_000, 30_000, 100_000, 300_000, 1_000_000]:
    Y = rng.standard_normal((m, 2))
    lab = rng.integers(0, 3, m)
    row = dict(m=m)
    if m <= 100_000:
        t0 = time.time()
        met = TwoLevelExtension(Y, lab, s=0.25, D=1.0, optimal=True)
        row["t_optimal"] = time.time() - t0
        row["distortion_optimal"] = met.L / met.eta
    t0 = time.time()
    fast = TwoLevelExtension(Y, lab, s=0.25, D=1.0, optimal=False)
    row["t_fast"] = time.time() - t0
    row["distortion_fast"] = fast.L / fast.eta
    Q = rng.uniform(Y.min(0), Y.max(0), size=(50, 2))
    t0 = time.time()
    for a in range(0, 50, 2):
        fast(Q[a], Q[a + 1])
    row["t_query"] = (time.time() - t0) / 25
    if m <= 10_000:
        t0 = time.time()
        R = np.where(lab[:, None] == lab[None, :], 0.25, 1.0)
        np.fill_diagonal(R, 0.0)
        pred = DBSCAN(eps=0.5, min_samples=5, metric="precomputed").fit_predict(R)
        row["t_dbscan_end_to_end"] = time.time() - t0
        row["ari_dbscan"] = ARI(lab, pred)
        del R
    rows.append(row)
    print(row, flush=True)

# conference fitted law for the epsilon-semimetric (Table 1 of the conference version)
for r in rows:
    r["eps_semimetric_extrapolated_s"] = 7e-9 * r["m"] ** 5.72
json.dump(rows, open(os.path.join(OUT, "e1.json"), "w"), indent=1)
