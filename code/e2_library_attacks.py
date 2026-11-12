"""E2: unmodified library clustering under the exact two-level metric (random targets).

Documented hyperparameters stay at library defaults (DBSCAN eps=0.5, min_samples=5, HDBSCAN
min_cluster_size=5, OPTICS min_samples=5, xi=0.05, spectral n_neighbors=10). The adversary only
supplies the metric. On the data the exact metric equals the two-level target (s=0.25, D=1.0),
so the precomputed matrix is the metric's restriction to Y.
"""
import json, os, time, warnings
import numpy as np
import kmedoids
from sklearn.cluster import DBSCAN, HDBSCAN, OPTICS, AgglomerativeClustering, SpectralClustering
from sklearn.metrics import adjusted_rand_score as ARI

warnings.filterwarnings("ignore")
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "results")
S, D = 0.25, 1.0


def algorithms(k, seed, with_optics=True):
    algs = {
        "DBSCAN": lambda R: DBSCAN(eps=0.5, min_samples=5, metric="precomputed").fit_predict(R),
        "HDBSCAN": lambda R: HDBSCAN(metric="precomputed").fit_predict(R),
        "single linkage": lambda R: AgglomerativeClustering(n_clusters=k, metric="precomputed", linkage="single").fit_predict(R),
        "complete linkage": lambda R: AgglomerativeClustering(n_clusters=k, metric="precomputed", linkage="complete").fit_predict(R),
        "average linkage": lambda R: AgglomerativeClustering(n_clusters=k, metric="precomputed", linkage="average").fit_predict(R),
        "k-medoids (FasterPAM)": lambda R: kmedoids.KMedoids(k, metric="precomputed", method="fasterpam",
                                                             init="random", random_state=seed).fit(R).labels_,
        "spectral (kNN graph)": lambda R: SpectralClustering(n_clusters=k, affinity="precomputed_nearest_neighbors",
                                                             random_state=seed).fit_predict(R),
    }
    if with_optics:
        algs["OPTICS"] = lambda R: OPTICS(metric="precomputed").fit_predict(R)
    return algs


rows = []
for m in [300, 3000]:
    for k in [2, 3, 5]:
        for seed in range(5):
            rng = np.random.default_rng(100 * m + 10 * k + seed)
            while True:
                lab = rng.integers(0, k, m)
                if np.bincount(lab, minlength=k).min() >= 12:
                    break
            R = np.where(lab[:, None] == lab[None, :], S, D)
            np.fill_diagonal(R, 0.0)
            for name, f in algorithms(k, seed).items():
                t0 = time.time()
                pred = f(R)
                rows.append(dict(m=m, k=k, seed=seed, alg=name, ari=float(ARI(lab, pred)),
                                 noise=int((np.asarray(pred) < 0).sum()), seconds=time.time() - t0))
            print(m, k, seed, [(r["alg"], round(r["ari"], 3)) for r in rows[-8:]], flush=True)

m, k, seed = 10_000, 3, 0
rng = np.random.default_rng(7)
lab = rng.integers(0, k, m)
R = np.where(lab[:, None] == lab[None, :], S, D)
np.fill_diagonal(R, 0.0)
for name, f in algorithms(k, seed, with_optics=False).items():
    if name in ("single linkage", "complete linkage"):
        continue
    t0 = time.time()
    pred = f(R)
    rows.append(dict(m=m, k=k, seed=seed, alg=name, ari=float(ARI(lab, pred)),
                     noise=int((np.asarray(pred) < 0).sum()), seconds=time.time() - t0))
    print(rows[-1], flush=True)
json.dump(rows, open(os.path.join(OUT, "e2.json"), "w"), indent=1)
