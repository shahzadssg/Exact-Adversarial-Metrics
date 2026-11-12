"""E4: black-box detection of exact adversarial metrics versus legitimate custom metrics.

Auditor model: query access to the deployed dissimilarity on any points, plus the standardised
features (reference geometry E). Three statistics:
  S_axiom   largest triangle defect on all triples of 400 data points and 200 random points,
            relative to the median distance
  S_embed   classical MDS residual: eigenvalue mass of B = -1/2 J D^2 J outside the top ell
            positive eigenvalues (zero iff D is Euclidean in R^ell)
  S_distort log of the distortion max(d/E) / min(d/E) over data pairs
Benign (H0): weighted Euclidean, Mahalanobis, Manhattan, Chebyshev, cosine, NCA.
Attacks (H1): minimal relocation of g in {1, 3, 10} protected members (random or stealthy
selection) with the exact extension, and a global two-level attack with random labels.
"""
import json, os, warnings
import numpy as np
import kmedoids
from scipy.spatial.distance import cdist
from sklearn.metrics import roc_auc_score
from sklearn.neighbors import NeighborhoodComponentsAnalysis
from datasets import adult, compas, german, standardize
from exactmetric import ExactExtension, TwoLevelExtension, pdist_matrix

warnings.filterwarnings("ignore")
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "results")
N_SUB, N_RAND, TRIALS = 400, 200, 20


def max_triangle_defect(Dm):
    best = 0.0
    for k in range(len(Dm)):
        V = Dm - Dm[:, [k]] - Dm[[k], :]
        best = max(best, float(V.max()))
    return best


def mds_residual(Dm, ell):
    n = len(Dm)
    J = np.eye(n) - np.ones((n, n)) / n
    B = -0.5 * J @ (Dm ** 2) @ J
    w = np.linalg.eigvalsh((B + B.T) / 2)[::-1]
    tot = np.abs(w).sum()
    return float((np.abs(w).sum() - np.clip(w[:ell], 0, None).sum()) / tot)


def stats(D_data, D_all, E_data, ell):
    iu = np.triu_indices(len(D_data), 1)
    r = D_data[iu] / E_data[iu]
    med = np.median(D_all[np.triu_indices(len(D_all), 1)])
    return dict(axiom=max_triangle_defect(D_all) / med,
                embed=mds_residual(D_data, ell),
                distort=float(np.log(r.max() / max(r.min(), 1e-300))))


def benign_metrics(Xs, ys, rng):
    ell = Xs.shape[1]
    w = np.exp(rng.uniform(np.log(1 / 3), np.log(3), ell))
    C = np.cov(Xs.T) + 1e-9 * np.eye(ell)
    VI = np.linalg.inv(C)
    nca = NeighborhoodComponentsAnalysis(n_components=ell, random_state=int(rng.integers(1 << 30))).fit(Xs, ys)
    A = nca.components_
    return {
        "weighted Euclidean": lambda P, Q: cdist(P * w, Q * w),
        "Mahalanobis": lambda P, Q: cdist(P, Q, "mahalanobis", VI=VI),
        "Manhattan": lambda P, Q: cdist(P, Q, "cityblock"),
        "Chebyshev": lambda P, Q: cdist(P, Q, "chebyshev"),
        "cosine": lambda P, Q: cdist(P, Q, "cosine"),
        "NCA": lambda P, Q: cdist(P @ A.T, Q @ A.T),
    }


def relocation_attack(Xs, a, g, strategy, rng, delta_frac=0.02):
    """Honest k-medoids on Xs; relocate g protected members of the favourable cluster."""
    E = pdist_matrix(Xs)
    r = kmedoids.fasterpam(E, 2, init="build", random_state=0, n_cpu=1)
    lab, med = np.asarray(r.labels), np.asarray(r.medoids)
    return lab, med, E


def run_dataset(ds, cols, rng):
    X = standardize(ds["X"] if cols is None else ds["X"][:, cols])
    a, ylab = ds["a"], ds["ylab"]
    U, idx, cnt = np.unique(X, axis=0, return_index=True, return_counts=True)
    pool = idx[cnt == 1] if (cnt == 1).sum() >= N_SUB else idx
    ell = X.shape[1]
    out = []
    for t in range(TRIALS):
        sub = rng.choice(pool, N_SUB, replace=False)
        Xs, As, Ys = X[sub], a[sub], ylab[sub]
        lo, hi = Xs.min(0), Xs.max(0)
        Rp = rng.uniform(lo, hi, (N_RAND, ell))
        Pall = np.vstack([Xs, Rp])
        E = pdist_matrix(Xs)
        for name, f in benign_metrics(Xs, Ys, rng).items():
            Dd, Da = f(Xs, Xs), f(Pall, Pall)
            np.fill_diagonal(Dd, 0); np.fill_diagonal(Da, 0)
            out.append(dict(dataset=ds["name"], trial=t, kind="benign", variant=name, **stats(Dd, Da, E, ell)))
        # honest k-medoids
        r = kmedoids.fasterpam(E, 2, init="build", random_state=0, n_cpu=1)
        lab, med = np.asarray(r.labels), np.asarray(r.medoids)
        F = int(np.argmax([Ys[lab == q].mean() if (lab == q).any() else -1 for q in (0, 1)]))
        mF, mU = Xs[med[F]], Xs[med[1 - F]]
        nhat = (mU - mF) / np.linalg.norm(mU - mF); c = 0.5 * (mF + mU)
        delta = 0.02 * np.linalg.norm(mU - mF)
        cand = np.setdiff1d(np.flatnonzero((As == 1) & (lab == F)), med)
        if len(cand) >= 10:
            kap = []
            for i in cand:
                up = Xs[i] + ((c - Xs[i]) @ nhat + delta) * nhat
                msk = np.arange(N_SUB) != i
                rr = np.linalg.norm(Xs[msk] - up, axis=1) / E[i, msk]
                kap.append(max(rr.max(), 1) / min(rr.min(), 1))
            orders = {"stealthy": cand[np.argsort(kap)], "random": rng.permutation(cand)}
            for strat, order in orders.items():
                for g in (1, 3, 10):
                    G = order[:g]
                    Xp = Xs.copy()
                    Xp[G] = Xs[G] + (((c - Xs[G]) @ nhat) + delta)[:, None] * nhat[None, :]
                    R = pdist_matrix(Xp)
                    ext = ExactExtension(Xs, R, E)
                    PR = ext.phi_many(Rp)
                    Prow = np.vstack([R, PR])            # Phi on data = rows of R
                    Da = cdist(Prow, Prow, 'chebyshev')
                    Da = np.maximum(Da, ext.eta * pdist_matrix(Pall))
                    lab2 = np.asarray(kmedoids.fasterpam(R, 2, init="build", random_state=0, n_cpu=1).labels)
                    moved = float((lab2[G] != lab2[med[F]]).mean())
                    out.append(dict(dataset=ds["name"], trial=t, kind="attack", variant=f"relocation-{strat}-g{g}",
                                    success=moved, **stats(R, Da, E, ell)))
        # global two-level attack, random labels, exact extension
        labs = rng.integers(0, 2, N_SUB)
        tl = TwoLevelExtension(Xs, labs, s=0.25, D=1.0)
        Rt = tl.rho()
        PR = np.array([tl.phi(x) for x in Rp])
        Prow = np.vstack([np.array([tl.phi(x) for x in Xs]), PR])
        Da = cdist(Prow, Prow, 'chebyshev')
        Da = np.maximum(Da, tl.eta * pdist_matrix(Pall))
        out.append(dict(dataset=ds["name"], trial=t, kind="attack", variant="two-level (random labels)",
                        success=1.0, **stats(Rt, Da, E, ell)))
        print(ds["name"], t, flush=True)
    return out


if __name__ == "__main__":
    rng = np.random.default_rng(42)
    rows = []
    for ds, cols in ((adult(n=None, seed=0), [0, 1, 2]), (compas(), None), (german(), None)):
        rows += run_dataset(ds, cols, rng)
        json.dump(rows, open(os.path.join(OUT, "e4.json"), "w"), indent=1)
    # AUCs: benign pool vs each attack variant, per statistic
    res = {}
    variants = sorted({r["variant"] for r in rows if r["kind"] == "attack"})
    for stat in ("axiom", "embed", "distort"):
        h0 = [r[stat] for r in rows if r["kind"] == "benign"]
        res[stat] = {}
        for v in variants:
            h1 = [r[stat] for r in rows if r["variant"] == v]
            y = [0] * len(h0) + [1] * len(h1)
            res[stat][v] = float(roc_auc_score(y, h0 + h1))
    json.dump(dict(rows=rows, auc=res), open(os.path.join(OUT, "e4.json"), "w"), indent=1)
    print(json.dumps(res, indent=1))
