"""E3: targeted fairness attacks on real data with an unmodified k-medoids pipeline.

Honest pipeline: standardised numeric features, Euclidean distance, FasterPAM k-medoids (k=2,
BUILD initialisation, fixed seed). The favourable cluster has the higher rate of the favourable
ground-truth outcome. The adversary moves protected-group members out of the favourable cluster
by a minimal relocation across the bisector of the two medoids, takes the Euclidean metric of the
relocated configuration as target rho (an exact metric on the data) and supplies its exact
McShane-Frechet extension (Theorem 2). Eligible targets have a unique feature vector, since no
metric on the feature space separates identical vectors.

The adversary knows the pipeline (white-box) and selects targets greedily: it keeps a candidate
only if the re-run pipeline keeps both medoids, flips every target so far and changes no other
decision. Medoids count as kept when their feature vectors are unchanged, since twins with
identical features induce identical decisions. FasterPAM runs single-threaded with a fixed seed,
because its parallel variant returns different twins and borderline medoids for different thread
counts. Candidate orders: border (smallest margin ratio mu), stealthy (smallest individual
distortion) and random.
"""
import json, os
import numpy as np
import kmedoids
from datasets import adult, compas, german, standardize
from exactmetric import pdist_matrix

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "results")


def fairness(dec, a, ylab):
    pp, pr = dec[a == 1].mean(), dec[a == 0].mean()
    tpr = lambda grp: dec[(a == grp) & (ylab == 1)].mean()
    fpr = lambda grp: dec[(a == grp) & (ylab == 0)].mean()
    eod = max(abs(tpr(0) - tpr(1)), abs(fpr(0) - fpr(1)))
    return dict(p_prot=float(pp), p_ref=float(pr), dpd=float(pr - pp), di=float(pp / pr), eod=float(eod))


def run_kmedoids(Dm, seed=0):
    r = kmedoids.fasterpam(Dm, 2, init="build", random_state=seed, n_cpu=1)
    return np.asarray(r.labels), np.asarray(r.medoids)


def bootstrap_dpd(dec, a, B=2000, seed=0):
    rng = np.random.default_rng(seed)
    n = len(dec)
    vals = np.empty(B)
    for b in range(B):
        idx = rng.integers(0, n, n)
        d, g = dec[idx], a[idx]
        vals[b] = d[g == 0].mean() - d[g == 1].mean()
    lo, hi = np.quantile(vals, [0.025, 0.975])
    return float(lo), float(hi)


def study(ds, cols=None, delta_frac=0.02, max_targets=100, max_tries=150, seed=1):
    X = standardize(ds["X"] if cols is None else ds["X"][:, cols])
    a, ylab = ds["a"], ds["ylab"]
    n = len(X)
    U, inv, cnt = np.unique(X, axis=0, return_inverse=True, return_counts=True)
    inv = inv.ravel()
    EU = pdist_matrix(U)
    lab, med = run_kmedoids(EU[inv][:, inv])
    rates = [ylab[lab == c].mean() for c in (0, 1)]
    F = int(np.argmax(rates)); Uc = 1 - F
    mF, mU = X[med[F]], X[med[Uc]]
    dec = (lab == F).astype(int)
    base = fairness(dec, a, ylab)
    lo, hi = bootstrap_dpd(dec, a)
    half = (hi - lo) / 2
    n_p = int((a == 1).sum())
    g_res = int(np.floor(half * n_p))
    need = base["p_prot"] - 0.8 * base["p_ref"]
    g_45 = int(np.floor(need * n_p)) + 1 if need > 0 else 0

    nhat = (mU - mF) / np.linalg.norm(mU - mF)
    c = 0.5 * (mF + mU)
    delta = delta_frac * np.linalg.norm(mU - mF)
    eF = np.linalg.norm(X - mF, axis=1); eU = np.linalg.norm(X - mU, axis=1)
    mu = np.where(eF > 0, eU / np.maximum(eF, 1e-300), np.inf)
    cand = np.flatnonzero((a == 1) & (dec == 1) & (cnt[inv] == 1))
    cand = np.setdiff1d(cand, med)
    kappa = np.empty(len(cand))
    for q, i in enumerate(cand):
        u = inv[i]
        up = U[u] + ((c - U[u]) @ nhat + delta) * nhat
        mask = np.arange(len(U)) != u
        r = np.linalg.norm(U[mask] - up, axis=1) / EU[u, mask]
        kappa[q] = max(r.max(), 1.0) / min(r.min(), 1.0)
    info = dict(dataset=ds["name"], n=n, ell=X.shape[1],
                features=(ds["names"] if cols is None else [ds["names"][j] for j in cols]),
                distinct=len(U), n_prot=n_p, n_prot_fav=int(((a == 1) & (dec == 1)).sum()),
                candidates=len(cand), fav_rates=[float(r) for r in rates], baseline=base,
                dpd_ci=[lo, hi], ci_half=half, g_res=g_res, g_45=g_45, groups=ds["groups"],
                medoid_distance=float(np.linalg.norm(mU - mF)))
    print(info, flush=True)

    def attack(G):
        Up = U.copy()
        ug = inv[np.asarray(G, int)]
        if len(ug):
            t = (c - U[ug]) @ nhat
            Up[ug] = U[ug] + (t + delta)[:, None] * nhat[None, :]
        RU = pdist_matrix(Up)
        lab2, med2 = run_kmedoids(RU[inv][:, inv])
        # Compare medoids by feature vector: identical twins induce identical decisions,
        # and which twin the solver returns is arbitrary.
        same = set(inv[med2].tolist()) == set(inv[med].tolist())
        F2 = int(lab2[med[F]]) if same else int(np.argmax([ylab[lab2 == q].mean() for q in (0, 1)]))
        dec2 = (lab2 == F2).astype(int)
        return RU, dec2, same

    rng = np.random.default_rng(seed)
    orders = {"border": cand[np.argsort(mu[cand])],
              "stealthy": cand[np.argsort(kappa)],
              "random": rng.permutation(cand)}
    iu = np.triu_indices(len(U), 1)
    rows = []
    for strat, order in orders.items():
        G = []
        tries = 0
        for i in order:
            if len(G) >= max_targets or tries >= max_tries:
                break
            tries += 1
            RU, dec2, same = attack(G + [int(i)])
            others = np.setdiff1d(np.arange(n), G + [int(i)])
            ok = same and dec2[G + [int(i)]].sum() == 0 and (dec2[others] == dec[others]).all()
            if not ok:
                continue
            G.append(int(i))
            ratio = RU[iu] / EU[iu]
            row = dict(dataset=ds["name"], strategy=strat, g=len(G), tries=tries,
                       distortion=float(ratio.max() / ratio.min()), lower_bound=float(mu[G].max()),
                       **fairness(dec2, a, ylab))
            rows.append(row)
        last = rows[-1] if rows and rows[-1]["strategy"] == strat else None
        print(strat, "accepted", len(G), "of", tries, "tries;",
              [(r["g"], round(r["dpd"], 3), round(r["distortion"], 2), round(r["lower_bound"], 2))
               for r in rows if r["strategy"] == strat and r["g"] in (1, 5, 10, 20, g_res, 50, 100, len(G))], flush=True)
    return info, rows


if __name__ == "__main__":
    allinfo, allrows = [], []
    for ds, cols in ((adult(n=5000, seed=0), [0, 1, 2]), (compas(), None), (german(), None)):
        info, rows = study(ds, cols=cols)
        allinfo.append(info); allrows += rows
        json.dump(dict(info=allinfo, rows=allrows), open(os.path.join(OUT, "e3.json"), "w"), indent=1)
