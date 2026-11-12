"""E0: re-examination of the epsilon-semimetric construction of the conference version.

Part A uses the saved artifact of the Adult experiment (As_m12_double.mat).
Part B runs the synthetic benchmark pipeline (benchmark_metric.m logic, no clipping)
in double precision with our own data draws, several seeds per m.
"""
import json, os, sys, time
import numpy as np
import scipy.io as sio
from scipy.linalg import qr, qr_delete
from adm_port import (adult_instance, two_level_targets, build_M, lift_pair, d_tilde,
                      triangle_defect, count_violations)

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "..", "results")
os.makedirs(OUT, exist_ok=True)
res = {}

# ---------------- Part A: Adult artifact ----------------
S = sio.loadmat(os.path.join(HERE, "..", "data", "artifacts", "As_m12_double.mat"))
A = S["A"].astype(float)
s = float(np.squeeze(S["s"]))
y, labels, race = adult_instance()
T, D, dmin, dmax, small, large = two_level_targets(y, labels, clip=True)
eps = dmin / 10
m = len(y); h = A.shape[0]
M, pairs = build_M(y, s)
q = np.array([(M[:, k] @ A) @ M[:, k] for k in range(h)])
tgt = np.array([T[i, j] for (i, j) in pairs])
fid = float(np.median(np.abs(q - tgt ** 2) / tgt ** 2))
tau, _ = triangle_defect(T)
nv, tot = count_violations(T, eps)

rng = np.random.default_rng(2026)
lo, hi = y.min(0), y.max(0)
P = rng.uniform(lo, hi, size=(40, 3))
dt = lambda a, b: d_tilde(a, b, s, A)
DP = np.zeros((40, 40))
for a in range(40):
    for b in range(a + 1, 40):
        DP[a, b] = DP[b, a] = dt(P[a], P[b])
fresh_def, fresh_v, fresh_t = -np.inf, 0, 0
for a in range(40):
    for b in range(40):
        for c in range(40):
            if len({a, b, c}) < 3:
                continue
            v = DP[a, b] - DP[a, c] - DP[c, b]
            fresh_def = max(fresh_def, v); fresh_t += 1; fresh_v += int(v > eps)
mixed_def, mixed_v, mixed_t = -np.inf, 0, 0
for a in range(40):
    dx = np.array([dt(P[a], y[i]) for i in range(m)])
    for i in range(m):
        for j in range(m):
            if i == j:
                continue
            for v in (T[i, j] - dx[i] - dx[j], dx[i] - T[i, j] - dx[j]):
                mixed_def = max(mixed_def, v); mixed_t += 1; mixed_v += int(v > eps)
jumps = []
for i in range(m):
    for t in range(5):
        u = rng.standard_normal(3); u /= np.linalg.norm(u)
        jumps.append(dt(y[i] + 1e-9 * u, y[i]))
jumps = np.array(jumps)
amp = []
for a in range(0, 40, 2):
    v = lift_pair(P[a], P[a + 1], s, h); v[:3] = 0.0
    amp.append(np.sqrt(v @ A @ v) / np.linalg.norm(v))
wA = np.linalg.eigvalsh(A)
res["artifact"] = dict(m_aug=m, h=h, eps=eps, s=s, port_fidelity_median_relerr=fid,
                       tau=tau, tau_over_eps=tau / eps, data_triples_beyond_eps=nv, data_triples=tot,
                       fresh_max_defect=fresh_def, fresh_beyond_eps=fresh_v, fresh_triples=fresh_t,
                       mixed_max_defect=mixed_def, mixed_beyond_eps=mixed_v, mixed_triples=mixed_t,
                       jump_1e9_max=float(jumps.max()), jump_1e9_frac_above_small=float((jumps > small).mean()),
                       noise_amp_median=float(np.median(amp)), noise_amp_max=float(np.max(amp)),
                       lambda_min=float(wA.min()), lambda_max=float(wA.max()),
                       dbscan_radius=0.5 * (small + large), max_target=float(T.max()), small=small)
print(json.dumps(res["artifact"], indent=1))

S32 = sio.loadmat(os.path.join(HERE, "..", "data", "artifacts", "As_m32_double.mat"))
w32 = np.linalg.eigvalsh(S32["A"].astype(float))
res["artifact_m32"] = dict(lambda_min=float(w32.min()), lambda_max=float(w32.max()), n_negative=int((w32 < 0).sum()))

# ---------------- Part B: double precision on the synthetic pipeline ----------------
def A_matlab_double(M, delta):
    hh = M.shape[0]
    Q, R = qr(M)
    Acc = np.zeros((hh, hh))
    for i in range(hh):
        Qd, _ = qr_delete(Q, R, i, 1, which="col")
        B = Qd[:, -1]
        Ai = np.outer(B, B)
        lam = (M[:, i] @ Ai) @ M[:, i]
        Acc += Ai / lam * delta[i] ** 2
    return Acc


def synthetic(mm, seed, ell=2, k=3):
    r = np.random.default_rng(seed)
    Y = r.standard_normal((mm - k, ell))
    while True:
        lab = r.integers(1, k + 1, mm - k)
        if (np.bincount(lab, minlength=k + 1)[1:] >= 2).all():
            break
    C = np.array([Y[lab == c].mean(0) for c in range(1, k + 1)])
    return np.vstack([C, Y]), np.concatenate([np.arange(1, k + 1), lab])


rows = []
for mm in [10, 20, 30, 40, 50]:
    for seed in range(5):
        Y, lab = synthetic(mm, seed=1000 * mm + seed)
        Tm, _, dmn, dmx, sm, lg = two_level_targets(Y, lab, clip=False)
        e = sm
        hh = mm * (mm - 1) // 2
        ss = 0.9 * e / np.sqrt(hh)
        Mm, prs = build_M(Y, ss)
        dl = np.array([Tm[i, j] for (i, j) in prs])
        t0 = time.time()
        Am = A_matlab_double(Mm, dl)
        w = np.linalg.eigvalsh((Am + Am.T) / 2)
        qq = np.array([(Mm[:, k] @ Am) @ Mm[:, k] for k in range(hh)])
        intra = np.array([lab[i] == lab[j] for (i, j) in prs])
        rel = np.abs(qq - dl ** 2) / dl ** 2
        rows.append(dict(m=mm, seed=seed, condM=float(np.linalg.cond(Mm)),
                         indefinite=bool(w.min() < 0),
                         imag_intra=int((qq[intra] < 0).sum()), n_intra=int(intra.sum()),
                         imag_inter=int((qq[~intra] < 0).sum()), n_inter=int((~intra).sum()),
                         relerr_intra_median=float(np.median(rel[intra])),
                         relerr_inter_median=float(np.median(rel[~intra])),
                         seconds=time.time() - t0))
        print(rows[-1])
res["double_precision"] = rows
json.dump(res, open(os.path.join(OUT, "e0.json"), "w"), indent=1)
