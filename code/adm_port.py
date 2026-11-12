"""Faithful Python port of the MATLAB epsilon-semimetric code (compute_metric.m,
epsilon_semimetric.m) plus helpers to rebuild the Adult instance of Table 6.

MATLAB rng(seed) for seed >= 1 equals numpy RandomState(seed) (MT19937,
init_genrand, 53-bit doubles). MATLAB int32() rounds half away from zero.
"""
import math
import numpy as np

ADULT_RAW = np.array([
    [26, 15, 20, 1, 0],
    [66,  9, 24, 1, 1],
    [45, 14, 40, 1, 1],
    [46,  4, 40, 1, 0],
    [55, 10, 50, 1, 0],
    [81,  5, 16, 0, 0],
    [27,  9, 40, 0, 0],
    [41, 12, 40, 0, 1],
    [33, 12, 55, 0, 0],
    [53, 11, 48, 0, 0],
], dtype=float)


def mprod(v):
    p = 1.0
    for a in v:
        p *= a
    return p


def msum(v):
    s = 0.0
    for a in v:
        s += a
    return s


def matlab_seed(a, b):
    """seed = int32(|prod(a)*sum(b)| / 10^floor(log10(.)) * 1e8)"""
    v = abs(mprod(a) * msum(b))
    if v == 0 or not math.isfinite(v):
        return 0
    v = v / (10.0 ** math.floor(math.log10(v))) * 1e8
    return int(math.floor(v + 0.5))  # int32 rounds half away from zero (v > 0)


def noise(seed, ex):
    if seed == 0:  # MATLAB rng(0) == twister seed 5489
        seed = 5489
    return np.random.RandomState(seed).random_sample(ex)


def lift_pair(x, y, s, h):
    """Return zi - zj in R^h exactly as epsilon_semimetric.m does."""
    ell = len(x)
    ex = h - ell
    zi = np.concatenate([x, s * noise(matlab_seed(x, y), ex)])
    zj = np.concatenate([y, s * noise(matlab_seed(y, x), ex)])
    return zi - zj


def build_M(Y, s):
    m, ell = Y.shape
    h = m * (m - 1) // 2
    M = np.zeros((h, h))
    pairs = []
    k = 0
    for i in range(m):
        for j in range(i + 1, m):
            M[:, k] = lift_pair(Y[i], Y[j], s, h)
            pairs.append((i, j))
            k += 1
    return M, pairs


def d_tilde(x, y, s, A):
    """epsilon_semimetric.m: returns sqrt(v A v') which may be complex."""
    h = A.shape[0]
    v = lift_pair(np.asarray(x, float), np.asarray(y, float), s, h)
    q = (v @ A) @ v
    return np.sqrt(q + 0j) if q < 0 else math.sqrt(q)


def adult_instance():
    """Rebuild y_aug, labels, target distances and epsilon of the Adult run."""
    raw = ADULT_RAW
    race_white = raw[:, 3].astype(int)
    X = raw[:, :3]
    mu = X.mean(axis=0)
    sd = X.std(axis=0, ddof=1)
    y = (X - mu) / sd
    m, ell = y.shape
    for i in range(1, m + 1):
        y[i - 1, :] += (i * 1e-8)
    k = 2
    adv = np.where(race_white == 1, 1, 2)
    C = np.zeros((k, ell))
    cnt = np.zeros(k)
    for i in range(m):
        C[adv[i] - 1] += y[i]
        cnt[adv[i] - 1] += 1
    C = C / cnt[:, None]
    y_aug = np.vstack([C, y])
    for i in range(1, k + 1):
        y_aug[i - 1, :] += ((m + i) * 1e-8)
    labels = np.concatenate([[1, 2], adv])
    return y_aug, labels, race_white


def two_level_targets(Y, labels, small_factor=0.005, large_factor=200.0, clip=True):
    m = len(Y)
    D = np.linalg.norm(Y[:, None, :] - Y[None, :, :], axis=2)
    iu = np.triu_indices(m, 1)
    dmin, dmax = D[iu].min(), D[iu].max()
    small, large = small_factor * dmin, large_factor * dmax
    T = np.where(labels[:, None] == labels[None, :], small, large)
    if clip:
        T = np.minimum(T, 0.99 * D)
    np.fill_diagonal(T, 0.0)
    return T, D, dmin, dmax, small, large


def triangle_defect(T):
    """max_{i,j,k} T_ij - T_ik - T_kj and the number of violating triples > tol."""
    m = T.shape[0]
    best = -np.inf
    arg = None
    for k in range(m):
        V = T - T[:, [k]] - T[[k], :]
        V[k, :] = -np.inf
        V[:, k] = -np.inf
        np.fill_diagonal(V, -np.inf)
        idx = np.unravel_index(np.argmax(V), V.shape)
        if V[idx] > best:
            best = V[idx]
            arg = (idx[0], idx[1], k)
    return best, arg


def count_violations(T, tol):
    m = T.shape[0]
    n = 0
    tot = 0
    for k in range(m):
        V = T - T[:, [k]] - T[[k], :]
        mask = np.ones_like(V, dtype=bool)
        mask[k, :] = False
        mask[:, k] = False
        np.fill_diagonal(mask, False)
        n += int((V[mask] > tol).sum())
        tot += int(mask.sum())
    return n, tot
