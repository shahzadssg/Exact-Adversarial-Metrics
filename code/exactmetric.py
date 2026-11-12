"""Exact adversarial metrics on R^ell.

ExactExtension: McShane/Frechet extension of a finite metric R on Y.
    Phi_k(x) = min_j ( R[j,k] + L ||x - y_j|| )
    d(x, y)  = max( ||Phi(x) - Phi(y)||_inf , eta ||x - y|| )
With L >= L* = max R_ij/E_ij and eta <= eta* = min R_ij/E_ij this is a metric on
R^ell, equals R on Y, and satisfies eta E <= d <= L E (optimal for L = L*, eta = eta*).

TwoLevelExtension: the same construction for the two-level target
    R_ij = s (same label), D (different label), 0 (i = j)
with an O(m) evaluation of Phi.
"""
import numpy as np


def pdist_matrix(X, Z=None, chunk=2048):
    """Euclidean distance matrix, computed in row chunks."""
    X = np.asarray(X, float)
    Z = X if Z is None else np.asarray(Z, float)
    out = np.empty((len(X), len(Z)))
    zz = (Z * Z).sum(1)
    for a in range(0, len(X), chunk):
        Xa = X[a:a + chunk]
        sq = (Xa * Xa).sum(1)[:, None] + zz[None, :] - 2.0 * Xa @ Z.T
        np.maximum(sq, 0.0, out=sq)
        out[a:a + chunk] = np.sqrt(sq)
    if Z is X:
        np.fill_diagonal(out, 0.0)
    return out


def lipschitz_constants(R, E):
    """L* and eta* of a target R relative to the Euclidean matrix E."""
    iu = np.triu_indices(len(R), 1)
    ratio = R[iu] / E[iu]
    return float(ratio.max()), float(ratio.min())


def triangle_defect(R):
    """tau(R) = max_{i,j,k} R_ij - R_ik - R_kj (vectorised over k)."""
    best = -np.inf
    for k in range(len(R)):
        V = R - R[:, [k]] - R[[k], :]
        V[k, :] = -np.inf
        V[:, k] = -np.inf
        np.fill_diagonal(V, -np.inf)
        best = max(best, float(V.max()))
    return best


class ExactExtension:
    def __init__(self, Y, R, E=None):
        self.Y = np.asarray(Y, float)
        self.R = np.asarray(R, float)
        E = pdist_matrix(self.Y) if E is None else E
        self.L, self.eta = lipschitz_constants(self.R, E)

    def phi(self, x):
        r = self.L * np.sqrt(((self.Y - x) ** 2).sum(1))
        return np.min(self.R + r[:, None], axis=0)

    def phi_many(self, X):
        return np.array([self.phi(x) for x in np.asarray(X, float)])

    def __call__(self, x, y):
        x = np.asarray(x, float)
        y = np.asarray(y, float)
        return max(float(np.abs(self.phi(x) - self.phi(y)).max()),
                   self.eta * float(np.linalg.norm(x - y)))

    def matrix(self, X):
        """d on an arbitrary finite set X (rows)."""
        from scipy.spatial.distance import cdist
        P = self.phi_many(X)
        D = cdist(P, P, "chebyshev")
        return np.maximum(D, self.eta * pdist_matrix(X))


class TwoLevelExtension:
    """Exact metric realising the two-level target of a labelling."""

    def __init__(self, Y, labels, s, D, optimal=True):
        self.Y = np.asarray(Y, float)
        self.lab = np.asarray(labels)
        self.s, self.D = float(s), float(D)
        self.classes = np.unique(self.lab)
        if optimal:
            self.L, self.eta = self._constants_exact()
        else:
            self.L, self.eta = self._constants_fast()

    def _constants_exact(self, chunk=512):
        """L* and eta* in O(m^2) time and O(chunk * m) memory."""
        m = len(self.Y)
        L, eta = 0.0, np.inf
        zz = (self.Y ** 2).sum(1)
        for a in range(0, m, chunk):
            Ya = self.Y[a:a + chunk]
            sq = (Ya ** 2).sum(1)[:, None] + zz[None, :] - 2.0 * Ya @ self.Y.T
            np.maximum(sq, 0.0, out=sq)
            E = np.sqrt(sq)
            same = self.lab[a:a + chunk, None] == self.lab[None, :]
            idx = np.arange(a, min(a + chunk, m))
            E[np.arange(len(idx)), idx] = np.nan
            T = np.where(same, self.s, self.D) / E
            L = max(L, float(np.nanmax(T)))
            eta = min(eta, float(np.nanmin(T)))
        return L, eta

    def _constants_fast(self):
        """Valid (not optimal) constants in O(m log m): closest pair and a diameter bound."""
        from scipy.spatial import cKDTree
        tree = cKDTree(self.Y)
        dd, _ = tree.query(self.Y, k=2)
        dmin = float(dd[:, 1].min())
        diam = 2.0 * float(np.sqrt(((self.Y - self.Y.mean(0)) ** 2).sum(1)).max())
        return max(self.s, self.D) / dmin, min(self.s, self.D) / diam

    def rho(self):
        R = np.where(self.lab[:, None] == self.lab[None, :], self.s, self.D)
        np.fill_diagonal(R, 0.0)
        return R

    def phi(self, x):
        e = self.L * np.sqrt(((self.Y - x) ** 2).sum(1))
        out = np.empty(len(self.Y))
        for c in self.classes:
            idx = np.flatnonzero(self.lab == c)
            ec = e[idx]
            o = np.argsort(ec)[:2]
            first, arg1 = ec[o[0]], idx[o[0]]
            second = ec[o[1]] if len(o) > 1 else np.inf
            others = e[self.lab != c]
            best_out = others.min() if len(others) else np.inf
            nn_same = np.where(idx == arg1, second, first)
            out[idx] = np.minimum(np.minimum(ec, self.s + nn_same), self.D + best_out)
        return out

    def __call__(self, x, y):
        x = np.asarray(x, float)
        y = np.asarray(y, float)
        return max(float(np.abs(self.phi(x) - self.phi(y)).max()),
                   self.eta * float(np.linalg.norm(x - y)))


def reflect_across_bisector(y, a, b):
    """Mirror point y across the perpendicular bisector of the segment [a, b]."""
    n = (b - a) / np.linalg.norm(b - a)
    c = 0.5 * (a + b)
    return y - 2.0 * ((y - c) @ n) * n
