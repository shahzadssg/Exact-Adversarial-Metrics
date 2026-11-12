"""Generate LaTeX tables from the experiment results (no hand transcription)."""
import json, os
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "..", "results")
TAB = os.path.join(HERE, "..", "paper", "tables")
os.makedirs(TAB, exist_ok=True)


def sci(x, digits=1):
    if x == 0:
        return "0"
    e = int(np.floor(np.log10(abs(x))))
    mant = x / 10 ** e
    return f"{mant:.{digits}f}\\times10^{{{e}}}"


def fmt(x):
    return "0" if x == 0 else f"{x:.3f}"


def write(name, body):
    open(os.path.join(TAB, name + ".tex"), "w").write(body)
    print("wrote", name)


# ---------------- Table 1: conference artefact ----------------
e0 = json.load(open(os.path.join(RES, "e0.json")))
a = e0["artifact"]
eps = a["eps"]
rows = [
    (r"Triangle defect $\tau$ of the clipped targets on the 12 construction points",
     f"${a['tau']:.2f}$ ($ {a['tau']/eps:.0f}\\eps$)"),
    (r"Data triples that violate the triangle inequality by more than $\eps$",
     f"{a['data_triples_beyond_eps']} of {a['data_triples']} ({100*a['data_triples_beyond_eps']/a['data_triples']:.1f}\\%)"),
    (r"Largest defect on triples of 40 fresh points",
     f"${a['fresh_max_defect']:.1f}$ ($ {a['fresh_max_defect']/eps:.0f}\\eps$)"),
    (r"Fresh triples beyond $\eps$",
     f"{a['fresh_beyond_eps']} of {a['fresh_triples']} ({100*a['fresh_beyond_eps']/a['fresh_triples']:.1f}\\%)"),
    (r"Largest defect on triples with one fresh and two data points",
     f"${a['mixed_max_defect']:.1f}$ ($ {a['mixed_max_defect']/eps:.0f}\\eps$)"),
    (r"Mixed triples beyond $\eps$",
     f"{a['mixed_beyond_eps']} of {a['mixed_triples']} ({100*a['mixed_beyond_eps']/a['mixed_triples']:.1f}\\%)"),
    (r"$\dt(y_i+10^{-9}u,\,y_i)$, maximum over 60 random displacements",
     f"${a['jump_1e9_max']:.1f}$"),
    (r"Displacements with $\dt$ above $\delta_{\mathrm{small}}=" + f"{a['small']:.4f}" + "$",
     f"{100*a['jump_1e9_frac_above_small']:.0f}\\%"),
    (r"Noise amplification $\norm{n}_q/\norm{n}_2$, median and maximum",
     f"{a['noise_amp_median']:.0f} and {a['noise_amp_max']:.0f}"),
    (r"DBSCAN radius $(\delta_{\mathrm{small}}+\delta_{\mathrm{large}})/2$ versus largest target",
     f"{a['dbscan_radius']:.0f} versus {a['max_target']:.2f}"),
]
body = r"""\begin{table}[t]
\centering
\caption{Behaviour of the $\eps$-semimetric of the conference Adult experiment~\cite{ahmad2026secrypt} ($m=12$ construction points, $\eps=""" + f"{eps:.4f}" + r"""$), evaluated with the matrix $A$ saved by the conference code. Fresh points are uniform in the bounding box of the data.}\label{tab:reexam}
\footnotesize
\begin{tabular}{p{0.66\textwidth}r}
\toprule
Test & Result\\
\midrule
""" + "\n".join(f"{t} & {v}\\\\" for t, v in rows) + r"""
\bottomrule
\end{tabular}
\end{table}
"""
write("reexam", body)

# ---------------- Table 2: double precision ----------------
dp = e0["double_precision"]
lines = []
for m in sorted({r["m"] for r in dp}):
    rs = [r for r in dp if r["m"] == m]
    h = m * (m - 1) // 2
    cond = np.median([r["condM"] for r in rs])
    indef = sum(r["indefinite"] for r in rs)
    im = sum(r["imag_intra"] for r in rs); tot = sum(r["n_intra"] for r in rs)
    rel_in = np.median([r["relerr_intra_median"] for r in rs])
    rel_out = max(r["relerr_inter_median"] for r in rs)
    rin = f"{rel_in:.2g}" if 1e-2 <= rel_in < 1e3 else sci(rel_in)
    lines.append(f"{m} & {h} & ${sci(cond)}$ & {indef}/5 & {im}/{tot} ({100*im/tot:.0f}\\%) & ${rin}$ & ${sci(rel_out)}$\\\\")
body = r"""\begin{table}[t]
\centering
\caption{The construction of~\cite{rass2024,ahmad2026secrypt} in double precision on the synthetic pipeline of the conference version (three random clusters in $\R^2$, five random draws per $m$). Columns report the median condition number of $V$, runs with an indefinite $A$, intra-cluster pairs whose distance comes out imaginary, and median relative errors $|q(v)-\rho^2|/\rho^2$ of intra-cluster and inter-cluster targets. Rounding dominates the counts, so they vary with the BLAS and LAPACK build.}\label{tab:double}
\footnotesize
\begin{tabular}{rrrrrrr}
\toprule
$m$ & $h$ & $\kappa(V)$ & indefinite & imaginary (intra) & rel.\ err.\ intra & rel.\ err.\ inter\\
\midrule
""" + "\n".join(lines) + r"""
\bottomrule
\end{tabular}
\end{table}
"""
write("double", body)

# ---------------- Table 3: scalability ----------------
e1 = json.load(open(os.path.join(RES, "e1.json")))
lines = []
for r in e1:
    m = r["m"]
    mm = f"$10^{{{int(np.log10(m))}}}$" if np.log10(m) == int(np.log10(m)) else f"${m//10**int(np.log10(m))}\\cdot10^{{{int(np.log10(m))}}}$"
    opt = f"{r['t_optimal']:.2f}" if "t_optimal" in r else "--"
    db = f"{r['t_dbscan_end_to_end']:.2f} / {r['ari_dbscan']:.3f}" if "t_dbscan_end_to_end" in r else "--"
    ext = r["eps_semimetric_extrapolated_s"]
    lines.append(f"{mm} & {opt} & {r['t_fast']:.3f} & {1000*r['t_query']:.1f} & {db} & $\\approx {sci(ext, 0).replace('1\\times','')}$\\\\")
body = r"""\begin{table}[t]
\centering
\caption{Cost of the exact two-level metric for random three-cluster targets in $\R^2$ (seconds on two CPU cores). ``Optimal'' computes $\Lst$ and $\est$ exactly in $O(m^2)$ time, ``near-optimal'' uses the $O(m\log m)$ constants. ``Query'' evaluates $d$ on a pair of new points. The last column extrapolates the fit $t\approx 7\cdot10^{-9}m^{5.72}$\,s of the conference version~\cite[Table~1]{ahmad2026secrypt}, which measured $11{,}038$\,s at $m=130$.}\label{tab:scalability}
\footnotesize
\begin{tabular}{rrrrrr}
\toprule
$m$ & optimal (s) & near-opt.\ (s) & query (ms) & DBSCAN (s) / ARI & $\eps$-semimetric (s)\\
\midrule
""" + "\n".join(lines) + r"""
\bottomrule
\end{tabular}
\end{table}
"""
write("scalability", body)

# ---------------- Table 4: library algorithms ----------------
e2 = json.load(open(os.path.join(RES, "e2.json")))
params = {
    "DBSCAN": r"$r=0.5$, $\mathit{minPts}=5$",
    "HDBSCAN": r"min.\ cluster size $5$",
    "OPTICS": r"$\mathit{minPts}=5$, $\xi=0.05$",
    "single linkage": r"$k$ clusters",
    "complete linkage": r"$k$ clusters",
    "average linkage": r"$k$ clusters",
    "k-medoids (FasterPAM)": r"$k$, random init.",
    "spectral (kNN graph)": r"$k$, $10$ neighbours",
}
names = {"k-medoids (FasterPAM)": "$k$-medoids (FasterPAM)", "spectral (kNN graph)": "spectral ($k$NN graph)"}
lines = []
for alg in params:
    small = [r for r in e2 if r["alg"] == alg and r["m"] <= 3000]
    big = [r for r in e2 if r["alg"] == alg and r["m"] == 10000]
    mn = min(r["ari"] for r in small)
    noise = sum(r["noise"] for r in small)
    bigs = f"{big[0]['ari']:.3f} & {big[0]['seconds']:.1f}" if big else "-- & --"
    lines.append(f"{names.get(alg, alg)} & {params[alg]} & {len(small)} & {mn:.3f} & {noise} & {bigs}\\\\")
body = r"""\begin{table}[t]
\centering
\caption{Unmodified library algorithms under the exact two-level metric ($s=0.25$, $D=1$) for random target partitions. Runs cover $m\in\{300,3000\}$, $k\in\{2,3,5\}$ and five seeds. Hyperparameters are library defaults. ARI is the adjusted Rand index between output and target, and ``noise'' counts points labelled as noise.}\label{tab:library}
\footnotesize
\begin{tabular}{llrrrrr}
\toprule
 & & \multicolumn{3}{c}{$m\leq 3000$} & \multicolumn{2}{c}{$m=10^4$, $k=3$}\\
\cmidrule(lr){3-5}\cmidrule(lr){6-7}
Algorithm & Hyperparameters & runs & min ARI & noise & ARI & time (s)\\
\midrule
""" + "\n".join(lines) + r"""
\bottomrule
\end{tabular}
\end{table}
"""
write("library", body)

# ---------------- Table 5: fairness ----------------
e3 = json.load(open(os.path.join(RES, "e3.json")))
cols5 = []
for info in e3["info"]:
    ds = info["dataset"]
    rows = [r for r in e3["rows"] if r["dataset"] == ds]
    best = {st: max([r["g"] for r in rows if r["strategy"] == st], default=0) for st in ("border", "stealthy", "random")}
    g = min(info["g_res"], best["stealthy"])
    r = [x for x in rows if x["strategy"] == "stealthy" and x["g"] == g][0]
    r10 = [x for x in rows if x["strategy"] == "stealthy" and x["g"] == 10][0]
    b = info["baseline"]
    lo, hi = info["dpd_ci"]
    cols5.append(dict(name=ds, n=str(info["n"]), prot=f"{info['groups'][0]} ({info['n_prot']})",
                      dpd=f"${b['dpd']:.3f}$", ci=f"$[{lo:.3f},\\,{hi:.3f}]$", di=f"${b['di']:.2f}$",
                      gres=str(info["g_res"]), flips=f"{best['border']}/{best['stealthy']}/{best['random']}",
                      g=str(g), dpd2=f"${r['dpd']:.3f}$", di2=f"${r['di']:.2f}$",
                      k10=f"${r10['distortion']:.2f}$ (${r10['lower_bound']:.2f}$)"))
spec = [("Records $n$", "n"), ("Protected group ($n_p$)", "prot"), ("Honest DPD", "dpd"),
        ("DPD, $95\\%$ interval", "ci"), ("Honest DI", "di"), ("$g_{\\mathrm{res}}$", "gres"),
        ("Clean flips (b/s/r)", "flips"), ("Evaluated flips $g$", "g"),
        ("DPD after $g$ flips", "dpd2"), ("DI after $g$ flips", "di2"),
        ("$\\kappa(d)$ at $g=10$ (floor)", "k10")]
lines = [f"{lab} & " + " & ".join(c[key] for c in cols5) + "\\\\" for lab, key in spec]
body = r"""\begin{table}[t]
\centering
\caption{Targeted attacks on $k$-medoids ($k=2$, FasterPAM, BUILD initialisation). $g_{\mathrm{res}}$ is the number of flips that shifts the DPD by the half-width of its $95\%$ bootstrap interval. Clean flips count targets that the border (b), stealthy (s) and random (r) adversaries move with unchanged medoids and no other changed decision, with at most $150$ candidates tried and $100$ targets moved. The evaluated stealthy attack uses $g=\min(g_{\mathrm{res}},\text{clean stealthy flips})$ targets. The last row reports the distortion after ten stealthy flips and, in parentheses, the floor of Theorem~\ref{thm:floor}.}\label{tab:fairness}
\footnotesize
\begin{tabular}{lrrr}
\toprule
 & Adult & COMPAS & German Credit\\
\midrule
""" + "\n".join(lines) + r"""
\bottomrule
\end{tabular}
\end{table}
"""
write("fairness", body)

# ---------------- Table 6: detection ----------------
p4 = os.path.join(RES, "e4.json")
if os.path.exists(p4):
    e4 = json.load(open(p4))
    if isinstance(e4, dict) and "auc" in e4:
        auc = e4["auc"]
        cols = ["relocation-stealthy-g1", "relocation-stealthy-g3", "relocation-stealthy-g10",
                "relocation-random-g1", "relocation-random-g3", "relocation-random-g10", "two-level (random labels)"]
        head = "Statistic & 1 & 3 & 10 & 1 & 3 & 10 & labels\\\\"
        lines = []
        for st, lab in (("axiom", r"$S_{\mathrm{axiom}}$"), ("embed", r"$S_{\mathrm{embed}}$"), ("distort", r"$S_{\mathrm{distort}}$")):
            lines.append(lab + " & " + " & ".join(f"{auc[st][c]:.2f}" for c in cols) + "\\\\")
        rows = e4["rows"]
        fams = ["weighted Euclidean", "Mahalanobis", "Manhattan", "Chebyshev", "cosine", "NCA"]
        med = []
        tol = e4.get("tolerance", 1e-9)
        cl = lambda x: x if x >= tol else 0.0
        for f in fams:
            v = [r for r in rows if r["variant"] == f]
            med.append((f, np.median([cl(r["axiom"]) for r in v]), np.median([cl(r["embed"]) for r in v]), np.median([np.exp(r["distort"]) for r in v])))
        for c in cols:
            v = [r for r in rows if r["variant"] == c]
            med.append((c, np.median([cl(r["axiom"]) for r in v]), np.median([cl(r["embed"]) for r in v]), np.median([np.exp(r["distort"]) for r in v])))
        body = r"""\begin{table}[t]
\centering
\caption{Black-box detection. Top: ROC AUC of each statistic for separating the pooled benign metrics from each attack (three datasets, 20 subsamples of 400 points each, $g$ relocated targets). An AUC of $0.5$ means chance, and values below $0.5$ mean that attacks look more benign than legitimate metrics. Values of $S_{\mathrm{axiom}}$ and $S_{\mathrm{embed}}$ below $10^{-9}$ count as zero, which absorbs floating-point rounding. Bottom: median statistic per metric.}\label{tab:detection}
\footnotesize
\begin{tabular}{lrrrrrrr}
\toprule
 & \multicolumn{3}{c}{stealthy, $g=$} & \multicolumn{3}{c}{random, $g=$} & two-level\\
\cmidrule(lr){2-4}\cmidrule(lr){5-7}\cmidrule(lr){8-8}
""" + head + "\n\\midrule\n" + "\n".join(lines) + r"""
\bottomrule
\end{tabular}

\vspace{2mm}
\begin{tabular}{lrrr}
\toprule
Metric & $S_{\mathrm{axiom}}$ & $S_{\mathrm{embed}}$ & $\kappa=e^{S_{\mathrm{distort}}}$\\
\midrule
""" + "\n".join(f"{n.replace('relocation-', 'attack, ').replace('-g', ', $g=').replace('two-level (random labels)', 'attack, two-level') + ('$' if '-g' in n else '')} & {fmt(ax)} & {fmt(em)} & {kp:,.2f}\\\\" for n, ax, em, kp in med) + r"""
\bottomrule
\end{tabular}
\end{table}
"""
        write("detection", body)
