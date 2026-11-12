# Exact Adversarial Metrics: code and results

This package reproduces every table and figure of

> Shahzad Ahmad, Stefan Rass and Enes Sovtic.
> **Exact Adversarial Metrics: Scalable Fairness Attacks on Clustering-Based Decisions and Committed Audits.**
> Extended version of the SECRYPT 2026 paper *Adversarial Distance Metrics: A Threat to Fairness in Clustering-Based Decision Systems*.

It contains the exact adversarial metric of Theorem 1, a Python port of the conference construction, the experiments E0 to E4, the reference results behind the paper and a notebook that runs everything and checks the results.

## Contents

```
README.md
Exact_Adversarial_Metrics.ipynb   runs E0 to E4, checks the results, shows tables and figures
code/
  exactmetric.py          exact extension (Theorem 1) and the two-level metric
  adm_port.py             port of the conference MATLAB code (compute_metric.m, epsilon_semimetric.m)
  datasets.py             loaders and preprocessing for Adult, COMPAS and German Credit
  e0_reexamine.py         E0: artefacts of the conference epsilon-semimetric
  e1_scalability.py       E1: cost of the exact construction
  e2_library_attacks.py   E2: unmodified library algorithms
  e3_fairness.py          E3: targeted fairness attacks
  e4_detection.py         E4: black-box detection
  e4_auc.py               E4: AUCs with a 1e-9 tolerance
  thread_check.py         FasterPAM thread-count check
  make_tables.py          Tables 1 to 6 as LaTeX
  make_figures.py         Figures 1 and 2
results/                  reference results behind the paper (e0.json to e4.json)
data/                     datasets and conference artefacts, from data.zip (not in this archive)
```

## Requirements

Python 3.12 or newer and

```
pip install numpy scipy pandas scikit-learn matplotlib kmedoids
```

scikit-learn must be version 1.3 or newer, because E2 uses HDBSCAN. The `kmedoids` package provides FasterPAM. The notebook installs missing packages itself.

The reference results come from Linux with Python 3.13.16, NumPy 2.5.3, SciPy 1.18.1, pandas 3.0.5, scikit-learn 1.9.1, matplotlib 3.11.2, kmedoids 0.5.5 and joblib 1.6.0, on two CPU cores with 7 GB of memory. On Windows, with Python 3.12.4 and scikit-learn 1.5.1, the notebook passed the reproducibility check as well.

## Data

The experiments read these files from `data/`.

| File | Source | Used by |
|---|---|---|
| `adult.data` | UCI Machine Learning Repository, Adult | E3, E4 |
| `german.data` | UCI Machine Learning Repository, Statlog (German Credit Data) | E3, E4 |
| `compas-scores-two-years.csv` | ProPublica, `compas-analysis` repository on GitHub | E3, E4, `thread_check.py` |
| `artifacts/As_m12_double.mat` | conference code, `Implementation/saved_variables` | E0 Part A |
| `artifacts/As_m32_double.mat` | conference code, `Implementation/saved_variables` | E0 Part A |

The two `.mat` files hold the matrices A that the conference code saved for its Adult experiment (m = 12) and for a further run (m = 32). E0 Part B, E1 and E2 use synthetic data only. `datasets.py` documents the preprocessing, and Section 6.1 of the paper summarises it.

`data.zip` contains this `data/` folder. Without it, the notebook tries to download the three datasets from the URLs in its setup cell. It then runs E0 Part B only, because the two artefacts come only with `data.zip`.

## Quick start with the notebook

1. Unzip this archive and `data.zip` into the same folder, so that `code/`, `data/` and `results/` sit side by side.
2. Start Jupyter in that folder and open `Exact_Adversarial_Metrics.ipynb`.
3. Run all cells.

The setup cell installs missing packages, finds the data and sets the working directory to `code/`. Re-running it is safe. The module cells write `exactmetric.py`, `adm_port.py` and `datasets.py` into `code/`, identical to the shipped files. The cells E0 to E4 write `results/e0.json` to `results/e4.json`. The section "Reproducibility check" compares the run with reference values stored in the notebook. A correct run prints

```
DIFF items: 0
```

Items marked `varies` depend on the hardware or on the BLAS and LAPACK build. The check lists them for information only (see the notes below). The notebook then draws Figures 1 and 2.

## Running the scripts

Run the scripts from `code/` in this order.

```
cd code
python e0_reexamine.py         # Tables 1 and 2, needs data/artifacts
python e1_scalability.py       # Table 3
python e2_library_attacks.py   # Table 4
python e3_fairness.py          # Table 5 and Fig. 1
python e4_detection.py         # Table 6 and Fig. 2
python e4_auc.py               # recomputes the E4 AUCs with a 1e-9 tolerance
python make_tables.py          # writes ../paper/tables/*.tex
python make_figures.py         # writes ../paper/figures/*.pdf and *.png
python thread_check.py         # optional, see the notes below
```

Each experiment writes its JSON file into `results/` and replaces the shipped reference file. Keep a copy of the archive if you want to compare JSON files. `make_tables.py` and `make_figures.py` read only `results/`, so they also rebuild the paper's tables and figures from the shipped results without running any experiment. E1 to E4 take several minutes each on the reference machine. `thread_check.py` takes under a minute.

## Mapping to the paper

| Script | Paper | Output |
|---|---|---|
| `e0_reexamine.py` | Section 4.4, Tables 1 and 2 | `results/e0.json` |
| `e1_scalability.py` | Section 6.2, Table 3 | `results/e1.json` |
| `e2_library_attacks.py` | Section 6.3, Table 4 | `results/e2.json` |
| `e3_fairness.py` | Section 6.4, Table 5, Fig. 1 | `results/e3.json` |
| `e4_detection.py`, `e4_auc.py` | Section 6.5, Table 6, Fig. 2 | `results/e4.json` |
| `thread_check.py` | Section 7.2 | console output |
| `make_tables.py` | Tables 1 to 6 | `paper/tables/*.tex` |
| `make_figures.py` | Figs. 1 and 2 | `paper/figures/*.pdf`, `*.png` |

`exactmetric.py` implements Theorem 1. `ExactExtension` extends a finite metric on the data to an exact metric on the feature space with optimal distortion. `TwoLevelExtension` specialises it to the two-level targets of Proposition 2 and evaluates Φ at a point in O(mℓ) time. `adm_port.py` ports the conference MATLAB construction line by line, and E0 uses it to measure the conference artefacts.

## Reproducibility notes

**Seeds.** Every experiment fixes its random seeds.

**FasterPAM threads.** E3 and E4 call `kmedoids.fasterpam` with `n_cpu=1`. With several threads, the solver can return different individuals with identical features as the same medoid, and its choice can change from run to run. `thread_check.py` shows this on the honest COMPAS pipeline. In every run, the medoid feature vectors and all decisions agree. E3 also compares medoids by feature vector, so its acceptance test does not depend on which of these individuals the solver returns. E2 does not fix the thread count. By Theorem 2(c), every converged FasterPAM run returns the target partition, so the thread count cannot change its result.

**Platform-dependent results.** Timings in Tables 3 and 4 depend on the hardware. The counts in Table 2 depend on the BLAS and LAPACK build, because rounding dominates them. The Windows run gave 23 indefinite matrices instead of 24, and every condition number agreed to five digits. The E4 AUCs can differ in the fourth decimal, and the check allows a difference of 0.005. All other checked quantities matched on both platforms.

**Windows and joblib.** joblib looks up the number of physical cores through `wmic`, which recent Windows versions no longer ship. The failed lookup prints a harmless traceback the first time scikit-learn needs it, during spectral clustering in E2. The notebook fills joblib's cache beforehand and avoids the message. When you run the scripts directly, ignore the message or set the environment variable `LOKY_MAX_CPU_COUNT` to a value below the number of logical cores.

**MATLAB random numbers.** For positive seeds, MATLAB's `rng(seed)` and NumPy's `RandomState(seed)` produce the same Mersenne Twister stream. `adm_port.py` therefore reproduces the lifted vectors of the conference code exactly. E0 Part B uses its own draws, because MATLAB's normal generator differs from NumPy's.

## Citation

If you use this code, please cite the conference paper.

```bibtex
@inproceedings{ahmad2026secrypt,
  author    = {Ahmad, Shahzad and Rass, Stefan and Sovtic, Enes},
  title     = {Adversarial Distance Metrics: A Threat to Fairness in Clustering-Based Decision Systems},
  booktitle = {Proceedings of the 23rd International Conference on Security and Cryptography ({SECRYPT} 2026)},
  editor    = {De Capitani di Vimercati, Sabrina and Samarati, Pierangela},
  publisher = {SCITEPRESS},
  address   = {Porto, Portugal},
  year      = {2026}
}
```

## Contact

Shahzad Ahmad, LIT Secure and Correct Systems Lab, Johannes Kepler University Linz, shahzad.ahmad@jku.at
