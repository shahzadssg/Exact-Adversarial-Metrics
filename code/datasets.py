"""Loaders for Adult, COMPAS (ProPublica filter) and German Credit.

Each loader returns a dict with
  X      raw numeric features (n x ell), float
  names  feature names
  a      protected-group indicator (1 = protected group)
  ylab   ground-truth outcome, 1 = favourable outcome (>50K, no recidivism, good credit)
  groups (protected name, reference name)
"""
import os
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "..", "data")


def adult(n=5000, seed=0):
    cols = ["age", "workclass", "fnlwgt", "education", "education_num", "marital", "occupation",
            "relationship", "race", "sex", "capital_gain", "capital_loss", "hours", "country", "income"]
    df = pd.read_csv(os.path.join(DATA, "adult.data"), header=None, names=cols,
                     skipinitialspace=True, na_values="?")
    df = df[df["race"].isin(["White", "Black"])].reset_index(drop=True)
    if n is not None and n < len(df):
        df = df.sample(n=n, random_state=seed).reset_index(drop=True)
    X = np.column_stack([df["age"], df["education_num"], df["hours"],
                         np.log1p(df["capital_gain"]), np.log1p(df["capital_loss"])]).astype(float)
    return dict(name="Adult", X=X,
                names=["age", "education", "hours/week", "log capital gain", "log capital loss"],
                a=(df["race"] == "Black").to_numpy().astype(int),
                ylab=(df["income"].str.strip() == ">50K").to_numpy().astype(int),
                groups=("Black", "White"))


def compas():
    df = pd.read_csv(os.path.join(DATA, "compas-scores-two-years.csv"))
    df = df[(df["days_b_screening_arrest"] <= 30) & (df["days_b_screening_arrest"] >= -30)
            & (df["is_recid"] != -1) & (df["c_charge_degree"] != "O") & (df["score_text"] != "N/A")]
    df = df[df["race"].isin(["African-American", "Caucasian"])].reset_index(drop=True)
    juv = df["juv_fel_count"] + df["juv_misd_count"] + df["juv_other_count"]
    X = np.column_stack([df["age"], df["priors_count"], juv,
                         (df["c_charge_degree"] == "F").astype(int)]).astype(float)
    return dict(name="COMPAS", X=X, names=["age", "priors", "juvenile counts", "felony charge"],
                a=(df["race"] == "African-American").to_numpy().astype(int),
                ylab=(df["two_year_recid"] == 0).to_numpy().astype(int),
                groups=("African-American", "Caucasian"))


def german():
    df = pd.read_csv(os.path.join(DATA, "german.data"), header=None, sep=r"\s+")
    X = np.column_stack([df[1], np.log(df[4]), df[7], df[10], df[12], df[15]]).astype(float)
    female = df[8].isin(["A92", "A95"]).to_numpy().astype(int)
    return dict(name="German Credit", X=X,
                names=["duration", "log amount", "installment rate", "residence", "age", "credits"],
                a=female, ylab=(df[20] == 1).to_numpy().astype(int),
                groups=("female", "male"))


def standardize(X):
    mu = X.mean(0)
    sd = X.std(0, ddof=0)
    sd[sd == 0] = 1.0
    return (X - mu) / sd


if __name__ == "__main__":
    for d in (adult(), compas(), german()):
        X = d["X"]
        u, inv, cnt = np.unique(X, axis=0, return_inverse=True, return_counts=True)
        unique_rows = int((cnt[inv.ravel()] == 1).sum())
        print(f"{d['name']}: n={len(X)} ell={X.shape[1]} protected={d['a'].sum()} "
              f"fav-rate={d['ylab'].mean():.3f} distinct={len(u)} rows-with-unique-vector={unique_rows}")
