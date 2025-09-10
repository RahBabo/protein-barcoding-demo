import argparse, os, numpy as np, pandas as pd
from statsmodels.stats.multitest import multipletests

def summarize(df):
    agg = df.groupby(["id","condition"])["count"].sum().unstack(fill_value=0)
    agg = agg.rename(columns=lambda c: c.lower())
    for c in ["control","target"]:
        if c not in agg.columns: agg[c] = 0
    agg["log2_fc"] = np.log2((agg["target"] + 1) / (agg["control"] + 1))
    return agg.reset_index()

def fdr_select(agg):
    x = agg["log2_fc"].values
    # simple two-sided z from centered distribution (toy only)
    z = (x - np.mean(x)) / (np.std(x) + 1e-8)
    p = 2 * (1 - 0.5 * (1 + np.math.erf(np.abs(z) / np.sqrt(2))))
    _, q, _, _ = multipletests(p, alpha=0.05, method="fdr_bh")
    out = agg.copy()
    out["z"] = z
    out["q"] = q
    return out.sort_values("q")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--counts", required=True)
    ap.add_argument("--out", default="results/selected_hits.csv")
    args = ap.parse_args()

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    df = pd.read_csv(args.counts)
    agg = summarize(df)
    res = fdr_select(agg)
    res.to_csv(args.out, index=False)
    print(res.head(10))

if __name__ == "__main__":
    main()
