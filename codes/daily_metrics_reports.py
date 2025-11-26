# -*- coding: utf-8 -*-
"""
@author: Jianpeng Cao
"""

# save as: daily_metrics_report.py
# usage:
#   python daily_metrics_report.py --csv path/to/your_backtest_daily.csv
#   (outputs: daily_metrics_summary.csv, component_means.csv, component_var_shares.csv)

import argparse
import numpy as np
import pandas as pd

def _pick_pnl_col(df: pd.DataFrame) -> str:
    cands = [c for c in df.columns if c.lower() in
             ("pnl_total","pnl","daily_pnl","pnl_sum","pnl_total_sum")]
    if not cands:
        raise ValueError("PnL column not found (expected: PnL_total / pnl / daily_pnl ...)")
    return cands[0]

def _t_stat(pnl: pd.Series):
    pnl = pd.to_numeric(pnl, errors="coerce").dropna()
    n = len(pnl)
    if n <= 1:
        return np.nan, np.nan
    mu = pnl.mean()
    sd = pnl.std(ddof=1)
    if sd <= 0:
        return np.nan, np.nan
    t = mu / (sd / np.sqrt(n))
    return t, n-1  # return only t and degrees of freedom (simple & robust; use scipy/statsmodels if p-value is needed)

def _ac1(pnl: pd.Series):
    pnl = pd.to_numeric(pnl, errors="coerce").dropna()
    if len(pnl) < 3:
        return np.nan
    return float(pnl.autocorr(lag=1))

def _max_drawdown_and_final(pnl: pd.Series):
    pnl = pd.to_numeric(pnl, errors="coerce").dropna()
    if len(pnl) == 0:
        return np.nan, np.nan
    cum = pnl.cumsum()
    dd = cum - cum.cummax()
    maxdd = float(dd.min())       # ≤ 0
    final = float(cum.iloc[-1])   # final cumulative PnL
    return maxdd, final

def _rmse(pnl: pd.Series):
    x = pd.to_numeric(pnl, errors="coerce").dropna()
    if len(x) == 0:
        return np.nan
    return float(np.sqrt(np.mean(np.square(x))))

def _component_means(df: pd.DataFrame, comp_cols):
    cols = [c for c in comp_cols if c in df.columns]
    if not cols:
        return pd.DataFrame()
    return df[cols].mean().to_frame(name="mean_per_day").T

def _component_var_shares(df: pd.DataFrame, comp_cols, include_cash_tc=True):
    cols = [c for c in comp_cols if c in df.columns]
    extra = []
    if include_cash_tc:
        for x in ("cash_pnl","TC","tc","transaction_cost"):
            if x in df.columns: extra.append(x)
    cols_all = cols + extra
    if not cols_all:
        return pd.DataFrame()

    # Total PnL (note: TC is a cost, so it should be subtracted; if daily_df already has PnL_total including cash/costs, use it directly)
    if "PnL_total" in df.columns:
        Y = pd.to_numeric(df["PnL_total"], errors="coerce")
    else:
        # If PnL_total is missing, construct it explicitly: sum(components + cash_pnl - TC)
        tmp = df[cols].sum(axis=1)
        if "cash_pnl" in df.columns:
            tmp = tmp + pd.to_numeric(df["cash_pnl"], errors="coerce").fillna(0.0)
        if "TC" in df.columns:
            tmp = tmp - pd.to_numeric(df["TC"], errors="coerce").fillna(0.0)
        Y = tmp

    Z = pd.concat([df[cols_all], Y.rename("_TOTAL")], axis=1).apply(pd.to_numeric, errors="coerce")
    cov = Z.cov(min_periods=1)
    varY = cov.loc["_TOTAL","_TOTAL"]
    if not np.isfinite(varY) or varY == 0.0:
        # Total variance is 0 (or unavailable), cannot allocate contributions
        return pd.DataFrame()

    shares = cov.loc[cols_all, "_TOTAL"] / varY
    out = shares.to_frame(name="var_share").T
    out.attrs["sum_shares"] = float(shares.sum())  # 应≈1
    return out

def summarize_daily_metrics(daily_csv: str, out_prefix: str = "daily_metrics"):
    df = pd.read_csv(daily_csv)
    pnl_col = _pick_pnl_col(df)
    pnl = pd.to_numeric(df[pnl_col], errors="coerce").dropna()

    # Core metrics
    n = len(pnl)
    mean = float(pnl.mean()) if n else np.nan
    std  = float(pnl.std(ddof=1)) if n>1 else np.nan
    rmse = _rmse(pnl)
    t, dfree = _t_stat(pnl)
    sharpe = (mean/std*np.sqrt(252)) if (std is not None and std>0 and np.isfinite(std)) else np.nan
    hit = float((pnl > 0).mean()) if n else np.nan
    ac1 = _ac1(pnl)
    maxdd, final = _max_drawdown_and_final(pnl)

    summary = pd.DataFrame([{
        "N_days": n,
        "mean": mean,
        "std": std,
        "RMSE": rmse,
        "t_stat": float(t) if t==t else np.nan,
        "t_df": dfree,
        "Sharpe_annualized": float(sharpe) if sharpe==sharpe else np.nan,
        "Hit_rate": hit,
        "AC1": ac1,
        "MaxDD": maxdd,
        "FinalCumPnL": final
    }])

    # Component statistics (based on the columns exported by your backtest)
    comp_cols = ["TH","UM_delta_hedged","SM","ME"]
    means = _component_means(df, comp_cols)
    shares = _component_var_shares(df, comp_cols, include_cash_tc=True)

    # Save files
    summary.to_csv(f"{out_prefix}_summary.csv", index=False)
    if not means.empty:
        means.to_csv(f"{out_prefix}_component_means.csv", index=False)
    if not shares.empty:
        shares.to_csv(f"{out_prefix}_component_var_shares.csv", index=False)

    # Print concise results
    print(summary.to_string(index=False))
    if not means.empty:
        print("\nComponent means (per day):")
        print(means.to_string(index=False))
    if not shares.empty:
        print("\nComponent variance shares (Cov(Xk,Y)/Var(Y), sum≈1):")
        print(shares.to_string(index=False))
        if "sum_shares" in shares.attrs:
            print("Check sum of shares:", shares.attrs["sum_shares"])

if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Compute daily PnL metrics and component contributions from backtest daily CSV")
    ap.add_argument("--csv", required=True, help="path to *_daily.csv (must include PnL_total column)")
    ap.add_argument("--out_prefix", default="daily_metrics")
    args = ap.parse_args()
    summarize_daily_metrics(args.csv, args.out_prefix)
