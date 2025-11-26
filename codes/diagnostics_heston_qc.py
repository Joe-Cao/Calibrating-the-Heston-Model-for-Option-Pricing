# diagnostics_heston_qc.py
# -*- coding: utf-8 -*-
"""
@author: Jianpeng Cao

Diagnostics for single-day Heston calibration:
- Fit-quality metrics (price/IV errors, within-spread ratio, bucket stats)
- No-arbitrage checks on the *full option chain* (monotonicity, convexity, calendar, parity)
Works with the companion modules:
  - pick_points_bs.py  (provides build_dataframe)
  - calibrate_heston_bs.py (provides Heston pricer + BS tools + calibrate_heston_one_day)
Author: Jianpeng Cao
"""

import argparse
import json
from dataclasses import asdict
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd

# === Import your existing modules (keep file names as in your repo) ===
from pick_points_bs import build_dataframe
from calibrate_heston_bs import (
    HestonParams, heston_price, implied_vol_from_price, bs_vega,
    calibrate_heston_one_day, #estimate_forward_by_parity, infer_r_minus_q_from_forward
    estimate_forward_and_r_by_regression
)

# ------------------------------
# Utility
# ------------------------------
def _ensure_datetime(df: pd.DataFrame, cols=("quote_date","expiry")) -> pd.DataFrame:
    for c in cols:
        if c in df.columns and not np.issubdtype(df[c].dtype, np.datetime64):
            df[c] = pd.to_datetime(df[c])
    return df

def _as_float(x, default=np.nan):
    try:
        v = float(x)
        if np.isfinite(v):
            return v
    except Exception:
        pass
    return default

# ------------------------------
# 1) Build per-point model outputs & errors
# ------------------------------
def compute_point_level_errors(
    day_sel: pd.DataFrame,
    params: HestonParams,
    full_day_raw: Optional[pd.DataFrame] = None,
    r_const: float = 0.0,
    q_const: float = 0.0,
) -> pd.DataFrame:
    """
    For each selected point, compute:
      - model price (Heston) using given params
      - model IV (via BS inversion)
      - market IV (via BS inversion)
      - price/IV errors and vega-weighted errors
    If full_day_raw is provided, infer (r-q) from put-call parity per expiry and
    set q = r_const - (r-q).
    """
    rows = []
    for _, row in day_sel.iterrows():
        S = _as_float(row['spot'])
        K = _as_float(row['strike'])
        T = _as_float(row['t_selected'])
        mid = _as_float(row['mid'])
        cp = str(row['cp']).upper()
        expiry = row['expiry'] if 'expiry' in row else None

        if not (np.isfinite(S) and np.isfinite(K) and np.isfinite(T) and np.isfinite(mid)):
            continue

        # infer carry if possible
        r_rate, q_rate = r_const, q_const
        if full_day_raw is not None and expiry is not None and T>0 and S>0:
            r_rate, q_rate = estimate_forward_and_r_by_regression(full_day_raw, expiry)

        p_model = float(heston_price(S, K, T, r_rate, q_rate, params, cp))
        iv_mkt = implied_vol_from_price(S, K, T, r_rate, q_rate, mid, cp)
        iv_mod = implied_vol_from_price(S, K, T, r_rate, q_rate, p_model, cp)
        vega = bs_vega(S, K, T, r_rate, q_rate, iv_mkt if iv_mkt and np.isfinite(iv_mkt) else 0.0)
        w_liq = _as_float(row.get('liq_weight', 1.0), 1.0)

        rows.append({
            **{c: row[c] for c in day_sel.columns if c in ['quote_date','expiry','t_selected','t_target','k','k_target','strike','cp','mid','bid','ask','spread_abs','spread_pct','spot']},
            'r': r_rate, 'q': q_rate,
            'price_model': p_model,
            'iv_mkt': iv_mkt if iv_mkt is not None else np.nan,
            'iv_model': iv_mod if iv_mod is not None else np.nan,
            'vega': vega,
            'w_liq': w_liq,
            'err_price': p_model - mid,
            'err_iv': (iv_mod - iv_mkt) if (iv_mkt is not None and iv_mod is not None) else np.nan
        })

    res = pd.DataFrame(rows)
    return res

def aggregate_fit_metrics(points: pd.DataFrame) -> Dict[str, float]:
    """Compute headline fit metrics from per-point error table."""
    out: Dict[str,float] = {}
    if points.empty:
        return out

    # weights: liquidity and/or 1/vega
    vega = points['vega'].replace([0, np.nan], np.nan)
    inv_vega = 1.0 / vega
    w = points['w_liq'].fillna(1.0) * inv_vega.replace([np.inf, -np.inf], np.nan)
    w = w.fillna(w.median() if np.isfinite(w.median()) else 1.0)

    # IV errors
    iv_err = points['err_iv']
    mask_iv = iv_err.abs().notna()
    if mask_iv.any():
        w_iv = w[mask_iv]
        norm_w_iv = w_iv / w_iv.sum()
        out['IVRMSE'] = float(np.sqrt(np.sum(norm_w_iv * iv_err[mask_iv]**2)))
        out['IVMAE']  = float(np.sum(norm_w_iv * iv_err[mask_iv].abs()))
    else:
        out['IVRMSE'] = np.nan
        out['IVMAE'] = np.nan

    # price errors (unweighted and liq-weighted)
    pe = points['err_price']
    w_liq = points['w_liq'].fillna(1.0)
    out['Price_RMSE'] = float(np.sqrt(np.average(pe**2, weights=w_liq)))
    out['Price_MAE']  = float(np.average(pe.abs(), weights=w_liq))

    # within-spread ratio (using mid +/- 0.5*spread_abs)
    if 'spread_abs' in points.columns:
        ok = (points['err_price'].abs() <= 0.5*points['spread_abs']).astype(float)
        out['Within_HalfSpread_Ratio'] = float(np.average(ok, weights=w_liq))
        nz_spread = points['spread_abs'] > 0
        if nz_spread.any():
            out['ZScore_mean'] = float(np.average((pe[nz_spread]/points['spread_abs'][nz_spread]), weights=w_liq[nz_spread]))
            out['ZScore_p95']  = float(np.quantile((pe[nz_spread]/points['spread_abs'][nz_spread]).abs(), 0.95))
    return out

def bucket_errors(points: pd.DataFrame, by=("t_target","k_target")) -> pd.DataFrame:
    """
    Produce bucket-level mean/STD for IV errors and price errors.
    """
    if points.empty or not all(c in points.columns for c in by):
        return pd.DataFrame()
    agg = points.groupby(list(by)).agg(
        n=('err_price','count'),
        iv_mae=('err_iv', lambda s: np.nanmean(np.abs(s))),
        iv_rmse=('err_iv', lambda s: np.sqrt(np.nanmean(s**2))),
        price_mae=('err_price', lambda s: np.nanmean(np.abs(s))),
        price_rmse=('err_price', lambda s: np.sqrt(np.nanmean(s**2)))
    ).reset_index()
    return agg

# ------------------------------
# 2) No-arbitrage checks on the full day chain
# ------------------------------
def _second_diff(x: np.ndarray) -> np.ndarray:
    return x[:-2] - 2*x[1:-1] + x[2:]

def arbitrage_checks_full_chain(full_day_raw: pd.DataFrame) -> Dict[str, float]:
    """
    Returns ratios of violations for several no-arbitrage properties:
      - bid<=ask
      - Call price non-increasing in strike (per expiry)
      - Put price non-decreasing in strike (per expiry)
      - Butterfly convexity for calls & puts (per expiry)
      - Calendar monotonicity at same strike (per K)
      - Simple parity residual stats at matched strikes
    Assumes full_day_raw has columns: quote_date, expiry, cp, strike, bid, ask, mid, underlying_close (spot)
    """
    out: Dict[str,float] = {}
    if full_day_raw is None or full_day_raw.empty:
        return out

    df = full_day_raw.copy()
    df.columns = [c.strip().lower() for c in df.columns]
    _ensure_datetime(df, ("quote_date","expiry"))
    # standard price columns
    if 'last_bid_price' in df.columns: df = df.rename(columns={'last_bid_price':'bid'})
    if 'last_ask_price' in df.columns: df = df.rename(columns={'last_ask_price':'ask'})
    if 'type' in df.columns and 'cp' not in df.columns:
        df['cp'] = df['type'].astype(str).str.upper().str[0]
    df['mid'] = np.where(
        np.isfinite(df.get('bid')) & np.isfinite(df.get('ask')) & (df['bid']>0) & (df['ask']>0),
        0.5*(df['bid']+df['ask']),
        pd.to_numeric(df.get('last', np.nan), errors='coerce')
    )
    df['strike'] = pd.to_numeric(df['strike'], errors='coerce')
    df = df[np.isfinite(df['strike'])]

    # 0) bid <= ask violations
    if {'bid','ask'}.issubset(df.columns):
        v = (df['bid'] - df['ask'] > 1e-12)
        out['Viol_bid_gt_ask_ratio'] = float(v.mean()) if len(v)>0 else np.nan

    # group by expiry for vertical structure checks
    def per_expiry_violations(sub: pd.DataFrame) -> Dict[str,int]:
        out_local = {'n_call_pairs':0,'viol_call_monotone':0,
                     'n_put_pairs':0,'viol_put_monotone':0,
                     'n_call_trip':0,'viol_call_bfly':0,
                     'n_put_trip':0,'viol_put_bfly':0}
        # Calls
        c = sub[sub['cp']=='C'].dropna(subset=['mid']).sort_values('strike')
        if len(c)>=2:
            diff = np.diff(c['mid'].values)
            out_local['n_call_pairs'] = diff.size
            out_local['viol_call_monotone'] = int((diff > 1e-8).sum())  # should be non-increasing
        if len(c)>=3:
            sd = _second_diff(c['mid'].values)
            out_local['n_call_trip'] = sd.size
            out_local['viol_call_bfly'] = int((sd < -1e-8).sum())       # convexity: second diff >= 0

        # Puts
        p = sub[sub['cp']=='P'].dropna(subset=['mid']).sort_values('strike')
        if len(p)>=2:
            diffp = np.diff(p['mid'].values)
            out_local['n_put_pairs'] = diffp.size
            out_local['viol_put_monotone'] = int((diffp < -1e-8).sum()) # should be non-decreasing
        if len(p)>=3:
            sdp = _second_diff(p['mid'].values)
            out_local['n_put_trip'] = sdp.size
            out_local['viol_put_bfly'] = int((sdp < -1e-8).sum())       # convexity: >=0
        return out_local

    agg = {'n_call_pairs':0,'viol_call_monotone':0,'n_put_pairs':0,'viol_put_monotone':0,
           'n_call_trip':0,'viol_call_bfly':0,'n_put_trip':0,'viol_put_bfly':0}
    
    for _, g in df.groupby('expiry'):
        loc = per_expiry_violations(g)
        for k,v in loc.items():
            agg[k] += v

    out['Viol_call_monotone_ratio'] = (agg['viol_call_monotone']/agg['n_call_pairs']) if agg['n_call_pairs']>0 else np.nan
    out['Viol_put_monotone_ratio']  = (agg['viol_put_monotone'] /agg['n_put_pairs'])  if agg['n_put_pairs']>0  else np.nan
    out['Viol_call_bfly_ratio']     = (agg['viol_call_bfly']    /agg['n_call_trip'])  if agg['n_call_trip']>0  else np.nan
    out['Viol_put_bfly_ratio']      = (agg['viol_put_bfly']     /agg['n_put_trip'])   if agg['n_put_trip']>0   else np.nan

    # 3) Calendar monotonicity at the same strike (for calls and puts separately)
    cal_viol_c = 0; cal_total_c = 0
    cal_viol_p = 0; cal_total_p = 0
    for K, gK in df.groupby('strike'):
        c = gK[gK['cp']=='C'].dropna(subset=['mid']).sort_values('expiry')
        if len(c)>=2:
            diff = np.diff(c['mid'].values)  # should be >= 0
            cal_total_c += diff.size
            cal_viol_c  += int((diff < -1e-8).sum())
        p = gK[gK['cp']=='P'].dropna(subset=['mid']).sort_values('expiry')
        if len(p)>=2:
            diffp = np.diff(p['mid'].values) # should be <= 0 (put decreases in T if q >= 0)
            cal_total_p += diffp.size
            cal_viol_p  += int((diffp > 1e-8).sum())
    out['Viol_call_calendar_ratio'] = (cal_viol_c / cal_total_c) if cal_total_c>0 else np.nan
    out['Viol_put_calendar_ratio']  = (cal_viol_p / cal_total_p) if cal_total_p>0 else np.nan

    # 4) Put-Call parity residuals at matched strikes
    par_list = []
    for exp, g in df.groupby('expiry'):
        cc = g[g['cp']=='C'][['strike','mid']].rename(columns={'mid':'c_mid'})
        pp = g[g['cp']=='P'][['strike','mid']].rename(columns={'mid':'p_mid'})
        j = pd.merge(cc, pp, on='strike', how='inner')
        if j.empty: 
            continue
        # Without curves, use naive parity C - P ?= S - K (carry ~ 0). Report dispersion.
        # If you have r,q per expiry you can plug them here.
        # We compute normalized residual by total half-spread proxy if available.
        res = (j['c_mid'] - j['p_mid']) - (df['underlying_close'].iloc[0] - j['strike'])
        par_list.append(res.values)
    if par_list:
        par = np.concatenate(par_list)
        out['Parity_resid_abs_med'] = float(np.median(np.abs(par)))
        out['Parity_resid_abs_p95'] = float(np.quantile(np.abs(par), 0.95))
    else:
        out['Parity_resid_abs_med'] = np.nan
        out['Parity_resid_abs_p95'] = np.nan

    return out

# ------------------------------
# 3) Orchestrator
# ------------------------------
def run_diagnostics(
    selected_csv: str,
    full_raw_csv: str,
    date: str,
    r_const: float = 0.0,
    q_const: float = 0.0,
    use_forward_to_infer_carry: bool = True,
    params_tuple: Optional[Tuple[float,float,float,float,float]] = None,
    out_prefix: str = "qc",
) -> Dict[str, object]:
    """
    If params_tuple is None, we will re-calibrate using your calibrate_heston_one_day on the selected day.
    Returns a dictionary of headline metrics and writes:
      - {out_prefix}_points.csv   (per-point errors)
      - {out_prefix}_buckets.csv  (bucket stats)
      - {out_prefix}_summary.json (headline + params + arbitrage ratios)
    """
    sel = pd.read_csv(selected_csv, parse_dates=['quote_date','expiry'])
    d0  = pd.to_datetime(date)           # 你想诊断的那天
    sel = sel[sel["quote_date"]==d0].copy()
    if full_raw_csv:
        full = build_dataframe(full_raw_csv)
        # restrict to the exact day in selected file
        d0 = pd.to_datetime(sel['quote_date'].iloc[0])
        full_day = full[full['quote_date']==d0].copy()
    else:
        full = None
        full_day = None

    # decide params
    if params_tuple is None:
        out = calibrate_heston_one_day(
            sel, full_day_raw=full_day,
            r_const=r_const, q_const=q_const,
            use_forward_to_infer_carry=use_forward_to_infer_carry
        )
        if not out.get('success', False):
            raise RuntimeError(f"Calibration failed: {out.get('message')}")
        params = out['params']
        calib_metrics = {k: out[k] for k in ['rmse_price','mae_price','n','message'] if k in out}
    else:
        params = HestonParams(*params_tuple)
        calib_metrics = {}

    # point-level
    points = compute_point_level_errors(sel, params, full_day, r_const, q_const)
    points.to_csv(f"{out_prefix}_points.csv", index=False)

    # aggregate & buckets
    head = aggregate_fit_metrics(points)
    buckets = bucket_errors(points)
    buckets.to_csv(f"{out_prefix}_buckets.csv", index=False)

    # arbitrage on full chain
    arb = arbitrage_checks_full_chain(full_day) if full_day is not None else {}

    # summary json
    summary = {
        'params': asdict(params),
        **calib_metrics,
        **head,
        **arb,
        'n_points': int(len(points))
    }
    with open(f"{out_prefix}_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    return {'summary': summary, 'points_path': f"{out_prefix}_points.csv", 'buckets_path': f"{out_prefix}_buckets.csv"}

# ------------------------------
# CLI
# ------------------------------
def parse_args():
    ap = argparse.ArgumentParser(description="Heston single-day calibration diagnostics & no-arbitrage checks")
    ap.add_argument("--selected_csv", required=True, help="Path to selected points CSV (from pick_points_bs)")
    ap.add_argument("--date", type=str, required=True, help="Date of the option data to calibrate")
    ap.add_argument("--full_raw_csv", required=True, help="Path to full raw chain CSV (one or multiple days)")
    ap.add_argument("--r", type=float, default=0.0, help="risk-free rate r")
    ap.add_argument("--q", type=float, default=0.0, help="dividend/borrow rate q")
    ap.add_argument("--no_forward_infer", action="store_true", help="Disable forward-based (r-q) inference")
    ap.add_argument("--params", type=str, default=None,
                    help="Optional manual params as 'kappa,theta,sigma,rho,v0' (if omitted we re-calibrate)")
    ap.add_argument("--out_prefix", type=str, default="qc", help="Prefix for output files")
    return ap.parse_args()

def main():
    args = parse_args()
    params_tuple = None
    if args.params:
        parts = [float(x) for x in args.params.split(",")]
        if len(parts) != 5:
            raise ValueError("params must be 5 comma-separated floats: kappa,theta,sigma,rho,v0")
        params_tuple = tuple(parts)

    res = run_diagnostics(
        selected_csv=args.selected_csv,
        full_raw_csv=args.full_raw_csv,
        date = args.date,
        r_const=args.r,
        q_const=args.q,
        use_forward_to_infer_carry=not args.no_forward_infer,
        params_tuple=params_tuple,
        out_prefix=args.out_prefix
    )
    print("=== Summary ===")
    for k,v in res['summary'].items():
        print(f"{k}: {v}")
    print(f"Wrote: {res['points_path']}  {res['buckets_path']}  {args.out_prefix}_summary.json")

if __name__ == "__main__":
    main()
