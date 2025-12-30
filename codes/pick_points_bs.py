# -*- coding: utf-8 -*-
"""
@author: Jianpeng Cao
"""
# pick_points.py
# -*- coding: utf-8 -*-
# pick_points_bs.py  — Point picker (BS/spot world: k = ln(K / S))
# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import argparse
from typing import Iterable

def build_dataframe(path: str) -> pd.DataFrame:
    """Read the raw CSV, standardize common fields, and compute: mid/spread/spot S/annualized time to expiry T/k=ln(K/S)."""
    df = pd.read_csv(path)
    df.columns = [c.strip().lower() for c in df.columns]

    # Liberal column-name mapping
    rename_map = {}
    if 'last_bid_price' in df.columns: rename_map['last_bid_price'] = 'bid'
    if 'last_ask_price' in df.columns: rename_map['last_ask_price'] = 'ask'
    if 'option_type' in df.columns and 'type' not in df.columns: rename_map['option_type'] = 'type'
    if 'underlying_last' in df.columns and 'underlying_close' not in df.columns:
        rename_map['underlying_last'] = 'underlying_close'
    df = df.rename(columns=rename_map)

    # Core fields & dtypes
    for col in ['quote_date','expiry']:
        df[col] = pd.to_datetime(df[col])
    df['strike'] = pd.to_numeric(df['strike'], errors='coerce')
    df['cp'] = df['type'].astype(str).str.upper().str[0].map({'C':'C','P':'P'})

    # Quotes & mid
    df['bid'] = pd.to_numeric(df.get('bid', np.nan), errors='coerce')
    df['ask'] = pd.to_numeric(df.get('ask', np.nan), errors='coerce')
    # Prefer bid/ask for mid; otherwise fall back to last
    df['mid'] = np.where(
        np.isfinite(df['bid']) & np.isfinite(df['ask']) & (df['bid']>0) & (df['ask']>0),
        0.5*(df['bid']+df['ask']),
        pd.to_numeric(df.get('last', np.nan), errors='coerce')
    )
    # Spread
    df['spread_abs'] = (df['ask'] - df['bid']).clip(lower=0)
    df['spread_pct'] = np.where(df['mid']>0, df['spread_abs']/df['mid'], np.nan)

    # Spot & maturity
    df['spot'] = pd.to_numeric(df.get('underlying_close', np.nan), errors='coerce')
    df['t'] = (df['expiry'] - df['quote_date']).dt.days / 365.0  # 可改为/252

    # Log moneyness (BS world)
    df['k'] = np.log(np.maximum(df['strike'], 1e-12) / np.maximum(df['spot'], 1e-12))

    # Liquidity columns (optional)
    df['total_volume']  = pd.to_numeric(df.get('total_volume',  np.nan), errors='coerce')
    df['open_interest'] = pd.to_numeric(df.get('open_interest', np.nan), errors='coerce')

    return df

def pick_points_bs(
    df: pd.DataFrame,
    T_targets: Iterable[float] = (0.02, 0.06, 0.10, 0.20, 0.30, 0.50, 1.00),
    k_targets_short: Iterable[float] = (-0.15, -0.08, -0.04, 0.00, 0.04, 0.08, 0.15),
    k_targets_long:  Iterable[float] = (-0.20, -0.12, -0.06, -0.03, 0.00, 0.03, 0.06, 0.12, 0.20),
    max_spread_pct: float = 0.05,          # Max acceptable relative spread (5%)
    prefer_side_by_k: bool = True,         # Prefer Put if k*<0; Call if k*>0
    date_start: str = None,
    date_end: str = None
) -> pd.DataFrame:
    dat = df.copy()
    if date_start: dat = dat[dat['quote_date'] >= pd.to_datetime(date_start)]
    if date_end:   dat = dat[dat['quote_date'] <= pd.to_datetime(date_end)]

    rows = []
    for d, day in dat.groupby('quote_date'):
        # Set of available expiries for the day
        exps = day[['expiry','t']].drop_duplicates().sort_values('t')
        if exps.empty: 
            continue

        for T_star in T_targets:
            # Find the actual expiry closest to the target tenor
            idx = (exps['t'] - T_star).abs().idxmin()
            expiry_sel = exps.loc[idx, 'expiry']
            t_sel = float(exps.loc[idx, 't'])

            sub = day[day['expiry']==expiry_sel].copy()
            if sub.empty: 
                continue

            # Use different k* grids for short vs. long tenors (short-tenor wings are noisier; use a narrower grid)
            k_targets = k_targets_short if t_sel <= 0.08 else k_targets_long

            # Soft filters: valid k/mid & spread not too large
            sub = sub[np.isfinite(sub['k']) & np.isfinite(sub['mid']) & (sub['mid']>0)]
            #sub = sub[(sub['spread_pct'].isna()) | (sub['spread_pct'] <= max_spread_pct)]
            if sub.empty:
                continue

            for k_star in k_targets:
                pref_cp = None
                if prefer_side_by_k:
                    pref_cp = 'P' if k_star < 0 else ('C' if k_star > 0 else None)

                cand = sub[sub['cp']==pref_cp] if pref_cp else sub
                if cand.empty:
                    cand = sub  # Fallback: if that side is missing, allow either side

                cand = cand.assign(k_dist=(cand['k'] - k_star).abs())

                # Selection rule: minimize |k - k*| → then lower spread → then higher volume
                pick = cand.sort_values(
                    ['k_dist','spread_pct','total_volume'],
                    ascending=[True, True, False]
                ).head(1)

                if pick.empty: 
                    continue

                r = pick.iloc[0].to_dict()
                r.update({
                    't_target':   float(T_star),
                    'k_target':   float(k_star),
                    't_selected': t_sel,
                    'k_distance': float(pick.iloc[0]['k_dist']),
                    'prefer_cp':  pref_cp
                })
                rows.append(r)

    sel = pd.DataFrame(rows)
    if sel.empty:
        return sel

    # Finalize output columns & simple weights
    cols = ['quote_date','expiry','t_selected','t_target','strike','cp','mid','bid','ask',
            'spread_abs','spread_pct','open_interest','total_volume','underlying_symbol',
            'spot','k','k_target','k_distance','prefer_cp']
    cols = [c for c in cols if c in sel.columns]
    sel = sel[cols].sort_values(['quote_date','t_target','k_target']).reset_index(drop=True)

    # Liquidity weight (usable in calibration)
    if 'spread_pct' in sel.columns:
        sel['liq_weight'] = 1.0 / sel['spread_pct'].clip(lower=0.001)
        sel['liq_weight'] = sel['liq_weight'].clip(upper=1e3)
    else:
        sel['liq_weight'] = 1.0

    return sel

def main():
    ap = argparse.ArgumentParser(description="Pick representative option points (BS world: k=ln(K/S)).")
    ap.add_argument("--input",  required=True, help="Path to options CSV")
    ap.add_argument("--output", required=True, help="Path to save selected points CSV")
    ap.add_argument("--start",  default=None,  help="Start date YYYY-MM-DD (optional)")
    ap.add_argument("--end",    default=None,  help="End date YYYY-MM-DD (optional)")
    ap.add_argument("--max_spread", type=float, default=0.05, help="Max relative spread (default 5%)")
    args = ap.parse_args()

    df = build_dataframe(args.input)
    sel = pick_points_bs(
        df,
        max_spread_pct=args.max_spread,
        date_start=args.start,
        date_end=args.end
    )
    sel.to_csv(args.output, index=False)
    print(f"Selected {len(sel)} rows -> {args.output}")

if __name__ == "__main__":
    main()

