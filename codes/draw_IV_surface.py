# -*- coding: utf-8 -*-
"""
@author: Jianpeng Cao
"""

# -*- coding: utf-8 -*-
"""
Plot market IV surface vs calibrated-model IV surface (Spot-S form).
- Market IV: invert Black–Scholes (with continuous dividend yield q) from mid prices
- Model  IV: call your calibrated model (IV or price); if price, invert via BS to IV
- Surfaces are built on a (log-moneyness k_S = ln(K/S), time-to-maturity T) grid via bin-averaging
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from calibrate_heston_bs import *

# ========= User inputs =========
csv_path   = r'C:\Users\caoxi\Desktop\SPX options data\Optsum_2017-01-03\Optsum_2017-01-03.csv'  # 你的原始期权CSV
trade_date = "2017-01-03"                                          # pick a specific day
r_annual   = 0.03    # annualized risk-free rate (replace with a curve per T if available)
q_annual   = 0.00    # annualized dividend/borrow yield

@dataclass
class HestonParams:
    kappa: float
    theta: float
    sigma: float
    rho: float
    v0: float
    
calibrated_params = HestonParams(2.79168, 0.049757, 0.81695, -0.815963, 0.01141)

# Grid and cleaning parameters
N_K, N_T = 25, 20             # grid density (k × T)
MAX_SPREAD_PCT = 0.10         # maximum acceptable relative spread (e.g., 10%)

# ========= Math utils (BS) =========
from math import log, sqrt, exp, erf

def norm_cdf(x): return 0.5*(1.0 + erf(x/np.sqrt(2.0)))

def bs_price(S, K, T, r, q, vol, cp):
    """Black–Scholes price with continuous dividend yield q; cp in {'C','P'}"""
    if T <= 0 or vol <= 0:
        intrinsic = max(0.0, S - K) if cp.upper()=='C' else max(0.0, K - S)
        # Discount to today (strictly this involves e^{-qT} and e^{-rT}, but as T→0 both ≈ 1)
        return intrinsic
    sig = vol*sqrt(T)
    d1  = (log(S/K) + (r - q + 0.5*vol*vol)*T) / sig
    d2  = d1 - sig
    if cp.upper()=='C':
        return S*exp(-q*T)*norm_cdf(d1) - K*exp(-r*T)*norm_cdf(d2)
    else:
        return K*exp(-r*T)*norm_cdf(-d2) - S*exp(-q*T)*norm_cdf(-d1)

def implied_vol_bs(S, K, T, r, q, price, cp, tol=1e-7, max_iter=100):
    """Invert Black–Scholes IV from price (bisection)."""
    if T <= 0 or price < 0: return np.nan
    lo, hi = 1e-6, 5.0
    plo = bs_price(S,K,T,r,q,lo,cp)
    phi = bs_price(S,K,T,r,q,hi,cp)
    # Quick reachability check
    if not (plo-1e-12 <= price <= phi+1e-12):
        return np.nan
    for _ in range(max_iter):
        mid = 0.5*(lo+hi)
        pm  = bs_price(S,K,T,r,q,mid,cp)
        if abs(pm - price) < tol:
            return mid
        if pm > price: hi = mid
        else:          lo = mid
    return mid

# ========= Your model hooks (fill one) =========
def model_price_at(S, K, T, calibrated_params, r, q, cp):
    """If you only have a price function (discounted to today), implement it here. Placeholder now: call Heston price."""
    return heston_price(S, K, T, r, q, calibrated_params, cp)

def model_iv_at(S, K, T, calibrated_params, r, q, cp):
    """If you already have a model that outputs IV directly, implement it here. Placeholder: infer IV from model price."""
    model_price = model_price_at(S, K, T, calibrated_params, r, q, cp)
    return implied_vol_bs(S, K, T, r, q, model_price, cp, tol=1e-7, max_iter=100)

# ========= Load & prep single-day data =========
df = pd.read_csv(csv_path)
df.columns = [c.strip().lower() for c in df.columns]

# Lenient normalization of column names
if 'last_bid_price' in df.columns and 'bid' not in df.columns:
    df = df.rename(columns={'last_bid_price':'bid'})
if 'last_ask_price' in df.columns and 'ask' not in df.columns:
    df = df.rename(columns={'last_ask_price':'ask'})
if 'option_type' in df.columns and 'type' not in df.columns:
    df = df.rename(columns={'option_type':'type'})
if 'underlying_last' in df.columns and 'underlying_close' not in df.columns:
    df = df.rename(columns={'underlying_last':'underlying_close'})

for col in ['quote_date','expiry']:
    df[col] = pd.to_datetime(df[col])
df = df[df['quote_date']==pd.to_datetime(trade_date)].copy()

df['bid'] = pd.to_numeric(df.get('bid', np.nan), errors='coerce')
df['ask'] = pd.to_numeric(df.get('ask', np.nan), errors='coerce')
df['mid'] = np.where(
    np.isfinite(df['bid']) & np.isfinite(df['ask']) & (df['bid']>0) & (df['ask']>0),
    0.5*(df['bid']+df['ask']),
    pd.to_numeric(df.get('last', np.nan), errors='coerce')
)
df['spread_pct'] = np.where(df['mid']>0, (df['ask']-df['bid']).clip(lower=0)/df['mid'], np.nan)
df['strike'] = pd.to_numeric(df['strike'], errors='coerce')
df['cp'] = df['type'].astype(str).str.upper().str[0].map({'C':'C','P':'P'})
df['spot'] = pd.to_numeric(df.get('underlying_close', np.nan), errors='coerce')

# If there are multiple spot values on the day, take a single representative value
S0 = float(df['spot'].dropna().iloc[0]) if df['spot'].notna().any() else np.nan

# Annualized time to maturity (365-day basis)
df['T'] = (df['expiry'] - df['quote_date']).dt.days/365.0
df = df[df['T']>0].copy()

# Log-moneyness relative to S (the S-form you requested)
df['kS'] = np.log(np.maximum(df['strike'], 1e-12) / np.maximum(S0, 1e-12))

# Filter out dirty/noisy points
df = df[np.isfinite(df['mid']) & np.isfinite(df['strike']) & np.isfinite(df['kS'])]
df = df[(df['spread_pct'].isna()) | (df['spread_pct'] <= MAX_SPREAD_PCT)].copy()

# ========= Market IV via BS =========
def row_market_iv(row):
    return implied_vol_bs(S0, row['strike'], row['T'], r_annual, q_annual, row['mid'], row['cp'])
df['iv_mkt'] = df.apply(row_market_iv, axis=1)

# ========= Model IV on same points =========
def row_model_iv(row):
    iv = model_iv_at(S0, row['strike'], row['T'], calibrated_params, r_annual, q_annual, row['cp'])
    if np.isfinite(iv):
        return iv
    price = model_price_at(S0, row['strike'], row['T'], calibrated_params, r_annual, q_annual, row['cp'])
    if price is None:
        return np.nan
    return implied_vol_bs(S0, row['strike'], row['T'], r_annual, q_annual, price, row['cp'])
df['iv_model'] = df.apply(row_model_iv, axis=1)

# ========= Build (kS, T) grid by bin-averaging =========
# Trim tails to avoid extreme-noise wings
k_lo, k_hi = df['kS'].quantile(0.05), df['kS'].quantile(0.95)
T_lo, T_hi = df['T'].quantile(0.05), df['T'].quantile(0.95)

k_edges = np.linspace(k_lo, k_hi, N_K+1)
T_edges = np.linspace(T_lo, T_hi, N_T+1)
k_centers = 0.5*(k_edges[:-1]+k_edges[1:])
T_centers = 0.5*(T_edges[:-1]+T_edges[1:])

df['k_bin'] = pd.cut(df['kS'], bins=k_edges, labels=False, include_lowest=True)
df['T_bin'] = pd.cut(df['T'],  bins=T_edges, labels=False, include_lowest=True)

pivot_mkt = df.pivot_table(index='T_bin', columns='k_bin', values='iv_mkt',   aggfunc='mean')
pivot_mod = df.pivot_table(index='T_bin', columns='k_bin', values='iv_model', aggfunc='mean')

def to_surface_df(pivot, name):
    arr = pivot.values
    KK, TT = np.meshgrid(k_centers[:arr.shape[1]], T_centers[:arr.shape[0]])
    return pd.DataFrame({'T':TT.ravel(), 'kS':KK.ravel(), name:arr.ravel()})

surf_mkt = to_surface_df(pivot_mkt, 'iv_mkt')
surf_mod = to_surface_df(pivot_mod, 'iv_model')
surf = pd.merge(surf_mkt, surf_mod, on=['T','kS'], how='outer')
surf['iv_diff'] = surf['iv_model'] - surf['iv_mkt']

# ========= Plot: three figures (no specific colors) =========
def plot_heat(surface_df, value_col, title):
    T_vals = np.sort(surface_df['T'].unique())
    k_vals = np.sort(surface_df['kS'].unique())
    grid = surface_df.pivot(index='T', columns='kS', values=value_col).values
    plt.figure()
    extent = [k_vals.min(), k_vals.max(), T_vals.min(), T_vals.max()]
    plt.imshow(grid, origin='lower', aspect='auto', extent=extent)
    plt.xlabel("log-moneyness k_S = ln(K/S)")
    plt.ylabel("time to maturity T (years)")
    plt.title(title)
    plt.colorbar(label=value_col)
    plt.tight_layout()
    plt.savefig(title, dpi=750, bbox_inches = 'tight')

plot_heat(surf.dropna(subset=['iv_mkt']),   'iv_mkt',   f"Market IV surface (S form) {trade_date}")
plot_heat(surf.dropna(subset=['iv_model']), 'iv_model', f"Model IV surface (S form) {trade_date}")
plot_heat(surf.dropna(subset=['iv_diff']),  'iv_diff',  f"Model − Market IV (S form) {trade_date}")

# ========= Save surfaces to CSV =========
out_csv = f"/mnt/data/iv_surfaces_S_form_{trade_date}.csv"
surf.to_csv(out_csv, index=False)
print(f"Saved surfaces to: {out_csv}")
