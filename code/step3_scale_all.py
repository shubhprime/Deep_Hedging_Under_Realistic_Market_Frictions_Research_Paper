"""
Step 3 Part F: Scale BS delta, Leland, and Whalley-Wilmott across the FULL dataset.

Run this LOCALLY in your project folder.
Usage: python3 step3_scale_all.py btc_options_train.csv train_results.csv
       python3 step3_scale_all.py btc_options_val.csv val_results.csv
       python3 step3_scale_all.py btc_options_test.csv test_results.csv

For each (symbol, sample_date) episode, runs all three hedging strategies and saves
ONE ROW PER EPISODE PER STRATEGY with summary stats (total cost, turnover, terminal
P&L, number of trades). This summary table is what step 4's evaluation metrics will
consume -- we don't need every hourly row, just the per-episode outcome of each strategy.
"""

import sys
import numpy as np
import pandas as pd
from scipy.stats import norm

INPUT_FILE = sys.argv[1] if len(sys.argv) > 1 else "btc_options_train.csv"
OUTPUT_FILE = sys.argv[2] if len(sys.argv) > 2 else "train_results.csv"

BTC_TRANSACTION_COST_RATE = 0.0005  # 5 bps round-trip, validated in step 3d/3e
DT_YEARS = 1 / (365 * 24)
RISK_AVERSION_LAMBDA = 60  # calibrated in step 3e
MIN_EPISODE_LENGTH = 4  # skip episodes with too few hourly observations to be meaningful

print(f"Loading {INPUT_FILE}...")
df = pd.read_csv(INPUT_FILE)
df["hour_bucket"] = pd.to_datetime(df["hour_bucket"])
df["sample_date"] = df["hour_bucket"].dt.date
df["T_years"] = df["time_to_maturity_days"] / 365
df["iv_decimal"] = df["mark_iv"] / 100
df["option_mid_usd"] = df["mid_price"] * df["underlying_price"]
print(f"Loaded {len(df)} rows.")


# --- Vectorized BS delta and gamma ---
def bs_delta_vec(S, K, T, sigma, is_call):
    valid = (T > 0) & (sigma > 0)
    d1 = np.where(valid, (np.log(S / K) + 0.5 * sigma ** 2 * T) / (sigma * np.sqrt(np.where(valid, T, 1))), 0)
    call_delta = norm.cdf(d1)
    put_delta = call_delta - 1.0
    delta = np.where(is_call, call_delta, put_delta)
    # expired/degenerate rows: use payoff indicator
    expired_call = np.where(is_call & ~valid, (S > K).astype(float), delta)
    expired = np.where(is_call, expired_call, np.where(~valid, -(S < K).astype(float), delta))
    return np.where(valid, delta, expired)


def bs_gamma_vec(S, K, T, sigma):
    valid = (T > 0) & (sigma > 0)
    d1 = np.where(valid, (np.log(S / K) + 0.5 * sigma ** 2 * T) / (sigma * np.sqrt(np.where(valid, T, 1))), 0)
    gamma = norm.pdf(d1) / (S * sigma * np.sqrt(np.where(valid, T, 1)))
    return np.where(valid, gamma, 0.0)


is_call = (df["type"] == "call").values
df["bs_delta"] = bs_delta_vec(df["underlying_price"].values, df["strike_price"].values, df["T_years"].values, df["iv_decimal"].values, is_call)
df["bs_gamma"] = bs_gamma_vec(df["underlying_price"].values, df["strike_price"].values, df["T_years"].values, df["iv_decimal"].values)

# Leland adjusted vol and delta
adj_factor = 1 + np.sqrt(2 / np.pi) * (BTC_TRANSACTION_COST_RATE / (df["iv_decimal"] * np.sqrt(DT_YEARS)))
df["leland_sigma"] = df["iv_decimal"] * np.sqrt(adj_factor)
df["leland_delta"] = bs_delta_vec(df["underlying_price"].values, df["strike_price"].values, df["T_years"].values, df["leland_sigma"].values, is_call)

# Whalley-Wilmott band half-width
df["band_half_width"] = (
    3 * BTC_TRANSACTION_COST_RATE * (df["underlying_price"] ** 2) * (df["bs_gamma"] ** 2) / (2 * RISK_AVERSION_LAMBDA)
) ** (1 / 3)

print("Computed deltas, gamma, Leland vol, WW band for all rows.")

results = []
n_episodes = 0
n_skipped = 0

for (symbol, sample_date), group in df.groupby(["symbol", "sample_date"]):
    group = group.sort_values("hour_bucket").reset_index(drop=True)
    n = len(group)
    if n < MIN_EPISODE_LENGTH:
        n_skipped += 1
        continue
    n_episodes += 1

    spot = group["underlying_price"].values
    option_mid_usd = group["option_mid_usd"].values
    option_pnl = -np.diff(option_mid_usd, prepend=option_mid_usd[0])
    option_pnl[0] = 0.0

    avg_moneyness = group["moneyness"].mean()
    avg_ttm = group["time_to_maturity_days"].mean()
    option_type = group["type"].iloc[0]

    # --- BS delta strategy (trade every step to exact target) ---
    bs_target = group["bs_delta"].values
    bs_trade = np.diff(bs_target, prepend=0.0)
    bs_cost = np.abs(bs_trade) * spot * (BTC_TRANSACTION_COST_RATE / 2)
    bs_position_lagged = np.concatenate(([0.0], bs_target[:-1]))
    bs_hedge_pnl = bs_position_lagged * np.diff(spot, prepend=spot[0])
    bs_hedge_pnl[0] = 0.0
    bs_total_pnl = (option_pnl + bs_hedge_pnl - bs_cost).sum()
    results.append({
        "symbol": symbol, "sample_date": sample_date, "option_type": option_type,
        "avg_moneyness": avg_moneyness, "avg_ttm_days": avg_ttm, "n_steps": n,
        "strategy": "bs_delta", "total_cost": bs_cost.sum(), "turnover": np.abs(bs_trade).sum(),
        "n_trades": int((np.abs(bs_trade) > 1e-9).sum()), "terminal_pnl": bs_total_pnl,
    })

    # --- Leland strategy (trade every step to Leland-adjusted target) ---
    le_target = group["leland_delta"].values
    le_trade = np.diff(le_target, prepend=0.0)
    le_cost = np.abs(le_trade) * spot * (BTC_TRANSACTION_COST_RATE / 2)
    le_position_lagged = np.concatenate(([0.0], le_target[:-1]))
    le_hedge_pnl = le_position_lagged * np.diff(spot, prepend=spot[0])
    le_hedge_pnl[0] = 0.0
    le_total_pnl = (option_pnl + le_hedge_pnl - le_cost).sum()
    results.append({
        "symbol": symbol, "sample_date": sample_date, "option_type": option_type,
        "avg_moneyness": avg_moneyness, "avg_ttm_days": avg_ttm, "n_steps": n,
        "strategy": "leland", "total_cost": le_cost.sum(), "turnover": np.abs(le_trade).sum(),
        "n_trades": int((np.abs(le_trade) > 1e-9).sum()), "terminal_pnl": le_total_pnl,
    })

    # --- Whalley-Wilmott strategy (sequential, path-dependent) ---
    ww_target = group["bs_delta"].values
    band = group["band_half_width"].values
    ww_position = np.zeros(n)
    ww_trade = np.zeros(n)
    ww_cost = np.zeros(n)
    ww_hedge_pnl = np.zeros(n)
    current = 0.0
    for i in range(n):
        lower, upper = ww_target[i] - band[i], ww_target[i] + band[i]
        if i == 0:
            new_pos = ww_target[i]
        elif current < lower:
            new_pos = lower
        elif current > upper:
            new_pos = upper
        else:
            new_pos = current
        trade = new_pos - current
        ww_trade[i] = trade
        ww_cost[i] = abs(trade) * spot[i] * (BTC_TRANSACTION_COST_RATE / 2)
        if i > 0:
            ww_hedge_pnl[i] = current * (spot[i] - spot[i - 1])
        current = new_pos
        ww_position[i] = current
    ww_total_pnl = (option_pnl + ww_hedge_pnl - ww_cost).sum()
    results.append({
        "symbol": symbol, "sample_date": sample_date, "option_type": option_type,
        "avg_moneyness": avg_moneyness, "avg_ttm_days": avg_ttm, "n_steps": n,
        "strategy": "whalley_wilmott", "total_cost": ww_cost.sum(), "turnover": np.abs(ww_trade).sum(),
        "n_trades": int((np.abs(ww_trade) > 1e-9).sum()), "terminal_pnl": ww_total_pnl,
    })

    if n_episodes % 1000 == 0:
        print(f"  Processed {n_episodes} episodes...")

print(f"\nDone. {n_episodes} episodes processed, {n_skipped} skipped (too short).")

results_df = pd.DataFrame(results)
results_df.to_csv(OUTPUT_FILE, index=False)
print(f"Saved {len(results_df)} rows ({n_episodes} episodes x 3 strategies) to {OUTPUT_FILE}")

print("\n=== Quick summary by strategy ===")
print(results_df.groupby("strategy")[["total_cost", "turnover", "n_trades", "terminal_pnl"]].mean())
