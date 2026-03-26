"""
Run this separately to monitor paper trading performance.
Usage: python logs/dashboard.py
"""

import pandas as pd
import numpy as np
import os

TRADES_LOG = "./logs/trades.csv"


def show_dashboard():
    if not os.path.exists(TRADES_LOG):
        print("No trades logged yet. Bot hasn't executed any trades.")
        return

    df = pd.read_csv(TRADES_LOG, parse_dates=["timestamp"])

    print("\n" + "="*55)
    print("  PAPER TRADING DASHBOARD")
    print("="*55)
    print(f"  Total entries logged : {len(df)}")
    print(f"  First trade          : {df['timestamp'].min()}")
    print(f"  Last trade           : {df['timestamp'].max()}")

    for asset in ["BTC", "ETH"]:
        asset_df = df[df["asset"] == asset]
        if len(asset_df) == 0:
            continue

        print(f"\n  {asset}:")
        print(f"    Total actions   : {len(asset_df)}")

        trades   = asset_df[asset_df["action"].isin(["LONG","SHORT"])]
        closes   = asset_df[asset_df["action"] == "CLOSE"]

        print(f"    Entries         : {len(trades)}")
        print(f"    Closes          : {len(closes)}")
        print(f"    Long entries    : {(trades['action']=='LONG').sum()}")
        print(f"    Short entries   : {(trades['action']=='SHORT').sum()}")

        if len(asset_df) > 0:
            latest_cap = asset_df["capital"].iloc[-1]
            first_cap  = asset_df["capital"].iloc[0]
            ret        = (latest_cap - first_cap) / first_cap * 100
            print(f"    Current capital : ${latest_cap:,.2f}")
            print(f"    Return          : {ret:+.2f}%")

        print(f"\n    Signal distribution:")
        print(f"      Avg P(buy)    : {asset_df['prob_buy'].mean():.3f}")
        print(f"      Avg P(sell)   : {asset_df['prob_sell'].mean():.3f}")
        print(f"      Avg P(hold)   : {asset_df['prob_hold'].mean():.3f}")

    print("\n" + "="*55)
    print("  Recent trades:")
    print(df.tail(10).to_string(index=False))


if __name__ == "__main__":
    show_dashboard()