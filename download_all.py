"""
Step 1c (v2): Download and process free first-of-month Deribit options data.
Now with: rate-limit retry/backoff, delay between requests, and automatic
resume (skips months already present in the master file -- no manual START editing).

Run this LOCALLY in your project folder.
pip install requests pandas python-dateutil
"""

import os
import time
import gzip
import shutil
import requests
import pandas as pd
from datetime import date
from dateutil.relativedelta import relativedelta

FULL_START = date(2020, 1, 1)
FULL_END = date(2024, 12, 1)
RAW_DIR = "raw_downloads"
MASTER_OUTPUT = "btc_options_hourly_master.csv"
CHUNK_SIZE = 500_000
DELAY_BETWEEN_REQUESTS_SEC = 5
MAX_RETRIES = 5

KEEP_COLS = [
    "symbol", "timestamp", "type", "strike_price", "expiration",
    "open_interest", "bid_price", "bid_amount", "bid_iv",
    "ask_price", "ask_amount", "ask_iv", "mark_price", "mark_iv",
    "underlying_price", "delta", "gamma", "vega", "theta",
]

os.makedirs(RAW_DIR, exist_ok=True)


def get_already_done_months():
    """Check the master file for which year-month combos are already present."""
    if not os.path.exists(MASTER_OUTPUT):
        return set()
    done = set()
    # only need the hour_bucket column, read in chunks to avoid loading everything
    for chunk in pd.read_csv(MASTER_OUTPUT, usecols=["hour_bucket"], chunksize=1_000_000):
        months = pd.to_datetime(chunk["hour_bucket"]).dt.to_period("M").astype(str)
        done.update(months.unique().tolist())
    return done


def download_with_retry(url, gz_path):
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            resp = requests.get(url, stream=True, timeout=120)
            if resp.status_code == 429:
                wait = 15 * attempt  # backoff: 15s, 30s, 45s...
                print(f"  Rate limited (429). Waiting {wait}s before retry {attempt}/{MAX_RETRIES}...")
                time.sleep(wait)
                continue
            resp.raise_for_status()
            with open(gz_path, "wb") as f:
                shutil.copyfileobj(resp.raw, f)
            return True
        except requests.exceptions.HTTPError as e:
            if getattr(e.response, "status_code", None) == 404:
                print("  No data available for this month (404). Skipping.")
                return False
            print(f"  HTTP error: {e}")
            return False
        except Exception as e:
            print(f"  Error: {e}. Retrying ({attempt}/{MAX_RETRIES})...")
            time.sleep(10)
    print("  Max retries exceeded. Giving up on this month.")
    return False


already_done = get_already_done_months()
print(f"Months already in master file: {sorted(already_done)}")

master_written = os.path.exists(MASTER_OUTPUT)
current = FULL_START
failed_months = []

while current <= FULL_END:
    year, month = current.year, current.month
    month_key = f"{year}-{month:02d}"

    if month_key in already_done:
        print(f"=== {month_key} === already done, skipping.")
        current += relativedelta(months=1)
        continue

    url = f"https://datasets.tardis.dev/v1/deribit/options_chain/{year}/{month:02d}/01/OPTIONS.csv.gz"
    gz_path = os.path.join(RAW_DIR, f"{year}-{month:02d}-01.csv.gz")

    print(f"\n=== {month_key} ===")
    print(f"Downloading {url} ...")

    success = download_with_retry(url, gz_path)
    if not success:
        failed_months.append(month_key)
        current += relativedelta(months=1)
        time.sleep(DELAY_BETWEEN_REQUESTS_SEC)
        continue

    print("  Downloaded. Processing...")

    try:
        monthly_snapshots = []
        with gzip.open(gz_path, "rt") as f:
            reader = pd.read_csv(f, usecols=lambda c: c in KEEP_COLS + ["symbol"], chunksize=CHUNK_SIZE)
            for chunk in reader:
                chunk = chunk[chunk["symbol"].str.startswith("BTC-")]
                if chunk.empty:
                    continue
                chunk["datetime"] = pd.to_datetime(chunk["timestamp"], unit="us")
                chunk["hour_bucket"] = chunk["datetime"].dt.floor("h")
                chunk_sorted = chunk.sort_values("datetime")
                last_per_hour = chunk_sorted.groupby(["symbol", "hour_bucket"], as_index=False).last()
                monthly_snapshots.append(last_per_hour)

        if monthly_snapshots:
            month_result = pd.concat(monthly_snapshots, ignore_index=True)
            month_result = month_result.sort_values("datetime").groupby(
                ["symbol", "hour_bucket"], as_index=False
            ).last()
            month_result.to_csv(
                MASTER_OUTPUT, mode="a", header=not master_written, index=False
            )
            master_written = True
            print(f"  Appended {len(month_result)} hourly rows to {MASTER_OUTPUT}")
        else:
            print("  No BTC rows found this month.")

    except Exception as e:
        print(f"  FAILED to process {month_key}: {e}")
        failed_months.append(month_key)

    finally:
        if os.path.exists(gz_path):
            os.remove(gz_path)
            print("  Raw file deleted.")

    current += relativedelta(months=1)
    time.sleep(DELAY_BETWEEN_REQUESTS_SEC)  # be polite to the server

print(f"\nDone. Master dataset saved to {MASTER_OUTPUT}")
if failed_months:
    print(f"These months FAILED and need re-running: {failed_months}")
else:
    print("All months succeeded.")