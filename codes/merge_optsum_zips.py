# -*- coding: utf-8 -*-
"""
@author: Jianpeng Cao
"""

# merge_optsum_zips.py
# -*- coding: utf-8 -*-
import os
import re
import csv
import sys
import glob
import zipfile
import pandas as pd
from pathlib import Path
from datetime import datetime

def extract_date_from_name(name: str):
    """
    Extract the date string '2017-01-03' from 'Optsum_2017-01-03.zip'
    or '.../Optsum_2017-01-03.csv'. Return None if extraction fails.
    """
    m = re.search(r'(\d{4}-\d{2}-\d{2})', name)
    return m.group(1) if m else None

def merge_optsum_zip_folder(
    in_dir: str,
    out_csv: str,
    start_date: str = "2017-01-03",
    end_date: str   = "2017-05-31",
    encoding: str = None,           # None = let pandas infer automatically; or specify 'utf-8','latin1','cp936', etc.
    chunksize: int = 0,             # 0 = read all at once; >0 = read in chunks (append), saving memory
    subset_columns: list | None = None,  # Provide a subset of column names to keep only the needed columns
) -> None:
    """
    Iterate over all Optsum_YYYY-MM-DD.zip files under in_dir, read the inner CSV,
    and merge them into out_csv.
    - Auto-populate column 'quote_date' from the filename (if the source file lacks it).
    - Normalize column names to lowercase and strip whitespace.
    - Support chunked writing to save memory.
    """
    in_dir = str(in_dir)
    out_csv = str(out_csv)
    Path(out_csv).parent.mkdir(parents=True, exist_ok=True)

    # Build the zip list filtered by date range
    zips = sorted(glob.glob(os.path.join(in_dir, "Optsum_*.zip")))
    if not zips:
        print(f"[WARN] No zip files found: {in_dir}/Optsum_*.zip")
        return

    if start_date:
        zips = [z for z in zips if extract_date_from_name(z) and extract_date_from_name(z) >= start_date]
    if end_date:
        zips = [z for z in zips if extract_date_from_name(z) and extract_date_from_name(z) <= end_date]

    if not zips:
        print("[WARN] No files after filtering.")
        return

    # Write the header only once when appending
    wrote_header = False
    total_rows = 0
    total_files = 0

    for zpath in zips:
        qdate = extract_date_from_name(zpath)
        try:
            with zipfile.ZipFile(zpath, 'r') as zf:
                # Find inner CSV (defaults to the same-named file)
                inner_csvs = [n for n in zf.namelist() if n.lower().endswith(".csv")]
                if not inner_csvs:
                    print(f"[SKIP] {zpath}: no CSV inside the archive")
                    continue

                # Prefer the same-named file; otherwise use the first CSV
                preferred = [n for n in inner_csvs if extract_date_from_name(n) == qdate]
                inner = preferred[0] if preferred else inner_csvs[0]

                with zf.open(inner) as f:
                    if chunksize and chunksize > 0:
                        # Read in chunks to save memory
                        for chunk in pd.read_csv(
                            f, encoding=encoding, low_memory=False, chunksize=chunksize
                        ):
                            # Normalize column names
                            chunk.columns = [c.strip().lower() for c in chunk.columns]
                            # Add 'quote_date' column if missing
                            if 'quote_date' not in chunk.columns:
                                chunk['quote_date'] = qdate
                            # Keep only required columns (if specified)
                            if subset_columns:
                                keep = [c for c in subset_columns if c in chunk.columns]
                                chunk = chunk[keep]
                            # Append to output CSV
                            chunk.to_csv(
                                out_csv,
                                mode='a',
                                index=False,
                                header=not wrote_header,
                                quoting=csv.QUOTE_MINIMAL
                            )
                            wrote_header = True
                            total_rows += len(chunk)
                    else:
                        # Read all at once
                        df = pd.read_csv(f, encoding=encoding, low_memory=False)
                        df.columns = [c.strip().lower() for c in df.columns]
                        if 'quote_date' not in df.columns:
                            df['quote_date'] = qdate
                        if subset_columns:
                            keep = [c for c in subset_columns if c in df.columns]
                            df = df[keep]
                        df.to_csv(
                            out_csv,
                            mode='a',
                            index=False,
                            header=not wrote_header,
                            quoting=csv.QUOTE_MINIMAL
                        )
                        wrote_header = True
                        total_rows += len(df)
                total_files += 1
                print(f"[OK] {qdate}  -> rows cum: {total_rows}")
        except Exception as e:
            print(f"[ERROR] Failed to process {zpath}: {e}")

    print(f"Done: merged {total_files} files, {total_rows} rows -> {out_csv}")

if __name__ == "__main__":
    # Example usage (adjust paths as needed):
    # Input directory: the folder containing Optsum_YYYY-MM-DD.zip files
    IN_DIR  = r"D:\your\folder\of\zips"
    # Output consolidated CSV
    OUT_CSV = r"D:\your\folder\merged\optsum_2017Q1_Q2.csv"

    # If memory is limited, set chunksize=100_000 to write in chunks;
    # if there are too many columns, you can keep only the commonly used ones.
    merge_optsum_zip_folder(
        in_dir=IN_DIR,
        out_csv=OUT_CSV,
        start_date="2017-01-03",
        end_date="2017-05-31",
        encoding=None,          # Let pandas infer automatically; if decoding errors occur, try 'utf-8'/'latin1'/'cp936'
        chunksize=0,            # 0 = read all at once; if memory-constrained, use 100_000
        subset_columns=None     # e.g., ['quote_date','expiry','type','strike','last_bid_price','last_ask_price','underlying_close','open_interest','total_volume']
    )
