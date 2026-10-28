"""
download_data.py
Script to acquire, validate, and preprocess Indian equities data (2015-2025).
Supports downloading directly or importing an existing CSV with columns:
  date,ticker,adjusted_close
"""

import argparse
import sys
from src.data import download_indian_equities, load_dataset, DEFAULT_INDIAN_TICKERS


def main():
    parser = argparse.ArgumentParser(description="Download and validate Indian Equities dataset.")
    parser.add_argument("--input-csv", type=str, default=None, help="Path to existing CSV with date,ticker,adjusted_close")
    parser.add_argument("--output", type=str, default="data/raw/indian_equities_2015_2025.csv", help="Output path for raw data")
    parser.add_argument("--start-date", type=str, default="2015-01-01", help="Start date (YYYY-MM-DD)")
    parser.add_argument("--end-date", type=str, default="2025-01-01", help="End date (YYYY-MM-DD)")

    args = parser.parse_args()

    if args.input_csv:
        print(f"Loading custom dataset from {args.input_csv}...")
        df = load_dataset(args.input_csv)
        is_synthetic = False
        print(f"Successfully loaded {len(df)} records across {df['ticker'].nunique()} tickers.")
    else:
        df, is_synthetic = download_indian_equities(
            tickers=DEFAULT_INDIAN_TICKERS,
            start_date=args.start_date,
            end_date=args.end_date,
            output_path=args.output,
        )

    print("\nDataset Summary:")
    print(f"- Total rows: {len(df)}")
    print(f"- Unique tickers ({df['ticker'].nunique()}): {sorted(df['ticker'].unique().tolist())}")
    print(f"- Date range: {df['date'].min()} to {df['date'].max()}")
    print(f"- Data source: {'SYNTHETIC TEST BENCHMARK' if is_synthetic else 'REAL YAHOO FINANCE NSE'}")
    print("\nSample records:")
    print(df.head())


if __name__ == "__main__":
    main()
