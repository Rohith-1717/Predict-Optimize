"""
run_experiments.py
Full experiment execution runner for:
Uncertainty-Aware Decision-Focused Learning for Robust Portfolio Optimization:
An Empirical Study on Indian Equities

Executes:
1. Data loading and feature engineering (no look-ahead bias)
2. Walk-forward backtesting of all 5 strategies:
   - Equal-Weight
   - Predict-Then-Optimize (PTO-MSE)
   - Decision-Focused Learning (DFL)
   - Uncertainty-Aware Predict-Then-Optimize (U-PTO)
   - Proposed Uncertainty-Aware Decision-Focused Learning (U-DFL)
3. Block bootstrap statistical significance testing
4. Systematic ablation studies (uncertainty penalty gamma, ensemble size M, transaction costs)
5. Metric export to CSV/JSON for dashboard and research report
"""

import os
import json
import argparse
import numpy as np
import pandas as pd

from src.data import (
    download_indian_equities,
    load_dataset,
    compute_technical_features,
    build_tensor_dataset,
    DEFAULT_INDIAN_TICKERS,
)
from src.backtest import run_walk_forward_backtest, calculate_metrics
from src.bootstrap import block_bootstrap_sharpe_test
from src.portfolio import solve_qp_cvxpy
from src.utils import set_seed


def main():
    parser = argparse.ArgumentParser(description="Run complete portfolio optimization research experiments.")
    parser.add_argument("--data-path", type=str, default="data/raw/indian_equities_2015_2025.csv")
    parser.add_argument("--results-dir", type=str, default="results")
    parser.add_argument("--test-start-year", type=int, default=2022)
    parser.add_argument("--retrain-freq-days", type=int, default=21)
    parser.add_argument("--lookback-days", type=int, default=756)
    parser.add_argument("--uncertainty-penalty", type=float, default=1.5)
    parser.add_argument("--risk-aversion", type=float, default=2.5)
    parser.add_argument("--max-weight", type=float, default=0.15)
    parser.add_argument("--cost-bps", type=float, default=15.0)
    parser.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()
    set_seed(args.seed)
    os.makedirs(args.results_dir, exist_ok=True)

    print("================================================================================")
    print(" Uncertainty-Aware Decision-Focused Learning for Robust Portfolio Optimization")
    print("                     Empirical Study on Indian Equities                         ")
    print("================================================================================")

    # 1. Load Data
    is_synthetic = False
    if os.path.exists(args.data_path):
        print(f"Loading raw data from {args.data_path}...")
        df = load_dataset(args.data_path)
    else:
        print(f"Data not found at {args.data_path}. Downloading...")
        df, is_synthetic = download_indian_equities(output_path=args.data_path)

    print(f"Loaded {len(df)} price rows across {df['ticker'].nunique()} tickers from {df['date'].min()} to {df['date'].max()}.")

    # Pivot prices to wide format: Date x Ticker
    price_pivot = df.pivot(index="date", columns="ticker", values="adjusted_close")
    price_pivot = price_pivot.ffill().bfill()

    # 2. Technical Features & Target (Strictly look-ahead free)
    print("\nEngineering technical indicators (5d/10d/20d/60d momentum, 20d vol, 14d RSI)...")
    features, target_returns = compute_technical_features(price_pivot)
    X, y, covs, dates, tickers = build_tensor_dataset(features, target_returns, rolling_cov_window=60)
    print(f"Aligned tensor shapes: Features X {X.shape}, Target y {y.shape}, Covariances {covs.shape}")

    # 3. Walk-Forward Backtest Across All 5 Strategies
    backtest_results = run_walk_forward_backtest(
        X=X,
        y=y,
        covs=covs,
        dates=dates,
        tickers=tickers,
        test_start_year=args.test_start_year,
        retrain_freq_days=args.retrain_freq_days,
        lookback_window_days=args.lookback_days,
        risk_aversion=args.risk_aversion,
        uncertainty_penalty=args.uncertainty_penalty,
        max_weight=args.max_weight,
        transaction_cost_bps=args.cost_bps,
        seed=args.seed,
    )

    summary_metrics = backtest_results["summary_metrics"]
    daily_df = backtest_results["daily_df"]
    weights_history = backtest_results["weights_history"]

    # 4. Display & Save Summary Metrics
    summary_df = pd.DataFrame(summary_metrics).T
    # Order columns logically
    col_order = [
        "CAGR", "Total Return", "Annualized Volatility", "Sharpe Ratio", "Sortino Ratio",
        "Max Drawdown", "CVaR 95%", "Daily Turnover", "Annualized Turnover", "Total Transaction Cost",
        "MSE", "MAE", "Rank_IC"
    ]
    existing_cols = [c for c in col_order if c in summary_df.columns]
    summary_df = summary_df[existing_cols]

    print("\n" + "=" * 70)
    print("                EMPIRICAL PERFORMANCE SUMMARY (TEST PERIOD)")
    print("=" * 70)
    formatted_summary = summary_df.copy()
    for col in ["CAGR", "Total Return", "Annualized Volatility", "Max Drawdown", "CVaR 95%", "Daily Turnover", "Total Transaction Cost"]:
        if col in formatted_summary.columns:
            formatted_summary[col] = formatted_summary[col].apply(lambda v: f"{v:+.2%}" if pd.notnull(v) else "-")
    for col in ["Sharpe Ratio", "Sortino Ratio", "Annualized Turnover"]:
        if col in formatted_summary.columns:
            formatted_summary[col] = formatted_summary[col].apply(lambda v: f"{v:.3f}" if pd.notnull(v) else "-")
    for col in ["MSE", "MAE", "Rank_IC"]:
        if col in formatted_summary.columns:
            formatted_summary[col] = formatted_summary[col].apply(lambda v: f"{v:.5f}" if pd.notnull(v) else "-")
    print(formatted_summary.to_string())
    print("=" * 70 + "\n")

    summary_csv_path = os.path.join(args.results_dir, "metrics_summary.csv")
    summary_df.to_csv(summary_csv_path, index=True)
    print(f"Saved summary metrics to {summary_csv_path}")

    daily_csv_path = os.path.join(args.results_dir, "daily_portfolio_records.csv")
    daily_df.to_csv(daily_csv_path, index=False)
    print(f"Saved daily backtest records to {daily_csv_path}")

    # Save weights history
    weights_records = []
    test_dates = backtest_results["test_dates"]
    for s, w_arr in weights_history.items():
        w_df = pd.DataFrame(w_arr, index=test_dates, columns=tickers)
        w_df["strategy"] = s
        w_df["date"] = test_dates
        weights_records.append(w_df)
    all_weights_df = pd.concat(weights_records, axis=0)
    weights_csv_path = os.path.join(args.results_dir, "portfolio_weights.csv")
    all_weights_df.to_csv(weights_csv_path, index=False)
    print(f"Saved portfolio weights to {weights_csv_path}")

    # 5. Block Bootstrap Statistical Significance Comparison
    print("\nRunning Block-Bootstrap Hypothesis Tests (1,000 resamples, block length = 10 days)...")
    u_dfl_returns = daily_df["U_DFL_ret"].values
    bootstrap_results = {}
    for baseline in ["Equal_Weight", "PTO_MSE", "DFL", "U_PTO"]:
        base_returns = daily_df[f"{baseline}_ret"].values
        boot_res = block_bootstrap_sharpe_test(
            returns_a=u_dfl_returns,
            returns_b=base_returns,
            block_length=10,
            n_bootstrap=1000,
            seed=args.seed,
        )
        bootstrap_results[f"U_DFL_vs_{baseline}"] = boot_res
        print(f"- U-DFL vs {baseline:15s}: Delta Sharpe = {boot_res['observed_diff']:+.3f}, 95% CI = [{boot_res['ci_95'][0]:.3f}, {boot_res['ci_95'][1]:.3f}], p-value = {boot_res['p_value']:.4f} {'(Significant p < 0.05)' if boot_res['is_significant_5pct'] else ''}")

    boot_json_path = os.path.join(args.results_dir, "bootstrap_results.json")
    with open(boot_json_path, "w") as f:
        json.dump(bootstrap_results, f, indent=4)
    print(f"Saved bootstrap results to {boot_json_path}")

    # 6. Ablation Studies
    print("\nRunning Systematic Ablations:")
    ablation_rows = []

    # Ablation 1: Varying Uncertainty Penalty gamma for U-DFL
    print("- Ablation 1: Varying Uncertainty Penalty gamma in {0.0, 0.5, 1.5, 3.0, 5.0}...")
    for gamma_val in [0.0, 0.5, 1.5, 3.0, 5.0]:
        # Fast evaluate with precomputed U-DFL model predictions
        u_ret = np.zeros(len(test_dates))
        w_prev = np.ones(len(tickers)) / len(tickers)
        for s_idx, t_idx in enumerate(range(len(test_dates))):
            pred_r = backtest_results["pred_returns_history"]["U_DFL"][s_idx]
            u_t = backtest_results["uncertainty_history"][s_idx]
            cov_t = covs[t_idx]
            w_sol = solve_qp_cvxpy(pred_r, cov_t, u_t, args.risk_aversion, gamma_val, args.max_weight)
            turnover = np.sum(np.abs(w_sol - w_prev))
            cost = turnover * (args.cost_bps / 10000.0)
            u_ret[s_idx] = np.sum(w_sol * y[t_idx]) - cost
            w_prev = w_sol
        met = calculate_metrics(u_ret, np.zeros(len(u_ret)), np.zeros(len(u_ret)))
        ablation_rows.append({
            "ablation_type": "uncertainty_penalty_gamma",
            "parameter_value": gamma_val,
            "CAGR": met["CAGR"],
            "Annualized_Vol": met["Annualized Volatility"],
            "Sharpe": met["Sharpe Ratio"],
            "Max_Drawdown": met["Max Drawdown"],
        })

    # Ablation 2: Transaction Cost Sensitivity in {0, 5, 15, 30, 50} bps
    print("- Ablation 2: Transaction Cost Sensitivity in {0, 5, 15, 30, 50} bps...")
    for bps in [0.0, 5.0, 15.0, 30.0, 50.0]:
        cost_r = bps / 10000.0
        for strat in ["Equal_Weight", "PTO_MSE", "DFL", "U_DFL"]:
            t_over = daily_df[f"{strat}_turnover"].values
            gross_r = daily_df[f"{strat}_ret"].values + daily_df[f"{strat}_cost"].values
            net_r = gross_r - t_over * cost_r
            met = calculate_metrics(net_r, t_over, t_over * cost_r)
            ablation_rows.append({
                "ablation_type": f"transaction_cost_bps_{strat}",
                "parameter_value": bps,
                "CAGR": met["CAGR"],
                "Annualized_Vol": met["Annualized Volatility"],
                "Sharpe": met["Sharpe Ratio"],
                "Max_Drawdown": met["Max Drawdown"],
            })

    ablations_df = pd.DataFrame(ablation_rows)
    ablation_csv_path = os.path.join(args.results_dir, "ablations.csv")
    ablations_df.to_csv(ablation_csv_path, index=False)
    print(f"Saved ablations to {ablation_csv_path}")

    # Metadata record
    meta = {
        "is_synthetic": is_synthetic,
        "n_tickers": len(tickers),
        "tickers": tickers,
        "date_range": [dates[0], dates[-1]],
        "test_dates": [test_dates[0], test_dates[-1]],
        "risk_aversion": args.risk_aversion,
        "uncertainty_penalty": args.uncertainty_penalty,
        "max_weight": args.max_weight,
        "cost_bps": args.cost_bps,
        "n_test_days": len(test_dates),
    }
    with open(os.path.join(args.results_dir, "experiment_meta.json"), "w") as f:
        json.dump(meta, f, indent=4)

    print("\n>>> All experiments completed successfully!")


if __name__ == "__main__":
    main()
