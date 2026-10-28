"""
src/backtest.py
Walk-forward backtesting engine, daily execution, transaction costs,
and comprehensive financial performance metrics.
"""

from typing import Dict, List, Tuple, Any, Optional
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
import torch

from src.model import MLPEnsemble
from src.portfolio import solve_qp_cvxpy
from src.dfl import train_pto_ensemble, train_dfl_ensemble
from src.utils import set_seed


def calculate_metrics(
    daily_net_returns: np.ndarray,
    turnover_series: np.ndarray,
    cost_series: np.ndarray,
    risk_free_rate: float = 0.06,  # 6% annual risk-free rate (standard Indian T-bill rate)
    pred_errors: Optional[Dict[str, float]] = None,
) -> Dict[str, float]:
    """
    Compute standard academic and practitioner portfolio performance metrics:
    - Annualized Return (CAGR)
    - Annualized Volatility
    - Sharpe Ratio
    - Sortino Ratio
    - Maximum Drawdown (MDD)
    - 95% Daily CVaR (Expected Shortfall)
    - Average Daily Turnover and Annualized Turnover
    - Total Transaction Costs
    """
    T = len(daily_net_returns)
    if T < 2:
        return {}

    # Cumulative wealth path
    cum_returns = np.cumprod(1.0 + daily_net_returns)
    total_return = cum_returns[-1] - 1.0
    cagr = (cum_returns[-1] ** (252.0 / T)) - 1.0

    # Volatility
    daily_vol = np.std(daily_net_returns, ddof=1)
    ann_vol = daily_vol * np.sqrt(252.0)

    # Sharpe ratio
    rf_daily = ((1.0 + risk_free_rate) ** (1.0 / 252.0)) - 1.0
    excess_returns = daily_net_returns - rf_daily
    if ann_vol > 1e-8:
        sharpe = (np.mean(excess_returns) / daily_vol) * np.sqrt(252.0)
    else:
        sharpe = 0.0

    # Sortino ratio (downside deviation relative to rf)
    downside_diff = np.minimum(daily_net_returns - rf_daily, 0.0)
    downside_dev = np.sqrt(np.mean(downside_diff ** 2)) * np.sqrt(252.0)
    if downside_dev > 1e-8:
        sortino = (np.mean(excess_returns) * 252.0) / downside_dev
    else:
        sortino = 0.0

    # Maximum Drawdown
    peaks = np.maximum.accumulate(cum_returns)
    drawdowns = (cum_returns - peaks) / peaks
    max_drawdown = float(np.min(drawdowns))  # negative value, e.g. -0.15

    # 95% Daily CVaR (Conditional Value-at-Risk / Expected Shortfall)
    var_95_threshold = np.percentile(daily_net_returns, 5.0)
    tail_losses = daily_net_returns[daily_net_returns <= var_95_threshold]
    if len(tail_losses) > 0:
        cvar_95 = float(-np.mean(tail_losses))  # reported as positive loss
    else:
        cvar_95 = float(-var_95_threshold)

    # Turnover & Costs
    avg_turnover = float(np.mean(turnover_series))
    ann_turnover = avg_turnover * 252.0
    total_costs = float(np.sum(cost_series))

    metrics = {
        "CAGR": float(cagr),
        "Total Return": float(total_return),
        "Annualized Volatility": float(ann_vol),
        "Sharpe Ratio": float(sharpe),
        "Sortino Ratio": float(sortino),
        "Max Drawdown": float(max_drawdown),
        "CVaR 95%": float(cvar_95),
        "Daily Turnover": float(avg_turnover),
        "Annualized Turnover": float(ann_turnover),
        "Total Transaction Cost": float(total_costs),
    }

    if pred_errors:
        metrics.update(pred_errors)

    return metrics


def run_walk_forward_backtest(
    X: np.ndarray,             # (T, N, F)
    y: np.ndarray,             # (T, N)
    covs: np.ndarray,          # (T, N, N)
    dates: List[str],
    tickers: List[str],
    test_start_year: int = 2022,
    retrain_freq_days: int = 21,  # Monthly walk-forward retraining (~21 trading days)
    lookback_window_days: int = 756,  # ~3 years of training history
    risk_aversion: float = 2.5,
    uncertainty_penalty: float = 1.5,
    max_weight: float = 0.15,
    transaction_cost_bps: float = 15.0,
    seed: int = 42,
) -> Dict[str, Any]:
    """
    Execute rigorous walk-forward backtest across all 5 strategies over the identical test period:
    1. Equal-Weight (1/N)
    2. Predict-Then-Optimize (PTO-MSE)
    3. Decision-Focused Learning (DFL)
    4. Uncertainty-Aware Predict-Then-Optimize (U-PTO)
    5. Proposed Uncertainty-Aware Decision-Focused Learning (U-DFL)
    """
    set_seed(seed)
    T, N, F = X.shape
    cost_rate = transaction_cost_bps / 10000.0

    # Determine test index start
    test_indices = [i for i, d in enumerate(dates) if int(d.split("-")[0]) >= test_start_year]
    if not test_indices:
        # Fallback to last 30% of data if dates don't span to test_start_year
        test_start_idx = int(0.70 * T)
    else:
        test_start_idx = test_indices[0]

    test_dates = dates[test_start_idx:]
    T_test = len(test_dates)

    print(f"\n=======================================================")
    print(f"Starting Walk-Forward Backtest:")
    print(f"- Total dates: {T} ({dates[0]} to {dates[-1]})")
    print(f"- Test period: {T_test} trading days ({test_dates[0]} to {test_dates[-1]})")
    print(f"- Retraining frequency: every {retrain_freq_days} days (monthly)")
    print(f"- Lookback window: {lookback_window_days} days (~3 years)")
    print(f"- Assets: {N} equities, Max Weight: {max_weight:.1%}")
    print(f"- Transaction cost: {transaction_cost_bps} bps")
    print(f"=======================================================\n")

    strategies = ["Equal_Weight", "PTO_MSE", "DFL", "U_PTO", "U_DFL"]

    # Storage for daily results
    daily_returns = {s: np.zeros(T_test) for s in strategies}
    daily_turnover = {s: np.zeros(T_test) for s in strategies}
    daily_costs = {s: np.zeros(T_test) for s in strategies}
    weights_history = {s: np.zeros((T_test, N)) for s in strategies}
    uncertainty_history = np.zeros((T_test, N))
    pred_returns_history = {s: np.zeros((T_test, N)) for s in ["PTO", "DFL", "U_DFL"]}

    # Track prior weights for turnover
    prev_weights = {s: np.ones(N) / N for s in strategies}

    # Model instances
    pto_model = None
    dfl_model = None
    udfl_model = None

    # Prediction error accumulators
    pred_errors = {
        "PTO": {"sq_err": [], "abs_err": [], "rank_ic": []},
        "DFL": {"sq_err": [], "abs_err": [], "rank_ic": []},
        "U_DFL": {"sq_err": [], "abs_err": [], "rank_ic": []},
    }

    for step, t in enumerate(range(test_start_idx, T)):
        # Check if retraining is needed (first test day or every retrain_freq_days)
        if step % retrain_freq_days == 0:
            train_start = max(0, t - lookback_window_days)
            train_end = t

            X_tr = X[train_start:train_end]
            y_tr = y[train_start:train_end]
            covs_tr = covs[train_start:train_end]

            print(f"[{test_dates[step]}] Retraining models on window {dates[train_start]} to {dates[train_end-1]} ({train_end - train_start} days)...")

            # 1. Train PTO ensemble (shared by Strategy 2 PTO and Strategy 4 U-PTO)
            pto_model = MLPEnsemble(n_members=5, input_dim=F, hidden_dim=32)
            train_pto_ensemble(pto_model, X_tr, y_tr, epochs=5, batch_size=64, lr=1e-3)

            # 2. Train DFL ensemble (gamma = 0)
            dfl_model = MLPEnsemble(n_members=5, input_dim=F, hidden_dim=32)
            train_dfl_ensemble(
                dfl_model, X_tr, y_tr, covs_tr,
                risk_aversion=risk_aversion, uncertainty_penalty=0.0,
                max_weight=max_weight, transaction_cost_bps=transaction_cost_bps,
                epochs=4, batch_size=64, lr=1e-3
            )

            # 3. Train U-DFL ensemble (proposed: gamma > 0)
            udfl_model = MLPEnsemble(n_members=5, input_dim=F, hidden_dim=32)
            train_dfl_ensemble(
                udfl_model, X_tr, y_tr, covs_tr,
                risk_aversion=risk_aversion, uncertainty_penalty=uncertainty_penalty,
                max_weight=max_weight, transaction_cost_bps=transaction_cost_bps,
                epochs=4, batch_size=64, lr=1e-3
            )

        # Daily test evaluation
        x_t = X[t]           # (N, F)
        y_realized = y[t]    # (N,)
        cov_t = covs[t]      # (N, N)

        # 1. Equal-Weight
        w_eq = np.ones(N) / N
        weights_history["Equal_Weight"][step] = w_eq

        # Inference from models
        pto_pred, pto_u = pto_model.predict_numpy(x_t)
        dfl_pred, dfl_u = dfl_model.predict_numpy(x_t)
        udfl_pred, udfl_u = udfl_model.predict_numpy(x_t)

        uncertainty_history[step] = udfl_u
        pred_returns_history["PTO"][step] = pto_pred
        pred_returns_history["DFL"][step] = dfl_pred
        pred_returns_history["U_DFL"][step] = udfl_pred

        # Record prediction quality
        for m_name, pred_val in [("PTO", pto_pred), ("DFL", dfl_pred), ("U_DFL", udfl_pred)]:
            pred_errors[m_name]["sq_err"].append(np.mean((pred_val - y_realized) ** 2))
            pred_errors[m_name]["abs_err"].append(np.mean(np.abs(pred_val - y_realized)))
            ic, _ = spearmanr(pred_val, y_realized)
            if not np.isnan(ic):
                pred_errors[m_name]["rank_ic"].append(ic)

        # 2. PTO (MSE trained, gamma = 0)
        w_pto = solve_qp_cvxpy(
            expected_returns=pto_pred,
            covariance_matrix=cov_t,
            uncertainty=None,
            risk_aversion=risk_aversion,
            uncertainty_penalty=0.0,
            max_weight=max_weight,
        )
        weights_history["PTO_MSE"][step] = w_pto

        # 3. DFL (DFL trained, gamma = 0)
        w_dfl = solve_qp_cvxpy(
            expected_returns=dfl_pred,
            covariance_matrix=cov_t,
            uncertainty=None,
            risk_aversion=risk_aversion,
            uncertainty_penalty=0.0,
            max_weight=max_weight,
        )
        weights_history["DFL"][step] = w_dfl

        # 4. U-PTO (MSE trained, gamma > 0)
        w_upto = solve_qp_cvxpy(
            expected_returns=pto_pred,
            covariance_matrix=cov_t,
            uncertainty=pto_u,
            risk_aversion=risk_aversion,
            uncertainty_penalty=uncertainty_penalty,
            max_weight=max_weight,
        )
        weights_history["U_PTO"][step] = w_upto

        # 5. U-DFL (Proposed: End-to-end uncertainty-aware DFL, gamma > 0)
        w_udfl = solve_qp_cvxpy(
            expected_returns=udfl_pred,
            covariance_matrix=cov_t,
            uncertainty=udfl_u,
            risk_aversion=risk_aversion,
            uncertainty_penalty=uncertainty_penalty,
            max_weight=max_weight,
        )
        weights_history["U_DFL"][step] = w_udfl

        # Compute realized returns, turnover, and transaction costs for all 5 strategies
        for s in strategies:
            w_curr = weights_history[s][step]
            w_old = prev_weights[s]

            # Turnover = sum |w_{i,t} - w_{i,t-1}|
            turnover = float(np.sum(np.abs(w_curr - w_old)))
            cost = turnover * cost_rate

            gross_ret = float(np.sum(w_curr * y_realized))
            net_ret = gross_ret - cost

            daily_returns[s][step] = net_ret
            daily_turnover[s][step] = turnover
            daily_costs[s][step] = cost

            # Update prior weight accounting for price drift
            end_val = w_curr * (1.0 + y_realized)
            tot_val = np.sum(end_val)
            prev_weights[s] = end_val / tot_val if tot_val > 0 else w_curr

    # Compute comprehensive metrics
    summary_metrics = {}
    for s in strategies:
        err_metrics = None
        if s == "PTO_MSE":
            err_metrics = {
                "MSE": float(np.mean(pred_errors["PTO"]["sq_err"])),
                "MAE": float(np.mean(pred_errors["PTO"]["abs_err"])),
                "Rank_IC": float(np.mean(pred_errors["PTO"]["rank_ic"])),
            }
        elif s == "DFL":
            err_metrics = {
                "MSE": float(np.mean(pred_errors["DFL"]["sq_err"])),
                "MAE": float(np.mean(pred_errors["DFL"]["abs_err"])),
                "Rank_IC": float(np.mean(pred_errors["DFL"]["rank_ic"])),
            }
        elif s == "U_DFL":
            err_metrics = {
                "MSE": float(np.mean(pred_errors["U_DFL"]["sq_err"])),
                "MAE": float(np.mean(pred_errors["U_DFL"]["abs_err"])),
                "Rank_IC": float(np.mean(pred_errors["U_DFL"]["rank_ic"])),
            }

        summary_metrics[s] = calculate_metrics(
            daily_net_returns=daily_returns[s],
            turnover_series=daily_turnover[s],
            cost_series=daily_costs[s],
            risk_free_rate=0.06,
            pred_errors=err_metrics,
        )

    # Format daily records into DataFrame
    daily_df = pd.DataFrame({"date": test_dates})
    for s in strategies:
        daily_df[f"{s}_ret"] = daily_returns[s]
        daily_df[f"{s}_cumret"] = np.cumprod(1.0 + daily_returns[s])
        daily_df[f"{s}_turnover"] = daily_turnover[s]
        daily_df[f"{s}_cost"] = daily_costs[s]

    results = {
        "summary_metrics": summary_metrics,
        "daily_df": daily_df,
        "weights_history": weights_history,
        "uncertainty_history": uncertainty_history,
        "pred_returns_history": pred_returns_history,
        "test_dates": test_dates,
        "tickers": tickers,
    }

    return results
