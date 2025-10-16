"""
Utility to evaluate optimal Salesforce feature lags per product/BU combination.

The script fits lightweight SARIMAX models with a single Salesforce regressor,
using the configuration's managed catalog to aggregate actuals. For each
product group and BU, it sweeps lags 0..max_lag (default 6) and reports the lag
with the lowest holdout MAE based on the configuration's TEST_SPLIT_MONTHS.
"""

from __future__ import annotations

import argparse
import importlib
import warnings
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd
from statsmodels.tsa.statespace.sarimax import SARIMAX

from sarima_model import SARIMAForecaster


def _load_config_class(dotted_path: str):
    """Resolve and return a configuration class from a dotted path string."""
    if "." not in dotted_path:
        raise ValueError("config class must be provided as module.ClassName")
    module_name, class_name = dotted_path.rsplit(".", 1)
    module = importlib.import_module(module_name)
    try:
        return getattr(module, class_name)
    except AttributeError as exc:
        raise ValueError(f"Unable to locate config class '{class_name}' in {module_name}") from exc


def _prepare_series(forecaster: SARIMAForecaster, series_key: str) -> Tuple[pd.Series, Dict]:
    """Return the actuals series and metadata for a given series key."""
    group_df = forecaster.combined_data.get(series_key)
    if group_df is None or group_df.empty:
        return pd.Series(dtype=float), {}

    metadata = forecaster.series_metadata.get(series_key, {})
    series = group_df.set_index("Month")["Actuals"].asfreq("MS")
    series = series.sort_index().astype(float)
    return series, metadata


def _extract_salesforce_features(
    forecaster: SARIMAForecaster,
    group_name: str,
    bu_code: Optional[str],
    index: pd.Index,
) -> pd.DataFrame:
    """Return the Salesforce feature matrix for the requested index."""
    exog = forecaster.get_exog_features_for_series(group_name, bu_code, index)
    if exog is None or exog.empty:
        return pd.DataFrame()
    salesforce_cols = [col for col in exog.columns if col.startswith("sf_")]
    if not salesforce_cols:
        return pd.DataFrame()
    return exog[salesforce_cols].astype(float)


def _evaluate_feature_lags(
    actuals: pd.Series,
    feature_series: pd.Series,
    order: Tuple[int, int, int, int, int, int, int],
    test_months: int,
    max_lag: int,
) -> Optional[Dict[str, float]]:
    """
    Evaluate lag candidates for a single Salesforce feature.

    Returns a dict containing the best lag and its MAE, or None if no lag could be
    evaluated successfully.
    """
    joined = pd.concat({"y": actuals, "feature": feature_series}, axis=1).dropna()
    if joined.empty or len(joined) <= test_months:
        return None

    results: List[Tuple[int, float]] = []
    for lag in range(max_lag + 1):
        shifted = joined["feature"].shift(lag)
        aligned = pd.concat({"y": joined["y"], "exog": shifted}, axis=1).dropna()
        if len(aligned) <= test_months:
            continue

        y_aligned = aligned["y"]
        exog_aligned = aligned["exog"]
        train_y = y_aligned.iloc[:-test_months]
        test_y = y_aligned.iloc[-test_months:]
        train_exog = exog_aligned.iloc[:-test_months].to_frame("exog")
        test_exog = exog_aligned.iloc[-test_months:].to_frame("exog")

        if train_y.empty or test_y.empty or train_exog.isnull().any().any() or test_exog.isnull().any().any():
            continue

        p, d, q, P, D, Q, s = order
        try:
            model = SARIMAX(
                train_y,
                exog=train_exog,
                order=(int(p), int(d), int(q)),
                seasonal_order=(int(P), int(D), int(Q), int(s)),
                trend="c",
                enforce_stationarity=False,
                enforce_invertibility=False,
            )
            fitted = model.fit(disp=False)
            forecast = fitted.get_forecast(steps=test_months, exog=test_exog)
            prediction = forecast.predicted_mean
            mae = float(np.mean(np.abs(prediction - test_y)))
            if np.isfinite(mae):
                results.append((lag, mae))
        except Exception as exc:  # pragma: no cover - defensive
            warnings.warn(f"SARIMAX failed for lag {lag}: {exc}", RuntimeWarning)
            continue

    if not results:
        return None

    best_lag, best_mae = min(results, key=lambda item: item[1])
    return {
        "best_lag": int(best_lag),
        "best_mae": float(best_mae),
        "lags_evaluated": len(results),
    }


def _ensure_output_directory(path: Path) -> None:
    """Create the parent directory for an output path if it doesn't exist."""
    parent = path.parent
    if not parent.exists():
        parent.mkdir(parents=True, exist_ok=True)


def analyze_lags(config_class: str, output_path: Path, max_lag: int) -> pd.DataFrame:
    """Run lag analysis and return the resulting DataFrame."""
    ConfigClass = _load_config_class(config_class)
    config = ConfigClass()
    if not getattr(config, "USE_SALESFORCE", False):
        warnings.warn(
            "Configuration has USE_SALESFORCE disabled; overriding to enable Salesforce data for lag analysis.",
            RuntimeWarning,
        )
        config.USE_SALESFORCE = True

    forecaster = SARIMAForecaster(config)
    forecaster.load_and_preprocess_data()
    forecaster.precompute_exogenous_store()

    if not forecaster.combined_data:
        raise RuntimeError("No combined product series are available for lag analysis.")

    results: List[Dict[str, object]] = []
    test_months = int(getattr(config, "TEST_SPLIT_MONTHS", 6))

    for series_key in sorted(forecaster.combined_data.keys()):
        actuals, metadata = _prepare_series(forecaster, series_key)
        if actuals.empty or len(actuals.dropna()) <= test_months:
            continue

        group_name = metadata.get("group", series_key)
        bu_code = metadata.get("bu_code")

        feature_matrix = _extract_salesforce_features(forecaster, group_name, bu_code, actuals.index)
        if feature_matrix.empty:
            continue

        order = forecaster.config.SARIMA_PARAMS.get(group_name, (1, 1, 1, 0, 0, 0, 12))
        for column in feature_matrix.columns:
            feature_series = feature_matrix[column]
            if feature_series.dropna().empty:
                continue

            evaluation = _evaluate_feature_lags(actuals, feature_series, order, test_months, max_lag)
            if not evaluation:
                continue

            results.append(
                {
                    "product_group": group_name,
                    "business_unit": bu_code or "",
                    "feature": column.replace("sf_", ""),
                    "best_lag": evaluation["best_lag"],
                    "validation_mae": evaluation["best_mae"],
                    "lags_evaluated": evaluation["lags_evaluated"],
                    "series_length": len(actuals),
                    "train_samples": max(len(actuals) - test_months, 0),
                    "test_samples": test_months,
                }
            )

    if not results:
        warnings.warn("No Salesforce features were evaluated; output will be empty.", RuntimeWarning)
        return pd.DataFrame(
            columns=[
                "product_group",
                "business_unit",
                "feature",
                "best_lag",
                "validation_mae",
                "lags_evaluated",
                "series_length",
                "train_samples",
                "test_samples",
            ]
        )

    frame = pd.DataFrame(results)
    frame = frame.sort_values(["product_group", "business_unit", "feature"]).reset_index(drop=True)
    _ensure_output_directory(output_path)
    frame.to_csv(output_path, index=False)
    return frame


def main(argv: Optional[Iterable[str]] = None) -> None:
    parser = argparse.ArgumentParser(description="Evaluate Salesforce feature lags per product/BU.")
    parser.add_argument(
        "--config-class",
        default="experiments.forecast_2025_test.experiment_config.Experiment2025Config",
        help="Dotted path to the configuration class to use.",
    )
    parser.add_argument(
        "--max-lag",
        type=int,
        default=6,
        help="Maximum lag (in months) to evaluate for each feature.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("analysis_outputs/salesforce_lag_analysis.csv"),
        help="Destination CSV file for the lag analysis report.",
    )

    args = parser.parse_args(argv)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=FutureWarning)
        warnings.simplefilter("ignore", category=UserWarning)
        frame = analyze_lags(args.config_class, args.output, args.max_lag)

    print(f"Lag analysis complete. Results written to {args.output} ({len(frame)} rows).")


if __name__ == "__main__":  # pragma: no cover
    main()
