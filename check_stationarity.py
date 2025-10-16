"""
Stationarity diagnostics for Salesforce demand-planning features.

This script evaluates each Salesforce feature (and aligned actual quantities)
for every product group / business-unit combination using ADF and KPSS tests.
"""

from __future__ import annotations

import argparse
import math
import warnings
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd
from statsmodels.tsa.stattools import adfuller, kpss

from analysis_utils import PreparedDataset, load_prepared_dataset


def _run_adf(series: pd.Series) -> Dict[str, Optional[float]]:
    """Execute the Augmented Dickey-Fuller test, returning metrics or NaNs if invalid."""
    try:
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            stat, pvalue, used_lag, n_obs, crit_values, _ = adfuller(series.dropna(), autolag="AIC")
        return {
            "adf_stat": float(stat),
            "adf_pvalue": float(pvalue),
            "adf_lags": int(used_lag),
            "adf_nobs": int(n_obs),
        }
    except Exception as exc:  # statsmodels raises ValueError for small samples, LinAlgError, etc.
        return {
            "adf_stat": math.nan,
            "adf_pvalue": math.nan,
            "adf_lags": math.nan,
            "adf_nobs": math.nan,
            "adf_error": str(exc),
        }


def _run_kpss(series: pd.Series) -> Dict[str, Optional[float]]:
    """Execute the KPSS test (level stationarity), returning metrics or NaNs if invalid."""
    try:
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            stat, pvalue, lags, crit_values = kpss(series.dropna(), regression="c", nlags="auto")
        return {
            "kpss_stat": float(stat),
            "kpss_pvalue": float(pvalue),
            "kpss_lags": int(lags),
        }
    except Exception as exc:
        return {
            "kpss_stat": math.nan,
            "kpss_pvalue": math.nan,
            "kpss_lags": math.nan,
            "kpss_error": str(exc),
        }


def _prepare_variables(dataset: PreparedDataset, include_actual: bool) -> List[str]:
    columns: List[str] = list(dataset.feature_columns)
    if include_actual and dataset.actual_column in dataset.data.columns:
        columns.append(dataset.actual_column)
    return columns


def _diagnose_series(series: pd.Series, min_obs: int) -> Dict[str, Optional[float]]:
    """Collect stationarity metrics for a single time series."""
    clean_series = series.dropna()
    n_obs = int(clean_series.shape[0])
    result: Dict[str, Optional[float]] = {"n_obs": n_obs, "is_constant": False, "notes": ""}

    if n_obs < min_obs:
        result["notes"] = f"Insufficient observations (<{min_obs})."
        return result

    if clean_series.nunique() <= 1:
        result["is_constant"] = True
        result["notes"] = "Series is constant; stationarity tests are not informative."
        return result

    adf_metrics = _run_adf(clean_series)
    kpss_metrics = _run_kpss(clean_series)

    result.update(adf_metrics)
    result.update(kpss_metrics)

    adf_pvalue = adf_metrics.get("adf_pvalue")
    if adf_pvalue is not None and not math.isnan(adf_pvalue):
        result["adf_stationary"] = adf_pvalue < 0.05
    else:
        result["adf_stationary"] = None

    kpss_pvalue = kpss_metrics.get("kpss_pvalue")
    if kpss_pvalue is not None and not math.isnan(kpss_pvalue):
        result["kpss_stationary"] = kpss_pvalue > 0.05
    else:
        result["kpss_stationary"] = None

    notes: List[str] = []
    if "adf_error" in adf_metrics:
        notes.append(f"ADF: {adf_metrics['adf_error']}")
    if "kpss_error" in kpss_metrics:
        notes.append(f"KPSS: {kpss_metrics['kpss_error']}")
    if notes:
        result["notes"] = " | ".join(notes)

    return result


def run_stationarity_checks(
    dataset: PreparedDataset,
    min_obs: int,
    include_actual: bool,
) -> pd.DataFrame:
    """Generate a long-form DataFrame with stationarity diagnostics per feature."""
    variables = _prepare_variables(dataset, include_actual=include_actual)
    records: List[Dict[str, Optional[float]]] = []

    for (group_key, bu), group_df in dataset.data.groupby(["group_key", "BU"]):
        ordered = group_df.sort_values("Date")
        for column in variables:
            if column not in ordered.columns:
                continue
            series = ordered[column]
            diagnostics = _diagnose_series(series, min_obs=min_obs)
            record = {
                "group_key": group_key,
                "BU": bu,
                "variable": column,
            }
            record.update(diagnostics)
            records.append(record)

    return pd.DataFrame.from_records(records)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run stationarity diagnostics on Salesforce feature time series.")
    parser.add_argument(
        "--salesforce-path",
        type=Path,
        default=Path("managed_data") / "salesforce_data_v2.csv",
        help="Path to the Salesforce feature snapshot CSV.",
    )
    parser.add_argument(
        "--actuals-path",
        type=Path,
        default=Path("managed_data") / "Actuals.csv",
        help="Path to the actuals (quantity) CSV.",
    )
    parser.add_argument(
        "--catalog-path",
        type=Path,
        default=Path("managed_data") / "product_catalog.csv",
        help="Path to the product catalog CSV.",
    )
    parser.add_argument(
        "--reference-path",
        type=Path,
        default=Path("managed_data") / "sf_product_reference_key.csv",
        help="Path to the Salesforce product mapping CSV.",
    )
    parser.add_argument(
        "--output-path",
        type=Path,
        default=Path("analysis_outputs-stationarity_results.csv"),
        help="Where to write the aggregated diagnostics.",
    )
    parser.add_argument(
        "--min-observations",
        type=int,
        default=8,
        help="Minimum non-null observations required before running tests.",
    )
    parser.add_argument(
        "--include-actuals",
        action="store_true",
        help="Include the aligned actual quantity series in the diagnostics.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    dataset = load_prepared_dataset(
        salesforce_path=args.salesforce_path,
        actuals_path=args.actuals_path,
        catalog_path=args.catalog_path,
        reference_path=args.reference_path,
    )

    results = run_stationarity_checks(
        dataset=dataset,
        min_obs=args.min_observations,
        include_actual=args.include_actuals,
    )

    output_path: Path = args.output_path
    output_path.parent.mkdir(parents=True, exist_ok=True)
    results.to_csv(output_path, index=False)

    print(f"Saved stationarity diagnostics to {output_path.resolve()}")
    print(
        f"Processed {results['group_key'].nunique()} product groups "
        f"across {results['BU'].nunique()} business units."
    )
    if dataset.missing_salesforce_mappings:
        print(
            f"Warning: {dataset.missing_salesforce_mappings} Salesforce rows could not be mapped to a catalog group."
        )
    if dataset.missing_actual_mappings:
        print(f"Warning: {dataset.missing_actual_mappings} actual rows lacked a catalog mapping.")


if __name__ == "__main__":
    main()
