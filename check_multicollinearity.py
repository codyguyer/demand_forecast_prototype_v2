"""
Multicollinearity diagnostics for Salesforce demand-planning features.

Generates per-product, per-BU correlation profiles and variance inflation
factors to highlight redundant feature information.
"""

from __future__ import annotations

import argparse
import math
import warnings
from itertools import combinations
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from statsmodels.stats.outliers_influence import variance_inflation_factor

from analysis_utils import PreparedDataset, load_prepared_dataset


def _valid_columns(
    frame: pd.DataFrame, candidate_columns: Sequence[str], min_obs: int, drop_constant: bool = True
) -> List[str]:
    valid: List[str] = []
    for column in candidate_columns:
        if column not in frame.columns:
            continue
        series = frame[column].dropna()
        if series.shape[0] < min_obs:
            continue
        if drop_constant and series.nunique() <= 1:
            continue
        valid.append(column)
    return valid


def _compute_correlations(
    frame: pd.DataFrame,
    columns: Sequence[str],
) -> List[Tuple[str, str, float]]:
    """Return upper-triangle pairwise correlations for the requested columns."""
    if len(columns) < 2:
        return []
    corr_matrix = frame[columns].corr()
    records: List[Tuple[str, str, float]] = []
    for left, right in combinations(columns, 2):
        value = corr_matrix.loc[left, right]
        records.append((left, right, float(value) if pd.notna(value) else math.nan))
    return records


def _compute_condition_number(matrix: np.ndarray) -> float:
    """Return the condition number of a feature matrix, or NaN if undefined."""
    if matrix.size == 0:
        return math.nan
    try:
        return float(np.linalg.cond(matrix))
    except Exception:
        return math.inf


def _standardise_matrix(frame: pd.DataFrame) -> np.ndarray:
    """Z-score standardise a feature frame for condition number diagnostics."""
    if frame.empty:
        return np.empty((0, 0))
    centered = frame - frame.mean(axis=0)
    scaled = centered / frame.std(axis=0, ddof=0)
    scaled = scaled.replace([np.inf, -np.inf], np.nan).dropna(axis=1, how="all")
    return scaled.dropna().to_numpy()


def _compute_vif(
    frame: pd.DataFrame,
    columns: Sequence[str],
) -> Tuple[List[Tuple[str, float]], float]:
    """Return per-column VIF values along with the matrix condition number."""
    if len(columns) < 2:
        return [], math.nan

    clean = frame[list(columns)].dropna()
    if clean.shape[0] < len(columns) + 1:
        return [], math.nan

    matrix = clean.to_numpy()
    matrix = np.column_stack([np.ones(matrix.shape[0]), matrix])

    values: List[Tuple[str, float]] = []
    for idx, column in enumerate(columns):
        try:
            vif_value = variance_inflation_factor(matrix, idx + 1)
        except Exception:
            vif_value = math.inf
        values.append((column, float(vif_value)))

    condition_number = _compute_condition_number(_standardise_matrix(clean[columns]))
    return values, condition_number


def analyze_multicollinearity(
    dataset: PreparedDataset,
    min_obs: int,
    include_actuals_in_corr: bool,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    corr_records: List[Dict[str, Optional[float]]] = []
    vif_records: List[Dict[str, Optional[float]]] = []

    for (group_key, bu), frame in dataset.data.groupby(["group_key", "BU"]):
        ordered = frame.sort_values("Date")
        feature_columns = _valid_columns(ordered, dataset.feature_columns, min_obs=min_obs)

        # Correlation diagnostics
        corr_columns = list(feature_columns)
        if include_actuals_in_corr and dataset.actual_column in ordered.columns:
            corr_columns_with_actual = _valid_columns(
                ordered, corr_columns + [dataset.actual_column], min_obs=min_obs, drop_constant=False
            )
            if len(corr_columns_with_actual) >= 2:
                corr_columns = corr_columns_with_actual
        correlation_pairs = _compute_correlations(ordered, corr_columns)
        for left, right, value in correlation_pairs:
            corr_records.append(
                {
                    "group_key": group_key,
                    "BU": bu,
                    "variable_x": left,
                    "variable_y": right,
                    "pearson_correlation": value,
                }
            )

        # VIF diagnostics
        vif_columns = _valid_columns(ordered, feature_columns, min_obs=min_obs, drop_constant=True)
        vif_values, condition_number = _compute_vif(ordered, vif_columns)
        if vif_values:
            for column, value in vif_values:
                vif_records.append(
                    {
                        "group_key": group_key,
                        "BU": bu,
                        "variable": column,
                        "vif": value,
                        "condition_number": condition_number,
                        "n_obs": int(ordered[column].dropna().shape[0]),
                    }
                )
        else:
            for column in feature_columns:
                vif_records.append(
                    {
                        "group_key": group_key,
                        "BU": bu,
                        "variable": column,
                        "vif": math.nan,
                        "condition_number": condition_number if not math.isnan(condition_number) else math.nan,
                        "n_obs": int(ordered[column].dropna().shape[0]) if column in ordered.columns else 0,
                        "notes": "Insufficient data for VIF calculation.",
                    }
                )

    correlation_df = pd.DataFrame.from_records(corr_records)
    vif_df = pd.DataFrame.from_records(vif_records)
    return correlation_df, vif_df


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Assess multicollinearity in Salesforce feature sets.")
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
        "--output-prefix",
        type=Path,
        default=Path("analysis_outputs") / "multicollinearity",
        help="Base path (without extension) for correlation and VIF result files.",
    )
    parser.add_argument(
        "--min-observations",
        type=int,
        default=8,
        help="Minimum non-null observations required to include a variable in diagnostics.",
    )
    parser.add_argument(
        "--include-actuals-in-corr",
        action="store_true",
        help="Include aligned actual quantities when computing correlations.",
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

    correlation_df, vif_df = analyze_multicollinearity(
        dataset=dataset,
        min_obs=args.min_observations,
        include_actuals_in_corr=args.include_actuals_in_corr,
    )

    output_prefix = Path(args.output_prefix)
    prefix_str = str(output_prefix)
    if prefix_str.lower().endswith(".csv"):
        prefix_str = prefix_str[:-4]
    corr_path = Path(f"{prefix_str}-correlations.csv")
    vif_path = Path(f"{prefix_str}-vif.csv")

    corr_path.parent.mkdir(parents=True, exist_ok=True)
    if vif_path.parent != corr_path.parent:
        vif_path.parent.mkdir(parents=True, exist_ok=True)

    correlation_df.to_csv(corr_path, index=False)
    vif_df.to_csv(vif_path, index=False)

    print(f"Saved correlation diagnostics to {corr_path.resolve()}")
    print(f"Saved VIF diagnostics to {vif_path.resolve()}")
    print(
        f"Evaluated {correlation_df['group_key'].nunique() if not correlation_df.empty else 0} product groups "
        f"for correlation and {vif_df['group_key'].nunique() if not vif_df.empty else 0} groups for VIF."
    )
    if dataset.missing_salesforce_mappings:
        print(
            f"Warning: {dataset.missing_salesforce_mappings} Salesforce rows could not be mapped to a catalog group."
        )
    if dataset.missing_actual_mappings:
        print(f"Warning: {dataset.missing_actual_mappings} actual rows lacked a catalog mapping.")


if __name__ == "__main__":
    # Some SciPy/NumPy warnings can be noisy when matrices are nearly singular.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        main()
