"""
Diagnostics for backlog exogenous variables.

This script evaluates stationarity (Augmented Dickey-Fuller) and multicollinearity
via Variance Inflation Factors for backlog-derived features on a per-product basis.
"""

from __future__ import annotations

import argparse
import math
from itertools import combinations
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.tsa.stattools import adfuller


DEFAULT_ALPHA = 0.05
DEFAULT_VIF_THRESHOLD = 5.0
FEATURE_COLUMNS = ("raw_data", "log_raw_data", "backlog_standardized")
TARGET_COLUMN = "actuals"


def parse_args() -> argparse.Namespace:
    script_dir = Path(__file__).resolve().parent
    upstream_dir = script_dir.parent

    parser = argparse.ArgumentParser(
        description=(
            "Assess stationarity and multicollinearity of backlog exogenous variables "
            "for each product (and BU, when available)."
        )
    )
    parser.add_argument(
        "--backlog-file",
        type=Path,
        default=upstream_dir / "managed_data" / "Backlog.csv",
        help="Path to the backlog CSV file (default: managed_data/Backlog.csv).",
    )
    parser.add_argument(
        "--actuals-file",
        type=Path,
        default=upstream_dir / "managed_data" / "Actuals.csv",
        help="Path to the actuals CSV file (default: managed_data/Actuals.csv).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=script_dir / "exog_check_stationary+collinear.csv",
        help="Destination CSV for diagnostic matrix.",
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=DEFAULT_ALPHA,
        help="Significance level for ADF stationarity test (default: 0.05).",
    )
    parser.add_argument(
        "--vif-threshold",
        type=float,
        default=DEFAULT_VIF_THRESHOLD,
        help="Threshold above which VIF indicates concerning multicollinearity (default: 5.0).",
    )
    parser.add_argument(
        "--min-observations",
        type=int,
        default=6,
        help="Minimum observations required to run tests for a series (default: 6).",
    )
    return parser.parse_args()


def load_backlog_dataframe(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Backlog file not found at: {path}")

    df = pd.read_csv(path)
    expected_columns = {"Product", "Month", "raw_data", "Backlog"}
    missing = expected_columns - set(df.columns)
    if missing:
        raise ValueError(
            f"Backlog CSV missing required columns: {sorted(missing)}. "
            "Ensure raw backlog data includes 'Product', 'Month', 'raw_data', and 'Backlog'."
        )

    df["Product"] = df["Product"].astype(str).str.strip()
    if "BU" in df.columns:
        df["BU"] = df["BU"].fillna("").astype(str).str.strip()
    else:
        df["BU"] = ""

    df["Month"] = pd.to_datetime(df["Month"], errors="coerce")
    df["raw_data"] = pd.to_numeric(df["raw_data"], errors="coerce")
    df["Backlog"] = pd.to_numeric(df["Backlog"], errors="coerce")

    df = df.dropna(subset=["Product", "Month", "raw_data"])
    df = df.sort_values(["Product", "BU", "Month"])
    return df


def load_actuals_dataframe(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Actuals file not found at: {path}")

    df = pd.read_csv(path)
    expected_columns = {"Product", "Month", "Actuals"}
    missing = expected_columns - set(df.columns)
    if missing:
        raise ValueError(
            f"Actuals CSV missing required columns: {sorted(missing)}. "
            "Ensure the file includes 'Product', 'Month', and 'Actuals'."
        )

    df["Product"] = df["Product"].astype(str).str.strip()
    if "BU" in df.columns:
        df["BU"] = df["BU"].fillna("").astype(str).str.strip()
    else:
        df["BU"] = ""

    df["Month"] = pd.to_datetime(df["Month"], errors="coerce")
    df["Actuals"] = pd.to_numeric(df["Actuals"], errors="coerce")

    df = df.dropna(subset=["Product", "Month"])
    df = df.sort_values(["Product", "BU", "Month"])
    return df


def build_feature_frame(group_df: pd.DataFrame) -> pd.DataFrame:
    base = (
        group_df.copy()
        .set_index("Month")
        .sort_index()
    )

    features = pd.DataFrame(index=base.index)
    features["raw_data"] = base["raw_data"].astype(float)
    features["backlog_standardized"] = base["Backlog"].astype(float)

    log_series = pd.Series(np.nan, index=features.index, dtype=float)
    positive_mask = features["raw_data"] > 0
    if positive_mask.any():
        log_series.loc[positive_mask] = np.log(features.loc[positive_mask, "raw_data"])
    features["log_raw_data"] = log_series

    # Reorder columns and drop rows where all features are missing.
    features = features[list(FEATURE_COLUMNS)].dropna(how="all")
    return features


def compute_adf(series: pd.Series, alpha: float, min_obs: int) -> Tuple[Dict[str, float], List[str]]:
    notes: List[str] = []
    clean = series.dropna()

    if len(clean) < max(3, min_obs):
        notes.append(f"ADF skipped (needs >= {max(3, min_obs)} observations, has {len(clean)}).")
        return {}, notes

    try:
        result = adfuller(clean, autolag="AIC")
    except ValueError as exc:
        notes.append(f"ADF failed: {exc}")
        return {}, notes

    statistic, pvalue, usedlag, nobs, critical_values, icbest = result
    stationary = bool(pvalue <= alpha)

    metrics = {
        "adf_statistic": float(statistic),
        "adf_pvalue": float(pvalue),
        "adf_used_lag": int(usedlag),
        "adf_nobs": int(nobs),
        "adf_icbest": float(icbest),
        "adf_stationary_flag": stationary,
        "adf_crit_1pct": float(critical_values.get("1%", np.nan)),
        "adf_crit_5pct": float(critical_values.get("5%", np.nan)),
        "adf_crit_10pct": float(critical_values.get("10%", np.nan)),
    }
    return metrics, notes


def compute_vif(
    feature_frame: pd.DataFrame, min_obs: int
) -> Tuple[Dict[str, float], int, List[str]]:
    notes: List[str] = []
    valid_columns = [
        col for col in feature_frame.columns if feature_frame[col].dropna().std(ddof=0) > 0
    ]

    if len(valid_columns) < 2:
        notes.append("VIF skipped (requires at least two features with non-zero variance).")
        return {col: math.nan for col in feature_frame.columns}, 0, notes

    matrix = feature_frame[valid_columns].dropna()
    obs = len(matrix)

    if obs <= len(valid_columns) or obs < min_obs:
        notes.append(
            f"VIF skipped (needs > {len(valid_columns)} complete rows; has {obs})."
        )
        return {col: math.nan for col in feature_frame.columns}, obs, notes

    matrix = sm.add_constant(matrix, has_constant="add")

    vif_values: Dict[str, float] = {col: math.nan for col in feature_frame.columns}
    for idx, column in enumerate(matrix.columns):
        if column == "const":
            continue
        try:
            with np.errstate(divide="ignore", invalid="ignore"):
                vif = float(variance_inflation_factor(matrix.values, idx))
        except Exception as exc:  # pragma: no cover - defensive
            notes.append(f"VIF failed for {column}: {exc}")
            vif = math.nan

        if math.isinf(vif):
            notes.append(f"VIF infinite for {column} (perfect collinearity detected).")
        vif_values[column] = vif

    return vif_values, obs, notes


def calculate_pairwise_correlations(feature_frame: pd.DataFrame) -> Dict[str, float]:
    corr_map: Dict[str, float] = {}
    cleaned = feature_frame.dropna()
    if cleaned.shape[0] < 2:
        return {f"corr_{a}__{b}": math.nan for a, b in combinations(feature_frame.columns, 2)}

    corr_matrix = cleaned.corr()
    for a, b in combinations(feature_frame.columns, 2):
        key = f"corr_{a}__{b}"
        if a in corr_matrix and b in corr_matrix.columns:
            corr_map[key] = float(corr_matrix.loc[a, b])
        else:
            corr_map[key] = math.nan
    return corr_map


def compute_feature_actual_correlations(
    feature_values: pd.DataFrame, target_series: pd.Series
) -> Tuple[Dict[str, float], List[str]]:
    corr_map: Dict[str, float] = {}
    notes: List[str] = []

    if target_series is None or target_series.dropna().shape[0] < 2:
        for feature in FEATURE_COLUMNS:
            corr_map[f"corr_{feature}__actuals"] = math.nan
        notes.append("Insufficient actuals data for feature-to-actual correlations.")
        return corr_map, notes

    combined = pd.concat([feature_values, target_series.rename(TARGET_COLUMN)], axis=1)

    for feature in FEATURE_COLUMNS:
        feature_key = f"corr_{feature}__actuals"
        pair = combined[[feature, TARGET_COLUMN]].dropna()
        if pair.shape[0] < 2:
            corr_map[feature_key] = math.nan
            continue
        corr_map[feature_key] = float(pair[feature].corr(pair[TARGET_COLUMN]))

    return corr_map, notes


def series_summary_stats(series: pd.Series) -> Dict[str, float]:
    clean = series.dropna()
    if clean.empty:
        return {
            "series_count": 0,
            "series_mean": math.nan,
            "series_std": math.nan,
            "series_min": math.nan,
            "series_max": math.nan,
        }

    return {
        "series_count": int(clean.count()),
        "series_mean": float(clean.mean()),
        "series_std": float(clean.std(ddof=0)),
        "series_min": float(clean.min()),
        "series_max": float(clean.max()),
    }


def analyze_group(
    product: str,
    bu: str,
    feature_frame: pd.DataFrame,
    target_series: pd.Series,
    alpha: float,
    vif_threshold: float,
    min_obs: int,
) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []

    if feature_frame.empty:
        return rows

    feature_values = feature_frame[list(FEATURE_COLUMNS)]

    adf_results: Dict[str, Dict[str, float]] = {}
    adf_notes: Dict[str, List[str]] = {}
    for feature in FEATURE_COLUMNS:
        metrics, notes = compute_adf(feature_values[feature], alpha=alpha, min_obs=min_obs)
        adf_results[feature] = metrics
        adf_notes[feature] = notes

    vif_values, vif_obs, vif_notes = compute_vif(feature_values, min_obs=min_obs)
    correlation_metrics = calculate_pairwise_correlations(feature_values)
    feature_actual_corrs, feature_actual_notes = compute_feature_actual_correlations(
        feature_values, target_series
    )
    corr_notes = []
    if all(math.isnan(value) for value in correlation_metrics.values()):
        corr_notes.append("Insufficient data for pairwise correlations.")

    for feature in FEATURE_COLUMNS:
        row: Dict[str, object] = {
            "product": product,
            "business_unit": bu,
            "feature": feature,
            "first_month": feature_frame.index.min(),
            "last_month": feature_frame.index.max(),
        }

        summary_stats = series_summary_stats(feature_values[feature])
        row.update(summary_stats)

        if adf_results[feature]:
            row.update(adf_results[feature])
        else:
            row["adf_statistic"] = math.nan
            row["adf_pvalue"] = math.nan
            row["adf_used_lag"] = math.nan
            row["adf_nobs"] = math.nan
            row["adf_icbest"] = math.nan
            row["adf_stationary_flag"] = False
            row["adf_crit_1pct"] = math.nan
            row["adf_crit_5pct"] = math.nan
            row["adf_crit_10pct"] = math.nan

        vif_value = vif_values.get(feature, math.nan)
        row["vif_value"] = vif_value
        row["vif_observations"] = vif_obs
        if math.isnan(vif_value):
            row["vif_flag_high"] = False
        else:
            row["vif_flag_high"] = bool(vif_value >= vif_threshold)

        for corr_key, corr_value in correlation_metrics.items():
            row[corr_key] = corr_value
        for corr_key, corr_value in feature_actual_corrs.items():
            row[corr_key] = corr_value

        note_entries: List[str] = []
        note_entries.extend(adf_notes.get(feature, []))
        note_entries.extend(vif_notes)
        note_entries.extend(corr_notes)
        note_entries.extend(feature_actual_notes)
        row["notes"] = "; ".join(dict.fromkeys(note_entries)) if note_entries else ""

        rows.append(row)

    return rows


def run_analysis(
    backlog_df: pd.DataFrame,
    actuals_df: pd.DataFrame,
    alpha: float,
    vif_threshold: float,
    min_obs: int,
) -> pd.DataFrame:
    results: List[Dict[str, object]] = []
    grouped = backlog_df.groupby(["Product", "BU"], sort=True)

    actuals_grouped = actuals_df.groupby(["Product", "BU"]) if not actuals_df.empty else []
    actuals_lookup: Dict[Tuple[str, str], pd.Series] = {}
    for (prod, bu), group in actuals_grouped:
        series = (
            group.dropna(subset=["Actuals"])
            .set_index("Month")
            .sort_index()["Actuals"]
        )
        actuals_lookup[(prod, bu)] = series

    for (product, bu), group in grouped:
        feature_frame = build_feature_frame(group)
        target_series = actuals_lookup.get((product, bu), pd.Series(dtype=float))
        feature_frame = feature_frame.join(
            target_series.rename(TARGET_COLUMN), how="left"
        )
        if TARGET_COLUMN not in feature_frame.columns:
            feature_frame[TARGET_COLUMN] = math.nan

        group_rows = analyze_group(
            product=product,
            bu=bu,
            feature_frame=feature_frame,
            target_series=feature_frame[TARGET_COLUMN],
            alpha=alpha,
            vif_threshold=vif_threshold,
            min_obs=min_obs,
        )
        results.extend(group_rows)

    if not results:
        return pd.DataFrame(columns=["product", "business_unit", "feature"])

    results_df = pd.DataFrame(results)
    results_df = results_df.sort_values(["product", "business_unit", "feature"]).reset_index(drop=True)
    return results_df


def main() -> None:
    args = parse_args()

    backlog_df = load_backlog_dataframe(args.backlog_file)
    actuals_df = load_actuals_dataframe(args.actuals_file)
    results_df = run_analysis(
        backlog_df=backlog_df,
        actuals_df=actuals_df,
        alpha=args.alpha,
        vif_threshold=args.vif_threshold,
        min_obs=args.min_observations,
    )

    output_path: Path = args.output
    output_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        results_df.to_csv(output_path, index=False)
    except PermissionError as exc:
        raise PermissionError(
            f"Unable to write results to {output_path}. Close any applications using the file and rerun."
        ) from exc

    total_products = results_df[["product", "business_unit"]].drop_duplicates().shape[0]
    total_rows = len(results_df)
    print(f"Diagnostics calculated for {total_products} product/BU combinations.")
    print(f"Generated {total_rows} feature-level rows.")
    print(f"Results saved to: {output_path}")


if __name__ == "__main__":
    main()
