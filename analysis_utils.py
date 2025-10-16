"""
Shared utilities for analyzing Salesforce-managed demand planning datasets.

These helpers load the managed CSV inputs, map Salesforce products onto
catalog group keys, and return aligned time series that downstream analysis
scripts can operate on.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence

import pandas as pd

BASE_DIR = Path(__file__).resolve().parent
DEFAULT_MANAGED_DATA_DIR = BASE_DIR / "managed_data"

# Canonical Salesforce feature columns available in the managed extract.
SALESFORCE_FEATURE_COLUMNS: Sequence[str] = (
    "Revenue",
    "Quantity",
    "Open_Opportunities",
    "New_Quotes",
    "Revenue_new_orders",
    "Quantity_new_orders",
)


@dataclass
class PreparedDataset:
    """Container for harmonised feature time series."""

    data: pd.DataFrame
    feature_columns: List[str]
    actual_column: str = "Actual_Quantity"
    missing_salesforce_mappings: int = 0
    missing_actual_mappings: int = 0


def _parse_multi_value(raw_value: Optional[str]) -> List[str]:
    """Split pipe/semicolon delimited catalog fields into trimmed values."""
    if raw_value in (None, ""):
        return []
    if isinstance(raw_value, str):
        text = raw_value.replace("\r", "").strip()
    else:
        text = str(raw_value).strip()
    if not text:
        return []
    for delimiter in ("|", ";", "\n"):
        if delimiter in text:
            parts = [part.strip() for part in text.split(delimiter)]
            return [part for part in parts if part]
    return [text]


def _build_sku_to_group_map(product_catalog: pd.DataFrame) -> Dict[str, str]:
    """Map individual SKU codes from the catalog onto group keys."""
    mapping: Dict[str, str] = {}
    for row in product_catalog.itertuples():
        sku_list = getattr(row, "sku_list", None)
        group_key = getattr(row, "group_key", None)
        if pd.isna(group_key) or sku_list in (None, ""):
            continue
        group = str(group_key).strip()
        if not group:
            continue
        for sku in _parse_multi_value(sku_list):
            # Preserve original casing and spacing after stripping padding.
            cleaned = sku.strip()
            if cleaned:
                mapping[cleaned] = group
    return mapping


def _build_group_to_bu_map(product_catalog: pd.DataFrame) -> Dict[str, List[str]]:
    """Collect the list of valid business unit codes for each product group."""
    mapping: Dict[str, List[str]] = {}
    for row in product_catalog.itertuples():
        group_key = getattr(row, "group_key", None)
        if pd.isna(group_key):
            continue
        group = str(group_key).strip()
        if not group:
            continue
        bu_raw = getattr(row, "business_unit_code", None)
        bu_values = _parse_multi_value(bu_raw)
        if not bu_values:
            continue
        cleaned_codes = [code.strip() for code in bu_values if code and str(code).strip()]
        if cleaned_codes:
            existing = mapping.setdefault(group, [])
            for code in cleaned_codes:
                if code not in existing:
                    existing.append(code)
    return mapping


def _normalise_numeric_columns(df: pd.DataFrame, columns: Iterable[str]) -> None:
    """Coerce the requested columns to numeric dtype in place."""
    for column in columns:
        if column in df.columns:
            series = df[column]
            if pd.api.types.is_string_dtype(series) or series.dtype == object:
                # Strip common currency formatting artefacts before coercion.
                cleaned = (
                    series.astype("string")
                    .str.strip()
                    .replace({"": pd.NA, "nan": pd.NA, "None": pd.NA}, regex=False)
                    .str.replace(r"[\$,]", "", regex=True)
                )
                df[column] = cleaned
            df[column] = pd.to_numeric(df[column], errors="coerce")


def _map_salesforce_products(
    salesforce_df: pd.DataFrame, reference_df: pd.DataFrame, known_groups: Sequence[str]
) -> pd.DataFrame:
    """Attach catalog group keys to Salesforce products using the reference table."""
    reference = reference_df.dropna(subset=["salesforce_product_name"]).copy()
    group_key_set = {
        str(value).strip()
        for value in known_groups
        if not pd.isna(value) and str(value).strip()
    }
    reference["salesforce_product_name"] = reference["salesforce_product_name"].astype("string").str.strip()
    reference["group_key"] = reference["group_key"].astype("string").str.strip()
    reference = reference[reference["group_key"].isin(group_key_set)]

    mapped = salesforce_df.merge(
        reference.rename(columns={"group_key": "sf_group_key"}),
        how="left",
        left_on="SF_Product_Name",
        right_on="salesforce_product_name",
    )

    mapped["SF_Product_Name"] = mapped["SF_Product_Name"].astype("string").str.strip()

    # Direct matches between product names and catalog group keys act as a fallback.
    direct_match_mask = mapped["sf_group_key"].isna() & mapped["SF_Product_Name"].isin(group_key_set)
    mapped.loc[direct_match_mask, "sf_group_key"] = mapped.loc[direct_match_mask, "SF_Product_Name"]

    mapped = mapped.rename(columns={"sf_group_key": "group_key"})
    mapped = mapped.drop(columns=["salesforce_product_name"], errors="ignore")
    return mapped


def load_prepared_dataset(
    salesforce_path: Path = DEFAULT_MANAGED_DATA_DIR / "salesforce_data_v2.csv",
    actuals_path: Path = DEFAULT_MANAGED_DATA_DIR / "Actuals.csv",
    catalog_path: Path = DEFAULT_MANAGED_DATA_DIR / "product_catalog.csv",
    reference_path: Path = DEFAULT_MANAGED_DATA_DIR / "sf_product_reference_key.csv",
) -> PreparedDataset:
    """
    Load managed Salesforce and actuals data, align them by (group_key, BU, month),
    and return a prepared dataset ready for analysis.
    """
    salesforce_df = pd.read_csv(salesforce_path)
    actuals_df = pd.read_csv(actuals_path)
    catalog_df = pd.read_csv(catalog_path)
    reference_df = pd.read_csv(reference_path)

    # Normalise catalog group keys and derive the valid set we will analyse.
    catalog_df["group_key"] = catalog_df["group_key"].astype("string").str.strip()
    valid_group_keys = {
        str(value).strip()
        for value in catalog_df["group_key"].dropna().unique().tolist()
        if str(value).strip()
    }

    sku_to_group = _build_sku_to_group_map(catalog_df)
    group_to_bu = _build_group_to_bu_map(catalog_df)

    # Map actuals onto catalog group keys.
    actuals_df["Product"] = actuals_df["Product"].astype("string").str.strip()
    actuals_df["group_key"] = actuals_df["Product"].map(sku_to_group)
    actuals_df["Date"] = pd.to_datetime(actuals_df["Month"])

    missing_actual_mappings = int(actuals_df["group_key"].isna().sum())
    actuals_df = actuals_df.dropna(subset=["group_key", "BU", "Date"])
    actuals_df["group_key"] = actuals_df["group_key"].astype("string").str.strip()
    actuals_df["BU"] = actuals_df["BU"].astype("string").str.strip()
    actuals_df = actuals_df[actuals_df["group_key"].isin(valid_group_keys)]
    actuals_df = actuals_df[
        actuals_df.apply(
            lambda row: row["BU"] in group_to_bu.get(row["group_key"], []),
            axis=1,
        )
    ]

    actuals_grouped = (
        actuals_df.groupby(["group_key", "BU", "Date"], as_index=False)["Actuals"].sum().rename(
            columns={"Actuals": "Actual_Quantity"}
        )
    )

    # Prepare Salesforce feature set (only keep products represented in the catalog).
    reference_df["group_key"] = reference_df["group_key"].astype("string").str.strip()
    reference_df = reference_df[reference_df["group_key"].isin(valid_group_keys)]
    valid_sf_names = {
        str(value).strip()
        for value in reference_df["salesforce_product_name"].astype("string").str.strip().dropna().unique().tolist()
        if str(value).strip()
    } | valid_group_keys
    salesforce_df["SF_Product_Name"] = salesforce_df["SF_Product_Name"].astype("string").str.strip()
    salesforce_df["BU"] = salesforce_df["BU"].astype("string").str.strip()
    salesforce_df = salesforce_df[salesforce_df["SF_Product_Name"].isin(valid_sf_names)]
    salesforce_df["Timestamp"] = pd.to_datetime(salesforce_df["Timestamp"])
    mapped_sf = _map_salesforce_products(salesforce_df, reference_df, catalog_df["group_key"])

    mapped_sf["group_key"] = mapped_sf["group_key"].astype("string").str.strip()
    missing_salesforce_mappings = int(mapped_sf["group_key"].isna().sum())
    mapped_sf = mapped_sf.dropna(subset=["group_key", "BU", "Timestamp"])
    mapped_sf["BU"] = mapped_sf["BU"].astype("string").str.strip()
    mapped_sf = mapped_sf[mapped_sf["group_key"].isin(valid_group_keys)]
    mapped_sf = mapped_sf[
        mapped_sf.apply(
            lambda row: row["BU"] in group_to_bu.get(row["group_key"], []),
            axis=1,
        )
    ]

    _normalise_numeric_columns(mapped_sf, SALESFORCE_FEATURE_COLUMNS)

    sf_grouped = (
        mapped_sf.groupby(["group_key", "BU", "Timestamp"], as_index=False)[list(SALESFORCE_FEATURE_COLUMNS)].sum()
    )
    sf_grouped = sf_grouped.rename(columns={"Timestamp": "Date"})

    # Join Salesforce features with actual quantities where available.
    combined = (
        sf_grouped.set_index(["group_key", "BU", "Date"])
        .join(
            actuals_grouped.set_index(["group_key", "BU", "Date"]),
            how="left",
        )
        .reset_index()
        .sort_values(["group_key", "BU", "Date"])
    )

    return PreparedDataset(
        data=combined,
        feature_columns=[column for column in SALESFORCE_FEATURE_COLUMNS if column in combined.columns],
        actual_column="Actual_Quantity",
        missing_salesforce_mappings=missing_salesforce_mappings,
        missing_actual_mappings=missing_actual_mappings,
    )


__all__ = [
    "PreparedDataset",
    "SALESFORCE_FEATURE_COLUMNS",
    "load_prepared_dataset",
]
