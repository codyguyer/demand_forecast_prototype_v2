"""
Backlog Integration Framework for SARIMA Model
Generates backlog-based exogenous features aligned with product groupings.
"""

import pandas as pd

from config import SARIMAConfig


class BacklogIntegration:
    """Prepare backlog series for use as exogenous regressors."""

    def __init__(self, config=None):
        self.config = config or SARIMAConfig()
        self.backlog_data = None
        self.processed_features = {}
        self.forecast_by_bu = bool(getattr(self.config, "FORECAST_BY_BU", False))
        self.bu_code_to_name = getattr(self.config, "BU_CODE_TO_NAME", {})

    @staticmethod
    def normalize_product_code(code):
        """Normalize product identifiers to ease mapping to configured groups."""
        if pd.isna(code):
            return ""
        normalized = str(code).strip().upper().replace(" ", "")
        for prefix in ("ITEM_", "ITEM", "PROD_", "PROD"):
            if normalized.startswith(prefix):
                normalized = normalized[len(prefix):]
                break
        return normalized.lstrip("_")

    def load_backlog_data(self, filepath_or_data):
        """Load backlog data from CSV or provided DataFrame."""
        if isinstance(filepath_or_data, str):
            data = pd.read_csv(filepath_or_data)
        else:
            data = filepath_or_data.copy()

        rename_map = {}
        for col in data.columns:
            normalized = col.strip().lower()
            if normalized in ("product", "product_code", "sku", "item"):
                rename_map[col] = "Product"
            elif normalized in ("month", "date"):
                rename_map[col] = "Month"
            elif normalized in ("bu", "business_unit", "business_unit_code"):
                rename_map[col] = "BU"
            elif normalized == "backlog":
                rename_map[col] = "Backlog"

        data = data.rename(columns=rename_map)

        required = {"Product", "Month", "Backlog"}
        missing = required - set(data.columns)
        if missing:
            raise ValueError(f"Backlog data missing required columns: {sorted(missing)}")

        data["Month"] = pd.to_datetime(data["Month"])
        data["Backlog"] = pd.to_numeric(data["Backlog"], errors="coerce").fillna(0.0)
        if "BU" in data.columns:
            data["BU"] = data["BU"].fillna("").astype(str).str.strip()
        else:
            data["BU"] = ""

        data["Normalized_Product"] = data["Product"].apply(self.normalize_product_code)
        data = data.sort_values(["Normalized_Product", "BU", "Month"])

        self.backlog_data = data
        print(f"Loaded {len(self.backlog_data)} backlog records")
        return self.backlog_data

    def process_backlog_features(self, product_groups):
        """Aggregate backlog into monthly features for each configured product group."""
        processed = {}

        if self.backlog_data is None or self.backlog_data.empty:
            print("No backlog data available to process.")
            self.processed_features = {}
            return {}

        backlog_df = self.backlog_data.copy()
        backlog_df["Month"] = backlog_df["Month"].dt.to_period("M").dt.to_timestamp()

        forecast_horizon_end = pd.to_datetime(self.config.CURRENT_MONTH) + pd.DateOffset(
            months=self.config.FORECAST_MONTHS
        )

        def build_feature_frame(slice_df):
            if slice_df.empty:
                return None

            monthly = slice_df.groupby("Month")["Backlog"].sum().sort_index()
            full_index = pd.date_range(
                start=monthly.index.min(),
                end=forecast_horizon_end,
                freq="MS"
            )
            monthly = monthly.reindex(full_index, fill_value=0.0)

            features = pd.DataFrame({"Date": monthly.index, "Backlog": monthly.values})
            features["Backlog_Lag1"] = features["Backlog"].shift(1).fillna(0.0)
            features["Backlog_Lag3"] = features["Backlog"].shift(3).fillna(0.0)
            features["Backlog_MA3"] = features["Backlog"].rolling(window=3, min_periods=1).mean()

            return features.fillna(0.0)

        for group_name, products in product_groups.items():
            normalized_targets = {self.normalize_product_code(p) for p in products}
            group_slice = backlog_df[
                backlog_df["Normalized_Product"].isin(normalized_targets)
            ].copy()

            if group_slice.empty:
                print(f"No backlog data found for {group_name} after normalization.")
                continue

            if self.forecast_by_bu and "BU" in group_slice.columns:
                for bu_code, bu_slice in group_slice.groupby("BU"):
                    if bu_slice.empty or not str(bu_code).strip():
                        continue
                    features = build_feature_frame(bu_slice)
                    if features is None:
                        continue
                    key = (group_name, str(bu_code).strip())
                    processed[key] = features
                    readable = self.bu_code_to_name.get(str(bu_code).strip(), bu_code)
                    print(f"  -> Backlog features prepared for {group_name} / BU {bu_code} ({readable})")
            else:
                features = build_feature_frame(group_slice)
                if features is None:
                    continue
                processed[group_name] = features
                print(f"  -> Backlog features prepared for {group_name}")

        self.processed_features = processed
        return processed

    def create_future_exog_features(self, group_name, forecast_start_date, forecast_months, bu_code=None):
        """Create future backlog feature frame for forecasting horizon."""
        key = (group_name, bu_code) if self.forecast_by_bu else group_name
        if key not in self.processed_features:
            return None

        features_df = self.processed_features[key].copy()

        if "Date" not in features_df.columns:
            print(f"Processed backlog features for {group_name} missing 'Date' column; cannot extend into future.")
            return None

        features_df = features_df.set_index("Date").sort_index()

        forecast_index = pd.date_range(
            start=forecast_start_date,
            periods=forecast_months,
            freq="MS"
        )

        future_df = features_df.reindex(forecast_index, fill_value=0.0)
        label = f"{group_name} ({self.bu_code_to_name.get(bu_code, bu_code)})" if bu_code else group_name
        print(f"Created future backlog features for {label}")

        return future_df

    def merge_with_actuals(self, actuals_data, group_name, bu_code=None):
        """Merge backlog features with actuals for modeling."""
        key = (group_name, bu_code) if self.forecast_by_bu else group_name
        label = f"{group_name} ({self.bu_code_to_name.get(bu_code, bu_code)})" if bu_code else group_name

        if isinstance(actuals_data, pd.Series):
            actuals_df = actuals_data.to_frame(name="Actuals")
        else:
            actuals_df = actuals_data.copy()

        if "Actuals" not in actuals_df.columns:
            raise ValueError("actuals_data must contain an 'Actuals' column.")

        actuals_reset = actuals_df.reset_index()
        index_name = actuals_df.index.name if actuals_df.index.name else "Month"
        if "Month" not in actuals_reset.columns:
            actuals_reset = actuals_reset.rename(columns={index_name: "Month"})

        if key not in self.processed_features:
            print(f"No backlog features available for {label}")
            actual_series = actuals_reset.set_index("Month")["Actuals"]
            return actual_series, None

        features_df = self.processed_features[key]

        merged = pd.merge(actuals_reset, features_df, left_on="Month", right_on="Date", how="left")

        feature_columns = getattr(self.config, "BACKLOG_FEATURES", ["Backlog"])
        merged = merged.sort_values("Month")

        for col in feature_columns:
            if col not in merged.columns:
                merged[col] = 0.0
            else:
                merged[col] = merged[col].fillna(0.0)

        merged = merged.set_index("Month")

        actual_series = merged["Actuals"]
        selected_features = merged[feature_columns]

        print(f"Merged {len(selected_features)} observations with backlog features for {label}")

        return actual_series, selected_features
