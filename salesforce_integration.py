"""
Salesforce Integration Framework for SARIMA Model
Future enhancement to incorporate pipeline data as external regressors
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple
from statsmodels.tsa.statespace.sarimax import SARIMAX
from config import SARIMAConfig

class SalesforceIntegration:
    """
    Framework for integrating Salesforce pipeline data into SARIMA forecasting
    """

    @staticmethod
    def normalize_product_code(code):
        """Normalize product identifiers so Salesforce matches actuals mapping."""
        if pd.isna(code):
            return ""
        normalized = str(code).strip().upper().replace(" ", "")
        for prefix in ("ITEM_", "ITEM", "PROD_", "PROD"):
            if normalized.startswith(prefix):
                normalized = normalized[len(prefix):]
                break
        normalized = normalized.lstrip("_")
        return normalized

    def __init__(self, config=None):
        self.config = config or SARIMAConfig()
        self.pipeline_data = None
        self.processed_features = None
        self.forecast_by_bu = bool(getattr(self.config, "FORECAST_BY_BU", False))
        self.bu_name_to_code = {
            str(k).strip().lower(): v for k, v in getattr(self.config, "BU_NAME_TO_CODE", {}).items()
        }
        self.bu_code_to_name = getattr(self.config, "BU_CODE_TO_NAME", {})
        self.group_key_lookup = {
            str(key).strip(): str(key).strip()
            for key in getattr(self.config, "PRODUCT_GROUPS", {}).keys()
            if str(key).strip()
        }
        self.group_key_lower_lookup = {key.lower(): original for original, key in self.group_key_lookup.items()}
        self.group_to_bu_codes = {
            group: tuple(codes)
            for group, codes in getattr(self.config, "GROUP_TO_BU_CODES", {}).items()
        }
        raw_sku_lookup = getattr(self.config, "SKU_TO_GROUP_MAP", {}) or {}
        self.raw_sku_lookup = {str(sku).strip(): group for sku, group in raw_sku_lookup.items() if str(sku).strip()}
        self.normalized_sku_lookup = {
            self.normalize_product_code(sku): group
            for sku, group in self.raw_sku_lookup.items()
            if self.normalize_product_code(sku)
        }
        self.reference_lookup = self._build_reference_lookup()
        self.normalized_reference_lookup = {
            self.normalize_product_code(name): group
            for name, group in self.reference_lookup.items()
            if self.normalize_product_code(name)
        }
        self.base_feature_columns: Sequence[str] = (
            "Revenue",
            "Quantity",
            "Open_Opportunities",
            "New_Quotes",
            "Revenue_new_orders",
            "Quantity_new_orders",
        )
        self.available_feature_columns: List[str] = []

    def _resolve_reference_path(self) -> Optional[Path]:
        raw_path = getattr(self.config, "SALESFORCE_REFERENCE_FILE", None)
        if not raw_path:
            return None
        try:
            path = Path(raw_path)
        except TypeError:
            return None
        if not path.is_absolute():
            path = Path(self.config.BASE_DIR) / path
        if not path.exists():
            print(f"Warning: Salesforce reference file not found at {path}")
            return None
        return path

    def _build_reference_lookup(self) -> Dict[str, str]:
        path = self._resolve_reference_path()
        if path is None:
            return {}
        try:
            reference_df = pd.read_csv(path)
        except Exception as exc:
            print(f"Warning: Failed to load Salesforce reference table from {path}: {exc}")
            return {}

        mapping: Dict[str, str] = {}
        valid_groups = set(self.group_key_lookup.keys())
        for row in reference_df.itertuples():
            group_raw = getattr(row, "group_key", None)
            name_raw = getattr(row, "salesforce_product_name", None)
            if not isinstance(group_raw, str) or not isinstance(name_raw, str):
                continue
            group = group_raw.strip()
            name = name_raw.strip()
            if not group or not name:
                continue
            if valid_groups and group not in valid_groups:
                continue
            key = name.lower()
            if key and key not in mapping:
                mapping[key] = group
        return mapping

    def _standardize_columns(self, dataframe: pd.DataFrame) -> pd.DataFrame:
        if dataframe is None:
            return dataframe

        working = dataframe.copy()
        fields = getattr(self.config, "SALESFORCE_FIELDS", {}) or {}

        canonical_targets = {
            "timestamp": "Timestamp",
            "product": "SF_Product_Name",
            "product_name": "SF_Product_Name",
            "business_unit": "Business_Unit_Code",
            "revenue": "Revenue",
            "quantity": "Quantity",
            "open_opportunities": "Open_Opportunities",
            "new_quotes": "New_Quotes",
            "revenue_new_orders": "Revenue_new_orders",
            "quantity_new_orders": "Quantity_new_orders",
        }

        fallback_sources = {
            "timestamp": ["Timestamp", "CloseDate", "Date"],
            "product": ["Product", "SF_Product_Name", "Product_Name"],
            "product_name": ["Product", "SF_Product_Name", "Product_Name"],
            "business_unit": ["Business_Unit_Code", "Business_Unit", "BU", "Business_Unit_Name"],
            "revenue": ["Revenue", "Weighted_Pipeline_Amount", "Weighted_Pipeline_Dollars"],
            "quantity": ["Quantity", "Weighted_Pipeline_Quantity"],
            "open_opportunities": ["Open_Opportunities"],
            "new_quotes": ["New_Quotes"],
            "revenue_new_orders": ["Revenue_new_orders"],
            "quantity_new_orders": ["Quantity_new_orders"],
        }

        rename_map: Dict[str, str] = {}

        def candidate_sources(key: str) -> List[str]:
            configured = fields.get(key)
            candidates: List[str] = []
            if isinstance(configured, str):
                candidates.append(configured)
            if key == "product":
                alt = fields.get("product_name")
                if isinstance(alt, str):
                    candidates.append(alt)
            if key == "product_name":
                alt = fields.get("product")
                if isinstance(alt, str):
                    candidates.append(alt)
            candidates.extend(fallback_sources.get(key, []))
            return [candidate for candidate in candidates if candidate]

        for key, target in canonical_targets.items():
            for candidate in candidate_sources(key):
                if candidate in working.columns:
                    rename_map[candidate] = target
                    break

        if "BU" in working.columns and "Business_Unit_Code" not in rename_map and "Business_Unit_Code" not in working.columns:
            rename_map["BU"] = "Business_Unit_Code"

        working = working.rename(columns=rename_map)

        if "SF_Product_Name" not in working.columns and "Product" in working.columns:
            working = working.rename(columns={"Product": "SF_Product_Name"})

        if "Business_Unit_Code" not in working.columns and "Business_Unit_Name" in working.columns:
            working["Business_Unit_Code"] = working["Business_Unit_Name"]

        if "BU" not in working.columns and "Business_Unit_Code" in working.columns:
            working["BU"] = working["Business_Unit_Code"]

        return working

    def _normalize_business_unit(self, value) -> Optional[str]:
        if pd.isna(value):
            return None
        text = str(value).strip()
        if not text:
            return None
        if text in self.bu_code_to_name:
            return text
        lookup_key = text.lower()
        if lookup_key in self.bu_name_to_code:
            return self.bu_name_to_code[lookup_key]
        return text.upper()

    def _filter_valid_business_units(self, dataframe: pd.DataFrame) -> pd.DataFrame:
        if not self.forecast_by_bu:
            return dataframe
        if not self.group_to_bu_codes:
            return dataframe

        def is_valid(row) -> bool:
            codes = self.group_to_bu_codes.get(row["group_key"])
            if not codes:
                return True
            return row["Business_Unit_Code"] in codes

        mask = dataframe.apply(is_valid, axis=1)
        dropped = int((~mask).sum())
        if dropped:
            print(f"Dropping {dropped} Salesforce rows with BUs not present in managed catalog")
        return dataframe[mask]

    def _resolve_group_key(self, product_name: Optional[str]) -> Optional[str]:
        if product_name in (None, ""):
            return None
        text = str(product_name).strip()
        if not text:
            return None

        lookup_key = text.lower()
        if lookup_key in self.reference_lookup:
            return self.reference_lookup[lookup_key]

        normalized = self.normalize_product_code(text)
        if normalized and normalized in self.normalized_reference_lookup:
            return self.normalized_reference_lookup[normalized]

        if lookup_key in self.group_key_lower_lookup:
            return self.group_key_lower_lookup[lookup_key]

        if text in self.raw_sku_lookup:
            return self.raw_sku_lookup[text]

        if normalized and normalized in self.normalized_sku_lookup:
            return self.normalized_sku_lookup[normalized]

        return None

    def _finalize_feature_frame(
        self, frame: pd.DataFrame, feature_columns: Sequence[str]
    ) -> Optional[pd.DataFrame]:
        if frame is None or frame.empty:
            return None

        trimmed = frame.copy()
        if "Date" not in trimmed.columns:
            return None
        trimmed = trimmed.dropna(subset=["Date"])
        if trimmed.empty:
            return None

        trimmed = trimmed.sort_values("Date").drop_duplicates("Date", keep="last")
        trimmed = trimmed.set_index("Date")

        start = trimmed.index.min()
        end_data = trimmed.index.max()

        current_month_raw = getattr(self.config, "CURRENT_MONTH", None)
        if current_month_raw:
            try:
                config_end = pd.to_datetime(current_month_raw)
            except Exception:
                config_end = end_data
        else:
            config_end = end_data

        end = max(end_data, config_end)

        month_index = pd.date_range(start=start, end=end, freq="MS")
        completed = trimmed.reindex(month_index, fill_value=0.0)

        completed = completed.astype(float)

        if getattr(self.config, "SALESFORCE_DIFFERENCE_FEATURES", False):
            for column in feature_columns:
                if column in completed.columns:
                    diff_col = f"{column}_Diff"
                    completed[diff_col] = completed[column].diff().fillna(0.0)

        completed = completed.reset_index().rename(columns={"index": "Date"})
        return completed

    def load_salesforce_data(self, filepath_or_data):
        """
        Load Salesforce opportunity data
        Align columns, map to managed catalog group keys, and normalise business units.
        """
        if isinstance(filepath_or_data, str):
            data_path = Path(filepath_or_data)
            self.pipeline_data = pd.read_csv(data_path)
        else:
            self.pipeline_data = filepath_or_data.copy()

        self.pipeline_data = self._standardize_columns(self.pipeline_data)

        if "Timestamp" not in self.pipeline_data.columns:
            raise ValueError("Salesforce data must include a timestamp column after standardisation.")
        if "SF_Product_Name" not in self.pipeline_data.columns:
            raise ValueError("Salesforce data must include product identifiers after standardisation.")

        self.pipeline_data["Timestamp"] = pd.to_datetime(
            self.pipeline_data["Timestamp"], errors="coerce"
        )
        self.pipeline_data = self.pipeline_data.dropna(subset=["Timestamp"])
        if self.pipeline_data.empty:
            print("Warning: Salesforce dataset is empty after parsing timestamps.")
            return self.pipeline_data

        self.pipeline_data["Timestamp"] = (
            self.pipeline_data["Timestamp"].dt.to_period("M").dt.to_timestamp()
        )
        self.pipeline_data["Date"] = self.pipeline_data["Timestamp"]
        self.pipeline_data["SF_Product_Name"] = (
            self.pipeline_data["SF_Product_Name"].astype(str).str.strip()
        )

        self.pipeline_data["group_key"] = self.pipeline_data["SF_Product_Name"].apply(
            self._resolve_group_key
        )
        missing_mask = self.pipeline_data["group_key"].isna()
        if missing_mask.any():
            missing_names = (
                self.pipeline_data.loc[missing_mask, "SF_Product_Name"]
                .dropna()
                .astype(str)
                .str.strip()
                .unique()
            )
            print(
                f"Warning: Salesforce records without a group_key mapping ({len(missing_names)} unique names). These rows will be dropped."
            )
            self.pipeline_data = self.pipeline_data[~missing_mask]

        if self.pipeline_data.empty:
            print("Warning: No Salesforce records remain after mapping product group keys.")
            return self.pipeline_data

        if "Business_Unit_Code" not in self.pipeline_data.columns:
            self.pipeline_data["Business_Unit_Code"] = self.pipeline_data.get("BU")

        self.pipeline_data["Business_Unit_Code"] = self.pipeline_data["Business_Unit_Code"].apply(
            self._normalize_business_unit
        )
        self.pipeline_data = self.pipeline_data.dropna(subset=["Business_Unit_Code"])
        if self.pipeline_data.empty:
            print("Warning: No Salesforce records remain after aligning business units.")
            return self.pipeline_data

        self.pipeline_data["Business_Unit_Code"] = (
            self.pipeline_data["Business_Unit_Code"].astype(str).str.strip()
        )
        self.pipeline_data["BU"] = self.pipeline_data["Business_Unit_Code"]

        self.pipeline_data = self._filter_valid_business_units(self.pipeline_data)
        if self.pipeline_data.empty:
            print("Warning: No Salesforce records remain after filtering invalid BU assignments.")
            return self.pipeline_data

        numeric_columns = [
            col
            for col in self.pipeline_data.columns
            if col in self.base_feature_columns
        ]
        # Include any additional numeric columns that may be useful for experiments
        additional_numeric = [
            col
            for col in self.pipeline_data.columns
            if col
            not in {
                "Timestamp",
                "Date",
                "SF_Product_Name",
                "Business_Unit_Code",
                "BU",
                "group_key",
            }
            and col not in numeric_columns
            and pd.api.types.is_numeric_dtype(self.pipeline_data[col])
        ]
        numeric_columns.extend(additional_numeric)

        for column in numeric_columns:
            self.pipeline_data[column] = pd.to_numeric(
                self.pipeline_data[column], errors="coerce"
            ).fillna(0.0)

        coverage_start = self.pipeline_data["Date"].min().strftime("%Y-%m")
        coverage_end = self.pipeline_data["Date"].max().strftime("%Y-%m")
        print(
            f"Loaded {len(self.pipeline_data)} Salesforce records covering {coverage_start} through {coverage_end}"
        )

        return self.pipeline_data

    def process_pipeline_features(self, product_groups):
        """
        Process Salesforce pipeline data into monthly features for each product group
        """
        print("Processing Salesforce pipeline features...")

        if self.pipeline_data is None or self.pipeline_data.empty:
            print('No Salesforce pipeline data available to process.')
            self.processed_features = {}
            return {}

        valid_groups = set(product_groups.keys())
        working = self.pipeline_data[self.pipeline_data["group_key"].isin(valid_groups)].copy()

        if working.empty:
            print("Warning: Salesforce data did not match any configured product groups.")
            self.processed_features = {}
            self.available_feature_columns = []
            return {}

        feature_columns = [col for col in self.base_feature_columns if col in working.columns]
        additional_numeric = [
            col
            for col in working.columns
            if col not in feature_columns + ["Date", "Timestamp", "group_key", "Business_Unit_Code", "BU", "SF_Product_Name"]
            and pd.api.types.is_numeric_dtype(working[col])
        ]
        feature_columns = list(dict.fromkeys(feature_columns + additional_numeric))

        if not feature_columns:
            print("Warning: No numeric Salesforce feature columns available after alignment.")
            self.processed_features = {}
            self.available_feature_columns = []
            return {}

        aggregated = (
            working.groupby(["group_key", "Business_Unit_Code", "Date"], as_index=False)[feature_columns]
            .sum()
            .sort_values(["group_key", "Business_Unit_Code", "Date"])
        )

        processed: Dict = {}
        observed_columns: set[str] = set()

        for group_name in product_groups.keys():
            print(f"Processing pipeline data for {group_name}")
            group_slice = aggregated[aggregated["group_key"] == group_name]
            if group_slice.empty:
                print(f"No Salesforce data found for group {group_name}.")
                continue

            if self.forecast_by_bu:
                bu_codes = (
                    group_slice["Business_Unit_Code"]
                    .dropna()
                    .astype(str)
                    .str.strip()
                    .unique()
                    .tolist()
                )
                bu_codes.sort()
                for bu_code in bu_codes:
                    bu_slice = group_slice[group_slice["Business_Unit_Code"] == bu_code]
                    feature_frame = self._finalize_feature_frame(
                        bu_slice.loc[:, ["Date"] + feature_columns], feature_columns
                    )
                    if feature_frame is None or feature_frame.empty:
                        continue
                    processed[(group_name, bu_code)] = feature_frame
                    observed_columns.update(col for col in feature_frame.columns if col != "Date")
                    readable_bu = self.bu_code_to_name.get(bu_code, bu_code)
                    print(
                        f"Prepared Salesforce feature frame for {group_name} ({readable_bu}) with {len(feature_frame)} monthly observations"
                    )
            else:
                combined_slice = (
                    group_slice.groupby("Date", as_index=False)[feature_columns].sum()
                )
                feature_frame = self._finalize_feature_frame(
                    combined_slice.loc[:, ["Date"] + feature_columns], feature_columns
                )
                if feature_frame is None or feature_frame.empty:
                    continue
                processed[group_name] = feature_frame
                observed_columns.update(col for col in feature_frame.columns if col != "Date")
                print(
                    f"Prepared Salesforce feature frame for {group_name} with {len(feature_frame)} monthly observations"
                )

        if not processed:
            print("Warning: Salesforce pipeline data did not yield any usable feature frames.")
            self.processed_features = {}
            self.available_feature_columns = []
            return {}

        self.processed_features = processed
        self.available_feature_columns = sorted(observed_columns)
        return processed


    def fit_sarima_with_exog(self, group_name, y_train, exog_train, sarima_params):
        """
        Fit SARIMA model with external regressors (Salesforce features)
        """
        print(f"Fitting SARIMAX model with external regressors for {group_name}")

        p, d, q, P, D, Q, s = sarima_params

        try:
            model = SARIMAX(
                y_train,
                exog=exog_train,
                order=(p, d, q),
                seasonal_order=(P, D, Q, s),
                trend='c',
                enforce_stationarity=False,
                enforce_invertibility=False
            )

            fitted_model = model.fit(disp=False)
            print(f"SARIMAX model fitted successfully for {group_name}")
            print(f"AIC: {fitted_model.aic:.2f}")

            # Print coefficient significance
            print("External regressor coefficients:")
            for i, col in enumerate(exog_train.columns):
                coef = fitted_model.params[f'x{i+1}']
                pvalue = fitted_model.pvalues[f'x{i+1}']
                significance = "***" if pvalue < 0.01 else "**" if pvalue < 0.05 else "*" if pvalue < 0.1 else ""
                print(f"  {col}: {coef:.4f} {significance} (p={pvalue:.3f})")

            return fitted_model

        except Exception as e:
            print(f"Error fitting SARIMAX model for {group_name}: {e}")
            return None

    def forecast_with_exog(self, model, exog_forecast, steps):
        """
        Generate forecast using external regressors
        """
        try:
            forecast_result = model.get_forecast(steps=steps, exog=exog_forecast)
            return {
                'forecast': forecast_result.predicted_mean,
                'conf_int_lower': forecast_result.conf_int().iloc[:, 0],
                'conf_int_upper': forecast_result.conf_int().iloc[:, 1]
            }
        except Exception as e:
            print(f"Error generating forecast with external regressors: {e}")
            return None

    def create_future_exog_features(self, group_name, forecast_start_date, forecast_months, bu_code=None):
        """
        Create future external regressor features for forecasting
        """
        key = (group_name, bu_code) if self.forecast_by_bu else group_name
        if key not in self.processed_features:
            return None

        features_df = self.processed_features[key].copy()
        readable_bu = self.bu_code_to_name.get(bu_code, bu_code) if bu_code else None

        if 'Date' not in features_df.columns:
            label = f"{group_name} ({readable_bu})" if readable_bu else group_name
            print(f"Processed features for {label} missing 'Date' column; cannot create future features.")
            return None

        features_df = features_df.set_index('Date').sort_index()

        forecast_index = pd.date_range(
            start=forecast_start_date,
            periods=forecast_months,
            freq='MS'
        )

        future_df = features_df.reindex(forecast_index)

        if future_df.isnull().any().any():
            future_df = future_df.fillna(0.0)
            label = f"{group_name} ({readable_bu})" if readable_bu else group_name
            print(f"Filled missing Salesforce feature values with zeros for {label} forecast horizon")

        label = f"{group_name} ({readable_bu})" if readable_bu else group_name
        print(f"Created future external features for {label} using Salesforce pipeline data")

        return future_df


    def merge_with_actuals(self, actuals_data, group_name, bu_code=None):
        """
        Merge Salesforce features with actual sales data for modeling
        """
        key = (group_name, bu_code) if self.forecast_by_bu else group_name
        label = f"{group_name} ({self.bu_code_to_name.get(bu_code, bu_code)})" if bu_code else group_name

        if key not in self.processed_features:
            print(f"No Salesforce features available for {label}")
            return actuals_data, None

        features_df = self.processed_features[key]

        # Merge on date
        merged = pd.merge(
            actuals_data.reset_index(),
            features_df,
            left_on='Month',
            right_on='Date',
            how='left'
        )

        # Ensure all expected feature columns exist and fill missing values
        feature_columns = [
            col for col in features_df.columns if col != 'Date'
        ]

        merged = merged.sort_values('Month')

        for col in feature_columns:
            merged[col] = pd.to_numeric(merged.get(col, 0.0), errors='coerce').fillna(0.0)

        merged = merged.set_index('Month')

        # Return actuals and features separately
        actuals_series = merged['Actuals']
        features_df = merged[feature_columns]

        print(f"Merged {len(features_df)} observations with Salesforce features for {label}")

        return actuals_series, features_df

# Example usage and integration instructions
class EnhancedSARIMAForecaster:
    """
    Enhanced SARIMA forecaster that can optionally use Salesforce data
    """

    @staticmethod
    def normalize_product_code(code):
        """Normalize product identifiers so Salesforce matches actuals mapping."""
        if pd.isna(code):
            return ""
        normalized = str(code).strip().upper().replace(" ", "")
        for prefix in ("ITEM_", "ITEM", "PROD_", "PROD"):
            if normalized.startswith(prefix):
                normalized = normalized[len(prefix):]
                break
        normalized = normalized.lstrip("_")
        return normalized

    def __init__(self, config=None, use_salesforce=False, salesforce_data_path=None):
        self.config = config or SARIMAConfig()
        self.use_salesforce = use_salesforce
        self.salesforce_integration = None

        if use_salesforce and salesforce_data_path:
            self.salesforce_integration = SalesforceIntegration(config)
            self.salesforce_integration.load_salesforce_data(salesforce_data_path)
            self.salesforce_integration.process_pipeline_features(self.config.PRODUCT_GROUPS)

    def fit_enhanced_model(self, group_name, actuals_data):
        """
        Fit model with or without Salesforce features based on configuration
        """
        sarima_params = self.config.SARIMA_PARAMS.get(group_name, (1, 1, 1, 1, 1, 1, 12))

        if self.use_salesforce and self.salesforce_integration:
            # Merge with Salesforce features
            y_series, exog_features = self.salesforce_integration.merge_with_actuals(
                actuals_data, group_name, None
            )

            if exog_features is not None and not exog_features.empty:
                # Split data
                split_point = len(y_series) - self.config.TEST_SPLIT_MONTHS
                y_train = y_series[:split_point]
                exog_train = exog_features[:split_point]

                # Fit SARIMAX model
                return self.salesforce_integration.fit_sarima_with_exog(
                    group_name, y_train, exog_train, sarima_params
                )

        # Fallback to standard SARIMA
        from sarima_model import SARIMAForecaster
        standard_forecaster = SARIMAForecaster(self.config)
        split_point = len(actuals_data) - self.config.TEST_SPLIT_MONTHS
        y_train = actuals_data[:split_point]

        return standard_forecaster.fit_sarima_model(group_name, y_train)
