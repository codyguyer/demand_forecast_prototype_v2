"""
SARIMA Time Series Forecasting Model
Built from scratch for product demand forecasting with configurable parameters
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')
from statistics import NormalDist

from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import adfuller
from statsmodels.stats.diagnostic import acorr_ljungbox
from statsmodels.tsa.forecasting.theta import ThetaModel
from statsmodels.tsa.holtwinters import ExponentialSmoothing

from config import SARIMAConfig
from salesforce_integration import SalesforceIntegration
from backlog_integration import BacklogIntegration

class SARIMAForecaster:
    """
    SARIMA forecasting model with configurable parameters for product demand
    """

    def __init__(self, config=None):
        self.config = config or SARIMAConfig()
        self.data = None
        self.models = {}
        self.forecasts = {}
        self.metrics = {}
        self.model_strategy = {}
        self.model_details = {}
        self.combined_data = {}
        self.series_metadata = {}
        self.salesforce_integration = None
        self.backlog_integration = None
        self.exogenous_store = None

        if self.config.USE_SALESFORCE and getattr(self.config, "SALESFORCE_DATA_FILE", None):
            try:
                self.salesforce_integration = SalesforceIntegration(self.config)
                self.salesforce_integration.load_salesforce_data(self.config.SALESFORCE_DATA_FILE)
                self.salesforce_integration.process_pipeline_features(self.config.PRODUCT_GROUPS)
                print("Salesforce pipeline features loaded for exogenous modeling")
            except Exception as exc:
                print(f"Warning: Unable to initialize Salesforce integration: {exc}")
                self.salesforce_integration = None

        if getattr(self.config, "USE_BACKLOG", False) and getattr(self.config, "BACKLOG_DATA_FILE", None):
            try:
                self.backlog_integration = BacklogIntegration(self.config)
                self.backlog_integration.load_backlog_data(self.config.BACKLOG_DATA_FILE)
                self.backlog_integration.process_backlog_features(self.config.PRODUCT_GROUPS)
                print("Backlog features loaded for exogenous modeling")
            except Exception as exc:
                print(f"Warning: Unable to initialize backlog integration: {exc}")
                self.backlog_integration = None

        self.precompute_exogenous_store()

    def _make_series_key(self, group_name, bu_code=None):
        """Create a unique key for a product group / BU combination."""
        if bu_code:
            return f"{group_name}__{bu_code}"
        return group_name

    def _format_series_label(self, group_name, bu_code=None):
        """Human friendly label for logs and plots."""
        if bu_code and getattr(self.config, "FORECAST_BY_BU", False):
            bu_name = getattr(self.config, "BU_CODE_TO_NAME", {}).get(bu_code, bu_code)
            return f"{group_name} ({bu_name})"
        return group_name

    @staticmethod
    def _safe_round(value, digits=2):
        """Safely round metric values while preserving NaN."""
        if value is None or pd.isna(value):
            return np.nan
        return round(float(value), digits)

    def _build_model_details(self, strategy, model, exog_columns=None):
        """Create a descriptor bundle for the fitted model."""
        if exog_columns is None:
            exog_columns = []
        elif isinstance(exog_columns, (pd.Index, np.ndarray)):
            exog_columns = [str(col) for col in exog_columns.tolist()]
        elif isinstance(exog_columns, (list, tuple)):
            exog_columns = [str(col) for col in exog_columns]
        else:
            exog_columns = [str(exog_columns)]

        details = {
            "model_name": strategy,
            "model_descriptor": strategy,
            "sarima_params": "N/A",
            "exogenous_used": exog_columns
        }

        if model is None:
            return details

        if strategy == 'ETS_THETA' and isinstance(model, dict):
            weights = model.get('weights', (0.5, 0.5))
            if not isinstance(weights, (list, tuple)) or len(weights) != 2:
                weights = (0.5, 0.5)
            theta_weight, ets_weight = weights
            try:
                theta_weight = float(theta_weight)
            except (TypeError, ValueError):
                theta_weight = np.nan
            try:
                ets_weight = float(ets_weight)
            except (TypeError, ValueError):
                ets_weight = np.nan
            details["model_descriptor"] = (
                f"ETS/Theta blend (theta={theta_weight:.2f}, ets={ets_weight:.2f})"
            )
            return details

        model_type = getattr(model, "_codex_model_type", None)
        order = getattr(model, "_codex_order", None)
        seasonal_order = getattr(model, "_codex_seasonal_order", None)

        if order is not None:
            order_str = f"({order[0]},{order[1]},{order[2]})"
        else:
            order_str = "N/A"

        if seasonal_order and isinstance(seasonal_order, tuple) and len(seasonal_order) == 4:
            seasonal_str = f"({seasonal_order[0]},{seasonal_order[1]},{seasonal_order[2]},{seasonal_order[3]})"
        else:
            seasonal_str = "None"

        if model_type == "SARIMA":
            descriptor = f"SARIMA{order_str}x{seasonal_str}"
            details["model_descriptor"] = descriptor
            details["sarima_params"] = descriptor
        elif model_type == "SARIMA_FALLBACK_ARIMA":
            descriptor = f"Fallback ARIMA{order_str}"
            details["model_descriptor"] = descriptor
        elif model_type == "ARIMA":
            descriptor = f"ARIMA{order_str}"
            details["model_descriptor"] = descriptor
        else:
            details["model_descriptor"] = strategy

        return details

    @staticmethod
    def _sanitize_label(label):
        """Sanitize label for filesystem use."""
        allowed = set("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_")
        return "".join(c if c in allowed else "_" for c in label)

    @staticmethod
    def _normalize_month_index(month_series):
        """Convert a heterogeneous Month column into pandas timestamps."""
        if month_series.empty:
            return pd.to_datetime(month_series)

        parsed = pd.to_datetime(month_series, errors="coerce")
        unresolved_mask = parsed.isna() & month_series.notna()
        if unresolved_mask.any():
            numeric_candidates = pd.to_numeric(
                month_series[unresolved_mask].astype(str).str.strip(),
                errors="coerce",
            )
            excel_mask = numeric_candidates.notna()
            if excel_mask.any():
                excel_dates = pd.to_datetime("1899-12-30") + pd.to_timedelta(
                    numeric_candidates[excel_mask].astype(int),
                    unit="D",
                )
                parsed.loc[numeric_candidates[excel_mask].index] = excel_dates.values

            remaining = parsed.isna() & month_series.notna()
            if remaining.any():
                samples = month_series[remaining].astype(str).unique().tolist()
                preview = ", ".join(samples[:5])
                raise ValueError(f"Unable to parse Month values: {preview}")

        return parsed

    def load_and_preprocess_data(self):
        """Load data and combine product groups according to replacement logic"""
        print("Loading and preprocessing data...")

        # Load data
        self.data = pd.read_csv(self.config.DATA_FILE)
        self.data['Month'] = self._normalize_month_index(self.data['Month'])
        self.data = self.data.sort_values(['Product', 'Month'])

        self.combined_data = {}
        self.series_metadata = {}

        forecast_by_bu = bool(getattr(self.config, "FORECAST_BY_BU", False))
        if forecast_by_bu and 'BU' not in self.data.columns:
            raise ValueError("FORECAST_BY_BU is enabled but the source data is missing the 'BU' column.")

        for group_name, products in self.config.PRODUCT_GROUPS.items():
            print(f"Combining products for {group_name}: {products}")

            group_data = self.data[self.data['Product'].isin(products)].copy()
            if group_data.empty:
                print(f"Warning: No records found for {group_name} using products {products}")
                continue

            if forecast_by_bu:
                for bu_code, bu_slice in group_data.groupby('BU'):
                    if bu_slice.empty:
                        continue
                    combined = bu_slice.groupby('Month')['Actuals'].sum().reset_index()
                    combined['Product_Group'] = group_name
                    combined['BU'] = bu_code

                    key = self._make_series_key(group_name, bu_code)
                    self.combined_data[key] = combined
                    self.series_metadata[key] = {
                        'group': group_name,
                        'bu_code': bu_code
                    }
                    print(f"  -> Added BU {bu_code} with {len(combined)} observations")
            else:
                combined = group_data.groupby('Month')['Actuals'].sum().reset_index()
                combined['Product_Group'] = group_name

                key = self._make_series_key(group_name)
                self.combined_data[key] = combined
                self.series_metadata[key] = {
                    'group': group_name,
                    'bu_code': None
                }

        print(f"Successfully prepared {len(self.combined_data)} product series")
        return self.combined_data

    def get_salesforce_feature_mode(self, group_name):
        """Determine which Salesforce feature set to use for a product group."""
        mode_config = getattr(self.config, "SALESFORCE_FEATURE_MODE_BY_GROUP", "quantity")

        if isinstance(mode_config, dict):
            if group_name in mode_config:
                return mode_config[group_name]
            return mode_config.get("default", "quantity")
        if isinstance(mode_config, str):
            return mode_config

        return "quantity"

    def filter_salesforce_features(self, features_df, group_name):
        """Filter Salesforce features based on configuration."""
        if features_df is None or features_df.empty:
            return None

        mode = self.get_salesforce_feature_mode(group_name)
        column_sets = getattr(self.config, "SALESFORCE_FEATURE_COLUMN_SETS", None)

        configured_columns = None
        if isinstance(column_sets, dict):
            if mode in column_sets:
                configured_columns = column_sets[mode]
            elif "default" in column_sets:
                configured_columns = column_sets["default"]

        if configured_columns:
            features_df = features_df.copy()
            missing = [col for col in configured_columns if col not in features_df.columns]

            if missing:
                print(f"Configured Salesforce features missing for {group_name} ({mode}): {missing}")
                for col in missing:
                    features_df[col] = 0.0

            return features_df[configured_columns]

        quantity_columns = [col for col in features_df.columns if "Quantity" in col]
        amount_columns = [col for col in features_df.columns if "Amount" in col or "Dollar" in col]

        if mode == "dollars" and amount_columns:
            return features_df[amount_columns]
        if mode == "quantity" and quantity_columns:
            return features_df[quantity_columns]

        if quantity_columns:
            return features_df[quantity_columns]
        if amount_columns:
            return features_df[amount_columns]

        return features_df

    def filter_backlog_features(self, features_df):
        """Filter backlog features based on configuration."""
        if features_df is None or features_df.empty:
            return None

        configured_columns = getattr(self.config, "BACKLOG_FEATURES", None)
        if configured_columns:
            features_df = features_df.copy()
            missing = [col for col in configured_columns if col not in features_df.columns]

            if missing:
                for col in missing:
                    features_df[col] = 0.0

            return features_df[configured_columns]

        return features_df

    @staticmethod
    def combine_exogenous_features(feature_frames, index=None):
        """Combine multiple exogenous feature dataframes."""
        valid_frames = [
            frame for frame in feature_frames
            if frame is not None and not frame.empty
        ]

        if not valid_frames:
            return None

        combined = pd.concat(valid_frames, axis=1)

        if index is not None:
            combined = combined.reindex(index, fill_value=0.0)

        return combined.fillna(0.0)

    def precompute_exogenous_store(self):
        """Build a wide exogenous feature matrix keyed by (Date, group, BU)."""
        stores = []

        def build_store(processed_dict, source_prefix):
            if not processed_dict:
                return None

            frames = []
            for key, frame in processed_dict.items():
                if frame is None or frame.empty:
                    continue

                local = frame.copy()
                if 'Date' not in local.columns:
                    local = local.reset_index()
                    date_col = 'Date'
                    if 'index' in local.columns and local['index'].dtype.kind == 'M':
                        date_col = 'index'
                    elif 'Month' in local.columns:
                        date_col = 'Month'
                    local = local.rename(columns={date_col: 'Date'})

                local['Date'] = pd.to_datetime(local['Date'])

                if isinstance(key, tuple):
                    group_name, bu_code = key
                else:
                    group_name = key
                    bu_code = ''

                if pd.isna(bu_code):
                    bu_code = ''
                else:
                    bu_code = str(bu_code).strip()

                local['Product_Group'] = group_name
                local['BU'] = bu_code
                frames.append(local)

            if not frames:
                return None

            store_df = pd.concat(frames, ignore_index=True)
            store_df['BU'] = store_df['BU'].fillna('')
            store_df = store_df.sort_values(['Date', 'Product_Group', 'BU'])

            feature_cols = [
                col for col in store_df.columns
                if col not in ('Date', 'Product_Group', 'BU')
            ]

            if not feature_cols:
                return None

            prefixed_columns = {
                col: f"{source_prefix}{col}"
                for col in feature_cols
            }

            store_df = store_df.rename(columns=prefixed_columns)
            ordered_cols = [prefixed_columns[col] for col in feature_cols]
            store_df = store_df.set_index(['Date', 'Product_Group', 'BU'])[ordered_cols]
            store_df.index.set_names(['Date', 'Product_Group', 'BU'], inplace=True)
            return store_df.astype(float)

        if self.salesforce_integration and getattr(self.salesforce_integration, "processed_features", None):
            sf_store = build_store(self.salesforce_integration.processed_features, "sf_")
            if sf_store is not None:
                stores.append(sf_store)

        if self.backlog_integration and getattr(self.backlog_integration, "processed_features", None):
            backlog_store = build_store(self.backlog_integration.processed_features, "backlog_")
            if backlog_store is not None:
                stores.append(backlog_store)

        if stores:
            combined_store = pd.concat(stores, axis=1).fillna(0.0)
            combined_store = combined_store.sort_index()
            combined_store.index.set_names(['Date', 'Product_Group', 'BU'], inplace=True)
            self.exogenous_store = combined_store
        else:
            self.exogenous_store = None

    def get_exog_features_for_series(self, group_name, bu_code, target_index):
        """Retrieve precomputed exogenous features for a given series and index."""
        if self.exogenous_store is None:
            return None

        if pd.isna(bu_code):
            bu_key = ''
        else:
            bu_key = str(bu_code).strip()

        try:
            series_matrix = self.exogenous_store.xs(
                (group_name, bu_key),
                level=('Product_Group', 'BU')
            )
        except KeyError:
            return None

        series_matrix = series_matrix.sort_index()

        frames = []

        def extract_and_filter(prefix, filter_fn, needs_group=False):
            cols = [col for col in series_matrix.columns if col.startswith(prefix)]
            if not cols:
                return None
            subset = series_matrix[cols].copy()
            subset.columns = [col[len(prefix):] for col in cols]
            if needs_group:
                filtered = filter_fn(subset, group_name)
            else:
                filtered = filter_fn(subset)

            if filtered is None or filtered.empty:
                return None

            if (
                needs_group
                and self.salesforce_integration
                and target_index is not None
                and len(target_index) > 0
            ):
                requested_start = target_index[0]
                last_available = filtered.index.max()
                if last_available is None or requested_start > last_available:
                    forecast_months = len(target_index)
                    future_df = self.salesforce_integration.create_future_exog_features(
                        group_name,
                        requested_start,
                        forecast_months,
                        bu_code,
                    )
                    if future_df is None:
                        return None

                    future_subset = future_df.reindex(target_index)
                    future_subset = future_subset[filtered.columns.intersection(future_subset.columns)]
                    future_subset = future_subset.fillna(method="ffill").fillna(method="bfill")
                    combined_future = pd.concat([filtered, future_subset], axis=0)
                    filtered = combined_future.loc[~combined_future.index.duplicated(keep="last")]

            filtered.columns = [f"{prefix}{col}" for col in filtered.columns]
            return filtered

        if self.salesforce_integration:
            sf_filtered = extract_and_filter("sf_", self.filter_salesforce_features, needs_group=True)
            if sf_filtered is not None:
                frames.append(sf_filtered)

        if self.backlog_integration:
            backlog_filtered = extract_and_filter("backlog_", self.filter_backlog_features, needs_group=False)
            if backlog_filtered is not None:
                frames.append(backlog_filtered)

        if not frames:
            return None

        combined = pd.concat(frames, axis=1).fillna(0.0)

        if target_index is not None:
            combined = combined.reindex(target_index, fill_value=0.0)
            if combined.empty or np.isclose(combined.abs().sum().sum(), 0.0):
                return None

        combined = combined.loc[:, sorted(combined.columns)]

        return combined.astype(float)

    def analyze_seasonality(self, series, group_name, bu_code=None):
        """Analyze seasonality patterns in the data"""
        label = self._format_series_label(group_name, bu_code)
        print(f"Analyzing seasonality for {label}...")

        if len(series) < self.config.MIN_OBSERVATIONS:
            print(f"Warning: {label} has insufficient data for reliable seasonal analysis")
            return False

        try:
            # Perform seasonal decomposition
            decomposition = seasonal_decompose(series, model='additive', period=12)

            # Calculate seasonal strength
            seasonal_var = np.var(decomposition.seasonal.dropna())
            residual_var = np.var(decomposition.resid.dropna())

            if residual_var > 0:
                seasonal_strength = seasonal_var / (seasonal_var + residual_var)
                print(f"Seasonal strength for {label}: {seasonal_strength:.3f}")

                return seasonal_strength > self.config.SEASONALITY_THRESHOLD

        except Exception as e:
            print(f"Error in seasonal analysis for {label}: {e}")

        return False

    def check_stationarity(self, series, group_name, bu_code=None):
        """Check if series is stationary using Augmented Dickey-Fuller test"""
        cleaned = series.dropna()
        label = self._format_series_label(group_name, bu_code)

        if cleaned.nunique() <= 1:
            print(f"Stationarity test for {label}: constant series, treating as stationary")
            return True

        if len(cleaned) < 3:
            print(f"Stationarity test for {label}: insufficient data, treating as stationary")
            return True

        try:
            result = adfuller(cleaned)
        except ValueError as exc:
            print(f"Warning: ADF failed for {label}: {exc}. Treating as stationary")
            return True

        p_value = result[1]
        print(f"Stationarity test for {label}: p-value = {p_value:.4f}")
        is_stationary = p_value < 0.05
        print(f"{label} is {'stationary' if is_stationary else 'non-stationary'}")

        return is_stationary

    def split_data(self, series):
        """Split data into train and test sets"""
        split_point = max(len(series) - self.config.TEST_SPLIT_MONTHS, 0)
        train = series.iloc[:split_point]
        test = series.iloc[split_point:]
        return train, test

    def refit_model_on_full_history(self, strategy, group_name, full_series, exog_features=None, label=None):
        """Refit the selected model using the complete history before forecasting."""
        display_name = label or self._format_series_label(group_name)
        try:
            if strategy == 'SARIMA':
                return self.fit_sarima_model(group_name, full_series, exog_features, label)
            if strategy == 'ARIMA':
                return self.fit_arima_model(group_name, full_series, exog_features, label)
            if strategy == 'ETS_THETA':
                return self.fit_theta_ets_model(group_name, full_series, label)
        except Exception as exc:
            print(f"Warning: Refit on full history failed for {display_name}: {exc}")
        return None

    def fit_sarima_model(self, group_name, train_data, exog_train=None, label=None):
        """Fit SARIMA model for a product group"""
        display_name = label or group_name
        print(f"Fitting SARIMA model for {display_name}...")

        # Get SARIMA parameters for this group
        params = self.config.SARIMA_PARAMS.get(group_name, (1, 1, 1, 1, 1, 1, 12))
        p, d, q, P, D, Q, s = params

        print(f"Using parameters: SARIMA({p},{d},{q})A-({P},{D},{Q},{s})")

        if exog_train is not None and exog_train.empty:
            exog_train = None

        if exog_train is not None:
            exog_train = exog_train.astype(float)

        try:
            model = SARIMAX(
                train_data,
                exog=exog_train,
                order=(p, d, q),
                seasonal_order=(P, D, Q, s),
                trend='c',
                enforce_stationarity=False,
                enforce_invertibility=True
            )

            fitted_model = model.fit(disp=False)
            print(f"Model fitted successfully for {display_name}")
            print(f"AIC: {fitted_model.aic:.2f}")
            setattr(fitted_model, "_codex_model_type", "SARIMA")
            setattr(fitted_model, "_codex_order", (p, d, q))
            setattr(fitted_model, "_codex_seasonal_order", (P, D, Q, s))

            return fitted_model

        except Exception as e:
            print(f"Error fitting model for {display_name}: {e}")
            print("Trying simpler ARIMA model...")
            try:
                fallback_model = SARIMAX(
                    train_data,
                    exog=exog_train,
                    order=(1, 1, 1),
                    trend='c',
                    enforce_stationarity=False,
                    enforce_invertibility=True
                )
                fitted_model = fallback_model.fit(disp=False)
                print(f"Fallback model fitted for {display_name}")
                setattr(fitted_model, "_codex_model_type", "SARIMA_FALLBACK_ARIMA")
                setattr(fitted_model, "_codex_order", (1, 1, 1))
                setattr(fitted_model, "_codex_seasonal_order", None)
                return fitted_model
            except Exception as e2:
                print(f"Fallback model also failed for {display_name}: {e2}")
                return None

    def should_use_short_series_arima(self, series, group_name):
        """Decide if the medium-history ARIMA fallback should be used."""
        if not getattr(self.config, "SHORT_SERIES_ARIMA_ENABLED", False):
            return False

        series_len = len(series)
        sarima_min = getattr(self.config, "SARIMA_MIN_HISTORY", None)
        arima_min = getattr(self.config, "ARIMA_MIN_HISTORY", getattr(self.config, "SHORT_SERIES_THRESHOLD", 24))

        if sarima_min is not None and series_len >= sarima_min:
            return False
        if series_len < arima_min:
            return False

        forced = getattr(self.config, "SHORT_SERIES_FORCE_GROUPS", None)
        if forced and group_name in forced:
            return True

        return True

    def should_use_theta_ets(self, series, group_name):
        """Decide if the very short-history ETS/Theta blend should be used."""
        if not getattr(self.config, "ETS_THETA_ENABLED", False):
            return False

        series_len = len(series)
        arima_min = getattr(self.config, "ARIMA_MIN_HISTORY", getattr(self.config, "SHORT_SERIES_THRESHOLD", 24))

        if series_len >= arima_min:
            return False

        forced = getattr(self.config, "ETS_THETA_FORCE_GROUPS", None)
        if forced and group_name in forced:
            return True

        min_obs = getattr(self.config, "MIN_OBSERVATIONS", 12)
        return series_len >= min_obs

    def determine_model_strategy(self, series, group_name):
        """Select which forecasting approach to use for a product group."""
        series_len = len(series)
        sarima_min = getattr(self.config, "SARIMA_MIN_HISTORY", None)
        arima_min = getattr(self.config, "ARIMA_MIN_HISTORY", getattr(self.config, "SHORT_SERIES_THRESHOLD", 24))

        if sarima_min is not None and series_len >= sarima_min:
            return 'SARIMA'

        if self.should_use_short_series_arima(series, group_name):
            return 'ARIMA'

        if self.should_use_theta_ets(series, group_name):
            return 'ETS_THETA'

        if sarima_min is None or series_len >= sarima_min:
            return 'SARIMA'

        if getattr(self.config, "SHORT_SERIES_ARIMA_ENABLED", False) and series_len >= arima_min:
            return 'ARIMA'

        if getattr(self.config, "ETS_THETA_ENABLED", False):
            return 'ETS_THETA'

        return 'SARIMA'

    def get_short_series_arima_order(self, group_name):
        """Retrieve the ARIMA(p,d,q) order for a short-history series."""
        params = getattr(self.config, "SHORT_SERIES_ARIMA_PARAMS", None) or {}
        if isinstance(params, dict) and group_name in params:
            return params[group_name]

        return getattr(self.config, "SHORT_SERIES_ARIMA_DEFAULT_ORDER", (1, 1, 1))

    def fit_theta_ets_model(self, group_name, train_data, label=None):
        """Fit an ETS/Theta blend model for very short history groups."""
        display_name = label or group_name
        weights = getattr(self.config, "ETS_THETA_WEIGHTS", (0.5, 0.5))
        if (not isinstance(weights, (list, tuple))) or len(weights) != 2:
            weights = (0.5, 0.5)
        theta_weight, ets_weight = weights
        total_weight = theta_weight + ets_weight
        if total_weight == 0:
            theta_weight = ets_weight = 0.5
            total_weight = 1.0
        theta_weight /= total_weight
        ets_weight /= total_weight

        theta_period_cfg = getattr(self.config, "ETS_THETA_PERIOD", 12) or 12
        theta_period = max(2, min(int(theta_period_cfg), max(2, len(train_data))))

        theta_result = None
        try:
            theta_model = ThetaModel(train_data, period=theta_period)
            theta_result = theta_model.fit()
        except Exception as exc:
            print(f"Warning: ThetaModel failed for {display_name}: {exc}")

        ets_result = None
        trend = getattr(self.config, "ETS_THETA_TREND", 'add')
        if trend in (None, 'none', 'None'):
            trend = None
        seasonal = getattr(self.config, "ETS_THETA_SEASONAL", None)
        if seasonal in (None, 'none', 'None'):
            seasonal = None
        seasonal_periods = getattr(self.config, "ETS_THETA_SEASONAL_PERIODS", None)
        if seasonal is not None and seasonal_periods is None:
            seasonal_periods = theta_period
        if seasonal_periods is not None and seasonal_periods > len(train_data):
            print(f"Warning: Not enough data for seasonal ETS component on {display_name}; disabling seasonality.")
            seasonal = None
            seasonal_periods = None
        damped = bool(getattr(self.config, "ETS_THETA_DAMPED", False))

        try:
            ets_model = ExponentialSmoothing(
                train_data,
                trend=trend,
                damped_trend=damped if trend is not None else False,
                seasonal=seasonal,
                seasonal_periods=seasonal_periods,
                initialization_method='estimated'
            )
            ets_result = ets_model.fit(optimized=True)
        except Exception as exc:
            print(f"Warning: ETS component failed for {display_name}: {exc}")

        if theta_result is None and ets_result is None:
            print(f"Error: Unable to fit ETS/Theta blend for {display_name}.")
            return None

        if theta_result is None:
            theta_weight, ets_weight = 0.0, 1.0
        elif ets_result is None:
            theta_weight, ets_weight = 1.0, 0.0

        resid_std = None
        if ets_result is not None and hasattr(ets_result, 'resid'):
            import numpy as np
            resid = np.asarray(ets_result.resid)
            resid = resid[~np.isnan(resid)]
            if resid.size > 1:
                resid_std = float(resid.std(ddof=1))

        return {
            'theta': theta_result,
            'ets': ets_result,
            'weights': (theta_weight, ets_weight),
            'resid_std': resid_std
        }

    def fit_arima_model(self, group_name, train_data, exog_train=None, label=None):
        """Fit a non-seasonal ARIMA model for short-history product groups."""
        order = self.get_short_series_arima_order(group_name)
        display_name = label or group_name
        print(f"Using short-history ARIMA{order} for {display_name} (no seasonal component)")

        if exog_train is not None and exog_train.empty:
            exog_train = None

        if exog_train is not None:
            exog_train = exog_train.astype(float)

        try:
            model = SARIMAX(
                train_data,
                exog=exog_train,
                order=order,
                seasonal_order=(0, 0, 0, 0),
                trend='c',
                enforce_stationarity=False,
                enforce_invertibility=True
            )

            fitted_model = model.fit(disp=False)
            print(f"Short-history ARIMA fitted successfully for {display_name}")
            print(f"AIC: {fitted_model.aic:.2f}")
            setattr(fitted_model, "_codex_model_type", "ARIMA")
            setattr(fitted_model, "_codex_order", order)
            setattr(fitted_model, "_codex_seasonal_order", None)
            return fitted_model

        except Exception as exc:
            print(f"Error fitting ARIMA fallback for {display_name}: {exc}")
            return None

    def calculate_metrics(self, actual, predicted, group_name):
        """Calculate forecast accuracy metrics for summary and exports."""
        # Remove any NaN values
        mask = ~(np.isnan(actual) | np.isnan(predicted))
        actual_clean = actual[mask]
        predicted_clean = predicted[mask]

        if len(actual_clean) == 0:
            return {
                "bias_percent": np.nan,
                "mae_percent": np.nan,
                "rmse": np.nan,
                "mape": np.nan,
                "cov": np.nan
            }

        # Bias percentage
        mean_actual = np.mean(actual_clean)
        if mean_actual != 0:
            bias_percent = (np.mean(predicted_clean - actual_clean) / mean_actual) * 100
        else:
            bias_percent = np.nan

        # MAE% defined as total absolute error divided by total actual demand
        total_actual = np.sum(np.abs(actual_clean))
        if total_actual == 0:
            mae_percent = np.nan
        else:
            total_abs_error = np.sum(np.abs(predicted_clean - actual_clean))
            mae_percent = (total_abs_error / total_actual) * 100

        # Root Mean Squared Error
        rmse = np.sqrt(np.mean((predicted_clean - actual_clean) ** 2))

        # Mean Absolute Percentage Error (ignore zero actuals)
        nonzero_mask = actual_clean != 0
        if np.any(nonzero_mask):
            mape = np.mean(
                np.abs((predicted_clean[nonzero_mask] - actual_clean[nonzero_mask]) / actual_clean[nonzero_mask])
            ) * 100
        else:
            mape = np.nan

        # Coefficient of Variation for actual demand
        if mean_actual != 0:
            cov = (np.std(actual_clean) / np.abs(mean_actual)) * 100
        else:
            cov = np.nan

        print(
            f"{group_name} - MAE%: {mae_percent:.2f}%, Bias%: {bias_percent:.2f}%, "
            f"RMSE: {rmse:.2f}, MAPE: {mape:.2f}%, CoV: {cov:.2f}%"
        )

        return {
            "bias_percent": bias_percent,
            "mae_percent": mae_percent,
            "rmse": rmse,
            "mape": mape,
            "cov": cov
        }

    def forecast_theta_ets(self, model_bundle, steps, start_index, group_name, bu_code=None, label=None):
        """Generate forecasts and confidence intervals for the ETS/Theta blend."""
        if model_bundle is None or steps <= 0:
            return None

        display_name = label or self._format_series_label(group_name, bu_code)

        forecast_index = pd.date_range(
            start=start_index + pd.DateOffset(months=1),
            periods=steps,
            freq='MS'
        )

        theta_res = model_bundle.get('theta')
        ets_res = model_bundle.get('ets')
        weight_theta, weight_ets = model_bundle.get('weights', (0.5, 0.5))

        if theta_res is not None:
            theta_forecast = theta_res.forecast(steps)
            theta_forecast = pd.Series(theta_forecast.values, index=forecast_index)
        else:
            theta_forecast = pd.Series(np.zeros(steps), index=forecast_index, dtype=float)

        if ets_res is not None:
            ets_forecast = ets_res.forecast(steps)
            ets_forecast = pd.Series(ets_forecast.values, index=forecast_index)
        else:
            ets_forecast = pd.Series(np.zeros(steps), index=forecast_index, dtype=float)

        blended = weight_theta * theta_forecast + weight_ets * ets_forecast

        lower = upper = None
        alpha = 1 - getattr(self.config, 'CONFIDENCE_LEVEL', 0.9)
        if theta_res is not None and hasattr(theta_res, 'prediction_intervals'):
            try:
                theta_intervals = theta_res.prediction_intervals(steps=steps, alpha=alpha)
                theta_lower = pd.Series(theta_intervals['lower'].values, index=forecast_index)
                theta_upper = pd.Series(theta_intervals['upper'].values, index=forecast_index)
                adjustment = blended - theta_forecast
                lower = theta_lower + adjustment
                upper = theta_upper + adjustment
            except Exception as exc:
                print(f"Warning: Theta intervals unavailable for {display_name}: {exc}")

        if lower is None or upper is None:
            resid_std = model_bundle.get('resid_std')
            if resid_std is not None and resid_std > 0:
                z_score = NormalDist().inv_cdf(0.5 + getattr(self.config, 'CONFIDENCE_LEVEL', 0.9) / 2)
                delta = z_score * resid_std
                lower = blended - delta
                upper = blended + delta
            else:
                lower = blended.copy()
                upper = blended.copy()

        if getattr(self.config, 'ENFORCE_NON_NEGATIVE_FORECASTS', False):
            blended = blended.clip(lower=0)
            lower = lower.clip(lower=0)
            upper = upper.clip(lower=0)

        return {
            'forecast': blended,
            'conf_int_lower': lower,
            'conf_int_upper': upper
        }

    def generate_forecast(self, model, steps, group_name, exog_future=None):
        """Generate forecast with confidence intervals"""
        print(f"Generating {steps}-month forecast for {group_name}...")

        try:
            forecast_result = model.get_forecast(steps=steps, exog=exog_future)
            forecast_mean = forecast_result.predicted_mean

            conf_int = forecast_result.conf_int(alpha=1 - self.config.CONFIDENCE_LEVEL).copy()

            if getattr(self.config, 'ENFORCE_NON_NEGATIVE_FORECASTS', False):
                forecast_mean = forecast_mean.clip(lower=0)
                conf_int.iloc[:, 0] = conf_int.iloc[:, 0].clip(lower=0)
                conf_int.iloc[:, 1] = conf_int.iloc[:, 1].clip(lower=0)

            return {
                'forecast': forecast_mean,
                'conf_int_lower': conf_int.iloc[:, 0],
                'conf_int_upper': conf_int.iloc[:, 1]
            }
        except Exception as e:
            print(f"Error generating forecast for {group_name}: {e}")
            return None

    def run_forecasting(self):
        """Main method to run the complete forecasting pipeline"""
        print("Starting SARIMA forecasting pipeline...")
        print("=" * 50)

        # Load and preprocess data
        self.load_and_preprocess_data()
        # Ensure exogenous store aligns with latest configuration
        self.precompute_exogenous_store()

        results_summary = []

        # Process each product group
        for series_key, group_data in self.combined_data.items():
            metadata = self.series_metadata.get(series_key, {})
            group_name = metadata.get('group', series_key)
            bu_code = metadata.get('bu_code')
            label = self._format_series_label(group_name, bu_code)
            exog_columns_used = []

            print(f"\nProcessing {label}...")
            print("-" * 30)

            # Prepare time series
            ts_data = group_data.set_index('Month')['Actuals']
            ts_data = ts_data.asfreq('MS')  # Monthly start frequency

            current_date = pd.to_datetime(self.config.CURRENT_MONTH)
            ts_data = ts_data[ts_data.index <= current_date]
            if getattr(self.config, 'TRIM_TRAILING_ZERO_PADDING', True):
                # Trim only trailing zeros that look like future padding, preserve internal zeros
                nonzero_idx = np.where(ts_data.to_numpy() != 0)[0]
                if nonzero_idx.size:
                    ts_data = ts_data.iloc[: nonzero_idx[-1] + 1]

            if len(ts_data) < 12:
                print(f"Warning: {label} has insufficient data. Skipping...")
                continue

            strategy = self.determine_model_strategy(ts_data, group_name)
            print(f"Selected {strategy} strategy for {label} (history={len(ts_data)} months)")

            forecast_start = ts_data.index[-1] + pd.DateOffset(months=1)

            exog_features = None
            exog_future = None

            if strategy != 'ETS_THETA':
                if self.exogenous_store is None:
                    print("No precomputed exogenous features available; proceeding without exogenous inputs.")
                else:
                    exog_features = self.get_exog_features_for_series(group_name, bu_code, ts_data.index)
                    if exog_features is None or exog_features.empty:
                        print(f"No exogenous features found for {label}; proceeding without exogenous inputs.")
                        exog_features = None
                    else:
                        future_index = pd.date_range(
                            start=forecast_start,
                            periods=self.config.FORECAST_MONTHS,
                            freq='MS'
                        )
                        exog_future = self.get_exog_features_for_series(group_name, bu_code, future_index)

                        if exog_future is None or exog_future.empty:
                            print(f"Exogenous features missing forecast horizon coverage for {label}; proceeding without exogenous inputs.")
                            exog_features = None
                            exog_future = None
                            exog_columns_used = []
                        else:
                            missing_cols = [col for col in exog_features.columns if col not in exog_future.columns]
                            for col in missing_cols:
                                exog_future[col] = 0.0
                            exog_future = exog_future[exog_features.columns]
                            exog_features = exog_features.astype(float)
                            exog_future = exog_future.astype(float)
                            exog_columns_used = list(exog_features.columns)

            ts_data = ts_data.astype(float)

            if strategy == 'SARIMA':
                # Analyze seasonality
                self.analyze_seasonality(ts_data, group_name, bu_code)

                # Check stationarity
                self.check_stationarity(ts_data, group_name, bu_code)
            else:
                print('Skipping detailed seasonality diagnostics for non-SARIMA strategy.')

            exog_train = None
            exog_test = None

            if strategy != 'ETS_THETA' and exog_features is not None:
                split_point = len(ts_data) - self.config.TEST_SPLIT_MONTHS
                if split_point <= 0:
                    print(f"Warning: Not enough observations after holdout for {label}. Skipping...")
                    continue

                train_data = ts_data.iloc[:split_point]
                test_data = ts_data.iloc[split_point:]

                exog_train = exog_features.iloc[:split_point]
                exog_test = exog_features.iloc[split_point:]
                exog_columns_used = list(exog_train.columns)

                if exog_future is not None:
                    missing_cols = [col for col in exog_train.columns if col not in exog_future.columns]
                    for col in missing_cols:
                        exog_future[col] = 0.0
                    exog_future = exog_future[exog_train.columns]
                else:
                    exog_train = None
                    exog_test = None
                    exog_columns_used = []
            else:
                train_data, test_data = self.split_data(ts_data)

            if len(train_data) == 0:
                print(f"Warning: {label} has no training data after split. Skipping...")
                continue

            if strategy == 'SARIMA':
                model = self.fit_sarima_model(group_name, train_data, exog_train, label)
            elif strategy == 'ARIMA':
                model = self.fit_arima_model(group_name, train_data, exog_train, label)
            else:
                model = self.fit_theta_ets_model(group_name, train_data, label)

            if model is None:
                print(f'Unable to fit model for {label}; skipping forecast.')
                continue

            self.model_strategy[series_key] = strategy
            model_for_forecast = model
            refit_needed = len(ts_data) > len(train_data)

            if len(test_data) > 0:
                if strategy == 'ETS_THETA':
                    test_result = self.forecast_theta_ets(model, len(test_data), train_data.index[-1], group_name, bu_code, label)
                    if test_result is None:
                        print(f"Unable to generate test forecast for {label}.")
                        continue
                    test_predictions = test_result['forecast']
                elif exog_test is not None and not exog_test.empty:
                    test_forecast = model.get_forecast(
                        steps=len(test_data),
                        exog=exog_test.fillna(0.0)
                    )
                    test_predictions = test_forecast.predicted_mean
                else:
                    test_forecast = model.get_forecast(steps=len(test_data))
                    test_predictions = test_forecast.predicted_mean

                self.metrics[series_key] = self.calculate_metrics(
                    test_data.values,
                    test_predictions.values if hasattr(test_predictions, 'values') else np.asarray(test_predictions),
                    label
                )
                if refit_needed:
                    print(f"Refitting {strategy} model on full history for {label} prior to final forecast...")
                    refit_model = self.refit_model_on_full_history(
                        strategy,
                        group_name,
                        ts_data,
                        exog_features,
                        label
                    )
                    if refit_model is not None:
                        model_for_forecast = refit_model
                    else:
                        print(f"Warning: Using holdout-trained model for {label} forecast; refit was unsuccessful.")
            elif refit_needed:
                refit_model = self.refit_model_on_full_history(
                    strategy,
                    group_name,
                    ts_data,
                    exog_features,
                    label
                )
                if refit_model is not None:
                    model_for_forecast = refit_model

            self.models[series_key] = model_for_forecast
            self.model_details[series_key] = self._build_model_details(strategy, model_for_forecast, exog_columns_used)

            if strategy == 'ETS_THETA':
                forecast = self.forecast_theta_ets(model_for_forecast, self.config.FORECAST_MONTHS, ts_data.index[-1], group_name, bu_code, label)
            else:
                forecast = self.generate_forecast(
                    model_for_forecast,
                    self.config.FORECAST_MONTHS,
                    label,
                    exog_future
                )

            if forecast:
                self.forecasts[series_key] = forecast

                forecast_dates = pd.date_range(
                    start=forecast_start,
                    periods=self.config.FORECAST_MONTHS,
                    freq='MS'
                )

                metrics = self.metrics.get(series_key, {})
                mae_percent = self._safe_round(metrics.get('mae_percent'))
                bias_percent = self._safe_round(metrics.get('bias_percent'))
                rmse = self._safe_round(metrics.get('rmse'))
                mape = self._safe_round(metrics.get('mape'))
                cov = self._safe_round(metrics.get('cov'))
                detail_bundle = self.model_details.get(series_key, {})
                model_used = detail_bundle.get('model_descriptor') or self.model_strategy.get(series_key, 'Unknown')
                sarima_params = detail_bundle.get('sarima_params') or 'N/A'
                exog_used = detail_bundle.get('exogenous_used') or []
                if isinstance(exog_used, (pd.Index, np.ndarray)):
                    exog_used_iter = [str(col) for col in exog_used.tolist()]
                elif isinstance(exog_used, (list, tuple, set)):
                    exog_used_iter = [str(col) for col in exog_used]
                else:
                    exog_used_iter = [str(exog_used)] if exog_used else []
                exog_used_iter = [item for item in (val.strip() for val in exog_used_iter) if item]
                exog_used_str = ", ".join(dict.fromkeys(exog_used_iter)) if exog_used_iter else "None"

                for i, date in enumerate(forecast_dates):
                    results_summary.append({
                        'Product_Group': group_name,
                        'BU': bu_code if bu_code else '',
                        'BU_Name': getattr(self.config, "BU_CODE_TO_NAME", {}).get(bu_code, '') if bu_code else '',
                        'Model_Type': self.model_strategy.get(series_key, 'SARIMA'),
                        'Model_Used': model_used,
                        'SARIMA_Params': sarima_params,
                        'Exogenous_Used': exog_used_str,
                        'Date': date.strftime('%Y-%m'),
                        'Forecast': round(forecast['forecast'].iloc[i], 0),
                        'Lower_CI': round(forecast['conf_int_lower'].iloc[i], 0),
                        'Upper_CI': round(forecast['conf_int_upper'].iloc[i], 0),
                        'MAE%': mae_percent,
                        'Bias%': bias_percent,
                        'RMSE': rmse,
                        'MAPE': mape,
                        'CoV': cov
                    })
        if self.config.SAVE_RESULTS and results_summary:
            results_df = pd.DataFrame(results_summary)
            output_path = Path(self.config.RESULTS_FILE)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            results_df.to_csv(output_path, index=False)
            print(f"\nResults saved to {output_path}")

        self.print_summary()

        return results_summary

    def print_summary(self):
        """Print summary of forecasting results"""
        print("\n" + "="*50)
        print("FORECASTING SUMMARY")
        print("="*50)

        print(f"Models fitted: {len(self.models)}")
        print(f"Forecasts generated: {len(self.forecasts)}")

        def label_for(key):
            meta = self.series_metadata.get(key, {})
            group = meta.get('group', key)
            bu_code = meta.get('bu_code')
            return self._format_series_label(group, bu_code)

        if self.metrics:
            print("\nModel Performance Metrics:")
            print("-" * 25)

            def format_metric(value, digits=2, suffix='%'):
                if value is None or pd.isna(value):
                    return "N/A"
                return f"{float(value):.{digits}f}{suffix}"

            for key, metrics in self.metrics.items():
                label = label_for(key)
                print(
                    f"{label}: "
                    f"MAE%={format_metric(metrics.get('mae_percent'))}, "
                    f"Bias%={format_metric(metrics.get('bias_percent'))}, "
                    f"RMSE={format_metric(metrics.get('rmse'), suffix='')}, "
                    f"MAPE={format_metric(metrics.get('mape'))}, "
                    f"CoV={format_metric(metrics.get('cov'))}"
                )

        if self.forecasts:
            print("\nNext 12-month forecasts generated for:")
            for key in self.forecasts.keys():
                label = label_for(key)
                detail = self.model_details.get(key, {})
                model_used = detail.get('model_descriptor') or self.model_strategy.get(key, 'SARIMA')
                print(f"  - {label} ({model_used})")


        print("="*50)
