"""
SARIMA Time Series Forecasting Model
Built from scratch for product demand forecasting with configurable parameters
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')
from statistics import NormalDist

from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import adfuller
from statsmodels.stats.diagnostic import acorr_ljungbox
from statsmodels.tsa.forecasting.theta import ThetaModel
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from sklearn.metrics import mean_absolute_error
import os

from config import SARIMAConfig
from salesforce_integration import SalesforceIntegration

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
        self.combined_data = {}
        self.salesforce_integration = None

        if self.config.USE_SALESFORCE and getattr(self.config, "SALESFORCE_DATA_FILE", None):
            try:
                self.salesforce_integration = SalesforceIntegration(self.config)
                self.salesforce_integration.load_salesforce_data(self.config.SALESFORCE_DATA_FILE)
                self.salesforce_integration.process_pipeline_features(self.config.PRODUCT_GROUPS)
                print("Salesforce pipeline features loaded for exogenous modeling")
            except Exception as exc:
                print(f"Warning: Unable to initialize Salesforce integration: {exc}")
                self.salesforce_integration = None

        # Create output directory if needed
        if self.config.OUTPUT_PLOTS and not os.path.exists(self.config.PLOTS_DIR):
            os.makedirs(self.config.PLOTS_DIR)

    def load_and_preprocess_data(self):
        """Load data and combine product groups according to replacement logic"""
        print("Loading and preprocessing data...")

        # Load data
        self.data = pd.read_csv(self.config.DATA_FILE)
        self.data['Month'] = pd.to_datetime(self.data['Month'])
        self.data = self.data.sort_values(['Product', 'Month'])

        # Combine product groups
        for group_name, products in self.config.PRODUCT_GROUPS.items():
            print(f"Combining products for {group_name}: {products}")

            # Get data for all products in this group
            group_data = self.data[self.data['Product'].isin(products)].copy()

            # Aggregate by month (sum actuals across products)
            combined = group_data.groupby('Month')['Actuals'].sum().reset_index()
            combined['Product_Group'] = group_name

            # Store combined data
            self.combined_data[group_name] = combined

        print(f"Successfully combined {len(self.combined_data)} product groups")
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

    def analyze_seasonality(self, series, group_name):
        """Analyze seasonality patterns in the data"""
        print(f"Analyzing seasonality for {group_name}...")

        if len(series) < self.config.MIN_OBSERVATIONS:
            print(f"Warning: {group_name} has insufficient data for reliable seasonal analysis")
            return False

        try:
            # Perform seasonal decomposition
            decomposition = seasonal_decompose(series, model='additive', period=12)

            # Calculate seasonal strength
            seasonal_var = np.var(decomposition.seasonal.dropna())
            residual_var = np.var(decomposition.resid.dropna())

            if residual_var > 0:
                seasonal_strength = seasonal_var / (seasonal_var + residual_var)
                print(f"Seasonal strength for {group_name}: {seasonal_strength:.3f}")

                # Plot decomposition if enabled
                if self.config.OUTPUT_PLOTS:
                    fig, axes = plt.subplots(4, 1, figsize=(12, 10))
                    decomposition.observed.plot(ax=axes[0], title=f'{group_name} - Original')
                    decomposition.trend.plot(ax=axes[1], title='Trend')
                    decomposition.seasonal.plot(ax=axes[2], title='Seasonal')
                    decomposition.resid.plot(ax=axes[3], title='Residual')
                    plt.tight_layout()
                    plt.savefig(f'{self.config.PLOTS_DIR}/{group_name}_decomposition.png')
                    plt.close()

                return seasonal_strength > self.config.SEASONALITY_THRESHOLD

        except Exception as e:
            print(f"Error in seasonal analysis for {group_name}: {e}")

        return False

    def check_stationarity(self, series, group_name):
        """Check if series is stationary using Augmented Dickey-Fuller test"""
        result = adfuller(series.dropna())
        p_value = result[1]

        print(f"Stationarity test for {group_name}: p-value = {p_value:.4f}")
        is_stationary = p_value < 0.05
        print(f"{group_name} is {'stationary' if is_stationary else 'non-stationary'}")

        return is_stationary

    def split_data(self, series):
        """Split data into train and test sets"""
        split_point = max(len(series) - self.config.TEST_SPLIT_MONTHS, 0)
        train = series.iloc[:split_point]
        test = series.iloc[split_point:]
        return train, test

    def fit_sarima_model(self, group_name, train_data, exog_train=None):
        """Fit SARIMA model for a product group"""
        print(f"Fitting SARIMA model for {group_name}...")

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
                enforce_stationarity=False,
                enforce_invertibility=False
            )

            fitted_model = model.fit(disp=False)
            print(f"Model fitted successfully for {group_name}")
            print(f"AIC: {fitted_model.aic:.2f}")

            return fitted_model

        except Exception as e:
            print(f"Error fitting model for {group_name}: {e}")
            print("Trying simpler ARIMA model...")
            try:
                fallback_model = SARIMAX(
                    train_data,
                    exog=exog_train,
                    order=(1, 1, 1),
                    enforce_stationarity=False,
                    enforce_invertibility=False
                )
                fitted_model = fallback_model.fit(disp=False)
                print(f"Fallback model fitted for {group_name}")
                return fitted_model
            except Exception as e2:
                print(f"Fallback model also failed for {group_name}: {e2}")
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

    def fit_theta_ets_model(self, group_name, train_data):
        """Fit an ETS/Theta blend model for very short history groups."""
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
            print(f"Warning: ThetaModel failed for {group_name}: {exc}")

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
            print(f"Warning: Not enough data for seasonal ETS component on {group_name}; disabling seasonality.")
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
            print(f"Warning: ETS component failed for {group_name}: {exc}")

        if theta_result is None and ets_result is None:
            print(f"Error: Unable to fit ETS/Theta blend for {group_name}.")
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

    def fit_arima_model(self, group_name, train_data, exog_train=None):
        """Fit a non-seasonal ARIMA model for short-history product groups."""
        order = self.get_short_series_arima_order(group_name)
        print(f"Using short-history ARIMA{order} for {group_name} (no seasonal component)")

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
                enforce_stationarity=False,
                enforce_invertibility=False
            )

            fitted_model = model.fit(disp=False)
            print(f"Short-history ARIMA fitted successfully for {group_name}")
            print(f"AIC: {fitted_model.aic:.2f}")
            return fitted_model

        except Exception as exc:
            print(f"Error fitting ARIMA fallback for {group_name}: {exc}")
            return None

    def calculate_metrics(self, actual, predicted, group_name):
        """Calculate bias% and MAE metrics"""
        # Remove any NaN values
        mask = ~(np.isnan(actual) | np.isnan(predicted))
        actual_clean = actual[mask]
        predicted_clean = predicted[mask]

        if len(actual_clean) == 0:
            return {"bias_percent": np.nan, "mae": np.nan}

        # Mean Absolute Error
        mae = mean_absolute_error(actual_clean, predicted_clean)

        # Bias percentage
        mean_actual = np.mean(actual_clean)
        if mean_actual != 0:
            bias_percent = (np.mean(predicted_clean - actual_clean) / mean_actual) * 100
        else:
            bias_percent = np.nan

        print(f"{group_name} - MAE: {mae:.2f}, Bias%: {bias_percent:.2f}%")

        return {"bias_percent": bias_percent, "mae": mae}

    def forecast_theta_ets(self, model_bundle, steps, start_index, group_name):
        """Generate forecasts and confidence intervals for the ETS/Theta blend."""
        if model_bundle is None or steps <= 0:
            return None

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
                print(f"Warning: Theta intervals unavailable for {group_name}: {exc}")

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

    def plot_results(self, group_name, historical_data, test_data, forecast_data):
        """Plot historical data, test predictions, and forecasts"""
        if not self.config.OUTPUT_PLOTS:
            return

        plt.figure(figsize=(15, 8))

        # Plot historical data
        plt.plot(historical_data.index, historical_data.values,
                label='Historical Data', color='blue', linewidth=2)

        # Plot test data
        if test_data is not None and len(test_data) > 0:
            plt.plot(test_data.index, test_data.values,
                    label='Test Data', color='green', linewidth=2)

        # Plot forecast
        if forecast_data:
            forecast_index = pd.date_range(
                start=historical_data.index[-1] + pd.DateOffset(months=1),
                periods=len(forecast_data['forecast']),
                freq='MS'
            )

            plt.plot(forecast_index, forecast_data['forecast'],
                    label='Forecast', color='red', linewidth=2)

            # Plot confidence intervals
            plt.fill_between(forecast_index,
                           forecast_data['conf_int_lower'],
                           forecast_data['conf_int_upper'],
                           alpha=0.3, color='red', label=f'{int(self.config.CONFIDENCE_LEVEL*100)}% Confidence Interval')

        model_label = self.model_strategy.get(group_name, 'SARIMA')
        plt.title(f'{group_name} - {model_label} Forecast')
        plt.xlabel('Date')
        plt.ylabel('Demand')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(f'{self.config.PLOTS_DIR}/{group_name}_forecast.png', dpi=300, bbox_inches='tight')
        plt.close()

    def run_forecasting(self):
        """Main method to run the complete forecasting pipeline"""
        print("Starting SARIMA forecasting pipeline...")
        print("=" * 50)

        # Load and preprocess data
        self.load_and_preprocess_data()

        results_summary = []

        # Process each product group
        for group_name, group_data in self.combined_data.items():
            print(f"\nProcessing {group_name}...")
            print("-" * 30)

            # Prepare time series
            ts_data = group_data.set_index('Month')['Actuals']
            ts_data = ts_data.asfreq('MS')  # Monthly start frequency

            current_date = pd.to_datetime(self.config.CURRENT_MONTH)
            ts_data = ts_data[ts_data.index <= current_date]
            ts_data = ts_data[ts_data > 0]  # Remove zero values at the end

            if len(ts_data) < 12:
                print(f"Warning: {group_name} has insufficient data. Skipping...")
                continue

            strategy = self.determine_model_strategy(ts_data, group_name)
            print(f"Selected {strategy} strategy for {group_name} (history={len(ts_data)} months)")

            forecast_start = ts_data.index[-1] + pd.DateOffset(months=1)

            exog_features = None
            exog_future = None

            if strategy != 'ETS_THETA' and self.salesforce_integration:
                actuals_with_index = ts_data.to_frame(name='Actuals')
                y_series, features = self.salesforce_integration.merge_with_actuals(
                    actuals_with_index, group_name
                )

                features = self.filter_salesforce_features(features, group_name) if features is not None else None

                if features is not None and not features.empty:
                    potential_future = self.salesforce_integration.create_future_exog_features(
                        group_name,
                        forecast_start,
                        self.config.FORECAST_MONTHS
                    )

                    if potential_future is not None and not potential_future.empty:
                        potential_future = self.filter_salesforce_features(potential_future, group_name)
                        if potential_future is not None and not potential_future.empty:
                            potential_future = potential_future.reindex(columns=features.columns, fill_value=0.0)
                            exog_features = features.astype(float)
                            exog_future = potential_future.astype(float).fillna(0.0)
                            ts_data = y_series.astype(float)
                        else:
                            print(f"Filtered Salesforce features lack future coverage for {group_name}; proceeding without exogenous inputs.")
                    else:
                        print(f"Salesforce features lack future coverage for {group_name}; proceeding without exogenous inputs.")
                else:
                    print(f"Salesforce features unavailable or filtered out for {group_name}; proceeding without exogenous inputs.")

            if exog_features is not None and (exog_future is None or exog_future.empty):
                print(f"Salesforce features missing forecast horizon coverage for {group_name}; proceeding without exogenous inputs.")
                exog_features = None
                exog_future = None

            if strategy == 'SARIMA':
                # Analyze seasonality
                self.analyze_seasonality(ts_data, group_name)

                # Check stationarity
                self.check_stationarity(ts_data, group_name)
            else:
                print('Skipping detailed seasonality diagnostics for non-SARIMA strategy.')

            exog_train = None
            exog_test = None

            if strategy != 'ETS_THETA' and exog_features is not None:
                split_point = len(ts_data) - self.config.TEST_SPLIT_MONTHS
                if split_point <= 0:
                    print(f"Warning: Not enough observations after holdout for {group_name}. Skipping...")
                    continue

                train_data = ts_data.iloc[:split_point]
                test_data = ts_data.iloc[split_point:]

                exog_train = exog_features.iloc[:split_point]
                exog_test = exog_features.iloc[split_point:]

                if exog_future is not None:
                    missing_cols = [col for col in exog_train.columns if col not in exog_future.columns]
                    for col in missing_cols:
                        exog_future[col] = 0.0
                    exog_future = exog_future[exog_train.columns]
                else:
                    exog_train = None
                    exog_test = None
            else:
                train_data, test_data = self.split_data(ts_data)

            if len(train_data) == 0:
                print(f"Warning: {group_name} has no training data after split. Skipping...")
                continue

            if strategy == 'SARIMA':
                model = self.fit_sarima_model(group_name, train_data, exog_train)
            elif strategy == 'ARIMA':
                model = self.fit_arima_model(group_name, train_data, exog_train)
            else:
                model = self.fit_theta_ets_model(group_name, train_data)

            if model is None:
                print(f'Unable to fit model for {group_name}; skipping forecast.')
                continue

            self.models[group_name] = model
            self.model_strategy[group_name] = strategy

            if len(test_data) > 0:
                if strategy == 'ETS_THETA':
                    test_result = self.forecast_theta_ets(model, len(test_data), train_data.index[-1], group_name)
                    if test_result is None:
                        print(f"Unable to generate test forecast for {group_name}.")
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

                self.metrics[group_name] = self.calculate_metrics(
                    test_data.values,
                    test_predictions.values if hasattr(test_predictions, 'values') else np.asarray(test_predictions),
                    group_name
                )

            if strategy == 'ETS_THETA':
                forecast = self.forecast_theta_ets(model, self.config.FORECAST_MONTHS, ts_data.index[-1], group_name)
            else:
                forecast = self.generate_forecast(
                    model,
                    self.config.FORECAST_MONTHS,
                    group_name,
                    exog_future
                )

            if forecast:
                self.forecasts[group_name] = forecast

                forecast_dates = pd.date_range(
                    start=forecast_start,
                    periods=self.config.FORECAST_MONTHS,
                    freq='MS'
                )

                for i, date in enumerate(forecast_dates):
                    results_summary.append({
                        'Product_Group': group_name,
                        'Model_Type': self.model_strategy.get(group_name, 'SARIMA'),
                        'Date': date.strftime('%Y-%m'),
                        'Forecast': round(forecast['forecast'].iloc[i], 0),
                        'Lower_CI': round(forecast['conf_int_lower'].iloc[i], 0),
                        'Upper_CI': round(forecast['conf_int_upper'].iloc[i], 0)
                    })
            self.plot_results(group_name, train_data, test_data, forecast)

        if self.config.SAVE_RESULTS and results_summary:
            results_df = pd.DataFrame(results_summary)
            results_df.to_csv(self.config.RESULTS_FILE, index=False)
            print(f"\nResults saved to {self.config.RESULTS_FILE}")

        self.print_summary()

        return results_summary

    def print_summary(self):
        """Print summary of forecasting results"""
        print("\n" + "="*50)
        print("FORECASTING SUMMARY")
        print("="*50)

        print(f"Models fitted: {len(self.models)}")
        print(f"Forecasts generated: {len(self.forecasts)}")

        if self.metrics:
            print("\nModel Performance Metrics:")
            print("-" * 25)
            for group, metrics in self.metrics.items():
                print(f"{group}: MAE={metrics['mae']:.2f}, Bias%={metrics['bias_percent']:.2f}%")

        if self.forecasts:
            print("\nNext 12-month forecasts generated for:")
            for group in self.forecasts.keys():
                model_used = self.model_strategy.get(group, 'SARIMA')
                print(f"  - {group} ({model_used})")


        print(f"\nPlots saved to: {self.config.PLOTS_DIR}/")
        print("="*50)
