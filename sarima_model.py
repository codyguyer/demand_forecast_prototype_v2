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

from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import adfuller
from statsmodels.stats.diagnostic import acorr_ljungbox
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

    def generate_forecast(self, model, steps, group_name, exog_future=None):
        """Generate forecast with confidence intervals"""
        print(f"Generating {steps}-month forecast for {group_name}...")

        try:
            forecast_result = model.get_forecast(steps=steps, exog=exog_future)
            forecast_mean = forecast_result.predicted_mean

            conf_int = forecast_result.conf_int(alpha=1 - self.config.CONFIDENCE_LEVEL)

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

        plt.title(f'{group_name} - SARIMA Forecast')
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

            forecast_start = ts_data.index[-1] + pd.DateOffset(months=1)

            exog_features = None
            exog_future = None

            if self.salesforce_integration:
                actuals_with_index = ts_data.to_frame(name='Actuals')
                y_series, features = self.salesforce_integration.merge_with_actuals(
                    actuals_with_index, group_name
                )

                if features is not None and not features.empty:
                    potential_future = self.salesforce_integration.create_future_exog_features(
                        group_name,
                        forecast_start,
                        self.config.FORECAST_MONTHS
                    )

                    if potential_future is not None and not potential_future.empty:
                        exog_features = features.astype(float)
                        exog_future = potential_future.astype(float).fillna(0.0)
                        ts_data = y_series.astype(float)
                    else:
                        print(f"Salesforce features lack future coverage for {group_name}; proceeding without exogenous inputs.")
                else:
                    print(f"Salesforce features unavailable for {group_name}; proceeding without exogenous inputs.")

            if exog_features is not None and (exog_future is None or exog_future.empty):
                print(f"Salesforce features missing forecast horizon coverage for {group_name}; proceeding without exogenous inputs.")
                exog_features = None
                exog_future = None

            # Analyze seasonality
            self.analyze_seasonality(ts_data, group_name)

            # Check stationarity
            self.check_stationarity(ts_data, group_name)

            exog_train = None
            exog_test = None

            if exog_features is not None:
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

            model = self.fit_sarima_model(group_name, train_data, exog_train)
            if model is None:
                continue

            self.models[group_name] = model

            if len(test_data) > 0:
                if exog_test is not None and not exog_test.empty:
                    test_forecast = model.get_forecast(
                        steps=len(test_data),
                        exog=exog_test.fillna(0.0)
                    )
                else:
                    test_forecast = model.get_forecast(steps=len(test_data))

                test_predictions = test_forecast.predicted_mean
                self.metrics[group_name] = self.calculate_metrics(
                    test_data.values, test_predictions.values, group_name
                )

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
            print(f"\nNext 12-month forecasts generated for:")
            for group in self.forecasts.keys():
                print(f"  • {group}")

        print(f"\nPlots saved to: {self.config.PLOTS_DIR}/")
        print("="*50)