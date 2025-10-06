"""
Salesforce Integration Framework for SARIMA Model
Future enhancement to incorporate pipeline data as external regressors
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
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

    def load_salesforce_data(self, filepath_or_data):
        """
        Load Salesforce opportunity data
        Expected columns: Product, Weighted_Pipeline_Amount, Weighted_Pipeline_Quantity, CloseDate
        """
        if isinstance(filepath_or_data, str):
            # Load from file
            self.pipeline_data = pd.read_csv(filepath_or_data)
        else:
            # Use provided DataFrame
            self.pipeline_data = filepath_or_data.copy()

        # Standardize column names
        column_mapping = {
            self.config.SALESFORCE_FIELDS['product']: 'Product',
            self.config.SALESFORCE_FIELDS['weighted_pipeline_dollars']: 'Weighted_Pipeline_Amount',
            self.config.SALESFORCE_FIELDS['weighted_pipeline_quantity']: 'Weighted_Pipeline_Quantity',
            self.config.SALESFORCE_FIELDS['close_date']: 'CloseDate'
        }

        self.pipeline_data = self.pipeline_data.rename(columns=column_mapping)
        self.pipeline_data['CloseDate'] = pd.to_datetime(self.pipeline_data['CloseDate'])

        if 'Product' in self.pipeline_data.columns:
            self.pipeline_data['Product'] = self.pipeline_data['Product'].astype(str)
            self.pipeline_data['Normalized_Product'] = self.pipeline_data['Product'].apply(self.normalize_product_code)
        else:
            self.pipeline_data['Normalized_Product'] = ''
            print('Warning: Salesforce data missing Product column; normalization skipped.')

        print(f"Loaded {len(self.pipeline_data)} Salesforce opportunities")
        return self.pipeline_data

    def process_pipeline_features(self, product_groups):
        """
        Process Salesforce pipeline data into monthly features for each product group
        """
        print("Processing Salesforce pipeline features...")

        monthly_features = {}

        if self.pipeline_data is None or self.pipeline_data.empty:
            print('No Salesforce pipeline data available to process.')
            self.processed_features = {}
            return {}

        if 'Normalized_Product' not in self.pipeline_data.columns and 'Product' in self.pipeline_data.columns:
            self.pipeline_data['Normalized_Product'] = self.pipeline_data['Product'].apply(self.normalize_product_code)

        forecast_horizon_end = pd.to_datetime(self.config.CURRENT_MONTH) + pd.DateOffset(months=self.config.FORECAST_MONTHS)

        for group_name, products in product_groups.items():
            print(f"Processing pipeline data for {group_name}")

            normalized_targets = {self.normalize_product_code(p) for p in products}

            # Filter pipeline data for this product group
            group_pipeline = self.pipeline_data[
                self.pipeline_data['Normalized_Product'].isin(normalized_targets)
            ].copy()

            if len(group_pipeline) == 0:
                print(f"No pipeline data found for {group_name} after normalization: {sorted(normalized_targets)}")
                continue

            missing_codes = normalized_targets.difference(set(group_pipeline['Normalized_Product'].unique()))
            if missing_codes:
                print(f"Partial Salesforce coverage for {group_name}; missing normalized codes: {sorted(missing_codes)}")

            # Create monthly aggregations
            group_pipeline['YearMonth'] = group_pipeline['CloseDate'].dt.to_period('M')

            monthly_agg = group_pipeline.groupby('YearMonth')[['Weighted_Pipeline_Amount', 'Weighted_Pipeline_Quantity']].sum()

            monthly_agg.index = monthly_agg.index.to_timestamp(how='start')

            full_index = pd.date_range(
                start=monthly_agg.index.min(),
                end=forecast_horizon_end,
                freq='MS'
            )

            monthly_agg = monthly_agg.reindex(full_index, fill_value=0.0)
            monthly_agg.index.name = 'Date'

            # Create lagged features (pipeline often leads actual sales)
            monthly_agg['Pipeline_Amount_Lag1'] = monthly_agg['Weighted_Pipeline_Amount'].shift(1)
            monthly_agg['Pipeline_Amount_Lag3'] = monthly_agg['Weighted_Pipeline_Amount'].shift(3)
            monthly_agg['Pipeline_Quantity_Lag1'] = monthly_agg['Weighted_Pipeline_Quantity'].shift(1)
            monthly_agg['Pipeline_Quantity_Lag3'] = monthly_agg['Weighted_Pipeline_Quantity'].shift(3)

            monthly_agg['Pipeline_Amount_MA3'] = monthly_agg['Weighted_Pipeline_Amount'].rolling(window=3, min_periods=1).mean()
            monthly_agg['Pipeline_Quantity_MA3'] = monthly_agg['Weighted_Pipeline_Quantity'].rolling(window=3, min_periods=1).mean()

            monthly_agg = monthly_agg.fillna(0.0).reset_index()

            monthly_features[group_name] = monthly_agg

        self.processed_features = monthly_features
        return monthly_features


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

    def create_future_exog_features(self, group_name, forecast_start_date, forecast_months):
        """
        Create future external regressor features for forecasting
        """
        if group_name not in self.processed_features:
            return None

        features_df = self.processed_features[group_name].copy()

        if 'Date' not in features_df.columns:
            print(f"Processed features for {group_name} missing 'Date' column; cannot create future features.")
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
            print(f"Filled missing Salesforce feature values with zeros for {group_name} forecast horizon")

        print(f"Created future external features for {group_name} using Salesforce pipeline data")

        return future_df


    def merge_with_actuals(self, actuals_data, group_name):
        """
        Merge Salesforce features with actual sales data for modeling
        """
        if group_name not in self.processed_features:
            print(f"No Salesforce features available for {group_name}")
            return actuals_data, None

        features_df = self.processed_features[group_name]

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
            'Weighted_Pipeline_Amount', 'Weighted_Pipeline_Quantity',
            'Pipeline_Amount_Lag1', 'Pipeline_Amount_Lag3',
            'Pipeline_Quantity_Lag1', 'Pipeline_Quantity_Lag3',
            'Pipeline_Amount_MA3', 'Pipeline_Quantity_MA3'
        ]

        merged = merged.sort_values('Month')

        for col in feature_columns:
            if col not in merged.columns:
                merged[col] = 0.0
            else:
                merged[col] = merged[col].fillna(0)

        merged = merged.set_index('Month')

        # Return actuals and features separately
        actuals_series = merged['Actuals']
        features_df = merged[feature_columns]

        print(f"Merged {len(features_df)} observations with Salesforce features for {group_name}")

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
                actuals_data, group_name
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
