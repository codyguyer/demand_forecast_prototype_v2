"""Configuration overrides for the 2025 forecast experiment without Salesforce exogenous data."""

from config import SARIMAConfig


class Experiment2025NoSFConfig(SARIMAConfig):
    """Use filtered 2022-2024 actuals only and isolate outputs."""

    DATA_FILE = "experiments/forecast_2025_test/Actuals_2022_2024.csv"
    CURRENT_MONTH = "2024-12"
    FORECAST_MONTHS = 12
    RESULTS_FILE = "experiments/forecast_2025_test/results_2025_experiment_no_sf.csv"
    PLOTS_DIR = "experiments/forecast_2025_test/plots_no_sf"
    USE_SALESFORCE = False
    SAVE_RESULTS = True
    OUTPUT_PLOTS = True
    TEST_SPLIT_MONTHS = 6
    ENFORCE_NON_NEGATIVE_FORECASTS = True
    SHORT_SERIES_ARIMA_ENABLED = True
    SARIMA_MIN_HISTORY = 36
    ARIMA_MIN_HISTORY = 24
    ETS_THETA_ENABLED = True
    ETS_THETA_WEIGHTS = (0.5, 0.5)
    ETS_THETA_PERIOD = 12
    ETS_THETA_TREND = 'add'
    ETS_THETA_SEASONAL = None
    ETS_THETA_SEASONAL_PERIODS = None
    ETS_THETA_DAMPED = False

