"""Configuration overrides for the 2025 forecast experiment."""

from config import SARIMAConfig


class Experiment2025Config(SARIMAConfig):
    """Use filtered 2022-2024 data and isolate all experiment artifacts."""

    DATA_FILE = "experiments/forecast_2025_test/Actuals_2022_2024.csv"
    CURRENT_MONTH = "2024-12"
    FORECAST_MONTHS = 12
    RESULTS_FILE = "experiments/forecast_2025_test/results_2025_experiment.csv"
    PLOTS_DIR = "experiments/forecast_2025_test/plots"
    USE_SALESFORCE = True
    SALESFORCE_DATA_FILE = "experiments/forecast_2025_test/salesforce_2022_2024.csv"
    SAVE_RESULTS = True
    OUTPUT_PLOTS = True
    TEST_SPLIT_MONTHS = 6
    ENFORCE_NON_NEGATIVE_FORECASTS = True
