"""
SARIMA Model Configuration
Easily adjustable parameters for forecasting models
"""

class SARIMAConfig:
    """Configuration class for SARIMA model parameters"""

    # Data configuration
    DATA_FILE = "Actuals.csv"
    FORECAST_MONTHS = 12
    CURRENT_MONTH = "2025-10"  # Update this based on current date
    USE_SALESFORCE = False
    SALESFORCE_DATA_FILE = "salesforce_data.csv"

    # Product grouping and replacement mapping
    PRODUCT_GROUPS = {
        "M11_dental": ["     ITEM_M11-040", "     ITEM_M11-050"],  # M11-040 being replaced by M11-050
        "M11_vet": ["     ITEM_M11-043", "     ITEM_M11-053"],  # M11-043 being replaced by M11-053
        "M9_dental": ["     ITEM_M9-040", "     ITEM_M9-050"],  # M9-040 being replaced by M9-050
        "M9_vet": ["     ITEM_M9-043", "     ITEM_M9-053"],  # M9-043 being replaced by M9-053
        "204": ["Prod_204"],
        "224": ["     Prod_224"],
        "ECG": ["ITEM_4-000-0080"],
        "Digital_Vital_Signs": ["     ITEM_4-000-0550"],
        "250": ["PROD_50"],
        "253": ["Prod_53"]
    }

    # SARIMA parameters (p,d,q,P,D,Q,s) for each product group
    # Format: (p, d, q, P, D, Q, s)
    # p,d,q: non-seasonal AR, differencing, MA order
    # P,D,Q: seasonal AR, differencing, MA order
    # s: seasonal period (12 for monthly data with annual seasonality)
    SARIMA_PARAMS = {
        "M11_dental": (0, 0, 1, 0, 1, 1, 12),
        "M11_vet": (0, 1, 1, 1, 1, 1, 12),
        "M9_dental": (0, 1, 1, 0, 0, 1, 12),
        "M9_vet": (1, 1, 1, 0, 1, 1, 12),
        "204": (0, 1, 1, 0, 1, 1, 12),
        "224": (1, 1, 1, 1, 1, 0, 12),
        "ECG": (2, 1, 2, 0, 1, 1, 12),
        "Digital_Vital_Signs": (0, 1, 2, 0, 1, 1, 12),
        "250": (1, 1, 1, 1, 1, 1, 12),
        "253": (0, 0, 1, 0, 1, 1, 12)
    }

    # Model evaluation parameters
    CONFIDENCE_LEVEL = 0.90  # 90% confidence intervals (±1.645 std dev)
    TEST_SPLIT_MONTHS = 6  # Number of months to hold out for testing

    # Future Salesforce integration parameters
    SALESFORCE_FIELDS = {
        "product": "Product Code",
        "weighted_pipeline_dollars": "Weighted Pipeline Dollars",
        "weighted_pipeline_quantity": "Weighted Pipeline Quantity",
        "close_date": "Close Date"
    }

    # Salesforce feature controls
    SALESFORCE_FEATURE_COLUMN_SETS = {
        "quantity": [
            "Weighted_Pipeline_Quantity",
            "Pipeline_Quantity_Lag1",
            "Pipeline_Quantity_Lag3",
            "Pipeline_Quantity_MA3"
        ],
        "dollars": [
            "Weighted_Pipeline_Amount",
            "Pipeline_Amount_Lag1",
            "Pipeline_Amount_Lag3",
            "Pipeline_Amount_MA3"
        ]
    }

    SALESFORCE_FEATURE_MODE_BY_GROUP = {
        "default": "quantity",
        "M11_dental": "quantity",
        "M11_vet": "quantity",
        "M9_dental": "quantity",
        "M9_vet": "quantity",
        "204": "quantity",
        "224": "quantity",
        "ECG": "quantity",
        "Digital_Vital_Signs": "quantity",
        "250": "quantity",
        "253": "quantity"
    }

    ENFORCE_NON_NEGATIVE_FORECASTS = True

    # Seasonality detection thresholds
    SEASONALITY_THRESHOLD = 0.3  # Minimum seasonal component strength to consider seasonal
    MIN_OBSERVATIONS = 24  # Minimum observations needed for reliable seasonal analysis

    # Output configuration
    OUTPUT_PLOTS = True
    SAVE_RESULTS = True
    RESULTS_FILE = "sarima_forecast_results.csv"
    PLOTS_DIR = "plots"
    # Short-history non-seasonal ARIMA fallback
    SHORT_SERIES_ARIMA_ENABLED = True
    SHORT_SERIES_THRESHOLD = 24
    SHORT_SERIES_ARIMA_DEFAULT_ORDER = (1, 1, 1)
    SHORT_SERIES_ARIMA_PARAMS = {}

    # Model selection thresholds
    SARIMA_MIN_HISTORY = 36
    ARIMA_MIN_HISTORY = 24

    # Very short-history ETS/Theta blend fallback
    ETS_THETA_ENABLED = False
    ETS_THETA_WEIGHTS = (0.5, 0.5)
    ETS_THETA_PERIOD = 12
    ETS_THETA_TREND = 'add'
    ETS_THETA_SEASONAL = None
    ETS_THETA_SEASONAL_PERIODS = None
    ETS_THETA_DAMPED = False
    ETS_THETA_FORCE_GROUPS = None

