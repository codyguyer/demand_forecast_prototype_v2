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
    USE_SALESFORCE = True
    SALESFORCE_DATA_FILE = "salesforce_data.csv"

    # Product grouping and replacement mapping
    PRODUCT_GROUPS = {
        "M11_dental": ["M11-040", "M11-050"],  # M11-040 being replaced by M11-050
        "M11_vet": ["M11-043", "M11-053"],  # M11-043 being replaced by M11-053
        "M9_dental": ["M9-040", "M9-050"],  # M9-040 being replaced by M9-050
        "M9_vet": ["M9-043", "M9-053"]  # M9-043 being replaced by M9-053
    }

    # SARIMA parameters (p,d,q,P,D,Q,s) for each product group
    # Format: (p, d, q, P, D, Q, s)
    # p,d,q: non-seasonal AR, differencing, MA order
    # P,D,Q: seasonal AR, differencing, MA order
    # s: seasonal period (12 for monthly data with annual seasonality)
    SARIMA_PARAMS = {
        "M11_dental": (1, 0, 1, 0, 0, 1, 12),
        "M11_vet": (0, 0, 1, 0, 1, 1, 12),
        "M9_dental": (0, 1, 1, 0, 0, 1, 12),
        "M9_vet": (1, 0, 1, 0, 0, 1, 12)
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
        "M9_vet": "quantity"
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
