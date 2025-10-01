"""
Main script to run SARIMA forecasting model
Execute this file to generate forecasts for all product groups
"""

from sarima_model import SARIMAForecaster
from config import SARIMAConfig
import pandas as pd

def main():
    """
    Main function to execute SARIMA forecasting pipeline
    """
    print("SARIMA Product Demand Forecasting")
    print("=" * 40)

    # Initialize configuration
    config = SARIMAConfig()

    # Display configuration
    print("\nCurrent Configuration:")
    print(f"  • Data file: {config.DATA_FILE}")
    print(f"  • Forecast months: {config.FORECAST_MONTHS}")
    print(f"  • Current month: {config.CURRENT_MONTH}")
    print(f"  • Confidence level: {config.CONFIDENCE_LEVEL*100:.0f}%")
    print(f"  • Test split: {config.TEST_SPLIT_MONTHS} months")
    if config.USE_SALESFORCE:
        print(f"  - Salesforce integration: ON ({config.SALESFORCE_DATA_FILE})")
    else:
        print("  - Salesforce integration: OFF")

    print("\nProduct Groups:")
    for group, products in config.PRODUCT_GROUPS.items():
        params = config.SARIMA_PARAMS.get(group, "Default")
        print(f"  • {group}: {products} -> SARIMA{params}")

    # Initialize and run forecaster
    forecaster = SARIMAForecaster(config)

    # Run the complete forecasting pipeline
    results = forecaster.run_forecasting()

    if results:
        print(f"\n[SUCCESS] Forecasting completed successfully!")
        print(f"  - Results saved to: {config.RESULTS_FILE}")
        print(f"  - Plots saved to: {config.PLOTS_DIR}/")

        # Display sample results
        results_df = pd.DataFrame(results)
        print(f"\nSample forecast results:")
        print(results_df.head(12).to_string(index=False))

    else:
        print("\n[ERROR] No forecasting results generated. Check your data and configuration.")

    return results

if __name__ == "__main__":
    main()