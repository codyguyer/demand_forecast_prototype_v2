"""Run the 2025 forecast experiment without Salesforce features."""

from sarima_model import SARIMAForecaster
from .no_salesforce_config import Experiment2025NoSFConfig


def main():
    """Execute the SARIMA pipeline using only historical actuals."""
    print("Running 2025 forecast experiment (no Salesforce)")
    print("=" * 48)

    config = Experiment2025NoSFConfig()
    forecaster = SARIMAForecaster(config)
    results = forecaster.run_forecasting()

    if results:
        print("\nExperiment completed successfully.")
        print(f"Detailed output: {config.RESULTS_FILE}")
    else:
        print("\nExperiment did not produce any results. Review logs above.")


if __name__ == "__main__":
    main()
