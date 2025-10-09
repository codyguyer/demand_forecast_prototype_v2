"""Standalone runner for the 2025 forecast experiment."""

from sarima_model import SARIMAForecaster
from .experiment_config import Experiment2025Config


def main():
    """Execute the SARIMA pipeline with experiment-specific settings."""
    print("Running 2025 forecast experiment")
    print("=" * 40)

    config = Experiment2025Config()
    forecaster = SARIMAForecaster(config)
    results = forecaster.run_forecasting()

    if results:
        print("\nExperiment completed successfully.")
        print(f"Detailed output: {config.RESULTS_FILE}")
    else:
        print("\nExperiment did not produce any results. Review logs above.")


if __name__ == "__main__":
    main()
