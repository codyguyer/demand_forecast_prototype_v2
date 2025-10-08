"""
Command-line helper to run SARIMA order selection across all configured series.
"""

import json
from pathlib import Path

import pandas as pd

from config import SARIMAConfig
from sarima_auto_selector import SARIMAOrderSelector


OUTPUT_FILE = Path("sarima_order_selection_results.csv")


def main() -> None:
    config = SARIMAConfig()

    print("Running SARIMA order selection...")
    df = pd.read_csv(config.DATA_FILE)

    selector = SARIMAOrderSelector(
        df,
        product_groups=config.PRODUCT_GROUPS,
        forecast_by_bu=getattr(config, "FORECAST_BY_BU", False),
    )

    results = selector.run()
    results_df = selector.results_to_dataframe(results)
    results_df.to_csv(OUTPUT_FILE, index=False)

    print(f"\nSelection results written to {OUTPUT_FILE.resolve()}")

    recommended_orders = {}
    for result in results:
        if result.status == "sarima_selected" and result.best_order and result.best_seasonal_order:
            order_display = tuple(result.best_order + result.best_seasonal_order)
            recommended_orders[result.series_key] = order_display

    if recommended_orders:
        print("\nRecommended SARIMA orders (p, d, q, P, D, Q, m):")
        print(json.dumps(recommended_orders, indent=2))
    else:
        print("\nNo SARIMA orders were selected; check warnings in the CSV output.")


if __name__ == "__main__":
    main()
