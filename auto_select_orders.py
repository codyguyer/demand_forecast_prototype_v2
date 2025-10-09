"""
Command-line helper to run SARIMA order selection across all configured series.
"""

import json
import time
from pathlib import Path

import pandas as pd

from config import SARIMAConfig
from sarima_auto_selector import SARIMAOrderSelector


OUTPUT_FILE = Path("sarima_order_selection_results.csv")


def main() -> None:
    config = SARIMAConfig()

    available_groups = list(config.PRODUCT_GROUPS.keys())
    print("Available product groups:")
    for name in available_groups:
        print(f"  - {name}")

    selection = input(
        "\nEnter a product group key (comma separated for multiple) or press Enter for all: "
    ).strip()

    if selection:
        requested = [item.strip() for item in selection.split(",") if item.strip()]
        invalid = [item for item in requested if item not in config.PRODUCT_GROUPS]
        if invalid:
            print(f"\nInvalid product group(s): {', '.join(invalid)}")
            print("Aborting order selection. Please rerun and choose from the listed options.")
            return
        product_groups = {key: config.PRODUCT_GROUPS[key] for key in requested}
    else:
        product_groups = config.PRODUCT_GROUPS

    if not product_groups:
        print("\nNo product groups selected; exiting.")
        return

    print(f"\nSelected product groups: {', '.join(product_groups.keys())}")

    print("Running SARIMA order selection...")
    start_time = time.perf_counter()

    df = pd.read_csv(config.DATA_FILE)

    selector = SARIMAOrderSelector(
        df,
        product_groups=product_groups,
        forecast_by_bu=getattr(config, "FORECAST_BY_BU", False),
    )

    results = selector.run()
    elapsed = time.perf_counter() - start_time

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

    print(f"\nOrder selection runtime: {elapsed:0.1f} seconds (~{elapsed/60:0.1f} minutes)")


if __name__ == "__main__":
    main()
