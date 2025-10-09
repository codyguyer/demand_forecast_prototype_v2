# Demand Forecast Prototype v2

Enhanced SARIMA-based demand forecasting pipeline that can incorporate Salesforce pipeline data as exogenous drivers. The project is optimized for monthly product-group forecasting, produces metrics and plots, and can be extended with additional data sources.

## Features
- Managed product catalog (`managed_data/product_catalog.csv`) that is the single source of truth for product groups, BU mappings, SARIMA orders, and Salesforce feature modes.
- Automatic aggregation of replacement products into consolidated demand series.
- Optional Salesforce opportunity ingestion with lagged and rolling feature engineering.
- SARIMAX training/testing with exogenous features when Salesforce data is present.
- Forecast generation with confidence intervals, CSV export, and plot creation.

## Requirements
- Python 3.9+
- See `requirements.txt` for the full list of dependencies.

Install dependencies (recommended in a virtual environment):

```bash
python -m venv .venv
.\.venv\Scripts\activate
pip install -r requirements.txt
```

## Configuration
Update configuration values in two places:
- `config.py` for data file paths, forecast horizon (`DATA_FILE`, `FORECAST_MONTHS`, `CURRENT_MONTH`), and integration toggles (`USE_SALESFORCE`, `SALESFORCE_FIELDS`, `SALESFORCE_DATA_FILE`, backlog options, etc.).
- `managed_data/product_catalog.csv` for every product-specific setting. If a row has any blank field, it is skipped at runtime and the product will not be forecast.

### Managed catalog workflow
1. Copy `managed_data/product_catalog.csv` to your preferred data store (CSV, database table, or SharePoint list) and keep the column structure.
2. Maintain one row per product group with delimited values (pipe `|`, semicolon `;`, or newline) for multi-SKU groupings.
3. Fill in the Salesforce feature mode (`salesforce_feature_mode`) for each product group (e.g., `quantity` or `dollars`); leaving it blank will skip the product.
4. Provide SARIMA order columns (`sarima_p` through `sarima_s`) and BU mapping columns (`business_unit_code`, `business_unit_name`). Any blank field triggers a skip, so complete rows are required for inclusion.
5. Restart scripts (or re-instantiate `SARIMAConfig`) after updating the catalog so the latest metadata loads before runtime.

## Usage
Run the main forecasting pipeline:

```bash
python main.py
```

The script will:
1. Load and preprocess historical demand data.
2. Optionally ingest Salesforce pipeline data and engineer exogenous features.
3. Train SARIMA/SARIMAX models with a rolling test split for evaluation.
4. Generate forward forecasts with confidence intervals.
5. Save results to `sarima_forecast_results.csv` (ignored by default) and plots in `plots/`.

### SARIMA Order Selection
Run the automatic SARIMA search to score candidate orders per product (and BU when enabled):

```bash
python auto_select_orders.py
```

The selector writes a detailed report to `sarima_order_selection_results.csv` (MAE per fold, AICc, warnings, Ljung-Box p-values) and prints a JSON snippet with the recommended `(p, d, q, P, D, Q, m)` tuple for each series. Use those tuples to update `SARIMA_PARAMS` in `config.py` once you are comfortable with the diagnostics.

## Salesforce Data Expectations
`salesforce_data.csv` should include the following columns (or the names mapped in `config.py`):
- Product Code
- Weighted Pipeline Dollars
- Weighted Pipeline Quantity
- Close Date

The integration aggregates opportunities monthly, builds lagged/rolling signals, and aligns future periods to match the forecast horizon.

## Repository Structure
- `main.py` ? CLI entry point for running the full pipeline.
- `config.py` ? Central configuration for data paths and modeling parameters.
- `sarima_model.py` ? Core SARIMA/SARIMAX modeling workflow.
- `salesforce_integration.py` ? Salesforce ingestion and feature engineering utilities.
- `requirements.txt` ? Python dependency list.

## Contributing
1. Fork the repository and create a feature branch.
2. Run `python -m py_compile *.py` and `python main.py` to validate changes.
3. Submit a pull request with a clear summary of updates.

## License
Internal Midmark Corporation prototype. Contact the maintainers before public distribution.
