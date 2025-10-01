# Demand Forecast Prototype v2

Enhanced SARIMA-based demand forecasting pipeline that can incorporate Salesforce pipeline data as exogenous drivers. The project is optimized for monthly product-group forecasting, produces metrics and plots, and can be extended with additional data sources.

## Features
- Configurable SARIMA parameters per product group via `config.py`.
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
Update `config.py` to control:
- Data file paths and forecast horizon (`DATA_FILE`, `FORECAST_MONTHS`, `CURRENT_MONTH`).
- Product group mappings (`PRODUCT_GROUPS`).
- SARIMA parameter grids (`SARIMA_PARAMS`).
- Salesforce integration toggle and field mapping (`USE_SALESFORCE`, `SALESFORCE_FIELDS`, `SALESFORCE_DATA_FILE`).

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

## Salesforce Data Expectations
`salesforce_data.csv` should include the following columns (or the names mapped in `config.py`):
- Product Code
- Weighted Pipeline Dollars
- Weighted Pipeline Quantity
- Close Date

The integration aggregates opportunities monthly, builds lagged/rolling signals, and aligns future periods to match the forecast horizon.

## Repository Structure
- `main.py` – CLI entry point for running the full pipeline.
- `config.py` – Central configuration for data paths and modeling parameters.
- `sarima_model.py` – Core SARIMA/SARIMAX modeling workflow.
- `salesforce_integration.py` – Salesforce ingestion and feature engineering utilities.
- `requirements.txt` – Python dependency list.

## Contributing
1. Fork the repository and create a feature branch.
2. Run `python -m py_compile *.py` and `python main.py` to validate changes.
3. Submit a pull request with a clear summary of updates.

## License
Internal Midmark Corporation prototype. Contact the maintainers before public distribution.
