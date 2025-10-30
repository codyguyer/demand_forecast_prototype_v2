# Demand Forecast Prototype v2

SARIMA-based demand forecasting pipeline tailored for monthly product-group planning. The system combines managed catalog metadata, Salesforce pipeline signals, and optional backlog features to fit, evaluate, and export forecasts with confidence intervals and diagnostics.

## Overview
- `SARIMAConfig` loads managed metadata, builds product-group/BU mappings, and exposes all modeling toggles from `config.py`.
- `SARIMAForecaster` aggregates historical demand, assembles exogenous regressors, selects an appropriate model strategy, and generates forecasts plus holdout metrics.
- Salesforce and backlog integrations contribute lagged, rolling, and weighted features that are dynamically filtered per product group.
- Results, diagnostics, and helper analyses are written to `analysis_outputs/` for downstream review.

## Data Inputs
### Managed catalog (`managed_data/product_catalog.csv`)
- Required columns: `group_key`, `sku_list`, `business_unit_code`, `business_unit_name`, `salesforce_feature_mode`, and all `sarima_*` order fields. Rows with missing required fields are skipped.
- Multi-SKU groupings may be pipe (`|`), semicolon (`;`), or newline separated; whitespace is preserved for intentional padding.
- `salesforce_feature_mode` determines which pipeline feature set to prefer (`quantity`, `dollars`, or `default`).
- Optional Salesforce lag overrides are read from the columns defined in `SALESFORCE_LAG_COLUMN_MAP`. Values support either a single integer lag (`3`) or BU-specific overrides (`default:2|ENT:4`).
- Update the catalog (and accompanying reference files) before instantiating `SARIMAConfig`; a fresh config reloads the latest metadata.

### Actual demand (`managed_data/Actuals.csv`)
- Expected columns: `Product`, `Month`, `Actuals`, and optionally `BU`.
- `SARIMAForecaster.load_and_preprocess_data` normalizes the `Month` column, groups SKUs into catalog-defined product groups, and, when `FORECAST_BY_BU` is enabled, splits each group by business-unit code.
- Demand history is truncated at `CURRENT_MONTH` and trailing all-zero padding is removed when `TRIM_TRAILING_ZERO_PADDING` is `True`.

### Salesforce pipeline (`managed_data/salesforce_data_v2.csv`)
- Column names are mapped through `SALESFORCE_FIELDS`; the defaults expect timestamp, product name, BU, revenue, quantity, and new-order metrics.
- Products are normalized and aligned to catalog groups using `managed_data/sf_product_reference_key.csv`, plus direct SKU lookups when available.
- Features are aggregated monthly, and the integration creates lagged, rolling, and weighted combinations per group/BU. Future periods are synthesized with a trailing six-month weighted average so exogenous coverage matches the forecast horizon.
- `SALESFORCE_FEATURE_COLUMN_SETS`, `SALESFORCE_FEATURE_MODE_BY_GROUP`, and `SALESFORCE_DIFFERENCE_FEATURES` control which columns enter each model.

### Backlog (`managed_data/Backlog.csv`)
- Expected columns (case-insensitive): `Product`, `Month`, `Backlog`, and optionally `BU`.
- Normalized backlog series generate `Backlog`, `Backlog_Lag1`, `Backlog_Lag3`, and `Backlog_MA3`. These are available when `USE_BACKLOG` is enabled.

### Reference tables
- `managed_data/sf_product_reference_key.csv` maps Salesforce product names back to catalog `group_key` values.
- Additional managed files can be added so long as `config.py` points `SALESFORCE_DATA_FILE`, `SALESFORCE_REFERENCE_FILE`, and `BACKLOG_DATA_FILE` at the correct locations.

## Configuration (`config.py`)
- Set file paths (`DATA_FILE`, `SALESFORCE_DATA_FILE`, `BACKLOG_DATA_FILE`), forecast settings (`FORECAST_MONTHS`, `CURRENT_MONTH`, `CONFIDENCE_LEVEL`, `TEST_SPLIT_MONTHS`), and integration toggles (`USE_SALESFORCE`, `USE_BACKLOG`, `FORECAST_BY_BU`, `OUTPUT_PLOTS`, `SAVE_RESULTS`).
- Model selection thresholds are governed by `SARIMA_MIN_HISTORY` (default 36 months) and `ARIMA_MIN_HISTORY` / `SHORT_SERIES_THRESHOLD` (default 24 months).
- Fallback behavior is controlled by `SHORT_SERIES_ARIMA_ENABLED`, `SHORT_SERIES_ARIMA_DEFAULT_ORDER`, and the optional `ETS_THETA_*` settings.
- `ENFORCE_NON_NEGATIVE_FORECASTS` clips negative forecasts and confidence bounds to zero. `TRIM_TRAILING_ZERO_PADDING` removes padded future zeros from training history.
- `RESULTS_FILE` defines the primary export (`analysis_outputs/sarima_forecast_results.csv` by default); update this if you need to relocate outputs.

## Forecasting Pipeline
1. Instantiate `SARIMAConfig` (loading catalog metadata, BU maps, SARIMA parameters, Salesforce lag instructions, etc.).
2. Read and normalize historical demand. SKUs listed in the catalog are aggregated into product-group totals and optionally broken out by BU.
3. Load Salesforce/backlog sources (if enabled), transform them into monthly features, and build a unified exogenous feature store keyed by `(Date, Product_Group, BU)` with `sf_` and `backlog_` prefixes.
4. Iterate over each series:
   - Trim history to the configured current month, ensure a regular monthly index, and drop trailing zero padding.
   - Choose a modeling strategy:
     - **SARIMA** when sufficient history exists (`>= SARIMA_MIN_HISTORY`) or when no fallback is required.
     - **ARIMA** (non-seasonal) for medium histories that fall below the SARIMA threshold but above `ARIMA_MIN_HISTORY`.
     - **ETS/Theta blend** for very short histories when `ETS_THETA_ENABLED` is true.
   - Pull aligned exogenous features for both the in-sample window and forecast horizon. When future Salesforce features are missing, the integration extrapolates them from the trailing six-month weighted average (weights `[0.4, 0.25, 0.15, 0.1, 0.06, 0.04]`).
   - Run diagnostics: seasonal strength via `seasonal_decompose` (compared against `SEASONALITY_THRESHOLD`) and Augmented Dickey-Fuller stationarity checks.
   - Split the series into train/test portions using `TEST_SPLIT_MONTHS`, fit the selected model, and measure holdout performance (MAE%, Bias%, RMSE, MAPE, CoV).
   - Refit on the full history before generating the forward forecast so long as the initial fit succeeded.
   - Produce point forecasts and confidence intervals, enforcing non-negativity when configured.
5. Aggregate results into a DataFrame and persist to `RESULTS_FILE`. A console summary lists fitted models, metrics, and the forecast recipients.

### Exogenous feature handling
- Salesforce and backlog feature matrices are combined only when data exists for the entire requested index; empty matrices gracefully fall back to univariate models.
- Columns are filtered per group using `SALESFORCE_FEATURE_COLUMN_SETS` and `BACKLOG_FEATURES`; missing configured columns are backfilled with zeros.
- Catalog-driven overrides (`SALESFORCE_LAGS_BY_GROUP`) can assign different lag structures per BU, allowing targeted pipeline features for volatile series.
- Exogenous columns recorded in the result file (field `Exogenous_Used`) capture the exact feature set used by each forecast.

### Outputs
- Primary forecast export: `analysis_outputs/sarima_forecast_results.csv` containing columns  
  `Product_Group`, `BU`, `BU_Name`, `Model_Type`, `Model_Used`, `SARIMA_Params`, `Exogenous_Used`, `Date`, `Forecast`, `Lower_CI`, `Upper_CI`, `MAE%`, `Bias%`, `RMSE`, `MAPE`, `CoV`.
- Optional plots are written to `plots/` when `OUTPUT_PLOTS` is enabled.
- Console logging prints seasonal strength, stationarity results, fitted model orders, evaluation metrics, and a final progress summary.

## Supporting scripts
- `auto_select_orders.py`: prompts for product groups and writes SARIMA order diagnostics to `analysis_outputs/sarima_order_selection_results.csv`, including Ljung-Box residual p-values and recommended `(p, d, q, P, D, Q, m)` tuples.
- `sarima_auto_selector.py`: underlying engine for order selection; can be imported for custom workflows.
- `analyze_salesforce_lags.py`: sweeps Salesforce feature lags per group/BU to identify lag values that minimize holdout MAE.
- `analysis_utils.py`: shared helpers that align managed data sources, map SKUs to catalog keys, and normalise numeric Salesforce features for ad-hoc analysis.
- `check_stationarity.py`: runs ADF and KPSS tests over Salesforce features (and optional actuals) to flag non-stationary signals.
- `check_multicollinearity.py`: computes variance inflation factors (VIF) for exogenous features to highlight redundant predictors.
- `check_multicollinearity.py` and `check_stationarity.py` both rely on `analysis_utils.load_prepared_dataset` for consistent alignment.
- `backlog_integration.py` / `salesforce_integration.py`: reusable integration layers for crafting exogenous inputs; they can be called from experiments in the `experiments/` directory.

## Usage
Run the main forecasting pipeline:

```bash
python main.py
```

Typical development workflow:
1. Ensure dependencies are installed (Python 3.9+; `pip install -r requirements.txt` inside a virtual environment).
2. Update managed data files and catalog entries.
3. (Optional) Run `python auto_select_orders.py` to refresh SARIMA orders and paste the suggested tuples into `config.SARIMA_PARAMS`.
4. Execute `python main.py` to regenerate forecasts.

## Development tips
- Validate syntax: `python -m py_compile *.py`.
- Spot-check the forecasting log (stdout) for seasonality/stationarity warnings or fallback activations.
- Forecast outputs are idempotent; rerunning with unchanged inputs overwrites the latest CSV in `analysis_outputs/`.

## License
Internal Midmark Corporation prototype. Contact the maintainers before public distribution.
