"""
SARIMA Model Configuration
Easily adjustable parameters for forecasting models
"""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple, Union


class SARIMAConfig:
    """Configuration class for SARIMA model parameters backed by a managed catalog."""

    BASE_DIR = Path(__file__).resolve().parent
    MANAGED_DATA_DIR = BASE_DIR / "managed_data"
    MANAGED_CATALOG_FILE = MANAGED_DATA_DIR / "product_catalog.csv"
    ORDER_COLUMN_NAMES: Tuple[str, ...] = (
        "sarima_p",
        "sarima_d",
        "sarima_q",
        "sarima_P",
        "sarima_D",
        "sarima_Q",
        "sarima_s",
    )
    REQUIRED_CATALOG_COLUMNS: Tuple[str, ...] = (
        "group_key",
        "sku_list",
        "business_unit_code",
        "business_unit_name",
        "salesforce_feature_mode",
    ) + ORDER_COLUMN_NAMES

    # Data configuration
    DATA_FILE = str(MANAGED_DATA_DIR / "Actuals.csv")
    FORECAST_MONTHS = 12
    CURRENT_MONTH = "2025-09"  # Use last fully completed month
    USE_SALESFORCE = False
    SALESFORCE_DATA_FILE = str(MANAGED_DATA_DIR / "salesforce_data_v2.csv")
    SALESFORCE_REFERENCE_FILE = str(MANAGED_DATA_DIR / "sf_product_reference_key.csv")
    USE_BACKLOG = False  # [ln(t)-ln(t-1)]*0.1 or (z-mean)/std and do not include any future months
    BACKLOG_DATA_FILE = str(MANAGED_DATA_DIR / "Backlog.csv")
    FORECAST_BY_BU = True

    # Managed metadata containers populated at runtime
    BUSINESS_UNITS: Sequence[str] = ()
    BU_CODE_TO_NAME: Dict[str, str] = {}
    BU_NAME_TO_CODE: Dict[str, str] = {}
    PRODUCT_GROUPS: Dict[str, List[str]] = {}
    SARIMA_PARAMS: Dict[str, Tuple[int, int, int, int, int, int, int]] = {}
    SALESFORCE_FEATURE_MODE_BY_GROUP: Dict[str, str] = {}
    GROUP_TO_BU_CODES: Dict[str, Tuple[str, ...]] = {}
    SKU_TO_GROUP_MAP: Dict[str, str] = {}

    # Model evaluation parameters
    CONFIDENCE_LEVEL = 0.90  # 90% confidence intervals (approx 1.645 std dev)
    TEST_SPLIT_MONTHS = 6  # Number of months to hold out for testing

    # Future Salesforce integration parameters
    SALESFORCE_FIELDS = {
        "timestamp": "Timestamp",
        "product_name": "SF_Product_Name",
        "business_unit": "BU",
        "revenue": "Revenue",
        "quantity": "Quantity",
        "new_quotes": "New_Quotes",
        "revenue_new_orders": "Revenue_new_orders",
        "quantity_new_orders": "Quantity_new_orders",
    }

    # Salesforce feature controls
    SALESFORCE_FEATURE_COLUMN_SETS = {
        "quantity": ["Quantity", "New_Quotes", "Quantity_new_orders"],
        "dollars": ["Revenue", "New_Quotes", "Revenue_new_orders"],
        "default": ["Quantity", "New_Quotes", "Quantity_new_orders"],
    }
    SALESFORCE_DIFFERENCE_FEATURES = False

    # Backlog feature controls
    BACKLOG_FEATURES = [
        "Backlog",
        "Backlog_Lag1",
        "Backlog_Lag3",
        "Backlog_MA3",
    ]

    ENFORCE_NON_NEGATIVE_FORECASTS = True
    TRIM_TRAILING_ZERO_PADDING = True  # Keep trailing placeholder zeros out while preserving internal zeros

    # Seasonality detection thresholds
    SEASONALITY_THRESHOLD = 0.3  # Minimum seasonal component strength to consider seasonal
    MIN_OBSERVATIONS = 24  # Minimum observations needed for reliable seasonal analysis

    # Output configuration
    OUTPUT_PLOTS = False
    SAVE_RESULTS = True
    RESULTS_FILE = "sarima_forecast_results.csv"
    PLOTS_DIR = "plots"

    # Short-history non-seasonal ARIMA fallback
    SHORT_SERIES_ARIMA_ENABLED = True
    SHORT_SERIES_THRESHOLD = 24
    SHORT_SERIES_ARIMA_DEFAULT_ORDER = (1, 1, 1)
    SHORT_SERIES_ARIMA_PARAMS: Dict[str, Tuple[int, int, int]] = {}

    # Model selection thresholds
    SARIMA_MIN_HISTORY = 36
    ARIMA_MIN_HISTORY = 24

    # Very short-history ETS/Theta blend fallback
    ETS_THETA_ENABLED = False
    ETS_THETA_WEIGHTS = (0.5, 0.5)
    ETS_THETA_PERIOD = 12
    ETS_THETA_TREND = "add"
    ETS_THETA_SEASONAL = None
    ETS_THETA_SEASONAL_PERIODS = None
    ETS_THETA_DAMPED = False
    ETS_THETA_FORCE_GROUPS = None

    def __init__(self) -> None:
        self.PRODUCT_GROUPS = {}
        self.SARIMA_PARAMS = {}
        self.BU_CODE_TO_NAME = {}
        self.BU_NAME_TO_CODE = {}
        self.BUSINESS_UNITS = ()
        self.SALESFORCE_FEATURE_MODE_BY_GROUP = {}
        self.GROUP_TO_BU_CODES = {}
        self.SKU_TO_GROUP_MAP = {}
        self._skipped_groups: List[str] = []
        self.managed_catalog_path = Path(self.MANAGED_CATALOG_FILE)

        self._load_managed_catalog()

        if self._skipped_groups:
            print("Skipping product groups due to incomplete managed catalog rows:")
            for message in self._skipped_groups:
                print(f"  - {message}")

        if not self.PRODUCT_GROUPS:
            print(
                f"Warning: No product groups loaded from {self.managed_catalog_path}. "
                "Forecasting steps will be skipped."
            )

    def _load_managed_catalog(self) -> None:
        """Populate catalog-driven metadata from CSV if available."""
        path = self.managed_catalog_path
        if not path.exists():
            print(f"Warning: Managed catalog file not found at {path}")
            return

        try:
            group_to_bu_codes: Dict[str, List[str]] = {}
            sku_to_group_map: Dict[str, str] = {}

            with path.open(newline="", encoding="utf-8-sig") as handle:
                reader = csv.DictReader(handle)
                if not reader.fieldnames:
                    print(f"Warning: Managed catalog at {path} is missing a header row")
                    return

                missing_headers = [
                    column for column in self.REQUIRED_CATALOG_COLUMNS if column not in reader.fieldnames
                ]
                if missing_headers:
                    raise ValueError(
                        f"Managed catalog missing required column(s): {', '.join(missing_headers)}"
                    )

                product_groups: Dict[str, List[str]] = {}
                sarima_params: Dict[str, Tuple[int, int, int, int, int, int, int]] = {}
                sf_modes: Dict[str, str] = {}
                bu_code_to_name: Dict[str, str] = {}
                bu_order: List[str] = []

                for line_number, row in enumerate(reader, start=2):
                    group_key = (row.get("group_key") or "").strip()
                    label = group_key or f"row {line_number}"

                    missing_values = [
                        column
                        for column in self.REQUIRED_CATALOG_COLUMNS
                        if not self._has_value(row.get(column))
                    ]
                    if missing_values:
                        self._skipped_groups.append(
                            f"{label} (blank columns: {', '.join(missing_values)})"
                        )
                        continue

                    sku_values = self._parse_catalog_list(row.get("sku_list"))
                    if not sku_values:
                        self._skipped_groups.append(f"{label} (no valid SKUs after parsing)")
                        continue

                    try:
                        order = self._parse_sarima_order(row)
                    except ValueError as exc:
                        self._skipped_groups.append(f"{label} ({exc})")
                        continue

                    product_groups[group_key] = sku_values
                    sarima_params[group_key] = order
                    sf_mode = str(row.get("salesforce_feature_mode")).strip()
                    sf_modes[group_key] = sf_mode

                    if sku_values:
                        for sku_entry in sku_values:
                            sku_clean = sku_entry.strip()
                            if sku_clean and sku_clean not in sku_to_group_map:
                                sku_to_group_map[sku_clean] = group_key

                    bu_codes = self._parse_catalog_list(row.get("business_unit_code"))
                    bu_names = self._parse_catalog_list(row.get("business_unit_name"))

                    for index, code in enumerate(bu_codes):
                        code_clean = code.strip()
                        if not code_clean:
                            continue

                        name = bu_names[index] if index < len(bu_names) else ""
                        name_clean = name.strip() if isinstance(name, str) else ""
                        if not name_clean:
                            name_clean = code_clean

                        existing_codes = group_to_bu_codes.setdefault(group_key, [])
                        if code_clean not in existing_codes:
                            existing_codes.append(code_clean)

                        if code_clean not in bu_code_to_name:
                            bu_order.append(code_clean)
                        bu_code_to_name[code_clean] = name_clean

                self.PRODUCT_GROUPS = product_groups
                self.SARIMA_PARAMS = sarima_params
                self.SALESFORCE_FEATURE_MODE_BY_GROUP = sf_modes

                if bu_code_to_name:
                    ordered_mapping = {code: bu_code_to_name[code] for code in bu_order}
                    self.BU_CODE_TO_NAME = ordered_mapping
                    self.BUSINESS_UNITS = tuple(ordered_mapping.keys())
                    self.BU_NAME_TO_CODE = {name: code for code, name in ordered_mapping.items()}
                else:
                    self.BU_CODE_TO_NAME = {}
                    self.BUSINESS_UNITS = ()
                    self.BU_NAME_TO_CODE = {}

                if group_to_bu_codes:
                    self.GROUP_TO_BU_CODES = {
                        group: tuple(codes) for group, codes in group_to_bu_codes.items()
                    }
                else:
                    self.GROUP_TO_BU_CODES = {}

                self.SKU_TO_GROUP_MAP = sku_to_group_map
        except ValueError:
            raise
        except Exception as exc:
            print(f"Warning: Failed to load managed catalog from {path}: {exc}")

    @staticmethod
    def _parse_catalog_list(raw_value: Optional[Union[str, Iterable[str]]]) -> List[str]:
        """
        Accept delimited strings or iterables of values from the managed catalog.
        Supports pipe, semicolon, and newline separated entries while preserving
        intentional leading spaces on product codes.
        """
        if raw_value is None:
            return []

        if isinstance(raw_value, (list, tuple)):
            return [
                SARIMAConfig._preserve_spacing(value)
                for value in raw_value
                if value not in (None, "")
            ]

        text = str(raw_value)
        if text == "":
            return []

        sanitized = text.replace("\r", "")
        for delimiter in ("|", ";", "\n"):
            if delimiter in sanitized:
                parts = sanitized.split(delimiter)
                break
        else:
            return [SARIMAConfig._preserve_spacing(sanitized)]

        return [
            SARIMAConfig._preserve_spacing(part)
            for part in parts
            if part not in (None, "")
        ]

    @staticmethod
    def _preserve_spacing(value: Union[str, int, float]) -> str:
        """Remove only trailing newline or carriage-return characters while keeping leading spaces."""
        if value is None:
            return ""
        if not isinstance(value, str):
            value = str(value)
        return value.replace("\r", "").rstrip("\n")

    @staticmethod
    def _has_value(value: Optional[Union[str, int, float]]) -> bool:
        """Return True when a catalog entry has a non-blank value."""
        if value is None:
            return False
        if isinstance(value, str):
            return bool(value.strip())
        return True

    def _parse_sarima_order(
        self, row: Dict[str, Optional[str]]
    ) -> Tuple[int, int, int, int, int, int, int]:
        """Extract SARIMA order tuple from a catalog row."""
        order_values: List[int] = []
        for column in self.ORDER_COLUMN_NAMES:
            raw = row.get(column) if row else None
            try:
                order_values.append(int(float(raw)))
            except (TypeError, ValueError) as exc:
                raise ValueError(f"invalid SARIMA order value '{raw}' in column '{column}'") from exc

        return tuple(order_values)
