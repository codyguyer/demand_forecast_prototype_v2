"""
Automatic SARIMA order selection across multiple demand series.

The selector follows the workflow requested by Cody:
  * Clean monthly series, fill internal gaps with zeros
  * Evaluate a compact SARIMA grid with expanding-window CV (h=1)
  * Rank candidates by MAE, tie-break with AICc while preferring simpler orders
  * Persist diagnostics including warning flags and Ljung-Box residual tests
  * Fallback to a simple Theta/seasonal naive blend when SARIMA is not viable
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple
import warnings

import numpy as np
import pandas as pd
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tools.sm_exceptions import ConvergenceWarning
from statsmodels.stats.diagnostic import acorr_ljungbox
from statsmodels.tsa.forecasting.theta import ThetaModel


@dataclass
class FoldResult:
    """Metrics and warnings per expanding-window fold."""

    fold: int
    train_end: pd.Timestamp
    mae: float
    actual: float
    forecast: float
    converged: bool
    warnings: List[str] = field(default_factory=list)


@dataclass
class CandidateResult:
    """Aggregated metrics for a SARIMA order candidate."""

    order: Tuple[int, int, int]
    seasonal_order: Tuple[int, int, int, int]
    mean_mae_h1: float
    mean_mae_h3: float
    mae_std_h1: float
    mae_std_h3: float
    aicc: float
    folds_h1: List[FoldResult]
    folds_h3: List[FoldResult]
    warning_messages: List[str]
    full_fit: Any  # statsmodels SARIMAXResults; kept for diagnostics

    @property
    def complexity(self) -> int:
        p, d, q = self.order
        P, D, Q, _ = self.seasonal_order
        return p + q + P + Q + d + D


@dataclass
class CandidateInitialScore:
    """Initial fit metrics used to shortlist SARIMA candidates."""

    order: Tuple[int, int, int]
    seasonal_order: Tuple[int, int, int, int]
    aicc: float
    pseudo_mae_last: float
    warnings: List[str]
    full_fit: Any


@dataclass
class SeriesSelectionResult:
    """Final selection payload per product series."""

    series_key: str
    group: str
    bu: Optional[str]
    status: str
    best_order: Optional[Tuple[int, int, int]] = None
    best_seasonal_order: Optional[Tuple[int, int, int, int]] = None
    mean_mae_h1: Optional[float] = None
    mae_h3: Optional[float] = None
    aicc: Optional[float] = None
    ljung_box_pvalue: Optional[float] = None
    warning_messages: List[str] = field(default_factory=list)
    fold_metrics: List[FoldResult] = field(default_factory=list)
    fallback_model: Optional[str] = None
    volatility_class: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Compact dict for DataFrame export."""
        return {
            "series_key": self.series_key,
            "group": self.group,
            "bu": self.bu or "",
            "status": self.status,
            "order": self.best_order or (),
            "seasonal_order": self.best_seasonal_order or (),
            "mean_mae_h1": self.mean_mae_h1,
            "mae_h3": self.mae_h3,
            "aicc": self.aicc,
            "ljung_box_pvalue": self.ljung_box_pvalue,
            "fallback_model": self.fallback_model or "",
            "volatility_class": self.volatility_class or "",
            "warnings": " | ".join(self.warning_messages),
        }


class SARIMAOrderSelector:
    """Coordinate SARIMA order selection across multiple demand series."""

    def __init__(
        self,
        data: pd.DataFrame,
        *,
        product_groups: Dict[str, Sequence[str]],
        forecast_by_bu: bool = False,
        season_length: Optional[int] = None,
        product_col: str = "Product",
        date_col: str = "Month",
        value_col: str = "Actuals",
        bu_col: str = "BU",
        burn_in: int = 24,
        min_folds: int = 5,
        max_folds: int = 6,
        horizon: int = 1,
        step: int = 1,
        maxiter: int = 200,
        max_retries: int = 2,
        enforce_stationarity: bool = True,
        enforce_invertibility: bool = True,
        ljung_box_lags: Sequence[int] = (12,),
        fallback_theta_period: Optional[int] = 12,
        volatility_threshold: float = 0.5,
        shortlist_top_k: int = 3,
        shortlist_extended_k: int = 5,
        shortlist_extended_min_obs: int = 48,
    ) -> None:
        self.data = data.copy()
        self.product_groups = self._normalize_product_groups(product_groups)
        self.forecast_by_bu = forecast_by_bu
        self.product_col = product_col
        self.date_col = date_col
        self.value_col = value_col
        self.bu_col = bu_col
        self.burn_in = burn_in
        self.horizon = horizon
        self.step = step
        self.min_folds = min_folds
        self.max_folds = max_folds
        self.maxiter = maxiter
        self.max_retries = max_retries
        self.enforce_stationarity = enforce_stationarity
        self.enforce_invertibility = enforce_invertibility
        self.ljung_box_lags = tuple(ljung_box_lags)
        self.fallback_theta_period = fallback_theta_period
        self.volatility_threshold = volatility_threshold
        self.shortlist_top_k = max(1, int(shortlist_top_k))
        self.shortlist_extended_k = max(self.shortlist_top_k, int(shortlist_extended_k))
        self.shortlist_extended_min_obs = max(1, int(shortlist_extended_min_obs))

        self._prepare_data()

        # Assume monthly cadence; infer if not specified
        if season_length:
            self.season_length = season_length
        else:
            self.season_length = self._infer_season_length()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def run(self) -> List[SeriesSelectionResult]:
        """Execute order selection for all configured product series."""
        results: List[SeriesSelectionResult] = []
        series_iterable = self._iter_series()

        for series_key, series in series_iterable:
            result = self._process_single_series(series_key, series)
            results.append(result)

        return results

    def results_to_dataframe(self, results: Iterable[SeriesSelectionResult]) -> pd.DataFrame:
        """Flatten selection results into a tabular DataFrame."""
        return pd.DataFrame([res.to_dict() for res in results])

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _normalize_product_groups(
        self, product_groups: Dict[str, Sequence[str]]
    ) -> Dict[str, List[str]]:
        """Strip whitespace from catalog SKUs to ensure matching against Actuals."""
        normalized: Dict[str, List[str]] = {}
        for group_name, items in product_groups.items():
            cleaned_items: List[str] = []
            for item in items:
                if item is None:
                    continue
                text = str(item).strip()
                if text:
                    cleaned_items.append(text)
            if cleaned_items:
                normalized[group_name] = cleaned_items
            else:
                normalized[group_name] = []
        return normalized

    def _prepare_data(self) -> None:
        """Ensure date column is datetime and values are numeric."""
        if not np.issubdtype(self.data[self.date_col].dtype, np.datetime64):
            raw_dates = self.data[self.date_col]
            parsed_dates = pd.to_datetime(raw_dates, errors="coerce")
            if parsed_dates.isna().any():
                numeric_mask = pd.to_numeric(raw_dates, errors="coerce").notna()
                if numeric_mask.any():
                    excel_converted = pd.to_datetime(
                        pd.to_numeric(raw_dates[numeric_mask]),
                        unit="D",
                        origin="1899-12-30",
                    )
                    parsed_dates.loc[numeric_mask] = excel_converted

            if parsed_dates.isna().any():
                bad_examples = list(map(str, raw_dates[parsed_dates.isna()].unique()[:5]))
                raise ValueError(
                    f"Unable to parse {self.date_col!r} values into datetimes; examples: {', '.join(bad_examples)}"
                )

            self.data[self.date_col] = parsed_dates

        self.data = self.data.sort_values([self.product_col, self.date_col])
        self.data[self.value_col] = pd.to_numeric(self.data[self.value_col], errors="coerce").fillna(0.0)
        if self.product_col in self.data.columns:
            normalized_products = (
                self.data[self.product_col]
                .astype(str)
                .str.strip()
            )
            normalized_products = normalized_products.replace({"nan": "", "None": ""})
            self.data[self.product_col] = normalized_products
        if self.forecast_by_bu and self.bu_col in self.data.columns:
            normalized_bu = (
                self.data[self.bu_col]
                .astype(str)
                .str.strip()
            )
            normalized_bu = normalized_bu.replace({"nan": "", "None": ""})
            self.data[self.bu_col] = normalized_bu

    def _infer_season_length(self) -> int:
        """Infer seasonality from month spacing (defaults to annual seasonality)."""
        if self.date_col not in self.data.columns:
            return 12
        sorted_dates = self.data[self.date_col].sort_values()
        if sorted_dates.empty:
            return 12
        deltas = sorted_dates.diff().dropna().value_counts()
        if deltas.empty:
            return 12
        most_common_delta = deltas.index[0]
        if most_common_delta.components.days in (28, 29, 30, 31):
            return 12  # monthly cadence
        # Fallback: assume annual seasonality for safety
        return 12

    def _iter_series(self) -> Iterable[Tuple[Tuple[str, Optional[str]], pd.Series]]:
        """Yield (series_key, series) pairs based on product grouping."""
        for group_name, product_ids in self.product_groups.items():
            mask = self.data[self.product_col].isin(product_ids)
            group_df = self.data.loc[mask]
            if group_df.empty:
                continue
            if self.forecast_by_bu and self.bu_col in group_df.columns:
                for bu_code, bu_df in group_df.groupby(self.bu_col):
                    series = self._build_series(bu_df)
                    if series is not None:
                        yield (group_name, bu_code), series
            else:
                series = self._build_series(group_df)
                if series is not None:
                    yield (group_name, None), series

    def _build_series(self, df: pd.DataFrame) -> Optional[pd.Series]:
        """Aggregate to a monthly series, filling missing months with zeros."""
        if df.empty:
            return None
        monthly = (
            df.groupby(self.date_col)[self.value_col]
            .sum()
            .sort_index()
        )
        if monthly.empty:
            return None
        start, end = monthly.index.min(), monthly.index.max()
        full_index = pd.date_range(start=start, end=end, freq="MS")
        series = monthly.reindex(full_index, fill_value=0.0)
        return series.astype(float)

    def _process_single_series(
        self,
        series_key: Tuple[str, Optional[str]],
        series: pd.Series,
    ) -> SeriesSelectionResult:
        """Evaluate the SARIMA grid for a single demand series."""
        group_name, bu_code = series_key

        is_volatile = self._is_series_volatile(series)

        if len(series) < max(self.burn_in, 12):
            fallback = self._fallback_forecast(series, series_key, reason="short_history")
            return fallback

        candidates = self._generate_candidates(len(series), is_volatile=is_volatile)
        initial_scores: List[CandidateInitialScore] = []

        for order, seasonal_order in candidates:
            initial_score = self._initial_candidate_score(series, order, seasonal_order)
            if initial_score:
                initial_scores.append(initial_score)

        if not initial_scores:
            fallback = self._fallback_forecast(series, series_key, reason="initial_screen_failed")
            return fallback

        shortlist_limit = self._determine_shortlist_size(len(series), is_volatile=is_volatile)
        initial_scores.sort(key=self._candidate_initial_sort_key)
        shortlisted = initial_scores[:shortlist_limit]

        candidate_results: List[CandidateResult] = []
        for shortlisted_score in shortlisted:
            candidate_result = self._evaluate_candidate(
                series,
                shortlisted_score.order,
                shortlisted_score.seasonal_order,
                is_volatile=is_volatile,
                prefit=shortlisted_score,
            )
            if candidate_result:
                candidate_results.append(candidate_result)

        if not candidate_results:
            fallback = self._fallback_forecast(series, series_key, reason="shortlist_evaluation_failed")
            return fallback

        best_candidate = self._select_best_candidate(candidate_results, is_volatile=is_volatile)
        best_result = self._build_series_result(
            series=series,
            series_key=series_key,
            candidate=best_candidate,
            is_volatile=is_volatile,
        )
        return best_result

    def _generate_candidates(
        self,
        n_obs: int,
        *,
        is_volatile: bool,
    ) -> List[Tuple[Tuple[int, int, int], Tuple[int, int, int, int]]]:
        """Construct a compact SARIMA search space with short-series safeguards."""
        pdq_grid: List[Tuple[int, int, int]] = []

        def add_pdq(order: Tuple[int, int, int]) -> None:
            if order != (0, 0, 0) and order not in pdq_grid:
                pdq_grid.append(order)

        core_orders = [
            (0, 0, 0),
            (1, 0, 0),
            (0, 0, 1),
            (1, 0, 1),
        ]
        extended_orders = [
            (2, 0, 0),
            (0, 0, 2),
        ]
        top_orders = [
            (2, 0, 1),
            (1, 0, 2),
            (2, 0, 2),
        ]

        volatility_orders = [
            (1, 1, 0),
            (0, 1, 1),
            (1, 1, 1),
            (2, 1, 1),
            (1, 1, 2),
            (2, 1, 2),
        ]

        for d in (0, 1):
            for base in core_orders:
                add_pdq((base[0], d, base[2]))
            if n_obs >= 48 or (is_volatile and n_obs >= 32):
                for ext in extended_orders:
                    add_pdq((ext[0], d, ext[2]))
            if n_obs >= 60 or (is_volatile and n_obs >= 36):
                for top in top_orders:
                    add_pdq((top[0], d, top[2]))

        if is_volatile:
            for extra in volatility_orders:
                add_pdq(extra)

        seasonal_grid: List[Tuple[int, int, int, int]] = []

        def add_seasonal(order: Tuple[int, int, int, int]) -> None:
            if order not in seasonal_grid:
                seasonal_grid.append(order)

        add_seasonal((0, 0, 0, self.season_length))

        has_one_season = n_obs >= self.season_length
        has_two_seasons = n_obs >= 2 * self.season_length
        has_three_seasons = n_obs >= 3 * self.season_length
        has_four_seasons = n_obs >= 4 * self.season_length
        limited_history = not has_four_seasons

        if has_two_seasons:
            add_seasonal((0, 1, 0, self.season_length))

        if has_two_seasons:
            add_seasonal((1, 0, 0, self.season_length))
            add_seasonal((0, 0, 1, self.season_length))

        if is_volatile and has_two_seasons:
            add_seasonal((0, 1, 1, self.season_length))

        if has_three_seasons and not limited_history:
            add_seasonal((1, 0, 1, self.season_length))

        if is_volatile and has_four_seasons:
            add_seasonal((1, 1, 0, self.season_length))

        if has_four_seasons:
            add_seasonal((1, 1, 1, self.season_length))

        seen_pairs: set[Tuple[Tuple[int, int, int], Tuple[int, int, int, int]]] = set()
        matched: List[Tuple[Tuple[int, int, int], Tuple[int, int, int, int]]] = []
        unmatched: List[Tuple[Tuple[int, int, int], Tuple[int, int, int, int]]] = []

        for order in pdq_grid:
            for seasonal in seasonal_grid:
                pair = (order, seasonal)
                if pair in seen_pairs:
                    continue
                seen_pairs.add(pair)
                if order[1] == seasonal[1]:
                    matched.append(pair)
                else:
                    unmatched.append(pair)

        def candidate_priority(
            pair: Tuple[Tuple[int, int, int], Tuple[int, int, int, int]]
        ) -> Tuple[int, Tuple[int, int, int], Tuple[int, int, int]]:
            order, seasonal = pair
            return (
                sum(order) + sum(seasonal[:3]),
                order,
                seasonal[:3],
            )

        matched.sort(key=candidate_priority)
        unmatched.sort(key=candidate_priority)

        if is_volatile:
            preferred_pairs: List[Tuple[Tuple[int, int, int], Tuple[int, int, int, int]]] = []
            seasonal_priority = [
                (0, 1, 1, self.season_length),
                (1, 1, 0, self.season_length),
                (1, 1, 1, self.season_length),
            ]
            order_priority = [
                (1, 1, 0),
                (0, 1, 1),
                (1, 1, 1),
                (2, 1, 1),
            ]
            for seasonal in seasonal_priority:
                for order in order_priority:
                    preferred_pairs.append((order, seasonal))
            prioritized: List[Tuple[Tuple[int, int, int], Tuple[int, int, int, int]]] = []
            for pair in preferred_pairs:
                if pair in matched:
                    matched.remove(pair)
                    prioritized.append(pair)
            matched = prioritized + matched
            return matched

        candidates = matched + unmatched
        return candidates

    def _determine_shortlist_size(self, n_obs: int, *, is_volatile: bool) -> int:
        """Determine how many candidates to keep for cross-validation."""
        if is_volatile or n_obs < self.shortlist_extended_min_obs:
            return self.shortlist_extended_k
        return self.shortlist_top_k

    @staticmethod
    def _candidate_initial_sort_key(score: CandidateInitialScore) -> Tuple[Any, ...]:
        """Ordering key for initial candidate screen."""
        aicc = score.aicc if np.isfinite(score.aicc) else float("inf")
        pseudo_mae = score.pseudo_mae_last if np.isfinite(score.pseudo_mae_last) else float("inf")
        complexity = sum(score.order) + sum(score.seasonal_order[:3])
        return (aicc, pseudo_mae, complexity, score.order, score.seasonal_order[:3])

    def _initial_candidate_score(
        self,
        series: pd.Series,
        order: Tuple[int, int, int],
        seasonal_order: Tuple[int, int, int, int],
    ) -> Optional[CandidateInitialScore]:
        """Fit a candidate once to collect metrics for shortlist ranking."""
        full_fit, warnings_list = self._fit_sarimax(series, order, seasonal_order)
        if full_fit is None:
            return None

        if not bool(full_fit.mle_retvals.get("converged", False)):
            return None

        aicc = getattr(full_fit, "aicc", np.nan)
        aicc_value = float(aicc) if np.isfinite(aicc) else float("inf")
        pseudo_mae = self._pseudo_mae_last(full_fit)

        return CandidateInitialScore(
            order=order,
            seasonal_order=seasonal_order,
            aicc=aicc_value,
            pseudo_mae_last=pseudo_mae,
            warnings=list(warnings_list),
            full_fit=full_fit,
        )

    @staticmethod
    def _pseudo_mae_last(model_fit: Any) -> float:
        """Approximate 1-step pseudo out-of-sample MAE using last filter residual."""
        try:
            residuals = np.asarray(model_fit.resid)
            if residuals.size == 0:
                return float("inf")
            last_residual = residuals[-1]
            if not np.isfinite(last_residual):
                return float("inf")
            return float(abs(last_residual))
        except Exception:
            return float("inf")

    def _evaluate_candidate(
        self,
        series: pd.Series,
        order: Tuple[int, int, int],
        seasonal_order: Tuple[int, int, int, int],
        *,
        is_volatile: bool,
        prefit: Optional[CandidateInitialScore] = None,
    ) -> Optional[CandidateResult]:
        """Evaluate a single SARIMA candidate via expanding-window CV."""
        folds_h1 = self._expanding_window_cv(series, order, seasonal_order)
        if not folds_h1:
            return None

        mae_values_h1 = [fold.mae for fold in folds_h1]
        mean_mae_h1 = float(np.mean(mae_values_h1))
        mae_std_h1 = float(np.std(mae_values_h1)) if len(mae_values_h1) > 1 else 0.0

        if is_volatile:
            folds_h3 = self._expanding_window_cv(series, order, seasonal_order, horizon=3)
            if folds_h3:
                mae_values_h3 = [fold.mae for fold in folds_h3]
                mean_mae_h3 = float(np.mean(mae_values_h3))
                mae_std_h3 = float(np.std(mae_values_h3)) if len(mae_values_h3) > 1 else 0.0
            else:
                # Penalise candidates that cannot support the longer horizon
                mean_mae_h3 = float("inf")
                mae_std_h3 = float("inf")
                folds_h3 = []
        else:
            folds_h3 = []
            mean_mae_h3 = np.nan
            mae_std_h3 = np.nan

        if prefit and prefit.full_fit is not None:
            full_fit = prefit.full_fit
            if not bool(full_fit.mle_retvals.get("converged", False)):
                return None
            warnings_full = list(prefit.warnings)
        else:
            full_fit, warnings_full = self._fit_sarimax(series, order, seasonal_order)
            if full_fit is None or full_fit.mle_retvals.get("converged", False) is False:
                return None

        warning_messages = [msg for fold in folds_h1 for msg in fold.warnings]
        if folds_h3:
            warning_messages.extend(msg for fold in folds_h3 for msg in fold.warnings)
        warning_messages.extend(warnings_full)
        warning_messages = list(dict.fromkeys(warning_messages))

        aicc = getattr(full_fit, "aicc", np.nan)

        return CandidateResult(
            order=order,
            seasonal_order=seasonal_order,
            mean_mae_h1=mean_mae_h1,
            mean_mae_h3=mean_mae_h3,
            mae_std_h1=mae_std_h1,
            mae_std_h3=mae_std_h3,
            aicc=float(aicc) if np.isfinite(aicc) else np.nan,
            folds_h1=folds_h1,
            folds_h3=folds_h3,
            warning_messages=warning_messages,
            full_fit=full_fit,
        )

    def _expanding_window_cv(
        self,
        series: pd.Series,
        order: Tuple[int, int, int],
        seasonal_order: Tuple[int, int, int, int],
        horizon: Optional[int] = None,
    ) -> List[FoldResult]:
        """Run expanding-window backtests for a candidate."""
        n_obs = len(series)
        fold_results: List[FoldResult] = []

        cv_horizon = horizon or self.horizon
        last_possible_start = n_obs - cv_horizon
        if last_possible_start <= self.burn_in:
            return fold_results

        fold_end_indices = list(range(self.burn_in, last_possible_start + 1, self.step))
        if not fold_end_indices:
            return fold_results
        if len(fold_end_indices) > self.max_folds:
            fold_end_indices = fold_end_indices[-self.max_folds :]

        for fold_num, train_end_idx in enumerate(fold_end_indices, start=1):
            train_slice = series.iloc[:train_end_idx]
            test_slice = series.iloc[train_end_idx : train_end_idx + cv_horizon]

            if test_slice.empty:
                continue

            model_fit, warnings_list = self._fit_sarimax(train_slice, order, seasonal_order)
            if model_fit is None:
                return []  # Bail out on full candidate if any fold cannot fit

            forecast = model_fit.get_forecast(steps=cv_horizon)
            predicted = forecast.predicted_mean.iloc[: len(test_slice)]
            mae = float(np.mean(np.abs(predicted.values - test_slice.values)))

            fold_results.append(
                FoldResult(
                    fold=fold_num,
                    train_end=train_slice.index[-1],
                    mae=mae,
                    actual=float(test_slice.iloc[-1]),
                    forecast=float(predicted.iloc[-1]),
                    converged=bool(model_fit.mle_retvals.get("converged", False)),
                    warnings=warnings_list,
                )
            )

        if len(fold_results) < min(self.min_folds, len(fold_end_indices)):
            return []

        return fold_results

    def _fit_sarimax(
        self,
        series: pd.Series,
        order: Tuple[int, int, int],
        seasonal_order: Tuple[int, int, int, int],
    ) -> Tuple[Optional[Any], List[str]]:
        """Fit SARIMAX with retries and capture warnings."""
        warnings_accumulator: List[str] = []

        for attempt in range(1, self.max_retries + 1):
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                try:
                    model = SARIMAX(
                        series,
                        order=order,
                        seasonal_order=seasonal_order,
                        trend='c',
                        enforce_stationarity=self.enforce_stationarity,
                        enforce_invertibility=self.enforce_invertibility,
                        use_exact_diffuse=True,
                    )
                    fit_result = model.fit(
                        disp=False,
                        maxiter=self.maxiter,
                    )
                except (ValueError, np.linalg.LinAlgError) as exc:
                    warnings_accumulator.append(f"fit_error: {exc}")
                    fit_result = None

            for warning in caught:
                msg = str(warning.message)
                warnings_accumulator.append(msg)

            if fit_result is not None and fit_result.mle_retvals.get("converged", False):
                return fit_result, warnings_accumulator

        return None, warnings_accumulator

    def _select_best_candidate(
        self,
        candidates: List[CandidateResult],
        *,
        is_volatile: bool,
    ) -> CandidateResult:
        """Rank candidates with volatility-aware MAE priorities."""

        def primary_mae(cand: CandidateResult) -> float:
            if is_volatile and np.isfinite(cand.mean_mae_h3):
                return cand.mean_mae_h3
            return cand.mean_mae_h1

        def blended_mae(cand: CandidateResult) -> float:
            mae_h3 = cand.mean_mae_h3 if np.isfinite(cand.mean_mae_h3) else cand.mean_mae_h1
            return 0.5 * (cand.mean_mae_h1 + mae_h3)

        candidates_sorted = sorted(
            candidates,
            key=lambda cand: (
                primary_mae(cand),
                blended_mae(cand),
                np.inf if cand.aicc is None else cand.aicc,
                cand.complexity,
            ),
        )
        return candidates_sorted[0]

    def _build_series_result(
        self,
        series: pd.Series,
        series_key: Tuple[str, Optional[str]],
        candidate: CandidateResult,
        is_volatile: bool,
    ) -> SeriesSelectionResult:
        """Assemble final metrics and diagnostics for the winning candidate."""
        group_name, bu_code = series_key
        unique_warnings = list(dict.fromkeys(candidate.warning_messages))
        result = SeriesSelectionResult(
            series_key=self._format_series_key(group_name, bu_code),
            group=group_name,
            bu=bu_code,
            status="sarima_selected",
            best_order=candidate.order,
            best_seasonal_order=candidate.seasonal_order,
            mean_mae_h1=candidate.mean_mae_h1,
            aicc=candidate.aicc,
            warning_messages=unique_warnings,
            fold_metrics=candidate.folds_h1,
            volatility_class="volatile" if is_volatile else "stable",
        )

        # Residual diagnostics on full fit
        residual_pvalue = self._ljung_box(candidate.full_fit)
        result.ljung_box_pvalue = residual_pvalue

        # Retain meta h=3 MAE from candidate evaluation
        result.mae_h3 = candidate.mean_mae_h3 if np.isfinite(candidate.mean_mae_h3) else None

        return result

    def _ljung_box(self, model_fit: Any) -> Optional[float]:
        """Compute Ljung-Box p-value at requested seasonal lags."""
        try:
            residuals = model_fit.resid
            residuals = residuals[np.isfinite(residuals)]
            if residuals.size == 0:
                return None
            lb = acorr_ljungbox(residuals, lags=list(self.ljung_box_lags), return_df=True)
            # Return worst-case (minimum) p-value across requested lags
            pvalue = float(lb["lb_pvalue"].min())
            return pvalue
        except Exception as exc:
            return None

    def _fallback_forecast(
        self,
        series: pd.Series,
        series_key: Tuple[str, Optional[str]],
        reason: str,
    ) -> SeriesSelectionResult:
        """Provide a fallback result when SARIMA selection is not viable."""
        group_name, bu_code = series_key
        key_label = self._format_series_key(group_name, bu_code)

        fallback_name = "Theta"
        mae_h1 = None
        mae_h3 = None
        warnings_list: List[str] = ["sarima_unavailable", reason]

        try:
            theta = ThetaModel(series, period=self.fallback_theta_period or self.season_length)
            theta_fit = theta.fit()
            forecast = theta_fit.forecast(max(self.horizon, 1))
            actual = series.iloc[-self.horizon :] if self.horizon > 0 else pd.Series(dtype=float)
            align_len = min(len(actual), len(forecast))
            if align_len > 0:
                mae_h1 = float(np.mean(np.abs(forecast.iloc[-align_len:].values - actual.iloc[-align_len:].values)))

            # Multi-step MAE
            forecast_h3 = theta_fit.forecast(3)
            actual_h3 = series.iloc[-3:]
            align_h3 = min(len(actual_h3), len(forecast_h3))
            if align_h3 > 0:
                mae_h3 = float(np.mean(np.abs(forecast_h3.iloc[-align_h3:].values - actual_h3.iloc[-align_h3:].values)))
        except Exception as exc:
            fallback_name = "SeasonalNaive"
            seasonal_lag = self.season_length
            if len(series) > seasonal_lag:
                naive_forecast = series.shift(seasonal_lag).iloc[-self.horizon :] if self.horizon > 0 else pd.Series(dtype=float)
                actual = series.iloc[-self.horizon :] if self.horizon > 0 else pd.Series(dtype=float)
                align_len = min(len(naive_forecast), len(actual))
                if align_len > 0:
                    mae_h1 = float(np.mean(np.abs(naive_forecast.iloc[-align_len:].values - actual.iloc[-align_len:].values)))
                naive_h3 = series.shift(seasonal_lag).iloc[-3:]
                actual_h3 = series.iloc[-3:]
                align_h3 = min(len(naive_h3), len(actual_h3))
                if align_h3 > 0:
                    mae_h3 = float(np.mean(np.abs(naive_h3.iloc[-align_h3:].values - actual_h3.iloc[-align_h3:].values)))
            warnings_list.append(f"fallback_error: {exc}")

        return SeriesSelectionResult(
            series_key=key_label,
            group=group_name,
            bu=bu_code,
            status="fallback",
            fallback_model=fallback_name,
            mean_mae_h1=mae_h1,
            mae_h3=mae_h3,
            warning_messages=warnings_list,
        )

    @staticmethod
    def _format_series_key(group_name: str, bu_code: Optional[str]) -> str:
        return f"{group_name}__{bu_code}" if bu_code else group_name

    def _is_series_volatile(self, series: pd.Series) -> bool:
        """Classify a series as volatile using coefficient of variation."""
        if series.empty:
            return False
        mean = float(series.mean())
        std = float(series.std())
        if mean <= 0:
            return True
        cv = std / mean
        return cv >= self.volatility_threshold


def run_order_selection(
    *,
    data_path: str,
    product_groups: Dict[str, Sequence[str]],
    forecast_by_bu: bool = False,
    output_path: Optional[str] = None,
) -> pd.DataFrame:
    """Convenience wrapper to execute SARIMA order selection end-to-end."""
    df = pd.read_csv(data_path)
    selector = SARIMAOrderSelector(
        df,
        product_groups=product_groups,
        forecast_by_bu=forecast_by_bu,
    )
    results = selector.run()
    results_df = selector.results_to_dataframe(results)
    if output_path:
        results_df.to_csv(output_path, index=False)
    return results_df
