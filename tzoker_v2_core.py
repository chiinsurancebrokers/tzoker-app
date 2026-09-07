"""Tzoker V2 core.

The V2 design separates three different ideas that the original app mixed together:
1) data quality / descriptive history,
2) probability calibration against the mathematically-known fair-game baseline, and
3) ticket/system coverage.

Historical frequency is never presented as a future probability unless it beats the
known baseline on a strictly later holdout sample. If it does not, V2 falls back to the
fair baseline (5/45 for each main number, 1/20 for each Joker number).
"""
from __future__ import annotations

import os
import random
from collections import Counter
from dataclasses import dataclass
from itertools import combinations
from math import comb
from typing import Iterable, Sequence

import numpy as np
import pandas as pd

MAIN_POOL = 45
MAIN_DRAW = 5
JOKER_POOL = 20
MAIN_BASELINE = MAIN_DRAW / MAIN_POOL
JOKER_BASELINE = 1 / JOKER_POOL
TICKET_PRICE_EUR = 1.0

# Known development rows previously mixed into the 2025 workbook.
PLACEHOLDER_ROWS = {
    ("2025-2921", (19, 23, 38, 39, 44), 15),
    ("2025-2922", (1, 3, 20, 24, 43), 18),
    ("2025-2923", (6, 15, 25, 28, 43), 11),
    ("2025-2924", (16, 17, 22, 31, 43), 20),
}

# Current fixed prizes shown by OPAP/Allwyn. Jackpot is variable.
_CATEGORY_SPECS = [
    ("5+1", 5, True, "Jackpot"),
    ("5", 5, False, "€100,000"),
    ("4+1", 4, True, "€2,500"),
    ("4", 4, False, "€50"),
    ("3+1", 3, True, "€50"),
    ("3", 3, False, "€2"),
    ("2+1", 2, True, "€2"),
    ("1+1", 1, True, "€1.50"),
    ("2", 2, False, "€1"),
]


def exact_category_probability(main_matches: int, joker_match: bool) -> float:
    """Probability of an *exact* prize-category event for one 5+1 line."""
    if not 0 <= main_matches <= MAIN_DRAW:
        raise ValueError("main_matches must be in 0..5")
    main_ways = comb(MAIN_DRAW, main_matches) * comb(MAIN_POOL - MAIN_DRAW, MAIN_DRAW - main_matches)
    joker_ways = 1 if joker_match else JOKER_POOL - 1
    return (main_ways * joker_ways) / (comb(MAIN_POOL, MAIN_DRAW) * JOKER_POOL)


def prize_categories() -> list[dict]:
    rows = []
    for key, m, jmatch, prize in _CATEGORY_SPECS:
        p = exact_category_probability(m, jmatch)
        rows.append({
            "key": key,
            "main_matches": m,
            "joker_match": jmatch,
            "probability": p,
            "odds_1_in": 1.0 / p,
            "prize": prize,
        })
    return rows


def _season_for_month(month: int) -> str:
    if month in (12, 1, 2):
        return "Winter"
    if month in (3, 4, 5):
        return "Spring"
    if month in (6, 7, 8):
        return "Summer"
    return "Autumn"


def _header_text(value) -> str:
    return str(value).strip().upper() if isinstance(value, str) else ""


def _parse_date(value):
    if pd.isna(value):
        return None
    try:
        if isinstance(value, str):
            # dayfirst=True handles historical Greek workbook dates without locking
            # the loader to exactly one text format.
            return pd.to_datetime(value, dayfirst=True, errors="raise")
        return pd.to_datetime(value, errors="raise")
    except Exception:
        return None


def _load_year_file(filepath: str, year: int) -> tuple[list[dict], dict]:
    raw = pd.read_excel(filepath, header=None)
    n_rows, n_cols = raw.shape
    scan_rows = min(12, n_rows)

    id_col = date_col = joker_col = None
    header_rows: list[int] = []
    for r in range(scan_rows):
        for c in range(n_cols):
            text = _header_text(raw.iat[r, c])
            if not text:
                continue
            if id_col is None and ("ΚΛΗΡ" in text or "ΔΙΑΓ" in text):
                id_col = c
                header_rows.append(r)
            if date_col is None and "ΗΜ" in text:
                date_col = c
                header_rows.append(r)
            if joker_col is None and "ΤΖΟΚΕΡ" in text:
                joker_col = c
                header_rows.append(r)

    # Historical files are known to use these fallback positions. We keep the
    # fallback, but report it so malformed future files are visible in the UI.
    used_fallback = any(x is None for x in (date_col, joker_col))
    id_col = 0 if id_col is None else id_col
    date_col = 1 if date_col is None else date_col
    joker_col = 7 if joker_col is None else joker_col
    if not (0 <= date_col < n_cols and 0 <= joker_col < n_cols and date_col < joker_col):
        return [], {
            "year": year, "rows": n_rows, "loaded": 0, "skipped": n_rows,
            "invalid_duplicates": 0, "fallback_header": used_fallback,
            "error": "Could not locate date/Joker columns",
        }

    main_cols = list(range(date_col + 1, joker_col))
    header_row = max(header_rows, default=0)
    data_start = header_row + 1
    for r in range(header_row + 1, min(header_row + 10, n_rows)):
        if _parse_date(raw.iat[r, date_col]) is not None:
            data_start = r
            break

    draws: list[dict] = []
    skipped = invalid_duplicates = 0
    for r in range(data_start, n_rows):
        row = raw.iloc[r]
        date = _parse_date(row.iloc[date_col])
        if date is None:
            if not pd.isna(row.iloc[date_col]):
                skipped += 1
            continue

        nums: list[int] = []
        for c in main_cols:
            value = row.iloc[c]
            if pd.isna(value):
                continue
            try:
                number = int(float(value))
            except (TypeError, ValueError):
                continue
            # Spacer columns can contain numeric-looking junk; validate only after
            # collecting the intended five values.
            nums.append(number)

        if len(nums) != MAIN_DRAW or not all(1 <= n <= MAIN_POOL for n in nums):
            skipped += 1
            continue
        if len(set(nums)) != MAIN_DRAW:
            invalid_duplicates += 1
            skipped += 1
            continue

        try:
            joker = int(float(row.iloc[joker_col]))
        except (TypeError, ValueError):
            skipped += 1
            continue
        if not 1 <= joker <= JOKER_POOL:
            skipped += 1
            continue

        id_value = row.iloc[id_col]
        if pd.isna(id_value):
            # Row number prevents two legitimate same-date draws from collapsing.
            local_id = f"row{r + 1}-{date:%Y%m%d}"
        else:
            try:
                local_id = str(int(float(id_value)))
            except (TypeError, ValueError):
                local_id = str(id_value).strip()
        draw_id = f"{year}-{local_id}"
        main_numbers = tuple(sorted(nums))
        if (draw_id, main_numbers, joker) in PLACEHOLDER_ROWS:
            continue

        draws.append({
            "draw_id": draw_id,
            "date": date,
            "year": int(year),
            "main_numbers": list(main_numbers),
            "joker": joker,
            "day_of_week": date.strftime("%A"),
            "season": _season_for_month(date.month),
            **{f"num{i+1}": n for i, n in enumerate(main_numbers)},
        })

    report = {
        "year": year,
        "rows": n_rows,
        "loaded": len(draws),
        "skipped": skipped,
        "invalid_duplicates": invalid_duplicates,
        "fallback_header": used_fallback,
        "error": "",
    }
    return draws, report


def load_all_draws(data_dir: str = ".", start_year: int = 1997, end_year: int = 2026):
    all_draws: list[dict] = []
    reports: list[dict] = []
    for year in range(start_year, end_year + 1):
        path = os.path.join(data_dir, f"Joker_{year}.xlsx")
        if not os.path.exists(path):
            continue
        draws, report = _load_year_file(path, year)
        reports.append(report)
        all_draws.extend(draws)

    if not all_draws:
        # Compatibility fallback: the repository's original loader is already proven
        # against the historical Joker_*.xlsx layouts used by this project.  V2's
        # stricter parser may reject an unfamiliar workbook layout; in that case, use
        # the legacy parser rather than returning an empty app, then re-validate the
        # resulting rows before probability analysis.
        try:
            from tzoker_core import load_all_draws as legacy_load_all_draws
            legacy_df, legacy_report = legacy_load_all_draws(
                data_dir=data_dir, start_year=start_year, end_year=end_year
            )
        except Exception as exc:
            fallback_report = pd.DataFrame(reports)
            if fallback_report.empty:
                fallback_report = pd.DataFrame([{
                    "year": None, "rows": 0, "loaded": 0, "skipped": 0,
                    "invalid_duplicates": 0, "fallback_header": True,
                    "error": f"V2 parser loaded no rows and legacy fallback failed: {exc}",
                }])
            else:
                fallback_report["error"] = fallback_report.get("error", "").astype(str) + \
                    f"; legacy fallback failed: {exc}"
            return pd.DataFrame(), fallback_report

        if legacy_df.empty:
            return legacy_df, legacy_report

        # Re-validate the legacy output so impossible draws cannot enter V2.
        valid_mask = legacy_df["main_numbers"].apply(
            lambda nums: (
                isinstance(nums, (list, tuple))
                and len(nums) == MAIN_DRAW
                and len(set(int(n) for n in nums)) == MAIN_DRAW
                and all(1 <= int(n) <= MAIN_POOL for n in nums)
            )
        ) & legacy_df["joker"].apply(lambda j: 1 <= int(j) <= JOKER_POOL)
        invalid_count = int((~valid_mask).sum())
        legacy_df = legacy_df.loc[valid_mask].copy()
        legacy_df = legacy_df.sort_values(["date", "draw_id"]).reset_index(drop=True)

        # Normalize the report columns expected by the V2 Data Quality page.
        legacy_report = legacy_report.copy()
        if "draws" in legacy_report.columns and "loaded" not in legacy_report.columns:
            legacy_report = legacy_report.rename(columns={"draws": "loaded"})
        if "skipped_rows" in legacy_report.columns and "skipped" not in legacy_report.columns:
            legacy_report = legacy_report.rename(columns={"skipped_rows": "skipped"})
        legacy_report["invalid_duplicates"] = 0
        if len(legacy_report):
            legacy_report.loc[legacy_report.index[-1], "invalid_duplicates"] = invalid_count
        legacy_report["fallback_header"] = True
        legacy_report["error"] = ""
        legacy_report["loader"] = "legacy_compatibility_fallback"
        return legacy_df, legacy_report

    df = pd.DataFrame(all_draws).sort_values(["date", "draw_id"]).reset_index(drop=True)
    before = len(df)
    # De-duplicate only by namespaced draw_id. Two separate draws are allowed to
    # share a date (and, however unlikely, could even share the same outcome).
    # Outcome-based de-duplication can silently erase a legitimate draw.
    df = df.drop_duplicates(subset=["draw_id"], keep="first").reset_index(drop=True)
    removed = before - len(df)
    if reports:
        reports[-1]["deduplicated_total"] = removed
    return df, pd.DataFrame(reports)


class HistoricalAnalyzer:
    def __init__(self, draws: pd.DataFrame):
        self.draws = draws.sort_values("date").reset_index(drop=True)

    def main_counts(self, last_n: int | None = None) -> Counter:
        df = self.draws.tail(last_n) if last_n else self.draws
        counts = Counter({n: 0 for n in range(1, MAIN_POOL + 1)})
        for nums in df["main_numbers"]:
            counts.update(nums)
        return counts

    def joker_counts(self, last_n: int | None = None) -> Counter:
        df = self.draws.tail(last_n) if last_n else self.draws
        counts = Counter({j: 0 for j in range(1, JOKER_POOL + 1)})
        counts.update(int(x) for x in df["joker"])
        return counts

    def main_table(self, last_n: int | None = None) -> pd.DataFrame:
        df = self.draws.tail(last_n) if last_n else self.draws
        counts = self.main_counts(last_n)
        rows = []
        for n in range(1, MAIN_POOL + 1):
            rate = counts[n] / len(df) if len(df) else 0.0
            rows.append({
                "number": n,
                "count": counts[n],
                "appearance_rate": rate,
                "expected_rate": MAIN_BASELINE,
                "deviation_pp": 100 * (rate - MAIN_BASELINE),
            })
        return pd.DataFrame(rows).sort_values(["count", "number"], ascending=[False, True])

    def overdue_main(self) -> dict[int, int]:
        last_seen: dict[int, int] = {}
        for idx, nums in enumerate(self.draws["main_numbers"]):
            for n in nums:
                last_seen[int(n)] = idx
        end = len(self.draws) - 1
        return {n: end - last_seen.get(n, -1) for n in range(1, MAIN_POOL + 1)}


@dataclass(frozen=True)
class ProbabilityReport:
    alpha: float
    tune_brier: float
    holdout_model_brier: float
    holdout_baseline_brier: float
    holdout_skill: float
    holdout_improvement_se: float
    holdout_z: float
    final_probabilities: dict[int, float]
    used_baseline_fallback: bool
    observations_tune: int
    observations_holdout: int


def _targets_main(df: pd.DataFrame) -> np.ndarray:
    y = np.zeros((len(df), MAIN_POOL), dtype=float)
    for i, nums in enumerate(df["main_numbers"]):
        y[i, np.asarray(nums, dtype=int) - 1] = 1.0
    return y


def _targets_joker(df: pd.DataFrame) -> np.ndarray:
    y = np.zeros((len(df), JOKER_POOL), dtype=float)
    for i, j in enumerate(df["joker"]):
        y[i, int(j) - 1] = 1.0
    return y


def _rolling_frequency_predictions(y: np.ndarray, window: int) -> tuple[np.ndarray, np.ndarray]:
    """Return (prediction rows, target rows), using only rows strictly before target."""
    if len(y) <= window:
        return np.empty((0, y.shape[1])), np.empty((0, y.shape[1]))
    preds = []
    targets = []
    cumsum = np.vstack([np.zeros((1, y.shape[1])), np.cumsum(y, axis=0)])
    for i in range(window, len(y)):
        counts = cumsum[i] - cumsum[i - window]
        preds.append(counts / window)
        targets.append(y[i])
    return np.asarray(preds), np.asarray(targets)


def brier_score(probabilities: np.ndarray, targets: np.ndarray) -> float:
    if probabilities.size == 0:
        return float("nan")
    return float(np.mean((probabilities - targets) ** 2))


def _fit_alpha(raw: np.ndarray, y: np.ndarray, baseline: float) -> tuple[float, float]:
    best_alpha = 0.0
    best_brier = float("inf")
    for alpha in np.linspace(0.0, 1.0, 101):
        p = baseline + alpha * (raw - baseline)
        score = brier_score(p, y)
        if score < best_brier - 1e-15:
            best_brier = score
            best_alpha = float(alpha)
    return best_alpha, best_brier


def calibrated_frequency_report(
    draws: pd.DataFrame,
    pool: str = "main",
    window: int = 150,
    holdout_fraction: float = 0.40,
    min_holdout_rows: int = 50,
) -> ProbabilityReport:
    """Calibrate rolling-frequency deviations, then test on a strictly later holdout.

    Alpha is fitted on the earlier tuning segment. If the chosen alpha fails to beat
    the exact fair baseline on the later holdout, final probabilities fall back to the
    baseline. This prevents the app from converting random historical noise into a
    confident-looking future forecast.
    """
    if pool not in {"main", "joker"}:
        raise ValueError("pool must be 'main' or 'joker'")
    y_all = _targets_main(draws) if pool == "main" else _targets_joker(draws)
    baseline = MAIN_BASELINE if pool == "main" else JOKER_BASELINE
    pool_size = MAIN_POOL if pool == "main" else JOKER_POOL
    raw, y = _rolling_frequency_predictions(y_all, window)
    if len(raw) < max(2 * min_holdout_rows, 20):
        final = {i: baseline for i in range(1, pool_size + 1)}
        return ProbabilityReport(0.0, float("nan"), float("nan"), float("nan"), 0.0,
                                 float("nan"), 0.0, final, True, 0, 0)

    holdout_n = max(min_holdout_rows, int(round(len(raw) * holdout_fraction)))
    holdout_n = min(holdout_n, len(raw) // 2)
    split = len(raw) - holdout_n
    tune_raw, tune_y = raw[:split], y[:split]
    test_raw, test_y = raw[split:], y[split:]

    alpha, tune_brier = _fit_alpha(tune_raw, tune_y, baseline)
    model_test = baseline + alpha * (test_raw - baseline)
    baseline_test = np.full_like(test_y, baseline)
    model_brier = brier_score(model_test, test_y)
    baseline_brier = brier_score(baseline_test, test_y)
    skill = 1.0 - model_brier / baseline_brier if baseline_brier > 0 else 0.0

    # Gate the apparent edge on a strictly later holdout *and* require it to be
    # larger than ordinary sampling noise. A tiny positive Brier difference can
    # occur by chance even in perfectly fair simulated draws.
    model_per_draw = np.mean((model_test - test_y) ** 2, axis=1)
    baseline_per_draw = np.mean((baseline_test - test_y) ** 2, axis=1)
    improvement = baseline_per_draw - model_per_draw
    improvement_se = float(np.std(improvement, ddof=1) / np.sqrt(len(improvement))) if len(improvement) > 1 else float("inf")
    improvement_mean = float(np.mean(improvement))
    z = improvement_mean / improvement_se if improvement_se > 0 and np.isfinite(improvement_se) else 0.0
    use_fallback = not (model_brier < baseline_brier and z >= 1.96)

    recent = y_all[-min(window, len(y_all)):].mean(axis=0) if len(y_all) else np.full(pool_size, baseline)
    final_alpha = 0.0 if use_fallback else alpha
    final_vec = baseline + final_alpha * (recent - baseline)
    # Numerical guard only; valid rolling frequencies already live in [0,1].
    final_vec = np.clip(final_vec, 0.0, 1.0)
    final = {i + 1: float(final_vec[i]) for i in range(pool_size)}

    return ProbabilityReport(
        alpha=alpha,
        tune_brier=tune_brier,
        holdout_model_brier=model_brier,
        holdout_baseline_brier=baseline_brier,
        holdout_skill=skill,
        holdout_improvement_se=improvement_se,
        holdout_z=z,
        final_probabilities=final,
        used_baseline_fallback=use_fallback,
        observations_tune=len(tune_raw),
        observations_holdout=len(test_raw),
    )


def top_probability_numbers(report: ProbabilityReport, n: int, rng: random.Random | None = None) -> list[int]:
    """Rank if probabilities differ; randomize ties (the expected fair-game case)."""
    if not 1 <= n <= MAIN_POOL:
        raise ValueError("n out of range")
    rng = rng or random.Random()
    buckets: dict[float, list[int]] = {}
    for number, p in report.final_probabilities.items():
        buckets.setdefault(round(p, 14), []).append(number)
    ranked: list[int] = []
    for p in sorted(buckets, reverse=True):
        group = buckets[p]
        rng.shuffle(group)
        ranked.extend(group)
    return sorted(ranked[:n])


def top_probability_jokers(report: ProbabilityReport, n: int, rng: random.Random | None = None) -> list[int]:
    if not 1 <= n <= JOKER_POOL:
        raise ValueError("n out of range")
    rng = rng or random.Random()
    buckets: dict[float, list[int]] = {}
    for joker, p in report.final_probabilities.items():
        buckets.setdefault(round(p, 14), []).append(joker)
    ranked: list[int] = []
    for p in sorted(buckets, reverse=True):
        group = buckets[p]
        rng.shuffle(group)
        ranked.extend(group)
    return sorted(ranked[:n])


def full_system(numbers: Sequence[int], jokers: Sequence[int]) -> tuple[list[tuple[int, ...]], int, float]:
    numbers = sorted(set(int(n) for n in numbers))
    jokers = sorted(set(int(j) for j in jokers))
    if len(numbers) < MAIN_DRAW:
        raise ValueError("Need at least five distinct main numbers")
    if not all(1 <= n <= MAIN_POOL for n in numbers):
        raise ValueError("Main numbers must be 1..45")
    if not jokers or not all(1 <= j <= JOKER_POOL for j in jokers):
        raise ValueError("Need at least one Joker number in 1..20")
    combos = list(combinations(numbers, MAIN_DRAW))
    lines = len(combos) * len(jokers)
    return combos, lines, lines * TICKET_PRICE_EUR

# ---------------------------------------------------------------------------
# V2.2 — all-history signal / persistence diagnostics
# ---------------------------------------------------------------------------
from math import erfc, sqrt


def _normal_two_sided_p(z: float) -> float:
    """Two-sided standard-normal p-value without a scipy dependency."""
    return float(erfc(abs(float(z)) / sqrt(2.0)))


def _bernoulli_z(count: int, n_draws: int, baseline: float) -> float:
    if n_draws <= 0:
        return 0.0
    se = sqrt(baseline * (1.0 - baseline) / n_draws)
    if se <= 0:
        return 0.0
    return (count / n_draws - baseline) / se


def _holm_adjust(p_values: dict[int, float]) -> dict[int, float]:
    """Holm family-wise-error correction for the 45 simultaneous number tests."""
    m = len(p_values)
    ordered = sorted(p_values.items(), key=lambda kv: kv[1])
    adjusted: dict[int, float] = {}
    running = 0.0
    for rank, (key, p) in enumerate(ordered):
        value = min(1.0, (m - rank) * float(p))
        running = max(running, value)
        adjusted[key] = running
    return adjusted


def all_history_bias_report(
    draws: pd.DataFrame,
    tune_fraction: float = 0.60,
    alpha_level: float = 0.05,
) -> pd.DataFrame:
    """Test persistence of each main number across chronological, independent eras.

    The first ``tune_fraction`` of draws is the discovery/tuning era and the remainder
    is a strictly later holdout era.  Every one of the 45 numbers is tested on holdout;
    holdout p-values are Holm-corrected for the 45 simultaneous tests.  A number is
    labelled validated only when its deviation has the same direction in both eras and
    its later-era Holm-adjusted p-value is below ``alpha_level``.

    This is a diagnostic for possible persistent non-uniformity, not a guarantee that a
    future draw will preserve any historical deviation.
    """
    df = draws.sort_values(["date", "draw_id"]).reset_index(drop=True)
    n = len(df)
    if n < 100:
        return pd.DataFrame()
    split = int(round(n * tune_fraction))
    split = max(50, min(split, n - 50))
    tune = df.iloc[:split]
    hold = df.iloc[split:]

    tune_counts = Counter({i: 0 for i in range(1, MAIN_POOL + 1)})
    hold_counts = Counter({i: 0 for i in range(1, MAIN_POOL + 1)})
    for nums in tune["main_numbers"]:
        tune_counts.update(int(x) for x in nums)
    for nums in hold["main_numbers"]:
        hold_counts.update(int(x) for x in nums)

    hold_p = {}
    interim = []
    for number in range(1, MAIN_POOL + 1):
        tc = tune_counts[number]
        hc = hold_counts[number]
        tz = _bernoulli_z(tc, len(tune), MAIN_BASELINE)
        hz = _bernoulli_z(hc, len(hold), MAIN_BASELINE)
        hp = _normal_two_sided_p(hz)
        hold_p[number] = hp
        tune_rate = tc / len(tune)
        hold_rate = hc / len(hold)
        tune_dev = tune_rate - MAIN_BASELINE
        hold_dev = hold_rate - MAIN_BASELINE
        interim.append({
            "number": number,
            "tune_draws": len(tune),
            "tune_count": tc,
            "tune_rate": tune_rate,
            "tune_deviation_pp": 100 * tune_dev,
            "tune_z": tz,
            "holdout_draws": len(hold),
            "holdout_count": hc,
            "holdout_rate": hold_rate,
            "holdout_deviation_pp": 100 * hold_dev,
            "holdout_z": hz,
            "holdout_p_raw": hp,
            "same_direction": bool(tune_dev * hold_dev > 0),
        })

    adjusted = _holm_adjust(hold_p)
    for row in interim:
        q = adjusted[row["number"]]
        row["holdout_p_holm"] = q
        if row["same_direction"] and q < alpha_level:
            direction = "UP" if row["holdout_deviation_pp"] > 0 else "DOWN"
            row["status"] = f"VALIDATED {direction}"
        elif row["same_direction"]:
            row["status"] = "same direction, not significant"
        else:
            row["status"] = "not persistent"

    out = pd.DataFrame(interim)
    out["_validated"] = out["status"].str.startswith("VALIDATED")
    out["_abs_holdout_z"] = out["holdout_z"].abs()
    out = out.sort_values(
        ["_validated", "same_direction", "_abs_holdout_z", "number"],
        ascending=[False, False, False, True],
    ).drop(columns=["_validated", "_abs_holdout_z"]).reset_index(drop=True)
    return out


def multi_window_number_report(
    draws: pd.DataFrame,
    windows: Sequence[int | None] = (100, 300, 1000, None),
) -> pd.DataFrame:
    """Side-by-side rates for every number across several horizons, including all-time."""
    rows = {n: {"number": n} for n in range(1, MAIN_POOL + 1)}
    for window in windows:
        subset = draws.tail(window) if window else draws
        label = "all_time" if window is None else f"last_{window}"
        counts = Counter({n: 0 for n in range(1, MAIN_POOL + 1)})
        for nums in subset["main_numbers"]:
            counts.update(int(x) for x in nums)
        n_draws = len(subset)
        for number in range(1, MAIN_POOL + 1):
            rate = counts[number] / n_draws if n_draws else 0.0
            rows[number][f"{label}_rate"] = rate
            rows[number][f"{label}_deviation_pp"] = 100 * (rate - MAIN_BASELINE)
            rows[number][f"{label}_z"] = _bernoulli_z(counts[number], n_draws, MAIN_BASELINE)
    result = pd.DataFrame(rows.values())
    if "all_time_z" in result.columns:
        result = result.assign(_abs=result["all_time_z"].abs()).sort_values(
            ["_abs", "number"], ascending=[False, True]
        ).drop(columns="_abs")
    return result.reset_index(drop=True)


def yearly_number_report(draws: pd.DataFrame, number: int) -> pd.DataFrame:
    """Year-by-year history for one selected main number across all loaded years."""
    if not 1 <= int(number) <= MAIN_POOL:
        raise ValueError("number must be 1..45")
    rows = []
    df = draws.sort_values("date")
    for year, grp in df.groupby(df["date"].dt.year):
        count = int(grp["main_numbers"].apply(lambda xs: int(number) in set(xs)).sum())
        n = len(grp)
        rate = count / n if n else 0.0
        z = _bernoulli_z(count, n, MAIN_BASELINE)
        rows.append({
            "year": int(year),
            "draws": n,
            "count": count,
            "appearance_rate": rate,
            "expected_rate": MAIN_BASELINE,
            "deviation_pp": 100 * (rate - MAIN_BASELINE),
            "z": z,
        })
    return pd.DataFrame(rows)


def block_persistence_report(draws: pd.DataFrame, block_size: int = 250) -> pd.DataFrame:
    """Check direction consistency in non-overlapping chronological blocks.

    This is deliberately descriptive.  It helps distinguish a one-window spike from a
    deviation that repeatedly reappears in separate eras.
    """
    if block_size < 50:
        raise ValueError("block_size must be >= 50")
    df = draws.sort_values(["date", "draw_id"]).reset_index(drop=True)
    blocks = []
    for start in range(0, len(df), block_size):
        block = df.iloc[start:start + block_size]
        if len(block) < max(50, block_size // 2):
            continue
        counts = Counter({n: 0 for n in range(1, MAIN_POOL + 1)})
        for nums in block["main_numbers"]:
            counts.update(int(x) for x in nums)
        blocks.append((start, block, counts))

    rows = []
    for number in range(1, MAIN_POOL + 1):
        deviations = []
        zs = []
        for _, block, counts in blocks:
            rate = counts[number] / len(block)
            deviations.append(rate - MAIN_BASELINE)
            zs.append(_bernoulli_z(counts[number], len(block), MAIN_BASELINE))
        positive = sum(d > 0 for d in deviations)
        negative = sum(d < 0 for d in deviations)
        same_side = max(positive, negative)
        direction = "UP" if positive > negative else "DOWN" if negative > positive else "MIXED"
        rows.append({
            "number": number,
            "blocks": len(blocks),
            "blocks_above_baseline": positive,
            "blocks_below_baseline": negative,
            "dominant_direction": direction,
            "direction_consistency": same_side / len(blocks) if blocks else 0.0,
            "mean_deviation_pp": 100 * float(np.mean(deviations)) if deviations else 0.0,
            "max_abs_block_z": max((abs(z) for z in zs), default=0.0),
        })
    return pd.DataFrame(rows).sort_values(
        ["direction_consistency", "max_abs_block_z", "number"],
        ascending=[False, False, True],
    ).reset_index(drop=True)

# ---------------------------------------------------------------------------
# V2.3 — fixed-era and non-overlapping replication diagnostics
# ---------------------------------------------------------------------------
DEFAULT_YEAR_ERAS = (
    (1997, 2002),
    (2003, 2008),
    (2009, 2014),
    (2015, 2020),
    (2021, 2026),
)


def _two_sided_sign_p(successes: int, trials: int) -> float:
    """Exact two-sided sign-test p-value under P(up)=P(down)=0.5."""
    if trials <= 0:
        return 1.0
    successes = max(successes, trials - successes)
    tail = sum(comb(trials, k) for k in range(successes, trials + 1)) / (2 ** trials)
    return min(1.0, 2.0 * tail)


def era_number_report(
    draws: pd.DataFrame,
    eras: Sequence[tuple[int, int]] = DEFAULT_YEAR_ERAS,
) -> pd.DataFrame:
    """Long-format per-number statistics for fixed, non-overlapping calendar eras."""
    if draws.empty:
        return pd.DataFrame()
    df = draws.sort_values(["date", "draw_id"]).copy()
    rows: list[dict] = []
    for era_index, (start_year, end_year) in enumerate(eras, start=1):
        era = df[(df["date"].dt.year >= start_year) & (df["date"].dt.year <= end_year)]
        if era.empty:
            continue
        counts = Counter({n: 0 for n in range(1, MAIN_POOL + 1)})
        for nums in era["main_numbers"]:
            counts.update(int(x) for x in nums)
        n_draws = len(era)
        for number in range(1, MAIN_POOL + 1):
            count = counts[number]
            rate = count / n_draws
            dev = rate - MAIN_BASELINE
            rows.append({
                "number": number,
                "era_index": era_index,
                "era": f"{start_year}-{end_year}",
                "start_year": start_year,
                "end_year": end_year,
                "start_date": era["date"].min(),
                "end_date": era["date"].max(),
                "draws": n_draws,
                "count": count,
                "appearance_rate": rate,
                "expected_rate": MAIN_BASELINE,
                "deviation_pp": 100 * dev,
                "z": _bernoulli_z(count, n_draws, MAIN_BASELINE),
                "direction": "UP" if dev > 0 else "DOWN" if dev < 0 else "AT BASELINE",
            })
    return pd.DataFrame(rows)


def multi_era_persistence_report(
    draws: pd.DataFrame,
    eras: Sequence[tuple[int, int]] = DEFAULT_YEAR_ERAS,
    consistency_threshold: float = 0.80,
    alpha_level: float = 0.05,
) -> pd.DataFrame:
    """Summarize whether deviations replicate across fixed non-overlapping eras.

    The sign test asks only whether the same side of the baseline repeats across eras.
    ``pooled_p_holm`` tests the overall deviation after correcting across all 45 numbers.
    A PERSISTENT CANDIDATE label requires both >=80% directional consistency and a
    Holm-significant pooled deviation, with no strong (|z|>=1.96) reversal in an era.

    This is still a historical anomaly diagnostic, not a future-probability claim.
    """
    long = era_number_report(draws, eras=eras)
    if long.empty:
        return pd.DataFrame()

    interim: list[dict] = []
    pooled_p: dict[int, float] = {}
    for number, grp in long.groupby("number", sort=True):
        grp = grp.sort_values("era_index")
        devs = grp["deviation_pp"].to_numpy(dtype=float)
        zs = grp["z"].to_numpy(dtype=float)
        draws_n = grp["draws"].to_numpy(dtype=int)
        counts_n = grp["count"].to_numpy(dtype=int)
        up = int(np.sum(devs > 0))
        down = int(np.sum(devs < 0))
        non_ties = up + down
        same_side = max(up, down)
        dominant = "UP" if up > down else "DOWN" if down > up else "MIXED"
        consistency = same_side / non_ties if non_ties else 0.0
        sign_p = _two_sided_sign_p(same_side, non_ties) if non_ties else 1.0

        pooled_draws = int(draws_n.sum())
        pooled_count = int(counts_n.sum())
        pooled_rate = pooled_count / pooled_draws if pooled_draws else 0.0
        pooled_z = _bernoulli_z(pooled_count, pooled_draws, MAIN_BASELINE)
        p_raw = _normal_two_sided_p(pooled_z)
        pooled_p[int(number)] = p_raw

        if dominant == "UP":
            strong_reversal = bool(np.any(zs <= -1.96))
        elif dominant == "DOWN":
            strong_reversal = bool(np.any(zs >= 1.96))
        else:
            strong_reversal = False

        interim.append({
            "number": int(number),
            "eras": len(grp),
            "eras_above_baseline": up,
            "eras_below_baseline": down,
            "dominant_direction": dominant,
            "direction_consistency": consistency,
            "sign_test_p": sign_p,
            "mean_deviation_pp": float(np.mean(devs)),
            "median_deviation_pp": float(np.median(devs)),
            "min_era_deviation_pp": float(np.min(devs)),
            "max_era_deviation_pp": float(np.max(devs)),
            "pooled_rate": pooled_rate,
            "pooled_deviation_pp": 100 * (pooled_rate - MAIN_BASELINE),
            "pooled_z": pooled_z,
            "pooled_p_raw": p_raw,
            "strong_reversal": strong_reversal,
        })

    adjusted = _holm_adjust(pooled_p)
    for row in interim:
        q = adjusted[row["number"]]
        row["pooled_p_holm"] = q
        if (
            row["dominant_direction"] in {"UP", "DOWN"}
            and row["direction_consistency"] >= consistency_threshold
            and q < alpha_level
            and not row["strong_reversal"]
        ):
            row["status"] = f"PERSISTENT {row['dominant_direction']} CANDIDATE"
        elif row["direction_consistency"] >= consistency_threshold:
            row["status"] = f"consistent {row['dominant_direction'].lower()}, not validated"
        else:
            row["status"] = "mixed across eras"

    out = pd.DataFrame(interim)
    out["_candidate"] = out["status"].str.startswith("PERSISTENT")
    out = out.sort_values(
        ["_candidate", "direction_consistency", "pooled_p_holm", "number"],
        ascending=[False, False, True, True],
    ).drop(columns="_candidate").reset_index(drop=True)
    return out


def block_detail_report(
    draws: pd.DataFrame,
    number: int,
    block_size: int = 500,
) -> pd.DataFrame:
    """Detailed statistics for one number in separate sequential draw blocks."""
    number = int(number)
    if not 1 <= number <= MAIN_POOL:
        raise ValueError("number must be 1..45")
    if block_size < 50:
        raise ValueError("block_size must be >= 50")

    df = draws.sort_values(["date", "draw_id"]).reset_index(drop=True)
    rows: list[dict] = []
    block_no = 0
    for start in range(0, len(df), block_size):
        block = df.iloc[start:start + block_size]
        if len(block) < max(50, block_size // 2):
            continue
        block_no += 1
        count = int(block["main_numbers"].apply(lambda xs: number in set(xs)).sum())
        rate = count / len(block)
        dev = rate - MAIN_BASELINE
        rows.append({
            "block": block_no,
            "draw_index_start": start + 1,
            "draw_index_end": start + len(block),
            "start_date": block["date"].min(),
            "end_date": block["date"].max(),
            "draws": len(block),
            "count": count,
            "appearance_rate": rate,
            "expected_rate": MAIN_BASELINE,
            "deviation_pp": 100 * dev,
            "z": _bernoulli_z(count, len(block), MAIN_BASELINE),
            "direction": "UP" if dev > 0 else "DOWN" if dev < 0 else "AT BASELINE",
        })
    return pd.DataFrame(rows)
