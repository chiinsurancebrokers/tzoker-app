"""
Core data loading and statistical analysis for the Tzoker (Τζόκερ) consolidated app.

Consolidates and fixes the loading logic previously duplicated across
app.py / new_app.py / new_app_horoscope.py:

  - Handles BOTH historical Excel layouts (pre-2000s "ΔΙΑΓ." format with a
    stray '&' spacer column, and the modern "ΚΛΗΡ."/Num1..Num5 format),
    by locating columns via header-text search instead of a fixed row/offset.
  - Recovers years 1997-1999 and 2002, which previously loaded ZERO draws
    because those files have no per-row draw ID (all rows were silently
    skipped).
  - Filters out four known placeholder/test rows (draw IDs 2921-2924) that
    were mixed into Joker_2025.xlsx during earlier development.
  - De-duplicates exact repeated rows (Joker_2025.xlsx contains each real
    draw twice).
  - Namespaces draw_id by year, since the older files re-use small
    sequential IDs (1, 2, 3...) every calendar year.
"""

import os
from collections import Counter
from itertools import combinations

import numpy as np
import pandas as pd

# Known placeholder/test rows a previous version of this project injected into
# Joker_2025.xlsx for UI testing. They are not real OPAP results.
PLACEHOLDER_ROWS = {
    ('2025-2921', (19, 23, 38, 39, 44), 15),
    ('2025-2922', (1, 3, 20, 24, 43), 18),
    ('2025-2923', (6, 15, 25, 28, 43), 11),
    ('2025-2924', (16, 17, 22, 31, 43), 20),
}

FIBONACCI = {1, 2, 3, 5, 8, 13, 21, 34}
PRIMES = {2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43}

# Official OPAP prize categories and approximate odds (1997-present rule set).
# Ordered by realistic hit probability, best to worst.
PRIZE_CATEGORIES = [
    {"key": "2",   "label": "2 matches",        "odds_1_in": 40,          "prize": "€1"},
    {"key": "1+1", "label": "1 + Joker",         "odds_1_in": 145,         "prize": "€1.50"},
    {"key": "2+1", "label": "2 + Joker",         "odds_1_in": 950,         "prize": "€2"},
    {"key": "3",   "label": "3 matches",         "odds_1_in": 1_060,       "prize": "€2"},
    {"key": "3+1", "label": "3 + Joker",         "odds_1_in": 22_600,      "prize": "€50"},
    {"key": "4",   "label": "4 matches",         "odds_1_in": 27_200,      "prize": "€50"},
    {"key": "4+1", "label": "4 + Joker",         "odds_1_in": 122_176,     "prize": "€2,500 (fixed)"},
    {"key": "5",   "label": "5 matches",         "odds_1_in": 1_221_759,   "prize": "€100,000 (fixed, capped €2M/draw)"},
    {"key": "5+1", "label": "5 + Joker (JACKPOT)", "odds_1_in": 24_435_180, "prize": "Jackpot, min. €1,000,000"},
]


def _season_for_month(month: int) -> str:
    if month in (12, 1, 2):
        return "Winter"
    if month in (3, 4, 5):
        return "Spring"
    if month in (6, 7, 8):
        return "Summer"
    return "Autumn"


def _load_year_file(filepath: str, year: str):
    """Parse a single Joker_<year>.xlsx, auto-detecting its column layout."""
    raw = pd.read_excel(filepath, header=None)
    n_rows = len(raw)

    id_col = date_col = joker_col = None
    header_row_idx = 0
    scan_rows = min(6, n_rows)

    for r in range(scan_rows):
        for c in range(raw.shape[1]):
            val = raw.iat[r, c]
            if not isinstance(val, str):
                continue
            if id_col is None and ("ΚΛΗΡ" in val or "ΔΙΑΓ" in val):
                id_col = c
                header_row_idx = max(header_row_idx, r)
            if date_col is None and "ΗΜ" in val:
                date_col = c
                header_row_idx = max(header_row_idx, r)
            if "ΤΖΟΚΕΡ" in val:
                joker_col = c
                header_row_idx = max(header_row_idx, r)

    id_col = 0 if id_col is None else id_col
    date_col = 1 if date_col is None else date_col
    joker_col = 7 if joker_col is None else joker_col
    main_cols = list(range(date_col + 1, joker_col))

    data_start = header_row_idx + 1
    for r in range(header_row_idx + 1, min(header_row_idx + 8, n_rows)):
        if pd.notna(raw.iat[r, date_col]):
            data_start = r
            break

    draws, errors = [], 0

    for r in range(data_start, n_rows):
        row = raw.iloc[r]
        date_val = row[date_col]
        if pd.isna(date_val):
            continue
        try:
            date = pd.to_datetime(date_val, format="%d/%m/%Y") if isinstance(date_val, str) \
                else pd.to_datetime(date_val)
        except Exception:
            errors += 1
            continue

        nums = []
        for c in main_cols:
            v = row[c]
            if pd.isna(v):
                continue
            try:
                nums.append(int(float(v)))
            except (ValueError, TypeError):
                pass
        if len(nums) != 5 or not all(1 <= n <= 45 for n in nums):
            errors += 1
            continue

        try:
            joker = int(float(row[joker_col]))
        except (ValueError, TypeError):
            errors += 1
            continue
        if not 1 <= joker <= 20:
            errors += 1
            continue

        id_val = row[id_col]
        if pd.isna(id_val):
            local_id = date.strftime("%Y%m%d")
        else:
            try:
                local_id = str(int(float(id_val)))
            except (ValueError, TypeError):
                local_id = str(id_val).strip()
        draw_id = f"{year}-{local_id}"

        main_numbers = tuple(sorted(nums))
        if (draw_id, main_numbers, joker) in PLACEHOLDER_ROWS:
            continue

        draws.append({
            "draw_id": draw_id,
            "date": date,
            "year": int(year),
            "num1": main_numbers[0], "num2": main_numbers[1], "num3": main_numbers[2],
            "num4": main_numbers[3], "num5": main_numbers[4],
            "joker": joker,
            "main_numbers": list(main_numbers),
            "day_of_week": date.strftime("%A"),
            "season": _season_for_month(date.month),
        })

    return draws, errors


def load_all_draws(data_dir: str = ".", start_year: int = 1997, end_year: int = 2025):
    """Load and clean every Joker_<year>.xlsx file found in data_dir."""
    all_draws = []
    load_report = []

    for year in range(start_year, end_year + 1):
        filepath = os.path.join(data_dir, f"Joker_{year}.xlsx")
        if not os.path.exists(filepath):
            continue
        draws, errors = _load_year_file(filepath, str(year))
        load_report.append({"year": year, "draws": len(draws), "skipped_rows": errors})
        all_draws.extend(draws)

    if not all_draws:
        return pd.DataFrame(), pd.DataFrame(load_report)

    df = pd.DataFrame(all_draws)
    df["_key"] = df.apply(lambda r: (r["date"], tuple(r["main_numbers"]), r["joker"]), axis=1)
    df = df.drop_duplicates(subset="_key").drop(columns="_key")
    df = df.sort_values("date").reset_index(drop=True)
    return df, pd.DataFrame(load_report)


class JokerAnalyzer:
    """Statistical analysis over the loaded draw history."""

    def __init__(self, all_draws: pd.DataFrame):
        self.all_draws = all_draws
        self.main_freq = Counter()
        self.joker_freq = Counter()
        self._calculate_frequencies()

    def _calculate_frequencies(self):
        for _, row in self.all_draws.iterrows():
            for n in row["main_numbers"]:
                self.main_freq[n] += 1
            self.joker_freq[row["joker"]] += 1

    def hot_numbers(self, top_n=10, last_n_draws=None):
        df = self.all_draws.tail(last_n_draws) if last_n_draws else self.all_draws
        c = Counter()
        for nums in df["main_numbers"]:
            c.update(nums)
        return c.most_common(top_n)

    def cold_numbers(self, top_n=10, last_n_draws=None):
        df = self.all_draws.tail(last_n_draws) if last_n_draws else self.all_draws
        c = Counter({n: 0 for n in range(1, 46)})
        for nums in df["main_numbers"]:
            c.update(nums)
        return sorted(c.items(), key=lambda x: x[1])[:top_n]

    def overdue_numbers(self, top_n=10):
        """Numbers ranked by how many draws since their last appearance."""
        last_seen = {}
        for idx, nums in enumerate(self.all_draws["main_numbers"]):
            for n in nums:
                last_seen[n] = idx
        total = len(self.all_draws)
        gaps = {n: total - 1 - last_seen.get(n, -1) for n in range(1, 46)}
        return sorted(gaps.items(), key=lambda x: -x[1])[:top_n]

    def joker_hot_cold(self, top_n=5):
        hot = self.joker_freq.most_common(top_n)
        cold = sorted(
            ({j: self.joker_freq.get(j, 0) for j in range(1, 21)}).items(),
            key=lambda x: x[1]
        )[:top_n]
        return hot, cold

    def pattern_stats(self, last_n_draws=200):
        """Sum range / odd-even / low-high distribution over recent draws, for scoring."""
        df = self.all_draws.tail(last_n_draws)
        sums = df["main_numbers"].apply(sum)
        odd_counts = df["main_numbers"].apply(lambda nums: sum(1 for n in nums if n % 2 == 1))
        low_counts = df["main_numbers"].apply(lambda nums: sum(1 for n in nums if n <= 23))
        return {
            "sum_mean": sums.mean(), "sum_std": sums.std(),
            "odd_mean": odd_counts.mean(),
            "low_mean": low_counts.mean(),
        }

    def score_numbers(self, recent_window=150):
        """
        Composite 0-1 score per number (1-45), blending:
          - overall historical frequency
          - recent-window frequency (last `recent_window` draws)
          - overdue gap (numbers "due" get a bonus, capped so it doesn't dominate)
        This is a descriptive statistical score over draws that already happened —
        it does not and cannot predict an independent future random draw.
        """
        total_draws = len(self.all_draws)
        if total_draws == 0:
            return {n: 0.0 for n in range(1, 46)}

        overall = {n: self.main_freq.get(n, 0) / total_draws for n in range(1, 46)}

        recent_df = self.all_draws.tail(recent_window)
        recent_c = Counter()
        for nums in recent_df["main_numbers"]:
            recent_c.update(nums)
        recent = {n: recent_c.get(n, 0) / max(1, len(recent_df)) for n in range(1, 46)}

        gaps = dict(self.overdue_numbers(top_n=45))
        max_gap = max(gaps.values()) if gaps else 1
        overdue = {n: gaps.get(n, 0) / max(1, max_gap) for n in range(1, 46)}

        score = {}
        for n in range(1, 46):
            score[n] = 0.45 * overall[n] / max(overall.values()) + \
                       0.35 * recent[n] / max(recent.values()) + \
                       0.20 * overdue[n]
        return score

    def score_jokers(self):
        total = sum(self.joker_freq.values()) or 1
        return {j: self.joker_freq.get(j, 0) / total for j in range(1, 21)}

    def backtest_strategy(self, picks=5, lookback_draws=300, window=150):
        """
        Honest backtest: for each historical draw in the lookback window, compute
        what the top-`picks` scored numbers (using only draws BEFORE that point)
        would have matched against the ACTUAL draw. Returns a hit-count histogram.
        This measures how the scoring heuristic would have fared historically —
        it is not a claim about future draws, which are independent random events.
        """
        n = len(self.all_draws)
        start = max(window + 1, n - lookback_draws)
        hit_histogram = Counter()
        joker_hit = 0
        tested = 0

        for i in range(start, n):
            history = self.all_draws.iloc[:i]
            if len(history) < window:
                continue
            temp_analyzer = JokerAnalyzer(history)
            scores = temp_analyzer.score_numbers(recent_window=window)
            top_picks = set(sorted(scores, key=scores.get, reverse=True)[:picks])

            actual = set(self.all_draws.iloc[i]["main_numbers"])
            matches = len(top_picks & actual)
            hit_histogram[matches] += 1

            jscores = temp_analyzer.score_jokers()
            top_joker = max(jscores, key=jscores.get)
            if top_joker == self.all_draws.iloc[i]["joker"]:
                joker_hit += 1
            tested += 1

        return hit_histogram, joker_hit, tested


def build_full_system(numbers, jokers):
    """
    A "full wheel" system: every possible 5-number sub-combination of the chosen
    `numbers`, each paired with every chosen joker. This is straightforward
    combinatorics (not OPAP's proprietary abbreviated/reduced-system tables) and
    gives a mathematically exact, verifiable guarantee: if all of `numbers` were
    the actual 5 drawn, every one of the C(len(numbers), 5) tickets is a 5-match
    winner; if k of them were drawn, exactly C(k, 5) tickets win category "5".
    """
    combos = list(combinations(sorted(numbers), 5))
    total_tickets = len(combos) * len(jokers)
    return combos, total_tickets


def min_guaranteed_matches_table(n_numbers):
    """For an n-number full system, show C(k,5) tickets guaranteed per match count k."""
    from math import comb
    rows = []
    for k in range(min(5, n_numbers), n_numbers + 1) if n_numbers >= 5 else []:
        pass
    rows = []
    for k in range(5, n_numbers + 1):
        rows.append({"if_k_of_your_numbers_drawn": k, "guaranteed_tickets_matching_5": comb(k, 5)})
    return rows
