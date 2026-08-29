# Tzoker Analysis (consolidated)

## Run it
```
pip install -r requirements.txt
streamlit run app.py
```
Keep `Joker_*.xlsx` in the same folder as `app.py` — the loader picks up any
`Joker_<year>.xlsx` files present (1997–2025 included here).

## What changed from the three original files
See the module docstrings at the top of `app.py` and `tzoker_core.py` for the
full list of bugs fixed and what was intentionally dropped (horoscope module,
OPAP's incomplete proprietary system tables, duplicate function defs).

Quick summary:
- **Fixed:** years 1997, 1998, 1999, 2002 previously loaded 0 draws each
  (silently) — now load correctly (~425 recovered draws).
- **Fixed:** 4 known placeholder/test rows in `Joker_2025.xlsx` (draw IDs
  2921–2924) are now filtered out instead of being treated as real results.
- **Fixed:** exact-duplicate rows in `Joker_2025.xlsx` are de-duplicated.
- **Removed:** `new_app.py`'s crash on `import analyzer` (missing module) and
  `new_app_horoscope.py`'s crash on a stray `self.all_draws` debug line at
  module scope — both are gone since this is a fresh, tested consolidation.
- **Re-centered predictions:** the Predictions tab now leads with the
  categories that actually have realistic odds — "5" (1 in 1.22M) and "4+1"
  (1 in 122K) — via a "core 5" pick plus an optional full-wheel system
  expansion, with the 5+1 jackpot pick shown last, clearly labeled as the
  long shot it is (1 in 24.4M).
- **New:** a Backtest tab that honestly replays the scoring strategy against
  history using only prior data at each point, so you can see how it would
  have performed — with a plain note that past draws don't predict future
  independent ones.
