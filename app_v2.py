"""Streamlit UI for the baseline-aware Tzoker V2 analysis app."""
import os
import random

import pandas as pd
import streamlit as st

from tzoker_v2_core import (
    HistoricalAnalyzer,
    MAIN_BASELINE,
    JOKER_BASELINE,
    calibrated_frequency_report,
    full_system,
    load_all_draws,
    prize_categories,
    top_probability_jokers,
    top_probability_numbers,
)

st.set_page_config(page_title="Tzoker Analysis V2", page_icon="🎰", layout="wide")


@st.cache_data(show_spinner=False)
def get_data():
    return load_all_draws(os.path.dirname(__file__) or ".")


@st.cache_data(show_spinner=False)
def get_model_reports(df, window):
    return (
        calibrated_frequency_report(df, "main", window=window),
        calibrated_frequency_report(df, "joker", window=window),
    )


def odds_table():
    rows = []
    for row in prize_categories():
        rows.append({
            "Category": row["key"],
            "Exact odds": f"1 in {row['odds_1_in']:,.1f}",
            "Current fixed prize": row["prize"],
        })
    return pd.DataFrame(rows)


def page_overview(df, report):
    st.header("🎰 Tzoker Analysis V2")
    st.info(
        "V2 separates historical patterns from future probabilities. A historical-frequency "
        "model is allowed to influence the displayed probabilities only if it beats the exact "
        "fair-game baseline on a later holdout period. Otherwise the app falls back to 5/45 "
        "for every main number and 1/20 for every Joker number."
    )
    c1, c2, c3 = st.columns(3)
    c1.metric("Draws loaded", f"{len(df):,}")
    c2.metric("Main-number baseline", f"{MAIN_BASELINE:.4%}")
    c3.metric("Joker baseline", f"{JOKER_BASELINE:.2%}")
    st.subheader("Exact single-line prize-category odds")
    st.dataframe(odds_table(), hide_index=True, width="stretch")
    with st.expander("Data load report"):
        st.dataframe(report, hide_index=True, width="stretch")


def page_history(df):
    st.header("📊 Historical statistics")
    st.caption("Descriptive only — these figures do not change the probability of the next fair draw.")
    analyzer = HistoricalAnalyzer(df)
    window = st.select_slider("Window", options=[100, 150, 300, 500, 1000, None], value=150,
                              format_func=lambda x: "All-time" if x is None else f"Last {x}")
    table = analyzer.main_table(window)
    st.dataframe(table.head(45), hide_index=True, width="stretch")
    st.bar_chart(table.set_index("number")["appearance_rate"])


def _report_metrics(label, report):
    st.subheader(label)
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Fitted α", f"{report.alpha:.2f}")
    c2.metric("Holdout model Brier", "n/a" if pd.isna(report.holdout_model_brier) else f"{report.holdout_model_brier:.6f}")
    c3.metric("Holdout baseline Brier", "n/a" if pd.isna(report.holdout_baseline_brier) else f"{report.holdout_baseline_brier:.6f}")
    c4.metric("Brier skill vs baseline", f"{report.holdout_skill:+.3%}")
    c5.metric("Holdout z", f"{report.holdout_z:.2f}")
    if report.used_baseline_fallback:
        st.warning("No statistically credible holdout edge: final probabilities are the exact fair baseline.")
    else:
        st.success("The historical-frequency deviation beat baseline on this holdout. V2 keeps only the validated shrinkage amount.")


def page_probability_lab(df):
    st.header("🧪 Probability Lab")
    st.write(
        "This is the key audit missing from the old app. The earlier part of history fits how much "
        "weight (α) to give rolling frequency. A strictly later holdout then decides whether that "
        "weight survives. Lower Brier score is better."
    )
    window = st.slider("Rolling history window", 50, 500, 150, 25)
    main_report, joker_report = get_model_reports(df, window)
    _report_metrics("Main numbers (45 Bernoulli probabilities per draw)", main_report)
    _report_metrics("Joker numbers (20 Bernoulli probabilities per draw)", joker_report)

    st.subheader("Final main-number probabilities")
    prob_df = pd.DataFrame([
        {"number": n, "probability": p, "baseline": MAIN_BASELINE, "delta_pp": 100 * (p - MAIN_BASELINE)}
        for n, p in main_report.final_probabilities.items()
    ]).sort_values(["probability", "number"], ascending=[False, True])
    st.dataframe(prob_df, hide_index=True, width="stretch")


def page_system_builder(df):
    st.header("🧮 System Builder")
    st.info(
        "For a fair draw, coverage — buying additional distinct lines — changes your chance of a hit. "
        "Choosing a historically 'hot' number instead of another number does not."
    )
    window = st.slider("Model window", 50, 500, 150, 25, key="system_window")
    main_report, joker_report = get_model_reports(df, window)
    n_numbers = st.slider("Main numbers in full system", 5, 10, 7)
    n_jokers = st.slider("Joker numbers", 1, 5, 1)

    if "v2_seed" not in st.session_state:
        st.session_state.v2_seed = random.SystemRandom().randrange(1, 2**31)
    if st.button("Generate a fresh tie-break/random pick"):
        st.session_state.v2_seed = random.SystemRandom().randrange(1, 2**31)
    rng = random.Random(st.session_state.v2_seed)

    numbers = top_probability_numbers(main_report, n_numbers, rng)
    jokers = top_probability_jokers(joker_report, n_jokers, rng)
    _, lines, cost = full_system(numbers, jokers)
    st.write("**Main numbers:**", numbers)
    st.write("**Joker number(s):**", jokers)
    c1, c2 = st.columns(2)
    c1.metric("Distinct lines", f"{lines:,}")
    c2.metric("Cost per draw", f"€{cost:,.2f}")
    if main_report.used_baseline_fallback and joker_report.used_baseline_fallback:
        st.caption("Both models failed to verify an edge, so all probabilities are tied and the actual number identities are randomized.")


def page_data_quality(report):
    st.header("🧹 Data quality")
    st.dataframe(report, hide_index=True, width="stretch")
    if "invalid_duplicates" in report.columns and report["invalid_duplicates"].sum() > 0:
        st.error("Rows containing duplicate main numbers were rejected; a valid Tzoker draw must contain five distinct main numbers.")
    if "fallback_header" in report.columns and report["fallback_header"].any():
        st.warning("At least one workbook used fallback column positions because headers were not fully detected.")


def main():
    df, report = get_data()
    if df.empty:
        st.error("No valid Joker_<year>.xlsx data files were loaded.")
        return
    page = st.sidebar.radio("Section", ["Overview", "Historical Stats", "Probability Lab", "System Builder", "Data Quality"])
    if page == "Overview":
        page_overview(df, report)
    elif page == "Historical Stats":
        page_history(df)
    elif page == "Probability Lab":
        page_probability_lab(df)
    elif page == "System Builder":
        page_system_builder(df)
    else:
        page_data_quality(report)


if __name__ == "__main__":
    main()
