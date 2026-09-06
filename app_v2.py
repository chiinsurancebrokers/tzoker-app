"""Streamlit UI for the baseline-aware Tzoker V2 analysis app."""
import os
import random

import pandas as pd
import streamlit as st

from tzoker_v2_core import (
    HistoricalAnalyzer,
    all_history_bias_report,
    block_persistence_report,
    MAIN_BASELINE,
    JOKER_BASELINE,
    calibrated_frequency_report,
    full_system,
    load_all_draws,
    prize_categories,
    top_probability_jokers,
    multi_window_number_report,
    yearly_number_report,
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



@st.cache_data(show_spinner=False)
def get_signal_tables(df, tune_fraction, block_size):
    return (
        all_history_bias_report(df, tune_fraction=tune_fraction),
        multi_window_number_report(df),
        block_persistence_report(df, block_size=block_size),
    )


def page_signal_lab(df):
    st.header("🔬 Signal Lab — all years")
    first_date = df["date"].min().date()
    last_date = df["date"].max().date()
    st.info(
        f"This page uses the complete loaded history ({first_date} to {last_date}, "
        f"{len(df):,} draws). It asks whether any number's deviation from 5/45 persists "
        "across independent time periods. It does not assume that a historically high "
        "frequency will continue."
    )

    c1, c2 = st.columns(2)
    with c1:
        tune_fraction = st.slider(
            "Discovery share of history", 0.50, 0.80, 0.60, 0.05,
            help="The first part of the chronology is used to identify the direction of a deviation. The rest is untouched holdout."
        )
    with c2:
        block_size = st.select_slider(
            "Non-overlapping block size", options=[100, 150, 200, 250, 300, 500], value=250,
            help="A separate descriptive persistence check across non-overlapping chronological blocks."
        )

    persistence, windows, blocks = get_signal_tables(df, tune_fraction, block_size)

    st.subheader("1. Independent-era persistence test")
    st.caption(
        "The first chronological era determines whether a number was above or below the fair baseline. "
        "The later era checks whether the same direction survives. Holdout p-values are Holm-adjusted "
        "because 45 numbers are tested at the same time."
    )
    if persistence.empty:
        st.warning("Not enough draws for the persistence test.")
    else:
        validated = persistence[persistence["status"].str.startswith("VALIDATED")]
        same_not_sig = persistence[persistence["status"] == "same direction, not significant"]
        m1, m2, m3 = st.columns(3)
        m1.metric("Numbers tested", len(persistence))
        m2.metric("Same direction in both eras", len(validated) + len(same_not_sig))
        m3.metric("Validated after 45-test correction", len(validated))
        if validated.empty:
            st.warning(
                "No main number shows a statistically validated persistent deviation across the two independent eras after correcting for 45 simultaneous tests."
            )
        else:
            st.success(
                f"{len(validated)} number(s) survived the later holdout and the 45-number multiple-testing correction. "
                "Treat this as an anomaly to investigate, not as a guaranteed future edge."
            )

        show = persistence.copy()
        for col in ["tune_rate", "holdout_rate"]:
            show[col] = (100 * show[col]).round(3)
        for col in ["tune_deviation_pp", "holdout_deviation_pp", "tune_z", "holdout_z"]:
            show[col] = show[col].round(3)
        show["holdout_p_holm"] = show["holdout_p_holm"].map(lambda x: f"{x:.4g}")
        show = show.rename(columns={
            "tune_rate": "discovery_rate_%",
            "holdout_rate": "holdout_rate_%",
            "holdout_p_holm": "holdout_Holm_p",
        })
        st.dataframe(
            show[[
                "number", "discovery_rate_%", "tune_deviation_pp", "tune_z",
                "holdout_rate_%", "holdout_deviation_pp", "holdout_z",
                "same_direction", "holdout_Holm_p", "status"
            ]],
            hide_index=True, width="stretch"
        )

    st.subheader("2. Multi-window comparison")
    st.caption(
        "Every number is compared over the last 100, 300 and 1,000 draws and across the complete history. "
        "This makes a one-window spike easy to distinguish from a longer-lived pattern."
    )
    win = windows.copy()
    for col in [c for c in win.columns if c.endswith("_rate")]:
        win[col] = (100 * win[col]).round(3)
    for col in [c for c in win.columns if c.endswith("_deviation_pp") or c.endswith("_z")]:
        win[col] = win[col].round(3)
    st.dataframe(win, hide_index=True, width="stretch")

    st.subheader("3. Non-overlapping block persistence")
    st.caption(
        f"The full history is cut into separate {block_size}-draw blocks. A number that is high only in one block "
        "is very different from one that stays on the same side of the baseline across many independent blocks."
    )
    block_show = blocks.copy()
    block_show["direction_consistency"] = (100 * block_show["direction_consistency"]).round(1)
    block_show["mean_deviation_pp"] = block_show["mean_deviation_pp"].round(3)
    block_show["max_abs_block_z"] = block_show["max_abs_block_z"].round(2)
    st.dataframe(block_show, hide_index=True, width="stretch")

    st.subheader("4. Inspect one number across every year")
    default_number = 29 if 29 in range(1, 46) else 1
    selected = st.selectbox("Main number", list(range(1, 46)), index=default_number - 1)
    yearly = yearly_number_report(df, selected)
    if not yearly.empty:
        overall_count = int(df["main_numbers"].apply(lambda xs: selected in set(xs)).sum())
        overall_rate = overall_count / len(df)
        a, b, c = st.columns(3)
        a.metric("All-time appearances", overall_count)
        b.metric("All-time appearance rate", f"{overall_rate:.3%}")
        c.metric("Fair baseline", f"{MAIN_BASELINE:.3%}")
        yearly_show = yearly.copy()
        yearly_show["appearance_rate"] = (100 * yearly_show["appearance_rate"]).round(2)
        yearly_show["expected_rate"] = (100 * yearly_show["expected_rate"]).round(2)
        yearly_show["deviation_pp"] = yearly_show["deviation_pp"].round(2)
        yearly_show["z"] = yearly_show["z"].round(2)
        st.dataframe(yearly_show, hide_index=True, width="stretch")
        chart = yearly.set_index("year")[["appearance_rate", "expected_rate"]]
        st.line_chart(chart)

    st.caption(
        "Interpretation rule: a number is not promoted into the prediction engine merely because it looks high historically. "
        "A future V2.x model may use a signal only after it survives independent holdout testing and the multiple-testing guard."
    )

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
        if not report.empty:
            st.caption("Loader diagnostics")
            st.dataframe(report, hide_index=True, width="stretch")
        return
    page = st.sidebar.radio("Section", ["Overview", "Historical Stats", "Probability Lab", "Signal Lab", "System Builder", "Data Quality"])
    if page == "Overview":
        page_overview(df, report)
    elif page == "Historical Stats":
        page_history(df)
    elif page == "Probability Lab":
        page_probability_lab(df)
    elif page == "Signal Lab":
        page_signal_lab(df)
    elif page == "System Builder":
        page_system_builder(df)
    else:
        page_data_quality(report)


if __name__ == "__main__":
    main()
