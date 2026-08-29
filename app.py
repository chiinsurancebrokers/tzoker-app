"""
Tzoker (Τζόκερ) Analysis — consolidated app.

Merges app.py / new_app.py / new_app_horoscope.py into one clean, working app,
fixes their data-loading and crash bugs, and re-centers the "prediction" feature
on the categories with realistic odds (5, then 4+1) rather than chasing the
5+1 jackpot outright, per your request.

Dropped from the original three files:
  - new_app.py's `import analyzer` (module never existed -> instant crash)
  - new_app_horoscope.py's stray `self.all_draws` debug lines at module scope
    (undefined `self` -> instant crash), and the horoscope/numerology/zodiac
    module generally: it doesn't feed the odds-based prediction goal below,
    and reintroducing it would reintroduce that file's crash surface.
  - Duplicate function definitions (create_number_display, create_progress_bar,
    display_debug_logs, save_submissions_to_file were each defined 2-3x).
  - OPAP's proprietary abbreviated "typified system" tables (several were
    incomplete/placeholder in the source anyway) — replaced with a
    mathematically exact full-wheel system generator, see tzoker_core.py.
"""

import json
import os
from datetime import datetime

import pandas as pd
import streamlit as st

from tzoker_core import (
    JokerAnalyzer,
    PRIZE_CATEGORIES,
    build_full_system,
    load_all_draws,
    min_guaranteed_matches_table,
)
from tzoker_llm import TICKET_PRICE_EUR, ask_claude_for_pick, build_analysis_payload, get_api_key
from tzoker_openai import get_openai_api_key, review_pick_with_chatgpt

st.set_page_config(page_title="Tzoker Analysis", page_icon="🎰", layout="wide")

SUBMISSIONS_FILE = "submissions_data.json"


# --------------------------------------------------------------------------
# Data loading (cached)
# --------------------------------------------------------------------------
@st.cache_data(show_spinner=False)
def get_data():
    df, report = load_all_draws(data_dir=os.path.dirname(__file__) or ".")
    return df, report


def load_submissions():
    if not os.path.exists(SUBMISSIONS_FILE):
        return []
    try:
        with open(SUBMISSIONS_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError):
        return []


def save_submissions(subs):
    with open(SUBMISSIONS_FILE, "w", encoding="utf-8") as f:
        json.dump(subs, f, indent=2, default=str)


# --------------------------------------------------------------------------
# UI helpers
# --------------------------------------------------------------------------
def number_badges(numbers, joker=None):
    cols = st.columns(len(numbers) + (1 if joker else 0))
    for i, n in enumerate(numbers):
        cols[i].markdown(
            f"<div style='text-align:center;background:#667eea;color:white;"
            f"border-radius:50%;width:44px;height:44px;line-height:44px;"
            f"font-weight:700;margin:auto'>{n}</div>",
            unsafe_allow_html=True,
        )
    if joker:
        cols[-1].markdown(
            f"<div style='text-align:center;background:#ee5a24;color:white;"
            f"border-radius:50%;width:44px;height:44px;line-height:44px;"
            f"font-weight:700;margin:auto'>{joker}</div>",
            unsafe_allow_html=True,
        )


def prize_table_df():
    rows = []
    for c in PRIZE_CATEGORIES:
        rows.append({
            "Category": c["label"],
            "Odds": f"1 in {c['odds_1_in']:,}",
            "Prize": c["prize"],
        })
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------
# Pages
# --------------------------------------------------------------------------
def page_dashboard(df, report, analyzer):
    st.header("📊 Dashboard")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total draws loaded", f"{len(df):,}")
    c2.metric("Date range", f"{df['date'].min().year}–{df['date'].max().year}")
    c3.metric("Years of files", f"{report['year'].nunique()}")
    c4.metric("Most recent draw", df["date"].max().strftime("%d/%m/%Y"))

    with st.expander("Data load report (per year)"):
        st.dataframe(report, width='stretch', hide_index=True)
        total_skipped = report["skipped_rows"].sum()
        if total_skipped:
            st.caption(
                f"{total_skipped} raw spreadsheet rows across all years could not be "
                f"parsed as a valid draw (blank rows, section headers, etc.) and were skipped."
            )

    st.subheader("Most / least frequent main numbers (all-time)")
    col1, col2 = st.columns(2)
    with col1:
        st.write("🔥 Hottest")
        st.dataframe(
            pd.DataFrame(analyzer.hot_numbers(10), columns=["Number", "Times drawn"]),
            hide_index=True, width='stretch',
        )
    with col2:
        st.write("🧊 Coldest")
        st.dataframe(
            pd.DataFrame(analyzer.cold_numbers(10), columns=["Number", "Times drawn"]),
            hide_index=True, width='stretch',
        )

    st.subheader("Most overdue numbers (draws since last seen)")
    st.dataframe(
        pd.DataFrame(analyzer.overdue_numbers(10), columns=["Number", "Draws since last seen"]),
        hide_index=True, width='stretch',
    )


def page_predictions(df, analyzer):
    st.header("🎯 Predictions — ranked by realistic odds")
    st.info(
        "Each Tzoker draw is an independent random event — nothing in past results "
        "changes the odds of the next one. The scores below describe *historical* "
        "frequency/recency patterns in your data; they are a way to organize a play, "
        "not a forecast. The categories are ordered by how likely they actually are to hit.",
        icon="ℹ️",
    )

    st.dataframe(prize_table_df(), hide_index=True, width='stretch')

    recent_window = st.slider("Recency window (draws)", 50, 500, 150, step=25)
    scores = analyzer.score_numbers(recent_window=recent_window)
    ranked = sorted(scores, key=scores.get, reverse=True)
    jscores = analyzer.score_jokers()
    ranked_jokers = sorted(jscores, key=jscores.get, reverse=True)

    st.subheader("① Core 5 — your best shot at category \"5\" and \"4+1\"")
    core5 = sorted(ranked[:5])
    number_badges(core5)
    st.caption(
        "These 5 are the strategy's top-scored numbers. Playing exactly these 5 gives "
        "you a shot at every category from '2' up through the jackpot — category "
        "\"5\" (1 in 1.22M) and \"4+1\" (1 in 122K) are the realistic targets, not 5+1 "
        "(1 in 24.4M)."
    )

    st.subheader("② System expansion — better odds of a 4+1 or 5 hit, at a cost")
    st.write(
        "A *system* means playing more than 5 numbers, entering every 5-number "
        "combination inside them. If several of your extra numbers are drawn, you "
        "win category 5 (or 4+1) multiple times over in the same draw."
    )
    n_extra = st.slider("How many numbers to play (5 = no system)", 5, 10, 7)
    system_numbers = sorted(ranked[:n_extra])
    number_badges(system_numbers)

    joker_count = st.slider("How many Joker numbers to cover", 1, 5, 1)
    system_jokers = sorted(ranked_jokers[:joker_count])
    st.write("Joker number(s):", ", ".join(str(j) for j in system_jokers))

    combos, total_tickets = build_full_system(system_numbers, system_jokers)
    st.metric("Total lines in this system", f"{total_tickets:,}")
    st.metric("Estimated cost", f"€{total_tickets * TICKET_PRICE_EUR:,.2f}")

    guarantee = min_guaranteed_matches_table(n_extra)
    if guarantee:
        st.write("**Guaranteed tickets matching category \"5\" if k of your numbers are drawn:**")
        st.dataframe(pd.DataFrame(guarantee), hide_index=True, width='stretch')
    st.caption(
        "This is a *full* system (every 5-number combination inside your chosen numbers), "
        "computed directly rather than reproduced from Allwyn's table. Allwyn also sells "
        "cheaper *reduced* systems that trade a 100% category-5 guarantee for lower cost — "
        "see the official guide: "
        "https://www.allwyn.gr/el/odigos/systimata-tzoker"
    )

    st.subheader("③ Full jackpot pick (5+1) — long shot, shown for completeness")
    st.write(f"Core 5 + top Joker ({ranked_jokers[0]}) — 1 in 24,435,180.")
    number_badges(core5, joker=ranked_jokers[0])


def page_ai_insights(analyzer):
    st.header("🤖 AI Insights (Claude + ChatGPT second opinion)")
    st.info(
        "Neither model predicts draws — each one is still an independent random event. "
        "Claude reads the same statistical scores shown in the Predictions tab and "
        "synthesizes a pick from within them (it can't invent its own numbers). "
        "ChatGPT then reviews that pick against the same data — checking its arithmetic "
        "and whether its wording overstates what a frequency score means — rather than "
        "proposing a competing pick of its own. Two models agreeing isn't stronger "
        "evidence a combination will win; it's just two reviews of the same numbers.",
        icon="ℹ️",
    )

    claude_key = get_api_key()
    if not claude_key:
        st.warning(
            "No Anthropic API key found. Set the `ANTHROPIC_API_KEY` environment variable "
            "(on Railway: Project → Variables), or paste one below for this session only "
            "(it is not saved anywhere).",
            icon="🔑",
        )
        pasted = st.text_input("Anthropic API key", type="password", key="_key_input_field")
        if pasted:
            st.session_state["_anthropic_key_input"] = pasted
            st.rerun()
        return

    c1, c2 = st.columns(2)
    with c1:
        target_category = st.selectbox("Target category", ["5", "4+1", "5+1"])
    with c2:
        budget_eur = st.number_input("Budget (€)", min_value=1.0, max_value=1000.0, value=10.0, step=1.0)

    if st.button("Ask Claude"):
        with st.spinner("Analyzing historical scores..."):
            result, error = ask_claude_for_pick(analyzer, target_category, budget_eur, claude_key)
        if error:
            st.error(error)
            st.session_state.pop("_claude_result", None)
        else:
            st.session_state["_claude_result"] = result
            st.session_state["_claude_payload"] = build_analysis_payload(
                analyzer, target_category, budget_eur
            )
            st.session_state.pop("_chatgpt_review", None)

    result = st.session_state.get("_claude_result")
    if not result:
        return

    st.subheader("Claude's recommended pick")
    number_badges(result["numbers"], joker=result["joker_numbers"][0]
                  if len(result["joker_numbers"]) == 1 else None)
    if len(result["joker_numbers"]) > 1:
        st.write("Joker numbers:", ", ".join(str(j) for j in result["joker_numbers"]))

    st.metric("System size", result.get("system_size", len(result["numbers"])))
    st.metric("Estimated cost", f"€{result.get('estimated_cost_eur', 0):,.2f}")

    st.write("**Pattern notes:**", result.get("pattern_notes", ""))
    st.write("**Rationale:**", result.get("rationale", ""))
    st.caption(result.get("caveat", ""))

    st.divider()
    st.subheader("Second opinion (ChatGPT)")

    openai_key = get_openai_api_key()
    if not openai_key:
        st.warning(
            "No OpenAI API key found. Set the `OPENAI_API_KEY` environment variable "
            "(on Railway: Project → Variables), or paste one below for this session only "
            "(it is not saved anywhere).",
            icon="🔑",
        )
        pasted_oa = st.text_input("OpenAI API key", type="password", key="_oa_key_input_field")
        if pasted_oa:
            st.session_state["_openai_key_input"] = pasted_oa
            st.rerun()
        return

    if st.button("Get ChatGPT's review"):
        with st.spinner("Checking Claude's pick..."):
            review, rev_error = review_pick_with_chatgpt(
                st.session_state["_claude_payload"], result, openai_key
            )
        if rev_error:
            st.error(rev_error)
        else:
            st.session_state["_chatgpt_review"] = review

    review = st.session_state.get("_chatgpt_review")
    if review:
        verdict = review.get("verdict", "unknown")
        if verdict == "pass":
            st.success(f"ChatGPT verdict: **pass** — {review.get('summary', '')}")
        else:
            st.warning(f"ChatGPT verdict: **{verdict}** — {review.get('summary', '')}")
        st.dataframe(pd.DataFrame(review.get("checks", [])), hide_index=True, width='stretch')


def page_backtest(analyzer):
    st.header("📈 Backtest the scoring strategy")
    st.warning(
        "This replays the scoring strategy against draws that already happened, "
        "using only the data available *before* each draw at the time. It tells you "
        "how the heuristic would have scored historically — since each draw is "
        "independent, it is not evidence about how future draws will go.",
        icon="⚠️",
    )
    lookback = st.slider("How many recent draws to backtest", 50, 1000, 300, step=50)
    if st.button("Run backtest"):
        with st.spinner("Replaying history..."):
            hist, joker_hits, tested = analyzer.backtest_strategy(picks=5, lookback_draws=lookback)
        if tested == 0:
            st.error("Not enough history to backtest with the current window.")
            return
        rows = [{"Numbers matched": k, "Occurrences": v, "Rate": f"{v / tested:.1%}"}
                for k, v in sorted(hist.items(), reverse=True)]
        st.dataframe(pd.DataFrame(rows), hide_index=True, width='stretch')
        st.metric("Draws tested", tested)
        st.metric("Top-scored Joker matched actual Joker", f"{joker_hits} ({joker_hits / tested:.1%})")


def page_submissions(df):
    st.header("🎟️ My submissions")
    subs = st.session_state.setdefault("submissions", load_submissions())

    with st.form("add_submission"):
        st.write("Log a play")
        c1, c2 = st.columns(2)
        with c1:
            nums_str = st.text_input("5 numbers (comma-separated, 1-45)", "")
        with c2:
            joker_str = st.text_input("Joker (1-20)", "")
        submitted = st.form_submit_button("Add")
        if submitted:
            try:
                nums = sorted(int(x.strip()) for x in nums_str.split(","))
                joker = int(joker_str.strip())
                assert len(nums) == 5 and all(1 <= n <= 45 for n in nums) and 1 <= joker <= 20
                subs.append({
                    "date": datetime.now().strftime("%Y-%m-%d"),
                    "numbers": nums,
                    "joker": joker,
                })
                save_submissions(subs)
                st.success("Added.")
            except (ValueError, AssertionError):
                st.error("Enter exactly 5 numbers (1-45) and one Joker number (1-20).")

    if not subs:
        st.caption("No submissions logged yet.")
        return

    st.subheader("History")
    for i, sub in enumerate(reversed(subs)):
        idx = len(subs) - 1 - i
        cols = st.columns([3, 1])
        with cols[0]:
            st.write(f"{sub['date']}")
            number_badges(sub["numbers"], joker=sub["joker"])
        with cols[1]:
            match = df[(df["date"] <= sub["date"])].tail(1)
            if st.button("Remove", key=f"rm_{idx}"):
                subs.pop(idx)
                save_submissions(subs)
                st.rerun()


def main():
    df, report = get_data()
    if df.empty:
        st.error(
            "No Joker_<year>.xlsx files found next to app.py. Place your historical "
            "draw files in this folder and reload."
        )
        return

    analyzer = JokerAnalyzer(df)

    st.sidebar.title("🎰 Tzoker Analysis")
    page = st.sidebar.radio(
        "Section",
        ["Dashboard", "Predictions", "AI Insights", "Backtest", "My Submissions"],
    )

    if page == "Dashboard":
        page_dashboard(df, report, analyzer)
    elif page == "Predictions":
        page_predictions(df, analyzer)
    elif page == "AI Insights":
        page_ai_insights(analyzer)
    elif page == "Backtest":
        page_backtest(analyzer)
    elif page == "My Submissions":
        page_submissions(df)


if __name__ == "__main__":
    main()
