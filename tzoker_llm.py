"""
Claude (Anthropic API) integration for the Tzoker app.

Important framing, kept consistent with the rest of this app: Tzoker draws
are independent random events, so no analysis - statistical or LLM - can
predict the next one. What Claude actually does here:

  1. Summarizes/explains the ALREADY-COMPUTED statistical scores from
     tzoker_core.JokerAnalyzer in plain language (it is not asked to invent
     its own numbers from nothing - it's given the real frequency/recency/
     overdue scores and picks from within them, so the output is grounded
     and auditable, not hallucinated).
  2. Given a target category and budget, recommends a system size using a
     precomputed cost table (Claude is not asked to do the combinatorics
     itself - that's exact math, done in Python, and just handed to it).

Requires ANTHROPIC_API_KEY to be set as an environment variable (Railway:
set it in the project's Variables tab) or in .streamlit/secrets.toml for
local runs. Never hardcode a key in source.
"""

import json
import os
from math import comb

import streamlit as st

from tzoker_core import PRIZE_CATEGORIES

MODEL = "claude-sonnet-5"
TICKET_PRICE_EUR = 1.00  # confirmed current Allwyn price per line, Aug 2026
_ODDS_BY_CATEGORY = {c["key"]: c["odds_1_in"] for c in PRIZE_CATEGORIES}


def get_api_key():
    """Look for the key in Streamlit secrets, then the environment, then session_state."""
    try:
        if "ANTHROPIC_API_KEY" in st.secrets:
            return st.secrets["ANTHROPIC_API_KEY"]
    except Exception:
        pass
    if os.environ.get("ANTHROPIC_API_KEY"):
        return os.environ["ANTHROPIC_API_KEY"]
    return st.session_state.get("_anthropic_key_input")


def system_cost_table(max_size=10):
    """Full-wheel system size -> combinations -> cost, for a single joker number."""
    rows = []
    for n in range(5, max_size + 1):
        combos = comb(n, 5)
        rows.append({
            "numbers_played": n,
            "combinations": combos,
            "cost_eur_1_joker": round(combos * TICKET_PRICE_EUR, 2),
        })
    return rows


def build_analysis_payload(analyzer, target_category, budget_eur):
    scores = analyzer.score_numbers()
    ranked = sorted(scores, key=scores.get, reverse=True)
    jscores = analyzer.score_jokers()
    ranked_jokers = sorted(jscores, key=jscores.get, reverse=True)

    top15 = ranked[:15]
    top5j = ranked_jokers[:5]

    return {
        "total_draws_in_history": len(analyzer.all_draws),
        "date_range": [
            str(analyzer.all_draws["date"].min().date()),
            str(analyzer.all_draws["date"].max().date()),
        ],
        "target_category": target_category,
        "budget_eur": budget_eur,
        "ticket_price_eur": TICKET_PRICE_EUR,
        "system_cost_table": system_cost_table(),
        "top_15_scored_numbers": [
            {"number": n, "score": round(scores[n], 4)} for n in top15
        ],
        "top_5_scored_jokers": [
            {"joker": j, "score": round(jscores[j], 4)} for j in top5j
        ],
        # Ground-truth per-number evidence for every candidate Claude is allowed to
        # choose from - the exact real counts/percentages/gaps, computed the same
        # way regardless of what Claude says about them, so its prose can be
        # checked against real numbers rather than trusted at face value.
        "evidence_for_candidate_numbers": analyzer.number_evidence(top15),
        "evidence_for_candidate_jokers": analyzer.joker_evidence(top5j),
        "distribution_stats_last_200_draws": analyzer.distribution_stats(last_n_draws=200),
    }


SYSTEM_PROMPT = """You are a statistics-literate assistant helping organize a Greek Tzoker \
(Τζόκερ) lottery play. You will be given: precomputed statistical scores, ground-truth \
per-number evidence (real counts/percentages/gaps), distribution stats, a system-cost \
table, a target_category, and a budget_eur.

Hard constraints, non-negotiable:
- Every Tzoker combination of 5 numbers from 45 is EXACTLY as likely to be drawn as any \
other - this is mathematically true regardless of any historical frequency. You must \
NEVER state or imply that a number or combination has "bigger odds", "better chances", \
is "more likely to win", or anything equivalent, for a FUTURE draw. The only valid odds \
figures are the fixed category odds (e.g. "5" is 1 in 1,221,759) - those don't change \
based on which numbers you pick.
- What you ARE allowed to say: a number "appeared more often historically" or "scores \
higher on the frequency/recency/overdue formula" - purely descriptive of the past, never \
predictive of the future. Every such claim must cite the specific value it's based on \
from evidence_for_candidate_numbers / evidence_for_candidate_jokers (e.g. "27 appeared \
in 11.4% of all draws, above the ~11.1% expected by chance" - not just "27 is hot").
- Do NOT invent numbers or statistics. Choose the numbers you recommend FROM the \
top_15_scored_numbers and top_5_scored_jokers lists you are given - do not substitute \
numbers that aren't in those lists.
- Do NOT do your own combinatorics. Pick the system size from the given \
system_cost_table rows whose cost fits within budget_eur; do not compute your own cost. \
(Your system_size and estimated_cost_eur will be independently recomputed and corrected \
in Python after you respond, so get them as close as you can, but do not worry if exact -
the numbers/joker_numbers you choose are what matters most.)
- If budget_eur is below the cheapest row in system_cost_table (a plain 5-number system), \
still return that cheapest 5-number system and say so plainly in rationale.
- target_category changes what you should optimize for:
    - "5": maximize system_size within budget (more numbers = more 5-number \
      sub-combinations = more chances at matching all 5). Use exactly 1 joker number - \
      extra joker coverage does not help this category.
    - "4+1": both the 5-number sub-combination AND the joker must be right, so balance \
      system_size against covering 2-3 joker numbers from top_5_scored_jokers, within \
      budget.
    - "5+1": explicitly state in rationale that this is the jackpot long-shot category \
      (odds 1 in 24,435,180 vs 1 in 1,221,759 for "5" and 1 in 122,176 for "4+1"), and \
      keep the system modest rather than spending the whole budget chasing it.

Respond with ONLY a JSON object (no markdown fences, no preamble), matching exactly:
{
  "numbers": [list of ints, length = chosen system size, drawn only from top_15_scored_numbers],
  "joker_numbers": [list of 1-3 ints, drawn only from top_5_scored_jokers],
  "system_size": int,
  "estimated_cost_eur": number,
  "pattern_notes": "1-3 sentences, each citing a specific value from the evidence fields, describing the observed historical pattern - never framed as future likelihood",
  "rationale": "1-3 sentences on why this system size fits the target category and budget",
  "caveat": "1 sentence restating that draws are independent and this doesn't predict the future"
}
"""


def ask_claude_for_pick(analyzer, target_category, budget_eur, api_key, model=MODEL):
    try:
        import anthropic
    except ImportError:
        return None, "The 'anthropic' package isn't installed. Add it to requirements.txt."

    payload = build_analysis_payload(analyzer, target_category, budget_eur)

    try:
        client = anthropic.Anthropic(api_key=api_key)
        response = client.messages.create(
            model=model,
            max_tokens=800,
            system=SYSTEM_PROMPT,
            messages=[{"role": "user", "content": json.dumps(payload)}],
        )
        raw_text = "".join(
            block.text for block in response.content if getattr(block, "type", None) == "text"
        )
    except Exception as e:
        return None, f"Claude API call failed: {e}"

    cleaned = raw_text.strip()
    if cleaned.startswith("```"):
        cleaned = cleaned.strip("`")
        if cleaned.startswith("json"):
            cleaned = cleaned[4:]
    try:
        result = json.loads(cleaned)
    except json.JSONDecodeError:
        return None, f"Couldn't parse Claude's response as JSON:\n\n{raw_text}"

    valid_numbers = {row["number"] for row in payload["top_15_scored_numbers"]}
    valid_jokers = {row["joker"] for row in payload["top_5_scored_jokers"]}
    numbers = [n for n in result.get("numbers", []) if n in valid_numbers]
    jokers = [j for j in result.get("joker_numbers", []) if j in valid_jokers]

    if len(numbers) < 5 or not jokers:
        return None, (
            "Claude's response didn't include enough valid numbers/jokers from the "
            f"provided list. Raw response:\n\n{raw_text}"
        )

    numbers = sorted(set(numbers))
    jokers = sorted(set(jokers))
    result["numbers"] = numbers
    result["joker_numbers"] = jokers

    # Attach the real evidence for exactly the numbers/jokers Claude actually chose,
    # computed independently in Python - this is what a reviewer (or the user) checks
    # Claude's prose against, rather than trusting Claude to have restated the figures
    # from the payload accurately.
    result["evidence_numbers"] = analyzer.number_evidence(numbers)
    result["evidence_jokers"] = analyzer.joker_evidence(jokers)

    # Don't trust the model's own arithmetic for these two fields - recompute them
    # deterministically from what it actually returned, exactly as build_full_system
    # would. This is the same math tzoker_core.build_full_system uses; duplicated
    # here (rather than imported) to avoid a circular import with tzoker_core.
    exact_combos = comb(len(numbers), 5)
    exact_cost = round(exact_combos * len(jokers) * TICKET_PRICE_EUR, 2)
    result["system_size"] = len(numbers)
    result["estimated_cost_eur"] = exact_cost
    result["over_budget"] = exact_cost > budget_eur

    # Don't rely on the model to remember the long-shot framing - enforce it.
    if target_category == "5+1":
        odds = _ODDS_BY_CATEGORY.get("5+1")
        result["caveat"] = (
            (result.get("caveat", "") or "").rstrip(". ") + ". "
            f"5+1 is the jackpot long shot: 1 in {odds:,} — far lower odds than "
            f"category \"5\" (1 in {_ODDS_BY_CATEGORY['5']:,}) or \"4+1\" "
            f"(1 in {_ODDS_BY_CATEGORY['4+1']:,})."
        ).strip()

    return result, None
