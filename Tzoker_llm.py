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

MODEL = "claude-sonnet-5"
TICKET_PRICE_EUR = 1.00  # confirmed current Allwyn price per line, Aug 2026


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
            {"number": n, "score": round(scores[n], 4)} for n in ranked[:15]
        ],
        "top_5_scored_jokers": [
            {"joker": j, "score": round(jscores[j], 4)} for j in ranked_jokers[:5]
        ],
        "pattern_stats_last_200_draws": {
            k: (round(v, 2) if v is not None else None)
            for k, v in analyzer.pattern_stats(last_n_draws=200).items()
        },
    }


SYSTEM_PROMPT = """You are a statistics-literate assistant helping organize a Greek Tzoker \
(Τζόκερ) lottery play. You will be given precomputed statistical scores over the real \
historical draw data, and a pre-built system-cost table.

Hard constraints, non-negotiable:
- Tzoker draws are independent random events. Nothing in past draws changes the odds of \
the next one. You must never claim or imply the numbers you pick are more likely to be \
drawn next.
- Do NOT invent numbers or statistics. Choose the numbers you recommend FROM the \
top_15_scored_numbers and top_5_scored_jokers lists you are given - do not substitute \
numbers that aren't in those lists.
- Do NOT do your own combinatorics. Pick the system size from the given \
system_cost_table rows whose cost fits within budget_eur; do not compute your own cost.
- Your job is to (a) briefly describe what the provided scores show in plain language, \
and (b) synthesize a concrete pick + system size that fits the stated budget and target \
category, then (c) restate the independence caveat.

Respond with ONLY a JSON object (no markdown fences, no preamble), matching exactly:
{
  "numbers": [list of ints, length = chosen system size, drawn only from top_15_scored_numbers],
  "joker_numbers": [list of 1-3 ints, drawn only from top_5_scored_jokers],
  "system_size": int,
  "estimated_cost_eur": number,
  "pattern_notes": "1-3 sentences describing the observed historical scores/frequency pattern",
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

    result["numbers"] = sorted(set(numbers))
    result["joker_numbers"] = sorted(set(jokers))
    return result, None
