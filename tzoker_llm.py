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
        # Only "5" and "5+1" have a prize that gets split among multiple winners
        # (5+1 is a variable jackpot; "5" is capped at EUR 2M total if >20 winners).
        # "4+1" and every other category is a FIXED payout regardless of how many
        # other people also win it - pot-splitting doesn't apply there at all.
        # Computed in Python, not left to the model to recall correctly.
        "prize_is_shared_for_this_category": target_category in ("5", "5+1"),
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
table, a target_category, prize_is_shared_for_this_category, and a budget_eur.

Your overall aim is a CONSISTENT, BUDGET-BOUNDED method applied the same way every time -
not chasing losses, not escalating stakes, not implying wins will exceed losses over time.
No selection method changes the fact that this is a negative-expected-value game; never
suggest otherwise, even implicitly, in pattern_notes or rationale.

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
- Pot-splitting tiebreak - ONLY apply this when prize_is_shared_for_this_category is \
true (categories "5" and "5+1"): among candidate numbers with similar scores, you may \
prefer numbers in the 32-45 range over 1-31 as a tiebreaker. Numbers 1-31 double as \
calendar dates and get picked disproportionately by other players via birthdates, so \
32-45 numbers reduce the chance of SPLITTING the prize with other winners if you win. \
This does NOT change your odds of winning - only what you'd receive if you do. If you \
use this reasoning, say so explicitly and label it as being about payout-if-you-win, not \
win probability. When prize_is_shared_for_this_category is false (e.g. "4+1", which pays \
a fixed EUR 2,500 regardless of how many others also win it), do NOT mention pot-splitting \
at all - it has no relevance and citing it would misstate how that category's prize works.
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


def _parse_claude_json(raw_text):
    """Strip markdown fences if present and parse JSON. Returns (dict, error)."""
    cleaned = raw_text.strip()
    if cleaned.startswith("```"):
        cleaned = cleaned.strip("`")
        if cleaned.startswith("json"):
            cleaned = cleaned[4:]
    try:
        return json.loads(cleaned), None
    except json.JSONDecodeError:
        return None, f"Couldn't parse Claude's response as JSON:\n\n{raw_text}"


def _validate_and_correct(result, raw_text, payload, analyzer, target_category, budget_eur):
    """
    Shared validation/correction pipeline used for both the initial pick and every
    chat-driven revision - every turn gets the exact same deterministic guarantees,
    not just the first one.
    """
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

    result["evidence_numbers"] = analyzer.number_evidence(numbers)
    result["evidence_jokers"] = analyzer.joker_evidence(jokers)

    exact_combos = comb(len(numbers), 5)
    exact_cost = round(exact_combos * len(jokers) * TICKET_PRICE_EUR, 2)
    result["system_size"] = len(numbers)
    result["estimated_cost_eur"] = exact_cost
    result["over_budget"] = exact_cost > budget_eur

    if target_category == "5+1":
        odds = _ODDS_BY_CATEGORY.get("5+1")
        result["caveat"] = (
            (result.get("caveat", "") or "").rstrip(". ") + ". "
            f"5+1 is the jackpot long shot: 1 in {odds:,} — far lower odds than "
            f"category \"5\" (1 in {_ODDS_BY_CATEGORY['5']:,}) or \"4+1\" "
            f"(1 in {_ODDS_BY_CATEGORY['4+1']:,})."
        ).strip()

    return result, None


CHAT_SYSTEM_PROMPT = SYSTEM_PROMPT + """

You are now in a follow-up conversation about the pick you already made. The user may
ask questions, or ask you to reconsider the pick - including based on a "cross_check"
field you may be given, showing how the currently-picked numbers rank (1=hottest of 45)
over other time windows (last 100 draws, last 1000, all-time) than the one you originally
scored against. This is still purely descriptive of the past, same as everything else -
a number ranking well in one window and poorly in another is not evidence either way
about the next draw, and you must not imply otherwise.

If the user asks you to reconsider, you may swap in a different number FROM
top_15_scored_numbers (never outside it), and should explain the swap in terms of the
historical pattern it reflects (e.g. "swapped 3 for 41: 3 ranks 37th all-time despite a
recent spike, while 41 ranks 7th all-time and 3rd recently"), not in terms of future
odds. If the user is just asking a question and not requesting a change, keep the
existing numbers/jokers exactly as they are.

Always respond with ONLY a JSON object (no markdown fences, no preamble), matching
exactly - always include the full pick even if unchanged:
{
  "reply": "conversational answer to the user's message, 1-4 sentences",
  "numbers_changed": true or false,
  "numbers": [current or revised list of ints from top_15_scored_numbers],
  "joker_numbers": [current or revised list of ints from top_5_scored_jokers],
  "system_size": int,
  "estimated_cost_eur": number,
  "pattern_notes": "as before",
  "rationale": "as before",
  "caveat": "as before"
}
"""


def ask_claude_chat(analyzer, payload, api_history, user_message, target_category, budget_eur,
                     api_key, cross_check=None, model=MODEL):
    """
    One turn of follow-up conversation about an existing pick. api_history is the list
    of prior {"role": ..., "content": ...} messages (Anthropic format) from this
    conversation so far - the caller owns and persists this across turns. Returns
    (result, updated_api_history, error).
    """
    try:
        import anthropic
    except ImportError:
        return None, api_history, "The 'anthropic' package isn't installed. Add it to requirements.txt."

    user_payload = {"user_message": user_message}
    if cross_check is not None:
        user_payload["cross_check"] = cross_check
    new_messages = api_history + [{"role": "user", "content": json.dumps(user_payload)}]

    try:
        client = anthropic.Anthropic(api_key=api_key)
        response = client.messages.create(
            model=model,
            max_tokens=800,
            system=CHAT_SYSTEM_PROMPT,
            messages=new_messages,
        )
        raw_text = "".join(
            block.text for block in response.content if getattr(block, "type", None) == "text"
        )
    except Exception as e:
        return None, api_history, f"Claude API call failed: {e}"

    result, error = _parse_claude_json(raw_text)
    if error:
        return None, api_history, error

    result, error = _validate_and_correct(result, raw_text, payload, analyzer, target_category, budget_eur)
    if error:
        return None, api_history, error

    updated_history = new_messages + [{"role": "assistant", "content": raw_text}]
    return result, updated_history, None


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

    result, error = _parse_claude_json(raw_text)
    if error:
        return None, error

    result, error = _validate_and_correct(result, raw_text, payload, analyzer, target_category, budget_eur)
    if error:
        return None, error

    return result, None
