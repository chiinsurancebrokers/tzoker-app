"""
OpenAI (ChatGPT) integration for the Tzoker app - second-opinion validator.

Deliberately scoped narrower than the Claude module: this does not generate
its own independent pick. Its only job is to check Claude's pick against the
same real statistical data and flag anything wrong or worth a second look
(numbers outside the scored list, poor budget fit, an overstated claim in
Claude's own text, etc). Two models agreeing isn't stronger evidence that a
combination will win - it's just two reviews of the same arithmetic.

Requires OPENAI_API_KEY (env var / secrets.toml / session paste), same
pattern as ANTHROPIC_API_KEY in tzoker_llm.py.
"""

import json
import os

import streamlit as st

OPENAI_MODEL = "gpt-5"


def get_openai_api_key():
    try:
        if "OPENAI_API_KEY" in st.secrets:
            return st.secrets["OPENAI_API_KEY"]
    except Exception:
        pass
    if os.environ.get("OPENAI_API_KEY"):
        return os.environ["OPENAI_API_KEY"]
    return st.session_state.get("_openai_key_input")


REVIEW_SYSTEM_PROMPT = """You are a skeptical reviewer checking another AI's Tzoker \
(Greek lottery) pick for correctness, NOT generating your own pick from scratch.

You are given:
- the same precomputed statistical scores (top_15_scored_numbers, top_5_scored_jokers),
  evidence_for_candidate_numbers/jokers, distribution_stats, system_cost_table,
  target_category, and budget_eur that the other AI was given
- the other AI's response, including evidence_numbers/evidence_jokers - the REAL
  counts/percentages/gaps for exactly the numbers it chose, attached independently in
  Python (not written by the other AI, so these are ground truth)

Note: system_size and estimated_cost_eur have already been independently recomputed in
Python from the returned numbers/joker_numbers, so you do not need to re-check that
arithmetic - it is guaranteed correct. Focus your review on things Python can't verify:

1. Cross-check every quantitative claim in pattern_notes against evidence_numbers /
   evidence_jokers. If pattern_notes says "27 appeared in 11.4% of draws" or similar,
   does that match the real overall_pct/recent_count/draws_since_last_seen in the
   evidence? Flag any number that's wrong or unsupported.
2. Does pattern_notes or rationale cross the line from "appeared more often
   historically" into implying the pick is more likely to WIN a FUTURE draw? This is
   the single most important check - every combination is equally likely regardless of
   history.
3. Does the strategy actually match target_category?
   - "5": system should favor a larger system_size (more numbers) with a single joker,
     since extra joker coverage doesn't help this category.
   - "4+1": should show some balance between system_size and 2-3 joker numbers.
   - "5+1": rationale or caveat must explicitly note this is the jackpot long-shot
     category with much lower odds than "5" or "4+1".
4. If over_budget is true, does rationale/caveat acknowledge the system exceeds the
   stated budget?
5. Is the independence caveat present and accurate?

Do not propose alternative numbers. Do not claim any pick is more or less likely to
win future draws - that would repeat the same mistake you're checking for.

Respond with ONLY a JSON object (no markdown fences), matching exactly:
{
  "verdict": "pass" | "issues_found",
  "checks": [
    {"check": "short name", "ok": true/false, "note": "1 sentence"}
  ],
  "summary": "1-2 sentence overall verdict in plain language"
}
"""


def review_pick_with_chatgpt(analysis_payload, claude_result, api_key, model=OPENAI_MODEL):
    try:
        import openai
    except ImportError:
        return None, "The 'openai' package isn't installed. Add it to requirements.txt."

    user_content = json.dumps({
        "given_to_other_ai": analysis_payload,
        "other_ai_response": claude_result,
    })

    try:
        client = openai.OpenAI(api_key=api_key)
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": REVIEW_SYSTEM_PROMPT},
                {"role": "user", "content": user_content},
            ],
        )
        raw_text = response.choices[0].message.content
    except Exception as e:
        return None, f"OpenAI API call failed: {e}"

    cleaned = raw_text.strip()
    if cleaned.startswith("```"):
        cleaned = cleaned.strip("`")
        if cleaned.startswith("json"):
            cleaned = cleaned[4:]
    try:
        result = json.loads(cleaned)
    except json.JSONDecodeError:
        return None, f"Couldn't parse ChatGPT's response as JSON:\n\n{raw_text}"

    if "verdict" not in result or "checks" not in result:
        return None, f"ChatGPT's response was missing expected fields:\n\n{raw_text}"

    return result, None
