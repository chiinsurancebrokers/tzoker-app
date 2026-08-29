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
- the same precomputed statistical scores (top_15_scored_numbers, top_5_scored_jokers)
  and system_cost_table that the other AI was given
- the other AI's response (numbers, joker_numbers, system_size, estimated_cost_eur,
  pattern_notes, rationale, caveat)

Check for concrete problems only:
1. Are all chosen numbers actually present in top_15_scored_numbers, and jokers in
   top_5_scored_jokers?
2. Does estimated_cost_eur actually match system_cost_table for that system_size?
3. Does the system_size fit within the stated budget_eur?
4. Does pattern_notes or rationale overstate what a frequency score can tell you - i.e.
   does it imply the numbers are "more likely" to be drawn next, rather than just
   describing historical frequency? Flag this if so - it's the single most important
   check.
5. Is the caveat about draw independence actually present and accurate?

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
