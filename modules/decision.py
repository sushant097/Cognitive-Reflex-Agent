from typing import List, Optional
from modules.perception import PerceptionResult
from modules.memory import MemoryItem
from modules.model_manager import ModelManager
from modules.tools import load_prompt
import re

from modules.historical_index import load_similar_examples

# Optional logging fallback
try:
    from agent import log
except ImportError:
    import datetime
    def log(stage: str, msg: str):
        now = datetime.datetime.now().strftime("%H:%M:%S")
        print(f"[{now}] [{stage}] {msg}")

model = ModelManager()


async def generate_plan(
    user_input: str,
    perception: PerceptionResult,
    memory_items: List[MemoryItem],
    tool_descriptions: Optional[str],
    prompt_path: str,
    step_num: int = 1,
    max_steps: int = 3,
) -> str:
    """Generates the full solve() function plan for the agent."""

    # Session-local memory (optional)
    memory_texts = "\n".join(f"- {m.text}" for m in memory_items) or "None"

    # 1) Try direct reuse from historical memory
    examples = load_similar_examples(user_input)

    if examples:
        lines = []
        for ex in examples:
            fa = ex.get("final_answer") or ""
            if len(fa) > 500:
                fa = fa[:500] + "... [truncated]"
            lines.append(
                f"- Past query: {ex['user_query']}\n"
                f"  Tools: {', '.join(ex['tools_used'])}\n"
                f"  Outcome: {fa}"
            )
        historical_examples = "\n".join(lines)
    else:
        historical_examples = "None available"

    prompt_template = load_prompt(prompt_path)

    # Format kwargs guarded by placeholder checks
    format_kwargs = {
        "tool_descriptions": tool_descriptions,
        "user_input": user_input
    }
    if "{memory_texts}" in prompt_template:
        format_kwargs["memory_texts"] = memory_texts

    if "{historical_examples}" in prompt_template:
        format_kwargs["historical_examples"] = historical_examples

    prompt = prompt_template.format(**format_kwargs)

    snippet = prompt[:2500]# .replace("\n", " ")
    log("plan", f"Prompt snippet: {snippet}...")

    try:
        raw = (await model.generate_text(prompt)).strip()
        log("plan", f"LLM output: {raw}")

        # If wrapped in ```...``` strip fences first
        if raw.startswith("```"):
            # remove leading and trailing backticks
            raw = raw.strip("`").strip()
            # optional "python" language tag
            if raw.lower().startswith("python"):
                raw = raw[len("python"):].strip()

        # ----------------------------------------------------
        # 1) If there is a direct FINAL_ANSWER or FURTHER_...
        #    anywhere in the output AND there is NO solve(),
        #    treat that as the planner giving us a direct answer.
        # ----------------------------------------------------
        has_solve = bool(re.search(r"^\s*(async\s+)?def\s+solve\s*\(", raw, re.MULTILINE))

        direct_final = re.search(r"^\s*(FINAL_ANSWER:.*)$", raw, re.MULTILINE)
        if direct_final and not has_solve:
            answer = direct_final.group(1).strip()
            log("plan", f"Using direct FINAL_ANSWER from planner: {answer}")
            return answer

        direct_fpr = re.search(r"^\s*(FURTHER_PROCESSING_REQUIRED:.*)$", raw, re.MULTILINE)
        if direct_fpr and not has_solve:
            answer = direct_fpr.group(1).strip()
            log("plan", f"Using direct FURTHER_PROCESSING_REQUIRED from planner: {answer}")
            return answer

        # ----------------------------------------------------
        # 2) Otherwise, expect a proper async def solve(...)
        # ----------------------------------------------------
        if has_solve:
            return raw  # valid plan code

        # ----------------------------------------------------
        # 3) Fallback: neither solve() nor a clean direct answer
        # ----------------------------------------------------
        log("plan", "⚠️ LLM did not return solve() or a clean FINAL_ANSWER/FURTHER_PROCESSING_REQUIRED. Defaulting to fallback.")
        return "FINAL_ANSWER: [Could not generate valid solve()]"

    except Exception as e:
        log("plan", f"⚠️ Planning failed: {e}")
        return "FINAL_ANSWER: [unknown]"