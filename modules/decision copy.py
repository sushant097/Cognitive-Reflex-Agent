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
    examples = load_similar_examples(user_input, top_k=3)

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

        # If fenced in ```python ... ```, extract
        if raw.startswith("```"):
            raw = raw.strip("`").strip()
            if raw.lower().startswith("python"):
                raw = raw[len("python"):].strip()

        # accept direct FINAL_ANSWER from planner
        if raw.startswith("FINAL_ANSWER:"):
            log("plan", "✅ Direct answer from planner, no solve() needed.")
            return raw

        if re.search(r"^\s*(async\s+)?def\s+solve\s*\(", raw, re.MULTILINE):
            return raw  # correct, it is a full function
        else:
            log("plan", "⚠️ LLM did not return a valid solve(). Defaulting to FINAL_ANSWER")
            return "FINAL_ANSWER: [Could not generate valid solve()]"

    except Exception as e:
        log("plan", f"⚠️ Planning failed: {e}")
        return "FINAL_ANSWER: [unknown]"
