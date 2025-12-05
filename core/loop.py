# modules/loop.py

import asyncio
import re
from pathlib import Path
from typing import Optional

from core import context
from modules.perception import run_perception
from modules.decision import generate_plan
from modules.action import run_python_sandbox
from modules.model_manager import ModelManager
from core.session import MultiMCP
from core.strategy import select_decision_prompt_path
from core.context import AgentContext
from modules.tools import summarize_tools

from modules.historical_index import (
    update_index_for_session,
    find_best_cached_answer,
)

try:
    from agent import log
except ImportError:
    import datetime

    def log(stage: str, msg: str):
        now = datetime.datetime.now().strftime("%H:%M:%S")
        print(f"[{now}] [{stage}] {msg}")


# 🔧 Global verbose toggle (will be set from profiles.yaml at runtime)
VERBOSE_LOG = False


def vlog(stage: str, msg: str) -> None:
    """Verbose logger: only prints when VERBOSE_LOG is True."""
    if VERBOSE_LOG:
        log(stage, msg)


class AgentLoop:
    def __init__(self, context: AgentContext):
        self.context = context
        self.mcp = self.context.dispatcher
        self.model = ModelManager()

        # ---- Read custom config from profiles.yaml (with safe defaults) ----
        cfg = getattr(self.context.agent_profile, "custom_config", None)

        self.verbose_logging = False
        self.jaccard_similarity_threshold = 0.80
        self.memory_index_file = "memory/historical_conversation_store.json"

        if cfg is not None:
            # profiles.yaml:
            # custom_config:
            #   jaccard_similarity_threshold: 0.85
            #   verbose_logging: true
            #   memory_index_file: "memory/historical_conversation_store.json"
            self.verbose_logging = getattr(cfg, "verbose_logging", False)
            self.jaccard_similarity_threshold = getattr(
                cfg, "jaccard_similarity_threshold", 0.85
            )
            self.memory_index_file = getattr(
                cfg, "memory_index_file", "memory/historical_conversation_store.json"
            )

        # Derive memory_root from memory_index_file path
        # e.g., "memory/historical_conversation_store.json" -> "memory"
        self.memory_root = str(Path(self.memory_index_file).parent)

        # set global verbose log flag
        global VERBOSE_LOG
        VERBOSE_LOG = bool(self.verbose_logging)
        # verbose log initial settings
        vlog(
            "initial params",
            f"Initialized with jaccard_similarity_threshold={self.jaccard_similarity_threshold}, "
            f"memory_index_file={self.memory_index_file}"
        )

    def _update_historical_index(self):
        """
        Incrementally update historical_conversation_store.json
        for the current session.
        """
        try:
            mm = self.context.memory  # MemoryManager
            update_index_for_session(mm.memory_path, mm.session_id)
        except Exception as e:
            log("history", f"⚠️ Failed to update historical index: {e}")

    async def _finalize_from_content(self, original_question: str, tool_result: str) -> str:
        """
        Turn raw tool output + user question into a concise final answer.

        This is called when we've hit the FURTHER_PROCESSING_REQUIRED limit.
        It is fully generic.
        """

        prompt = f"""
You are a careful but concise assistant.

The user asked:
{original_question}

You have the following context from tools (search results, excerpts, or documents).
It may include URLs, summaries, long snippets, or even messages like "no results found":

{tool_result}

Follow this process strictly:

1. Understand what the user is actually asking for.

2. Read the context carefully and look for ANY information that helps answer the question.

3. If the context clearly contains enough information to answer, give a direct,
   specific answer in 1–2 sentences.

4. If the context is partial but still gives clues, synthesize the best possible
   answer and mention that it is based on limited information.

5. Only answer exactly "unknown" (or equivalent) if:
   - The context is truly unrelated or explicitly says no info was found, AND
   - After reading everything, you find no useful evidence.

Important:
- Prefer giving a best-effort answer over saying "unknown" whenever the
  context contains any relevant evidence.
- Do NOT repeat large chunks of the context. Just state the conclusion.
- Final output must be ONLY the answer text.

Now, provide your final answer. If long answer multi-paragraph, summarize in one sentence.
"""

        try:
            text = await self.model.generate_text(prompt)
            return text.strip()
        except Exception as e:
            log("loop", f"⚠️ Finalization failed: {e}")
            # Fallback: at least return some of the raw tool_result
            return tool_result[:2000]

    async def run(self):
        # ---------------------------------------------------------
        # Sync global VERBOSE_LOG with profiles.yaml
        # ---------------------------------------------------------
        global VERBOSE_LOG
        VERBOSE_LOG = bool(self.verbose_logging)

        # ---------------------------------------------------------
        # 0) Try SEMANTIC CACHE first (no perception, no tools)
        # ---------------------------------------------------------
        original_query = self.context.user_input or ""

        # Use threshold + memory_root from profiles.yaml
        semantic_hit = find_best_cached_answer(
            user_query=original_query,
            min_similarity=self.jaccard_similarity_threshold,
            memory_root=self.memory_root,
        )

        if semantic_hit:
            log("loop", "🔁 Cache hit – returning stored FINAL_ANSWER (no new history).")
            # semantic_hit already starts with FINAL_ANSWER:
            self.context.final_answer = semantic_hit.strip()
            # Log into current session memory so this turn is represented
            # self.context.memory.add_final_answer(self.context.final_answer)
            # self._update_historical_index()
            return {"status": "done", "result": self.context.final_answer}

        # ---------------------------------------------------------
        # 1) Normal multi-step loop
        # ---------------------------------------------------------

        max_steps = self.context.agent_profile.strategy.max_steps
        allowed_fpr_uses = max_steps - 1  # e.g., 2 when max_steps = 3
        self.further_processing_uses = 0

        for step in range(max_steps):
            vlog("loop", f"🔁 Step {step+1}/{max_steps} starting...")
            self.context.step = step
            lifelines_left = self.context.agent_profile.strategy.max_lifelines_per_step

            while lifelines_left >= 0:
                # === Perception ===
                user_input_override = getattr(self.context, "user_input_override", None)
                perception = await run_perception(
                    context=self.context,
                    user_input=user_input_override or self.context.user_input,
                )

                vlog("perception", f"{perception}")

                selected_servers = perception.selected_servers
                selected_tools = self.mcp.get_tools_from_servers(selected_servers)

                # Check if we are currently in a content summarization step (Step 2/3)
                # This is true if user_input_override is set (i.e., we have content to process).
                is_summarizing = bool(getattr(self.context, "user_input_override", None))

                # FIX: Only abort if we are NOT summarizing AND no tools were selected.
                if not selected_tools and not is_summarizing:
                    log("loop", "⚠️ No tools selected — aborting step.")
                    break


                effective_user_input = user_input_override or self.context.user_input

                # === Planning ===
                tool_descriptions = summarize_tools(selected_tools)
                prompt_path = select_decision_prompt_path(
                    planning_mode=self.context.agent_profile.strategy.planning_mode,
                    exploration_mode=self.context.agent_profile.strategy.exploration_mode,
                )

                plan = await generate_plan(
                    user_input=effective_user_input,
                    perception=perception,
                    memory_items=self.context.memory.get_session_items(),
                    tool_descriptions=tool_descriptions,
                    prompt_path=prompt_path,
                    step_num=step + 1,
                    max_steps=max_steps,
                )
                vlog("plan", f"{plan}")

                # === Execution ===

                # 0) Direct FINAL_ANSWER / FURTHER_PROCESSING from planner (no sandbox)
                if isinstance(plan, str) and (
                    plan.startswith("FINAL_ANSWER:")
                    or plan.startswith("FURTHER_PROCESSING_REQUIRED:")
                ):
                    log("loop", "✅ Planner returned direct answer, skipping sandbox.")
                    self.context.final_answer = plan
                    self.context.memory.add_final_answer(self.context.final_answer)
                    self._update_historical_index()
                    return {"status": "done", "result": self.context.final_answer}

                # 1) Normal case: LLM returned a solve() function (code plan)
                if re.search(r"^\s*(async\s+)?def\s+solve\s*\(", plan, re.MULTILINE):
                    vlog("loop", "[loop] Detected solve() plan — running sandboxed...")

                    self.context.log_subtask(tool_name="solve_sandbox", status="pending")
                    result = await run_python_sandbox(plan, dispatcher=self.mcp)

                    success = False
                    if isinstance(result, str):
                        result = result.strip()
                        if result.startswith("FINAL_ANSWER:"):
                            success = True
                            self.context.final_answer = result
                            self.context.update_subtask_status("solve_sandbox", "success")
                            self.context.memory.add_tool_output(
                                tool_name="solve_sandbox",
                                tool_args={"plan": plan},
                                tool_result={"result": result},
                                success=True,
                                tags=["sandbox"],
                            )
                            self.context.memory.add_final_answer(self.context.final_answer)
                            self._update_historical_index()
                            return {"status": "done", "result": self.context.final_answer}

                        elif result.startswith("FURTHER_PROCESSING_REQUIRED:"):
                            content = result.split("FURTHER_PROCESSING_REQUIRED:")[1].strip()
                            self.further_processing_uses += 1

                            if self.further_processing_uses <= allowed_fpr_uses:
                                # Forward intermediate result into next step
                                self.context.user_input_override = (
                                    f"Original user task: {self.context.user_input}\n\n"
                                    f"Your last tool produced this result:\n\n"
                                    f"{content}\n\n"
                                    f"If this fully answers the task, return:\n"
                                    f"FINAL_ANSWER: your answer\n\n"
                                    f"Otherwise, return the next FUNCTION_CALL."
                                )
                                vlog(
                                    "loop",
                                    "📨 Forwarding intermediate result to next step.",
                                )
                                vlog(
                                    "loop",
                                    f"🔁 Continuing based on FURTHER_PROCESSING_REQUIRED — Step {step+1} continues...",
                                )
                                break  # go to next step

                            else:
                                # Exceeded FPR budget: summarize and stop
                                log(
                                    "loop",
                                    "⚠️ FURTHER_PROCESSING_REQUIRED exceeded budget — forcing FINAL_ANSWER via summarization.",
                                )
                                final_answer = await self._finalize_from_content(
                                    original_question=self.context.user_input,
                                    tool_result=content,
                                )
                                self.context.final_answer = f"FINAL_ANSWER: {final_answer}"
                                self.context.memory.add_final_answer(self.context.final_answer)
                                self._update_historical_index()
                                return {
                                    "status": "done",
                                    "result": self.context.final_answer,
                                }

                        elif result.startswith("[sandbox error:"):
                            success = False
                            self.context.final_answer = "FINAL_ANSWER: [Execution failed]"
                        else:
                            success = True
                            self.context.final_answer = f"FINAL_ANSWER: {result}"
                    else:
                        self.context.final_answer = f"FINAL_ANSWER: {result}"

                    if success:
                        self.context.update_subtask_status("solve_sandbox", "success")
                    else:
                        self.context.update_subtask_status("solve_sandbox", "failure")

                    self.context.memory.add_tool_output(
                        tool_name="solve_sandbox",
                        tool_args={"plan": plan},
                        tool_result={"result": result},
                        success=success,
                        tags=["sandbox"],
                    )

                    if success and "FURTHER_PROCESSING_REQUIRED:" not in result:
                        self.context.memory.add_final_answer(self.context.final_answer)
                        self._update_historical_index()
                        return {"status": "done", "result": self.context.final_answer}
                    else:
                        lifelines_left -= 1
                        vlog("loop", f"🛠 Retrying... Lifelines left: {lifelines_left}")
                        continue
                else:
                    vlog(
                        "loop",
                        f"⚠️ Invalid plan detected — retrying... Lifelines left: {lifelines_left-1}",
                    )
                    lifelines_left -= 1
                    continue

        log("loop", "⚠️ Max steps reached without finding final answer.")
        self.context.final_answer = "FINAL_ANSWER: [Max steps reached]"
        self.context.memory.add_final_answer(self.context.final_answer)
        self._update_historical_index()
        return {"status": "done", "result": self.context.final_answer}
