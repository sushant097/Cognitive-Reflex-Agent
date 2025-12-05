import json
import re
import difflib
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

# -------------------------------------------------------------------
# Optional logging from agent.py
# -------------------------------------------------------------------
try:
    from agent import log
except ImportError:
    import datetime

    def log(stage: str, msg: str) -> None:
        now = datetime.datetime.now().strftime("%H:%M:%S")
        print(f"[{now}] [{stage}] {msg}")


# -------------------------------------------------------------------
# Optional YAML support (for config/profiles.yaml)
# -------------------------------------------------------------------
try:
    import yaml
except ImportError:  # pragma: no cover - very small fallback
    yaml = None


# -------------------------------------------------------------------
# Data model
# -------------------------------------------------------------------

@dataclass
class HistoricalExample:
    session_id: str
    turn_index: int
    user_query: str
    final_answer: str
    tools_used: List[str]
    successful_tools: List[str]
    tags: List[str]
    keywords: List[str]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "session_id": self.session_id,
            "turn_index": self.turn_index,
            "user_query": self.user_query,
            "final_answer": self.final_answer,
            "tools_used": self.tools_used,
            "successful_tools": self.successful_tools,
            "tags": self.tags,
            "keywords": self.keywords,
        }


# -------------------------------------------------------------------
# Config helpers
# -------------------------------------------------------------------

_DEFAULT_INDEX_PATH = Path("memory") / "historical_conversation_store.json"


def _load_profiles_custom_config() -> Dict[str, Any]:
    """
    Load custom_config from config/profiles.yaml, if available.

    Expected structure:

      custom_config:
        jaccard_similarity_threshold: 0.85
        verbose_logging: true
        memory_index_file: "memory/historical_conversation_store.json"
    """
    if yaml is None:
        return {}

    profiles_path = Path("config") / "profiles.yaml"
    if not profiles_path.exists():
        return {}

    try:
        with profiles_path.open("r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
    except Exception as e:
        log("history", f"Failed to read profiles.yaml: {e}")
        return {}

    if not isinstance(data, dict):
        return {}

    cc = data.get("custom_config") or {}
    return cc if isinstance(cc, dict) else {}


def _get_index_path() -> Path:
    """
    Single source of truth for where the historical index lives.

    Priority:
    1. custom_config.memory_index_file from config/profiles.yaml
    2. Fallback: memory/historical_conversation_store.json
    """
    cfg = _load_profiles_custom_config()
    path_str = cfg.get("memory_index_file")
    if isinstance(path_str, str) and path_str.strip():
        return Path(path_str.strip())
    return _DEFAULT_INDEX_PATH

def _get_top_k_value() -> int:
    """
    Get the top_k value from custom_config.memory_index_file from config/profiles.yaml
    Fallback to 3 if not set.
    """
    cfg = _load_profiles_custom_config()
    top_k = cfg.get("top_k_similar_examples")
    if isinstance(top_k, int) and top_k > 0:
        return top_k
    return 3


# -------------------------------------------------------------------
# Text normalization + similarity
# -------------------------------------------------------------------

_STOPWORDS = {
    "the", "is", "a", "an", "of", "and", "or", "to", "in", "on", "for",
    "with", "at", "by", "from", "as", "about", "what", "which", "who",
    "how", "much", "many", "when", "where", "why", "do", "does", "did",
    "his", "her", "their", "its", "this", "that", "these", "those",
}


def _normalize_text(text: str) -> List[str]:
    """
    Very lightweight tokenizer + stopword removal for keyword extraction.
    """
    text = text.lower()
    tokens = re.findall(r"[a-z0-9]+", text)
    return [t for t in tokens if t not in _STOPWORDS]


def _normalize_for_similarity(text: str) -> str:
    """
    Normalize a string for semantic-ish similarity comparison.
    """
    text = text.lower().strip()
    text = re.sub(r"\s+", " ", text)
    return text


def _string_similarity(a: str, b: str) -> float:
    """
    Character-level similarity (SequenceMatcher), for paraphrase-style checks.
    """
    return difflib.SequenceMatcher(
        None,
        _normalize_for_similarity(a),
        _normalize_for_similarity(b),
    ).ratio()


# -------------------------------------------------------------------
# Index I/O
# -------------------------------------------------------------------

def _load_index(index_path: Optional[Path] = None) -> List[Dict[str, Any]]:
    """
    Load the global historical index as a list of dicts.
    """
    if index_path is None:
        index_path = _get_index_path()

    if not index_path.exists():
        return []

    try:
        with index_path.open("r", encoding="utf-8") as f:
            data = json.load(f)
        return data if isinstance(data, list) else []
    except Exception as e:
        log("history", f"Failed to load historical index: {e}")
        return []


def _save_index(index_path: Optional[Path], items: List[Dict[str, Any]]) -> None:
    """
    Save the historical index to disk.
    """
    if index_path is None:
        index_path = _get_index_path()

    index_path.parent.mkdir(parents=True, exist_ok=True)
    with index_path.open("w", encoding="utf-8") as f:
        json.dump(items, f, indent=2, ensure_ascii=False)


# -------------------------------------------------------------------
# Parsing session memory
# -------------------------------------------------------------------

def _extract_user_query(run_text: str) -> Optional[str]:
    """
    Example run_metadata text:
      "Started new session with input: What is X? at 2025-11-28T20:07:31.568388"
    """
    marker = "Started new session with input:"
    if marker not in run_text:
        return None
    after = run_text.split(marker, 1)[1].strip()
    if " at " in after:
        after = after.split(" at ", 1)[0].strip()
    return after or None


def _extract_final_answer(entry: Dict[str, Any]) -> Optional[str]:
    """
    Look for a clean FINAL_ANSWER inside a tool_output entry.
    We avoid indexing partial / failed runs.
    """
    # Prefer explicit "final_answer" field if present
    fa = entry.get("final_answer")
    if isinstance(fa, str) and fa.startswith("FINAL_ANSWER:"):
        return fa

    # Otherwise look into tool_result.result
    tool_result = entry.get("tool_result") or {}
    result_text = tool_result.get("result")
    if isinstance(result_text, str) and result_text.startswith("FINAL_ANSWER:"):
        return result_text

    return None


def _should_index_final_answer(fa: str) -> bool:
    """
    Filter out "junk" FINAL_ANSWER strings.
    """
    if not fa.startswith("FINAL_ANSWER:"):
        return False
    lowered = fa.lower()
    for bad in ["could not generate", "unknown", "unexpected"]:
        if bad in lowered:
            return False
    return True


def _parse_session_memory(memory_path: Path, session_id: str) -> List[HistoricalExample]:
    """
    Read the per-session memory file and extract (query, FINAL_ANSWER) pairs.
    """
    try:
        with memory_path.open("r", encoding="utf-8") as f:
            events = json.load(f)
    except Exception as e:
        log("history", f"Failed to read {memory_path}: {e}")
        return []

    if not isinstance(events, list):
        log("history", f"Memory file {memory_path} not a list; skipping.")
        return []

    examples: List[HistoricalExample] = []
    i = 0
    turn_idx = 0

    while i < len(events):
        evt = events[i]
        i += 1

        if evt.get("type") != "run_metadata":
            continue

        txt = evt.get("text") or ""
        if "Started new session with input:" not in txt:
            continue

        user_query = _extract_user_query(txt)
        if not user_query:
            continue

        final_answer: Optional[str] = None
        tools_used: List[str] = []
        successful_tools: List[str] = []
        tags: List[str] = []

        j = i
        while j < len(events):
            evt2 = events[j]
            if evt2.get("type") == "run_metadata":
                break

            # 1) Tools and tags as before
            if evt2.get("type") == "tool_output":
                tool_name = evt2.get("tool_name")
                if tool_name:
                    tools_used.append(tool_name)
                    if evt2.get("success") is True:
                        successful_tools.append(tool_name)

                fa = _extract_final_answer(evt2)
                if fa:
                    final_answer = fa

                evt_tags = evt2.get("tags") or []
                if isinstance(evt_tags, list):
                    tags.extend(str(t) for t in evt_tags)

            # 2) NEW: always check for final_answer in ANY event
            fa = _extract_final_answer(evt2)
            if fa:
                final_answer = fa

            j += 1

        i = j

        if not final_answer or not _should_index_final_answer(final_answer):
            continue

        keywords = _normalize_text(user_query)
        examples.append(
            HistoricalExample(
                session_id=session_id,
                turn_index=turn_idx,
                user_query=user_query,
                final_answer=final_answer,
                tools_used=list(dict.fromkeys(tools_used)),
                successful_tools=list(dict.fromkeys(successful_tools)),
                tags=list(dict.fromkeys(tags)),
                keywords=keywords,
            )
        )
        turn_idx += 1

    return examples


# -------------------------------------------------------------------
# UPDATE INDEX
# -------------------------------------------------------------------

def update_index_for_session(memory_path_str: str, session_id: str) -> None:
    """
    Incrementally update the global historical index for this session.

    NOTE: The actual index file path is determined by config/profiles.yaml
    (custom_config.memory_index_file) with a safe fallback.
    """
    memory_path = Path(memory_path_str)
    index_path = _get_index_path()

    try:
        new_examples = _parse_session_memory(memory_path, session_id)
        if not new_examples:
            log("history", f"No indexable examples found for session {session_id}.")
            return

        index = _load_index(index_path)
        existing = {(it.get("session_id"), it.get("turn_index")) for it in index}
        added = 0

        for ex in new_examples:
            key = (ex.session_id, ex.turn_index)
            if key in existing:
                continue
            index.append(ex.to_dict())
            existing.add(key)
            added += 1

        if added:
            _save_index(index_path, index)
            log("history", f"✅ Updated historical index for session {session_id} (added {added} entries).")
        else:
            log("history", f"Historical index already up-to-date for session {session_id}.")
    except Exception as e:
        log("history", f"Failed to update index for session {session_id}: {e}")


# -------------------------------------------------------------------
# SIMPLE JACCARD (for ranking similar queries)
# -------------------------------------------------------------------

def _jaccard_similarity(a: List[str], b: List[str]) -> float:
    sa, sb = set(a), set(b)
    if not sa or not sb:
        return 0.0
    return len(sa & sb) / len(sa | sb)


# -------------------------------------------------------------------
# LOAD TOP-K SIMILAR EXAMPLES FOR DECISION PROMPT
# -------------------------------------------------------------------

def load_similar_examples(
    user_query: str,
    top_k: int = _get_top_k_value(), # Get from config/profiles.yaml
    memory_root: Optional[str] = None,  # kept for backwards compat, but ignored
) -> List[Dict[str, Any]]:
    """
    Return up to `top_k` historical examples most similar to `user_query`
    using Jaccard similarity over keyword sets.

    Path to the index file is taken from config/profiles.yaml (if present).
    """
    index_path = _get_index_path()
    index = _load_index(index_path)

    if not index:
        return []

    qkw = _normalize_text(user_query)
    scored: List[Any] = []

    for item in index:
        fa = item.get("final_answer")
        uq = item.get("user_query")

        if not isinstance(fa, str) or not isinstance(uq, str):
            continue
        if not fa.startswith("FINAL_ANSWER:"):
            continue
        if fa.strip() == "FINAL_ANSWER: [Could not generate valid solve()]":
            continue

        kw = item.get("keywords") or _normalize_text(uq)
        sim = _jaccard_similarity(qkw, kw)
        if sim > 0:
            scored.append((sim, item))

    if not scored:
        return []

    scored.sort(key=lambda x: x[0], reverse=True)
    top_items = [it for _, it in scored[:top_k]]

    try:
        preview = ", ".join(
            f'"{item.get("user_query")}" (sim={score:.2f})'
            for score, item in scored[:top_k]
        )
        log("history", f"Similar examples for '{user_query}': {preview}")
    except Exception:
        pass

    return top_items


# -------------------------------------------------------------------
# SEMANTIC FAST-PATH FOR DIRECT CACHE ANSWERS
# -------------------------------------------------------------------

def find_best_cached_answer(
    user_query: str,
    min_similarity: float,
    memory_root: str = "memory",  # kept for signature compatibility, ignored
) -> Optional[str]:
    """
    Returns the FULL `FINAL_ANSWER: ...` string if the user query
    is a paraphrase of any past query, based on string similarity.

    min_similarity is typically driven by profiles.yaml (read in loop.py).
    """
    index_path = _get_index_path()
    index = _load_index(index_path)

    if not index:
        return None

    best: Optional[str] = None
    best_sim = 0.0

    for item in index:
        uq = item.get("user_query")
        fa = item.get("final_answer")

        if not isinstance(uq, str) or not isinstance(fa, str):
            continue
        if not fa.startswith("FINAL_ANSWER:"):
            continue

        sim = _string_similarity(user_query, uq)
        if sim > best_sim:
            best_sim = sim
            best = fa

    if best and best_sim >= min_similarity:
        log("history", f"⚡ Semantic HIT (sim={best_sim:.2f}) for: {user_query}")
        return best

    return None
