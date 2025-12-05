from typing import Optional
import re

class HeuristicResult:
    def __init__(self, allowed: bool, text: str, reason: Optional[str] = None):
        self.allowed = allowed
        self.text = text
        self.reason = reason

BANNED_WORDS = {"badword1", "badword2"}
BLOCKED_DOMAINS = {"gmail.com", "drive.google.com", "localhost"}

def apply_query_heuristics(user_input: str) -> HeuristicResult:
    text = user_input.strip()

    # 1) empty / too short
    if len(text.split()) < 3:
        return HeuristicResult(False, "FINAL_ANSWER: Please add a bit more detail to your request.")

    # 2) too long
    if len(text) > 3000:
        return HeuristicResult(False, "FINAL_ANSWER: Your request is quite long. Please narrow it down.")

    # 3) blocked domains
    # (simple example)
    for domain in BLOCKED_DOMAINS:
        if domain in text:
            return HeuristicResult(False, "FINAL_ANSWER: For privacy reasons I can’t access that site. Please paste the relevant text instead.")

    # 4) harmful scripts
    if re.search(r"rm -rf|powershell|bypass antivirus", text, re.IGNORECASE):
        return HeuristicResult(False, "FINAL_ANSWER: I can’t help with harmful or unsafe scripts.")

    # 5) confidential patterns
    if re.search(r"(password|secret key|confidential)", text, re.IGNORECASE):
        return HeuristicResult(False, "FINAL_ANSWER: This looks confidential. Please remove secrets and try again.")

    return HeuristicResult(True, text)

def apply_answer_heuristics(answer: str) -> str:
    # 9) banned words filter (simple)
    filtered = answer
    for w in BANNED_WORDS:
        filtered = re.sub(w, "*" * len(w), filtered, flags=re.IGNORECASE)
    # 8) long / blob output check
    if len(filtered) > 4000:
        return f"FURTHER_PROCESSING_REQUIRED: {filtered[:1000]}..."
    return filtered