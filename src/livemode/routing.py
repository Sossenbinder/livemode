"""A deterministic rule that sends questions to Claude Code without the local model.

A 4B model answers some real questions itself, even when its prompt forbids it.
Whisper ends questions with a question mark, so questions are easy to find. Only
small talk questions about the bot stay with the local model.
"""

import re

_SMALL_TALK_QUESTIONS = re.compile(
    r"\b(how are you|how're you|how are things|how is it going|how's it going|how do you do"
    r"|what's up|whats up|what is up|you okay|are you okay|are you there|can you hear me"
    r"|what's your name|what is your name|who are you|are you a robot|are you real)\b"
)
# Short reactions that end with a question mark.
_REACTIONS = re.compile(r"(oh )?(really|right|seriously|isn't it|is it|no way|you think so)\?")


def is_question_for_claude(text: str) -> bool:
    """Return True if text is a real question that must go to Claude Code."""
    text = text.strip()
    if not text.endswith("?"):
        return False
    # Use only the last sentence: "Thanks! How are you?" is small talk.
    last = re.split(r"(?<=[.!?])\s+", text)[-1].lower().replace(",", "")
    return not (_SMALL_TALK_QUESTIONS.search(last) or _REACTIONS.fullmatch(last))
