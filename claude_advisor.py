"""Answers real questions with Claude and yields the answer one sentence at a time."""

import re
from collections.abc import AsyncIterator
from datetime import date

import anthropic

SYSTEM_PROMPT = """\
You answer questions inside a live voice conversation. A text-to-speech voice reads your answer aloud.

- Speak in plain, natural sentences. Do not use markdown, lists, headings, code, URLs or emojis.
- Give the answer first. Keep it to two to four sentences unless the user asks for detail.
- Write numbers, units and symbols the way a person says them.
- If the question is unclear, give your best answer and say what you assumed."""

# A sentence ends at . ! or ? followed by whitespace. Short fragments stay in the
# buffer so the voice does not speak a single word on its own.
_SENTENCE_END = re.compile(r"(?<=[.!?])\s+")
_MIN_SENTENCE_CHARS = 20
_MARKDOWN = re.compile(r"[*_#`]+|^\s*[-•]\s+", re.MULTILINE)


class ClaudeAdvisor:
    def __init__(self, model: str, effort: str):
        # Reads ANTHROPIC_API_KEY from the environment.
        self._client = anthropic.AsyncAnthropic()
        self._model = model
        self._effort = effort

    async def answer(self, question: str, transcript: str) -> AsyncIterator[str]:
        """Stream Claude's answer and yield each complete sentence as it arrives."""
        prompt = question
        if transcript:
            prompt = (
                f"<conversation_so_far>\n{transcript}\n</conversation_so_far>\n\n"
                f"Question: {question}"
            )

        buffer = ""
        async with self._client.beta.messages.stream(
            model=self._model,
            max_tokens=16000,
            system=f"{SYSTEM_PROMPT}\n\nToday's date is {date.today():%A, %d %B %Y}.",
            messages=[{"role": "user", "content": prompt}],
            output_config={"effort": self._effort},
            # Re-run a refused request on a suitable fallback model inside the same stream.
            betas=["server-side-fallback-2026-07-01"],
            fallbacks="default",
        ) as stream:
            async for text in stream.text_stream:
                buffer += text
                *complete, buffer = _split_sentences(buffer)
                for sentence in complete:
                    yield sentence

            final = await stream.get_final_message()

        if final.stop_reason == "refusal":
            yield "Sorry, I can't answer that one."
            return

        rest = _clean(buffer)
        if rest:
            yield rest


def _split_sentences(text: str) -> list[str]:
    """Return the complete sentences of text, followed by the unfinished remainder."""
    parts = _SENTENCE_END.split(text)
    sentences: list[str] = []
    pending = ""
    for part in parts[:-1]:
        pending = f"{pending} {part}".strip()
        if len(pending) >= _MIN_SENTENCE_CHARS:
            cleaned = _clean(pending)
            if cleaned:
                sentences.append(cleaned)
            pending = ""
    # Keep the space after a short sentence that stays in the buffer.
    remainder = f"{pending} {parts[-1]}" if pending else parts[-1]
    return [*sentences, remainder]


def _clean(text: str) -> str:
    return _MARKDOWN.sub("", text).strip()
