"""Check which sentences the local model sends to Claude. Needs a running Ollama.

Run with:  uv run scripts/route_test.py
"""

import asyncio
import sys
import time
from pathlib import Path

from openai import AsyncOpenAI

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import bot  # noqa: E402

TOOL = {
    "type": "function",
    "function": {
        "name": "ask_claude",
        "description": bot.ask_claude.__doc__.splitlines()[0],
        "parameters": {
            "type": "object",
            "properties": {"question": {"type": "string"}},
            "required": ["question"],
        },
    },
}

# (sentence, True if it must go to Claude)
CASES = [
    ("Hey, how is it going?", False),
    ("I am a bit tired today.", False),
    ("Haha, that's a good one.", False),
    ("Thanks, that's cool!", False),
    ("Why is the sky blue?", True),
    ("Should I learn Rust or Go?", True),
    ("Is it better to rent or buy a flat?", True),
    ("What year did the Berlin wall fall?", True),
    ("Can you recommend a good sci-fi book?", True),
    ("What do you think about electric cars?", True),
]


async def main() -> None:
    client = AsyncOpenAI(base_url=bot.OLLAMA_URL, api_key="ollama")
    correct = 0
    for sentence, expect_claude in CASES:
        start = time.perf_counter()
        response = await client.chat.completions.create(
            model=bot.OLLAMA_MODEL,
            messages=[
                {"role": "system", "content": bot.LOCAL_PROMPT},
                {"role": "user", "content": sentence},
            ],
            tools=[TOOL],
        )
        message = response.choices[0].message
        routed = bool(message.tool_calls)
        correct += routed == expect_claude
        result = f"CLAUDE {message.tool_calls[0].function.arguments}" if routed else f"LOCAL  {message.content!r}"
        mark = "ok " if routed == expect_claude else "BAD"
        print(f"{mark} {time.perf_counter() - start:4.2f}s  {sentence!r:44} {result}")
    print(f"\n{correct}/{len(CASES)} routed as expected with {bot.OLLAMA_MODEL}")


asyncio.run(main())
