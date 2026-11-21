"""Check which sentences the local model sends to Claude Code. Needs a running Ollama.

Run with:  uv run scripts/route_test.py
"""

import asyncio
import time

from openai import AsyncOpenAI

from livemode import config
from livemode.prompts import LOCAL_PROMPT
from livemode.routing import is_question_for_claude

TOOL = {
    "type": "function",
    "function": {
        "name": "ask_claude_code",
        "description": (
            "Send the user's last message to Claude Code. Claude works on it and speaks the answer."
        ),
        "parameters": {"type": "object", "properties": {}},
    },
}

# (sentence, True if it must go to Claude Code)
CASES = [
    ("Hey, how is it going?", False),
    ("How are you doing?", False),
    ("Oh, really?", False),
    ("I am a bit tired today.", False),
    ("Haha, that's a good one.", False),
    ("Thanks, that's cool!", False),
    ("Nice, good job.", False),
    ("Run the tests and tell me what fails.", True),
    ("Can you rename that function to load config?", True),
    ("What does this project do?", True),
    ("Commit the changes.", True),
    ("Why is the build so slow?", True),
    ("Why is the sky blue?", True),
    ("What do you think about electric cars?", True),
]


async def main() -> None:
    client = AsyncOpenAI(base_url=f"{config.OLLAMA_HOST}/v1", api_key="ollama")
    correct = 0
    for sentence, expect_claude in CASES:
        start = time.perf_counter()
        if is_question_for_claude(sentence):
            correct += expect_claude
            mark = "ok " if expect_claude else "BAD"
            print(f"{mark} 0.00s  {sentence!r:48} CLAUDE CODE (question rule)")
            continue
        response = await client.chat.completions.create(
            model=config.OLLAMA_MODEL,
            messages=[
                {"role": "system", "content": LOCAL_PROMPT},
                {"role": "user", "content": sentence},
            ],
            tools=[TOOL],
            temperature=0,
        )
        message = response.choices[0].message
        routed = bool(message.tool_calls)
        correct += routed == expect_claude
        result = "CLAUDE CODE" if routed else f"LOCAL  {message.content!r}"
        mark = "ok " if routed == expect_claude else "BAD"
        print(f"{mark} {time.perf_counter() - start:4.2f}s  {sentence!r:48} {result}")
    print(f"\n{correct}/{len(CASES)} routed as expected with {config.OLLAMA_MODEL}")


asyncio.run(main())
