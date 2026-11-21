"""Livemode: a live voice conversation that runs on local models.

Whisper (speech to text), an Ollama model and Kokoro (text to speech) run on this
machine. The local model handles small talk. When the user asks a real question,
the local model calls the ask_claude tool, and Claude's answer is spoken directly.

Run with:  uv run bot.py   then open http://localhost:7860
"""

import os
import re

import anthropic
from dotenv import load_dotenv
from loguru import logger
from pipecat.adapters.llm.schemas.direct_function import tool_options
from pipecat.audio.vad.silero import SileroVADAnalyzer
from pipecat.frames.frames import (
    FunctionCallResultProperties,
    LLMRunFrame,
    TTSSpeakFrame,
)
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.worker import (
    PipelineParams,
    PipelineWorker,
    ProcessorUnusablePolicy,
)
from pipecat.processors.aggregators.llm_context import LLMContext
from pipecat.processors.aggregators.llm_response_universal import (
    LLMContextAggregatorPair,
    LLMUserAggregatorParams,
)
from pipecat.runner.types import RunnerArguments
from pipecat.runner.utils import create_transport
from pipecat.services.kokoro.tts import KokoroTTSService
from pipecat.services.llm_service import FunctionCallParams
from pipecat.services.ollama.llm import OLLamaLLMService
from pipecat.services.whisper.stt import WhisperSTTService
from pipecat.transports.base_transport import BaseTransport, TransportParams
from pipecat.utils.text.transforms import strip_markdown
from pipecat.workers.runner import WorkerRunner

import cuda_libs
from claude_advisor import ClaudeAdvisor

load_dotenv(override=True)
# CTranslate2 loads cuBLAS and cuDNN at the first transcription. Make the CUDA 12 copies findable.
cuda_libs.preload()

OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "qwen3:4b-instruct")
OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434/v1")
WHISPER_MODEL = os.getenv("WHISPER_MODEL", "Systran/faster-distil-whisper-medium.en")
WHISPER_DEVICE = os.getenv("WHISPER_DEVICE", "auto")
KOKORO_VOICE = os.getenv("KOKORO_VOICE", "af_heart")
CLAUDE_MODEL = os.getenv("CLAUDE_MODEL", "claude-opus-5-5")
CLAUDE_EFFORT = os.getenv("CLAUDE_EFFORT", "low")
TRANSCRIPT_TURNS = 10

# Emoji, pictographs and dingbats. Small local models add them even when told not to.
_EMOJI = re.compile("[\U0001F000-\U0001FAFF\u2600-\u27BF\uFE0F\u200D]+")

LOCAL_PROMPT = """\
You are Live, a friendly voice companion in a spoken conversation with the user. \
A text-to-speech voice reads your replies aloud.

Decide for each user message:
1. If the message asks a question or makes a request, call the ask_claude tool. This \
includes questions about facts, advice, opinions, comparisons, choices and plans, even \
easy ones. Do not reply with text. Do not answer it yourself, not even partly.
2. The only questions you answer yourself are questions about you or about the \
conversation, such as "How are you?" or "What's your name?".
3. Everything else is small talk: greetings, reactions, thanks, feelings, jokes. Reply \
yourself in one or two short, casual sentences.

Never use emojis, lists or markdown: the voice cannot speak them.

When you call ask_claude, pass the question so that it makes sense on its own: replace \
words like "it" or "that" with what they refer to. When ask_claude returns, the user \
already heard the answer. Do not repeat it.

Examples:
- "Hey, how's it going?" -> reply yourself.
- "Haha, that's a good one." -> reply yourself.
- "Why is the sky blue?" -> ask_claude.
- "Should I take the train or the car?" -> ask_claude.
- "And how long does that take?" -> ask_claude, with the topic named in the question."""


@tool_options(timeout_secs=120)
async def ask_claude(params: FunctionCallParams, question: str):
    """Ask Claude, a far more knowledgeable assistant, a real question. Claude's answer is spoken to the user directly.

    Args:
        question: The user's question, rewritten so that it makes sense without the conversation.
    """
    advisor: ClaudeAdvisor = params.app_resources["advisor"]
    logger.info(f"ask_claude: {question}")

    spoken = 0
    status = "answered"
    try:
        async for sentence in advisor.answer(question, _transcript(params.context)):
            await params.llm.push_frame(TTSSpeakFrame(sentence))
            spoken += 1
    except anthropic.AuthenticationError:
        logger.error("Claude rejected the API key. Check ANTHROPIC_API_KEY in .env.")
        status = "failed"
    except anthropic.RateLimitError:
        logger.warning("Claude rate limit reached.")
        status = "failed"
    except (anthropic.APIStatusError, anthropic.APIConnectionError) as e:
        logger.error(f"Claude request failed: {e}")
        status = "failed"
    except Exception:
        # The handler must always return a result. Without one, the local model
        # answers the question itself. A missing API key raises a TypeError here.
        logger.exception("ask_claude failed")
        status = "failed"

    if status == "failed" and spoken == 0:
        await params.llm.push_frame(TTSSpeakFrame("Sorry, I couldn't reach Claude just now."))

    note = (
        "The user already heard Claude's answer. Do not repeat it."
        if status == "answered"
        else "Claude was not reachable, and the user was told so."
    )
    await params.result_callback(
        {"status": status, "note": note},
        properties=FunctionCallResultProperties(run_llm=False),
    )


async def strip_emojis(text: str, text_type: str) -> str:
    return _EMOJI.sub("", text)


def _transcript(context: LLMContext) -> str:
    """Return the last spoken turns as plain text, for Claude's context."""
    lines = []
    for message in context.get_messages():
        role = message.get("role") if isinstance(message, dict) else None
        if role not in ("user", "assistant"):
            continue
        content = message.get("content")
        if isinstance(content, list):
            content = " ".join(
                part.get("text", "") for part in content if isinstance(part, dict)
            )
        if isinstance(content, str) and content.strip():
            lines.append(f"{role}: {content.strip()}")
    return "\n".join(lines[-TRANSCRIPT_TURNS:])


transport_params = {
    "webrtc": lambda: TransportParams(audio_in_enabled=True, audio_out_enabled=True),
}


async def run_bot(transport: BaseTransport, runner_args: RunnerArguments):
    stt = WhisperSTTService(
        device=WHISPER_DEVICE,
        settings=WhisperSTTService.Settings(model=WHISPER_MODEL),
    )
    llm = OLLamaLLMService(
        base_url=OLLAMA_URL,
        settings=OLLamaLLMService.Settings(model=OLLAMA_MODEL, system_instruction=LOCAL_PROMPT),
    )
    tts = KokoroTTSService(
        settings=KokoroTTSService.Settings(voice=KOKORO_VOICE),
        text_transforms=[("*", strip_markdown), ("*", strip_emojis)],
    )

    @llm.event_handler("on_function_calls_started")
    async def on_function_calls_started(service, function_calls):
        # Covers the wait for Claude's first sentence.
        await tts.queue_frame(TTSSpeakFrame("Let me check.", append_to_context=False))

    context = LLMContext(tools=[ask_claude])
    user_aggregator, assistant_aggregator = LLMContextAggregatorPair(
        context,
        # The default stop strategy is the local Smart Turn v3 model.
        user_params=LLMUserAggregatorParams(vad_analyzer=SileroVADAnalyzer()),
    )

    pipeline = Pipeline(
        [
            transport.input(),
            stt,
            user_aggregator,
            llm,
            tts,
            transport.output(),
            assistant_aggregator,
        ]
    )

    worker = PipelineWorker(
        pipeline,
        params=PipelineParams(enable_metrics=True),
        idle_timeout_secs=runner_args.pipeline_idle_timeout_secs,
        processor_unusable_policy=ProcessorUnusablePolicy.END,
        app_resources={"advisor": ClaudeAdvisor(CLAUDE_MODEL, CLAUDE_EFFORT)},
    )
    runner = WorkerRunner(handle_sigint=runner_args.handle_sigint)
    await runner.add_workers(worker)

    @transport.event_handler("on_client_connected")
    async def on_client_connected(transport, client):
        logger.info("Client connected")
        context.add_message({"role": "user", "content": "(The user joined. Greet them briefly.)"})
        await worker.queue_frames([LLMRunFrame()])

    @transport.event_handler("on_client_disconnected")
    async def on_client_disconnected(transport, client):
        logger.info("Client disconnected")
        await runner.cancel()

    await runner.run()


async def bot(runner_args: RunnerArguments):
    """Entry point that the Pipecat runner calls for each browser connection."""
    transport = await create_transport(runner_args, transport_params)
    await run_bot(transport, runner_args)


def warm_up_whisper() -> None:
    """Run one transcription so that CUDA is ready before the first user turn."""
    import numpy as np
    from faster_whisper import WhisperModel

    logger.info("Warming up Whisper...")
    model = WhisperModel(
        WhisperSTTService.Settings(model=WHISPER_MODEL).model, device=WHISPER_DEVICE
    )
    list(model.transcribe(np.zeros(16000, dtype=np.float32), language="en")[0])
    del model


if __name__ == "__main__":
    from pipecat.runner.run import main

    if not os.getenv("ANTHROPIC_API_KEY"):
        logger.warning("ANTHROPIC_API_KEY is not set. Real questions will fail. See .env.example.")
    warm_up_whisper()
    main()
