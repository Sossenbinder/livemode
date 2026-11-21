"""The voice pipeline: microphone, Whisper, the local model, Kokoro and the speaker.

The local model does the small talk. Tasks and real questions go to Claude Code
through the channel bridge, and Claude's replies come back as speech.
"""

import asyncio
import re

from loguru import logger

from livemode import config, consent, cuda_libs
from livemode.channel import Bridge
from livemode.prompts import LOCAL_PROMPT
from livemode.routing import is_question_for_claude

GREETING = "Voice mode is on."

# Emoji, pictographs and dingbats. Small local models add them even when told not to.
_EMOJI = re.compile("[\U0001f000-\U0001faff☀-➿️‍]+")

_MAX_UNCLEAR_ANSWERS = 2


async def strip_emojis(text: str, text_type: str) -> str:
    return _EMOJI.sub("", text)


def last_user_text(context) -> str:
    """Return what the user said since the bot last spoke.

    A long sentence with a pause can arrive as several user messages, so this joins
    all user messages after the last assistant message.
    """
    parts: list[str] = []
    for message in reversed(context.get_messages()):
        role = message.get("role") if isinstance(message, dict) else None
        if role == "user":
            content = message.get("content")
            if isinstance(content, list):
                content = " ".join(p.get("text", "") for p in content if isinstance(p, dict))
            if isinstance(content, str) and content.strip():
                parts.append(content.strip())
        elif parts and role in ("assistant", "tool"):
            break
    return " ".join(reversed(parts))


def warm_up_whisper() -> None:
    """Download Whisper if necessary and run one transcription, so that CUDA is ready."""
    import numpy as np
    from faster_whisper import WhisperModel

    logger.info(f"Loading Whisper ({config.WHISPER_MODEL})...")
    model = WhisperModel(config.WHISPER_MODEL, device=config.WHISPER_DEVICE)
    list(model.transcribe(np.zeros(16000, dtype=np.float32), language="en")[0])
    del model


async def run(bridge: Bridge) -> None:
    """Run the voice pipeline until it is cancelled. Errors go to the log and to Claude."""
    try:
        await _run(bridge)
    except asyncio.CancelledError:
        raise
    except Exception as e:
        logger.exception("The voice pipeline failed")
        await bridge.ready.wait()
        await bridge.send_to_claude(
            f"(livemode system message, not from the user) The voice mode failed to start: {e}. "
            "Tell the user in the terminal. The log is in "
            f"{config.STATE_DIR / 'livemode.log'}."
        )


async def _run(bridge: Bridge) -> None:
    # Wait until Claude Code is connected. Model loading blocks for a few seconds,
    # and the MCP handshake must not wait for it.
    await bridge.ready.wait()

    cuda_libs.preload()
    await asyncio.to_thread(warm_up_whisper)

    from pipecat.adapters.llm.schemas.direct_function import tool_options
    from pipecat.audio.vad.silero import SileroVADAnalyzer
    from pipecat.frames.frames import (
        FunctionCallResultProperties,
        LLMContextFrame,
        TTSSpeakFrame,
    )
    from pipecat.pipeline.pipeline import Pipeline
    from pipecat.pipeline.worker import PipelineParams, PipelineWorker, ProcessorUnusablePolicy
    from pipecat.processors.aggregators.llm_context import LLMContext
    from pipecat.processors.aggregators.llm_response_universal import (
        LLMContextAggregatorPair,
        LLMUserAggregatorParams,
    )
    from pipecat.processors.frame_processor import FrameProcessor
    from pipecat.services.kokoro.tts import KokoroTTSService
    from pipecat.services.llm_service import FunctionCallParams
    from pipecat.services.ollama.llm import OLLamaLLMService
    from pipecat.services.whisper.stt import WhisperSTTService
    from pipecat.transports.local.audio import LocalAudioTransport, LocalAudioTransportParams
    from pipecat.turns.user_mute.always_user_mute_strategy import AlwaysUserMuteStrategy
    from pipecat.utils.text.transforms import strip_markdown
    from pipecat.workers.runner import WorkerRunner

    @tool_options(timeout_secs=10)
    async def ask_claude_code(params: FunctionCallParams):
        """Send the user's last message to Claude Code. Claude works on it and speaks the answer."""
        text = last_user_text(params.context)
        logger.info(f"To Claude Code: {text}")
        sent = bool(text) and await bridge.send_to_claude(text)
        if not sent:
            await params.llm.push_frame(TTSSpeakFrame("Claude Code is not connected."))
        await params.result_callback(
            {"status": "sent" if sent else "failed"},
            properties=FunctionCallResultProperties(run_llm=False),
        )

    class TurnRouter(FrameProcessor):
        """Handle a user turn before the local model, where a fixed rule is safer.

        - While a permission question is open, the turn is the answer.
        - A real question goes straight to Claude Code.
        - Everything else goes to the local model.
        """

        def __init__(self):
            super().__init__()
            self._unclear = 0

        async def process_frame(self, frame, direction):
            await super().process_frame(frame, direction)
            if isinstance(frame, LLMContextFrame):
                text = last_user_text(frame.context)
                if bridge.permissions:
                    await self._answer(text)
                    return
                if is_question_for_claude(text):
                    logger.info(f"To Claude Code (question rule): {text}")
                    sent = await bridge.send_to_claude(text)
                    # This filler goes into the context. It ends the user turn there,
                    # so the next turn does not repeat this question.
                    reply = "On it." if sent else "Claude Code is not connected."
                    await self.push_frame(TTSSpeakFrame(reply))
                    return
            await self.push_frame(frame, direction)

        async def _answer(self, text: str):
            request = bridge.permissions[0]
            decision = consent.parse(text)
            if decision is None:
                self._unclear += 1
                if self._unclear < _MAX_UNCLEAR_ANSWERS:
                    await self.push_frame(
                        TTSSpeakFrame("Please say yes or no.", append_to_context=False)
                    )
                    return
                # Leave the decision to the terminal dialog.
                bridge.permissions.popleft()
                await self.push_frame(
                    TTSSpeakFrame("Okay, answer it in the terminal.", append_to_context=False)
                )
            else:
                bridge.permissions.popleft()
                await bridge.answer_permission(request, decision)
                await self.push_frame(
                    TTSSpeakFrame("Allowed." if decision else "Denied.", append_to_context=False)
                )
            self._unclear = 0
            if bridge.permissions:
                await self.push_frame(_question(bridge.permissions[0]))

    def _question(request) -> "TTSSpeakFrame":
        what = (request.description or request.tool_name).rstrip(". ")
        return TTSSpeakFrame(
            f"Claude asks for permission: {what}. Should I allow it?", append_to_context=False
        )

    transport = LocalAudioTransport(
        LocalAudioTransportParams(audio_in_enabled=True, audio_out_enabled=True)
    )
    stt = WhisperSTTService(
        device=config.WHISPER_DEVICE,
        settings=WhisperSTTService.Settings(model=config.WHISPER_MODEL),
    )
    llm = OLLamaLLMService(
        base_url=f"{config.OLLAMA_HOST}/v1",
        settings=OLLamaLLMService.Settings(
            model=config.OLLAMA_MODEL,
            system_instruction=LOCAL_PROMPT,
            # Sampling makes the routing decision random. At 0, it is stable.
            temperature=0,
        ),
    )
    tts = KokoroTTSService(
        settings=KokoroTTSService.Settings(voice=config.KOKORO_VOICE),
        text_transforms=[("*", strip_markdown), ("*", strip_emojis)],
    )

    @llm.event_handler("on_function_calls_started")
    async def on_function_calls_started(service, function_calls):
        await tts.queue_frame(TTSSpeakFrame("On it.", append_to_context=False))

    context = LLMContext(tools=[ask_claude_code])
    user_aggregator, assistant_aggregator = LLMContextAggregatorPair(
        context,
        user_params=LLMUserAggregatorParams(
            vad_analyzer=SileroVADAnalyzer(),
            # Without headphones, the microphone hears the bot. Mute it while the bot talks.
            user_mute_strategies=[] if config.HEADPHONES else [AlwaysUserMuteStrategy()],
        ),
    )

    pipeline = Pipeline(
        [
            transport.input(),
            stt,
            user_aggregator,
            TurnRouter(),
            llm,
            tts,
            transport.output(),
            assistant_aggregator,
        ]
    )
    worker = PipelineWorker(
        pipeline,
        params=PipelineParams(enable_metrics=True),
        idle_timeout_secs=None,
        processor_unusable_policy=ProcessorUnusablePolicy.END,
    )

    async def speak(text: str) -> None:
        # Claude's reply goes into the local context, so the small talk knows about it.
        await worker.queue_frames([TTSSpeakFrame(text)])

    async def on_permission() -> None:
        if len(bridge.permissions) == 1:
            await worker.queue_frames([_question(bridge.permissions[0])])

    bridge.speak = speak
    bridge.on_permission = on_permission if config.VOICE_PERMISSIONS else None
    await worker.queue_frames([TTSSpeakFrame(GREETING, append_to_context=False)])

    runner = WorkerRunner(handle_sigint=False)
    await runner.add_workers(worker)
    logger.info("Voice mode is running.")
    try:
        await runner.run()
    finally:
        bridge.speak = None
        bridge.on_permission = None
