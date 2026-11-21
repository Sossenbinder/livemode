"""The voice pipeline: microphone, Whisper, the local model, Kokoro and the speaker.

The local model does the small talk. Tasks and real questions go to Claude Code
through the channel bridge, and Claude's replies come back as speech.
"""

import asyncio
import gc
import json
import re
import urllib.request

from loguru import logger

from livemode import config, consent, cuda_libs
from livemode.channel import Bridge
from livemode.prompts import LOCAL_PROMPT
from livemode.routing import is_question_for_claude

GREETING = "Voice mode is on."
FAREWELL = "Voice mode is off."

# Spoken commands that turn the voice off. The voice cannot hear "on" while it is off.
_VOICE_OFF = re.compile(
    r"\b(voice( mode)? off|turn (the |off )?voice( mode)?( off)?|stop listening|go to sleep)\b"
)

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


def compute_type(device: str) -> str:
    # int8 is the fastest type on a CPU.
    return "int8" if device == "cpu" else "default"


def warm_up_whisper(device: str = config.WHISPER_DEVICE) -> str:
    """Download Whisper if necessary and run one transcription, so that CUDA is ready.

    Return the device that works. If the GPU fails, for example because a game uses
    its memory, Whisper runs on the CPU. That is slower, but the voice mode works.
    """
    import numpy as np
    from faster_whisper import WhisperModel

    logger.info(f"Loading Whisper ({config.WHISPER_MODEL}, device {device})...")
    try:
        model = WhisperModel(config.WHISPER_MODEL, device=device, compute_type=compute_type(device))
        list(model.transcribe(np.zeros(16000, dtype=np.float32), language="en")[0])
        del model
    except RuntimeError as e:
        if device == "cpu":
            raise
        logger.warning(f"Whisper failed on device {device} ({e}). Whisper uses the CPU now.")
        return warm_up_whisper("cpu")
    return device


def is_voice_off_command(text: str) -> bool:
    return bool(_VOICE_OFF.search(text.lower().replace(",", "")))


def warm_up_ollama(keep_alive: str = "30m") -> None:
    """Load the local model now. Ollama unloads a model after 5 minutes without use.

    keep_alive "0" unloads the model instead, which frees its memory.
    """
    logger.info(f"Ollama {config.OLLAMA_MODEL}: keep_alive {keep_alive}")
    request = urllib.request.Request(
        f"{config.OLLAMA_HOST}/api/generate",
        # A request without a prompt only loads or unloads the model.
        data=json.dumps({"model": config.OLLAMA_MODEL, "keep_alive": keep_alive}).encode(),
        headers={"Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(request, timeout=120) as resp:
            resp.read()
    except OSError as e:
        raise RuntimeError(
            f"Ollama at {config.OLLAMA_HOST} did not load {config.OLLAMA_MODEL} ({e}). "
            "Start Ollama with `ollama serve`, then run `livemode setup`."
        ) from e


class VoiceController:
    """Starts and stops the voice pipeline. Off releases the microphone and the models."""

    def __init__(self, bridge: Bridge):
        self._bridge = bridge
        self._task: asyncio.Task | None = None
        self._lock = asyncio.Lock()
        self._pending: set[asyncio.Task] = set()
        bridge.set_voice = self.set

    @property
    def is_on(self) -> bool:
        return self._task is not None and not self._task.done()

    async def set(self, state: str) -> str:
        async with self._lock:
            want_on = not self.is_on if state == "toggle" else state == "on"
            if want_on and not self.is_on:
                self._task = asyncio.create_task(self._run())
            elif not want_on and self.is_on:
                await self._stop()
            return "on" if want_on else "off"

    def schedule(self, coro) -> None:
        # Keep a reference, so that the task is not garbage collected while it runs.
        task = asyncio.create_task(coro)
        self._pending.add(task)
        task.add_done_callback(self._pending.discard)

    async def turn_off_after_farewell(self) -> None:
        # Let the farewell play before the speaker closes.
        await asyncio.sleep(2)
        await self.set("off")

    async def _stop(self) -> None:
        task, self._task = self._task, None
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass
        # The pipeline objects hold Whisper. Collect them now, so that the GPU memory is free.
        gc.collect()
        await asyncio.to_thread(_unload_ollama)
        logger.info("Voice mode is off.")

    async def _run(self) -> None:
        try:
            await _run_pipeline(self._bridge, self)
        except asyncio.CancelledError:
            raise
        except Exception as e:
            logger.exception("The voice pipeline failed")
            await self._bridge.send_to_claude(
                f"(livemode system message, not from the user) The voice mode failed: {e}. Tell "
                "the user in the terminal, and do not investigate unless they ask. The log is "
                f"in {config.STATE_DIR / 'livemode.log'}."
            )


def _unload_ollama() -> None:
    try:
        warm_up_ollama(keep_alive="0")
    except RuntimeError as e:
        logger.warning(f"Ollama did not unload the model: {e}")


def _say(text: str) -> None:
    """Speak one sentence through the speaker, without the pipeline and the microphone."""
    import numpy as np
    import pyaudio
    from kokoro_onnx import Kokoro
    from pipecat.services.kokoro import tts

    kokoro = Kokoro(
        str(tts.KOKORO_CACHE_DIR / "kokoro-v1.0.onnx"),
        str(tts.KOKORO_CACHE_DIR / "voices-v1.0.bin"),
    )
    samples, rate = kokoro.create(text, voice=config.KOKORO_VOICE, lang="en-us")
    audio = pyaudio.PyAudio()
    stream = audio.open(format=pyaudio.paInt16, channels=1, rate=rate, output=True)
    try:
        stream.write((np.clip(samples, -1, 1) * 32767).astype(np.int16).tobytes())
    finally:
        stream.stop_stream()
        stream.close()
        audio.terminate()


async def warn(text: str) -> None:
    """Speak a warning when the voice cannot start. Errors go to the log only."""
    try:
        await asyncio.to_thread(_say, text)
    except Exception:
        logger.exception("The spoken warning failed")


async def run(bridge: Bridge) -> None:
    """Turn the voice on when Claude Code is connected, then wait until the server ends."""
    # Model loading blocks for a few seconds, and the MCP handshake must not wait for it.
    await bridge.ready.wait()
    controller = VoiceController(bridge)
    await controller.set("on")
    try:
        await asyncio.Event().wait()
    finally:
        if controller.is_on:
            await controller.set("off")


async def _run_pipeline(bridge: Bridge, controller: VoiceController) -> None:
    cuda_libs.preload()
    await asyncio.to_thread(warm_up_ollama)
    whisper_device = await asyncio.to_thread(warm_up_whisper)

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
                if is_voice_off_command(text):
                    logger.info(f"Voice off by voice command: {text}")
                    await self.push_frame(TTSSpeakFrame(FAREWELL, append_to_context=False))
                    controller.schedule(controller.turn_off_after_farewell())
                    return
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
        device=whisper_device,
        compute_type=compute_type(whisper_device),
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
        bridge.permissions.clear()
        await asyncio.shield(runner.cancel())
