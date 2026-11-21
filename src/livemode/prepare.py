"""`livemode setup`: get everything ready, so that Claude Code can start the channel fast.

Claude Code gives an MCP server 30 seconds to connect. The model downloads take longer,
so they must happen here and not at the first start.
"""

import json
import urllib.error
import urllib.request

from livemode import config, cuda_libs
from livemode.cli import START_COMMAND


def _step(text: str) -> None:
    print(f"\n==> {text}", flush=True)


def _check_ollama() -> bool:
    _step(f"Ollama at {config.OLLAMA_HOST}")
    try:
        with urllib.request.urlopen(f"{config.OLLAMA_HOST}/api/version", timeout=5) as resp:
            print(f"Ollama {json.load(resp).get('version', '?')} runs.")
    except (urllib.error.URLError, OSError):
        print(
            "Ollama does not answer. Install it from https://ollama.com, "
            "then start it with `ollama serve`."
        )
        return False

    print(f"Pulling {config.OLLAMA_MODEL}...")
    request = urllib.request.Request(
        f"{config.OLLAMA_HOST}/api/pull",
        data=json.dumps({"model": config.OLLAMA_MODEL}).encode(),
        headers={"Content-Type": "application/json"},
    )
    last = ""
    with urllib.request.urlopen(request, timeout=3600) as resp:
        for line in resp:
            event = json.loads(line)
            if "error" in event:
                print(f"Ollama error: {event['error']}")
                return False
            status = event.get("status", "")
            if event.get("total"):
                status = f"{status} {100 * event.get('completed', 0) // event['total']}%"
            if status != last:
                print(f"  {status}", flush=True)
                last = status
    return True


def _prepare_whisper() -> bool:
    _step(f"Whisper ({config.WHISPER_MODEL}, device {config.WHISPER_DEVICE})")
    cuda_libs.preload()
    from livemode.voice import warm_up_whisper

    try:
        device = warm_up_whisper()
    except Exception as e:
        print(f"Whisper failed: {e}")
        return False
    print(f"Whisper is ready on device {device}.")
    if device == "cpu" and config.WHISPER_DEVICE != "cpu":
        print("The GPU failed, so Whisper uses the CPU. The log above gives the reason.")
    return True


def _prepare_kokoro() -> bool:
    _step("Kokoro")
    from pipecat.services.kokoro import tts

    try:
        tts._ensure_model_files(
            tts.KOKORO_CACHE_DIR / "kokoro-v1.0.onnx", tts.KOKORO_CACHE_DIR / "voices-v1.0.bin"
        )
    except Exception as e:
        print(f"The Kokoro download failed: {e}")
        return False
    print(f"Kokoro is ready in {tts.KOKORO_CACHE_DIR}.")
    return True


def _check_audio() -> bool:
    _step("Audio devices")
    import pyaudio

    audio = pyaudio.PyAudio()
    ok = True
    for kind, getter in (
        ("Microphone", audio.get_default_input_device_info),
        ("Speaker", audio.get_default_output_device_info),
    ):
        try:
            print(f"{kind}: {getter()['name']}")
        except OSError:
            print(f"{kind}: none found")
            ok = False
    audio.terminate()
    return ok


def run() -> int:
    results = [_check_ollama(), _prepare_whisper(), _prepare_kokoro(), _check_audio()]
    if not all(results):
        print("\nSetup is not complete. Correct the problems above and run `livemode setup` again.")
        return 1
    print(f"\nSetup is complete. Start Claude Code with:\n  {START_COMMAND}")
    if not config.HEADPHONES:
        print(
            "\nWith speakers, the microphone is off while the bot talks. With headphones, set "
            "LIVEMODE_HEADPHONES=1, so you can interrupt it."
        )
    return 0
