"""Settings. Each value can be changed with an environment variable."""

import os
from pathlib import Path


def _flag(name: str) -> bool:
    return os.getenv(name, "").strip().lower() in ("1", "true", "yes", "on")


OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "qwen3:4b-instruct")


def _ollama_host() -> str:
    # The variable that the ollama CLI reads. It can have no scheme, for example "127.0.0.1:11434".
    host = os.getenv("OLLAMA_HOST", "").strip() or "http://localhost:11434"
    if "://" not in host:
        host = f"http://{host}"
    return host.rstrip("/")


OLLAMA_HOST = _ollama_host()
WHISPER_MODEL = os.getenv("WHISPER_MODEL", "Systran/faster-distil-whisper-medium.en")
WHISPER_DEVICE = os.getenv("WHISPER_DEVICE", "auto")
KOKORO_VOICE = os.getenv("KOKORO_VOICE", "af_heart")

# With headphones, the microphone cannot hear the bot, so the user can interrupt it.
# With speakers, the bot mutes the microphone while it talks.
HEADPHONES = _flag("LIVEMODE_HEADPHONES")

# Spoken answers to Claude Code's permission prompts.
VOICE_PERMISSIONS = not _flag("LIVEMODE_NO_VOICE_PERMISSIONS")

STATE_DIR = Path(os.getenv("XDG_STATE_HOME", Path.home() / ".local" / "state")) / "livemode"

# The name of the MCP server in the plugin. Claude Code names its tools mcp__<server>__<tool>.
SERVER_NAME = "livemode"
