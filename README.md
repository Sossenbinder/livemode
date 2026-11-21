# Livemode

A live voice conversation that runs on local models. A small local model does the small talk. When you ask a real question, it sends the question to Claude, and Claude's answer is spoken to you.

```
browser mic ─WebRTC─▶ Silero VAD + Smart Turn ─▶ Whisper ─▶ Ollama model ─▶ Kokoro ─WebRTC─▶ browser speaker
                                                                │  ▲
                                                     ask_claude │  │ Claude's answer, sentence by sentence
                                                                ▼  │
                                                          Anthropic API
```

Everything except the Claude call runs on this machine. The app uses [Pipecat](https://github.com/pipecat-ai/pipecat) for the streaming, the turn detection and the interruptions.

## Setup

1. Start Ollama and download the local model:
   ```bash
   ollama serve            # or: sudo systemctl start ollama
   ollama pull qwen3:4b-instruct
   ```
2. Copy `.env.example` to `.env`. Then set `ANTHROPIC_API_KEY`.
3. Start the app:
   ```bash
   uv run bot.py -t webrtc
   ```
4. Open http://localhost:7860 and click **Connect**.

The first start downloads the Whisper model (about 800 MB) and the Kokoro model (about 340 MB).

Use headphones or the browser tab. The browser cancels the echo, so you can interrupt the bot while it talks.

## How the routing works

The local model (`qwen3:4b-instruct`) gets one tool, `ask_claude`. Its prompt in `bot.py` (`LOCAL_PROMPT`) tells it to:

- Send every question or request to Claude.
- Answer only questions about itself, such as "How are you?".
- Do greetings, reactions and other small talk itself.

When the tool starts, the bot says "Let me check." Then `claude_advisor.py` streams Claude's answer, and each sentence goes to the voice as soon as it is complete. Claude also gets the last ten turns of the conversation, so follow-up questions work.

A 4B model sometimes routes wrong. To check a change to the prompt or the model, run:

```bash
uv run scripts/route_test.py
```

## Configuration

Set these in `.env`:

| Variable | Default | Notes |
|---|---|---|
| `ANTHROPIC_API_KEY` | (none) | Required for real questions. |
| `CLAUDE_MODEL` | `claude-opus-5-5` | `claude-sonnet-5-5` or `claude-haiku-5-5` start to speak sooner. |
| `CLAUDE_EFFORT` | `low` | Higher effort gives better answers, but you wait longer. |
| `OLLAMA_MODEL` | `qwen3:4b-instruct` | `qwen3:8b` routes more reliably, but uses about 5 GB of VRAM. |
| `OLLAMA_URL` | `http://localhost:11434/v1` | |
| `WHISPER_MODEL` | `Systran/faster-distil-whisper-medium.en` | English only. |
| `WHISPER_DEVICE` | `auto` | Set to `cpu` if the GPU has no free memory. |
| `KOKORO_VOICE` | `af_heart` | Other voices: `am_michael`, `bf_emma`, `bm_george`, and more. |

## Notes

- The app supports English only. Kokoro has no German voice.
- The system has CUDA 13, but faster-whisper needs the CUDA 12 cuBLAS and cuDNN libraries. The `nvidia-*-cu12` packages supply them, and `cuda_libs.py` loads them at startup.
- Kokoro runs on the CPU. It needs about 0.5 s before each sentence starts to play. This is now the largest delay in a local turn.
