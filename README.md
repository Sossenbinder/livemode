# Livemode

A local voice mode for Claude Code. You talk, and a small local model talks back. When you give a task or ask a real question, the request goes into your running Claude Code session. Claude does the work in the terminal as usual, and then a voice speaks a short summary of the result.

```
microphone ─▶ Silero VAD + Smart Turn ─▶ Whisper ─▶ question? ──yes──────────────┐
                                                       │ no                       │
                                                       ▼                          ▼
                                                 Ollama model ──ask_claude_code──▶ channel event
                                                       │ small talk                 │
                                                       ▼                            ▼
speaker ◀──────────────────────────────────────── Kokoro ◀──── reply tool ◀── Claude Code session
```

Speech recognition, the local model and the voice run on your machine. Only Claude Code uses the network.

Livemode is a [Claude Code channel](https://code.claude.com/docs/en/channels): an MCP server that sends events into the session that you started in the terminal. You see everything that Claude does, and you can type at any time.

> [!WARNING]
> Channels are a research preview. Custom channels start only with the `--dangerously-load-development-channels` flag, and a Claude Code update can change how they work. Channels need a claude.ai or Console login. They do not work with Bedrock, Vertex or Foundry, or with `claude -p`. Livemode was tested with Claude Code 2.1.319.

## What you need

- Linux. macOS can work, but nobody has tested it. Windows is not supported.
- [uv](https://docs.astral.sh/uv/)
- [Ollama](https://ollama.com), running (`ollama serve`)
- On Linux: PortAudio with its headers, and a C compiler, because PyAudio has no Linux wheel.
  - Arch: `sudo pacman -S portaudio base-devel`
  - Debian/Ubuntu: `sudo apt install portaudio19-dev build-essential`
- A microphone and a speaker or headphones.
- An NVIDIA GPU is optional. Without one, Whisper runs on the CPU and is slower.

## Install

1. Install the `livemode` command:
   ```bash
   uv tool install git+https://github.com/Sossenbinder/livemode
   ```
2. Download the models and check your setup. This pulls the Ollama model (about 2.5 GB), Whisper (about 800 MB) and Kokoro (about 340 MB):
   ```bash
   livemode setup
   ```
3. Install the plugin in Claude Code:
   ```
   /plugin marketplace add Sossenbinder/livemode
   /plugin install livemode@livemode
   ```

Do step 2 before the first start. Claude Code gives an MCP server 30 seconds to connect, and the downloads take longer.

## Use

Start Claude Code with the channel:

```bash
claude --dangerously-load-development-channels plugin:livemode@livemode
```

Type the channel name exactly: `plugin:livemode@livemode`, with no period or other character after it. Claude Code ignores a channel with a wrong name and does not tell the server. Livemode then reads the start command, keeps the microphone off, and speaks a warning.

Save the command as a shell function, so that you do not type it each time. In fish:

```fish
function claude-voice; claude --dangerously-load-development-channels plugin:livemode@livemode $argv; end
funcsave claude-voice
```

Confirm the warning about development channels. After a few seconds, the voice says "Voice mode is on." Then talk:

- **Small talk** ("Hey, how's it going?") stays with the local model and gets a fast answer.
- **Questions** ("Why is the build so slow?") and **tasks** ("Run the tests and tell me what fails.") go to Claude. The bot says "On it." Claude works in the terminal and then speaks a summary.
- **Permission prompts**: when Claude needs your approval, the voice asks "Claude asks for permission: … Should I allow it?" Say "yes" or "no". The terminal dialog stays open too, and the first answer wins.

Each request that goes to Claude shows in the conversation as a `← livemode: …` row with your words. Claude's spoken answer is the `reply` tool call under it. Small talk with the local model does not show in the conversation.

With speakers, the microphone is off while the bot talks, so the bot does not hear itself. Then you cannot interrupt it. With headphones, start Claude Code with `LIVEMODE_HEADPHONES=1` in front of the command, so you can interrupt the bot.

### Turn the voice off and on

| To | Do |
|---|---|
| Turn the voice off | Type `/livemode:voice off`, or say "voice off" |
| Turn the voice on again | Type `/livemode:voice on` |
| Toggle it | Type `/livemode:voice` |

When the voice is off, the microphone is free, Ollama unloads the model, and Whisper frees most of its GPU memory. Claude then answers in the terminal only.

The voice runs only in a session that you start with the channel flag. In other sessions, the plugin does not open the microphone and gives Claude no tools.

## Configuration

Set these environment variables in the shell before you start Claude Code. Claude Code gives its environment to the livemode server.

| Variable | Default | Notes |
|---|---|---|
| `LIVEMODE_HEADPHONES` | off | `1` keeps the microphone on while the bot talks, so you can interrupt it. |
| `LIVEMODE_NO_VOICE_PERMISSIONS` | off | `1` turns off spoken permission prompts. You answer them in the terminal. |
| `OLLAMA_MODEL` | `qwen3:4b-instruct` | Run `livemode setup` again after a change. |
| `OLLAMA_HOST` | `http://localhost:11434` | The same variable that the `ollama` CLI reads. |
| `WHISPER_MODEL` | `Systran/faster-distil-whisper-medium.en` | English only. |
| `WHISPER_DEVICE` | `auto` | If the GPU fails, for example because it has no free memory, Whisper uses the CPU. |
| `KOKORO_VOICE` | `af_heart` | Other voices: `am_michael`, `bf_emma`, `bm_george`, and more. |

The log is in `~/.local/state/livemode/livemode.log`.

## How it works

- **Routing:** Whisper ends questions with a question mark. A fixed rule sends each question to Claude, except small talk questions about the bot, such as "How are you?". For all other turns, the local model decides. It does the small talk itself, or it calls the `ask_claude_code` tool. The tool sends your words to Claude without changes. See `src/livemode/routing.py` and `LOCAL_PROMPT` in `src/livemode/prompts.py`.
- **Answers:** The channel tells Claude to answer each voice message with the `reply` tool, in two or three sentences written for the ear. Kokoro speaks the text. The channel approves its own `reply` tool, so you do not hear a permission question before each answer.
- **Permissions:** A fixed yes/no word list reads your answer. The local model does not, because a wrong "yes" is a security risk. If the answer is not clear twice, the question goes back to the terminal.
- **Protocol:** The server uses the MCP handshake protocol. The 2026-07-28 revision has no path for notifications that the server sends on its own, so Claude Code skips channels on it.

## Limits

- English only. Kokoro has no German voice in this setup, and the Whisper model is English only.
- A person or a sound near the microphone can approve a permission prompt. Set `LIVEMODE_NO_VOICE_PERMISSIONS=1` if that is a risk for you.
- The 4B model can still send a small talk sentence to Claude, or answer a command itself. Check a change to the prompt with `uv run scripts/route_test.py`.
- Whisper and the Ollama model need about 4 GB of GPU memory together. If a game uses the GPU, Ollama moves part of the model to the CPU, and Whisper can fall back to the CPU.

## Development

```bash
git clone https://github.com/Sossenbinder/livemode && cd livemode
uv sync
uv run scripts/route_test.py      # routing test, needs Ollama
```

To test the local code without the plugin, put a server named `livemode` into the `.mcp.json` of a test project, with the command `uv` and the arguments `run --directory /path/to/livemode livemode serve`. Then start `claude --dangerously-load-development-channels server:livemode` there.

To load the plugin from the folder for one session, use `claude --plugin-dir plugins/livemode --dangerously-load-development-channels plugin:livemode@inline`.

## License

MIT
