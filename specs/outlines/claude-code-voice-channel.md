# Outline: Livemode as a voice channel for Claude Code

## Context

Livemode is a local voice pipeline (Silero + Smart Turn, Whisper, Ollama `qwen3:4b-instruct`, Kokoro) in Pipecat. Today it runs as a browser app, and the local model sends real questions to the Anthropic API. The user wants a voice mode inside the Claude Code session that runs in the terminal, and a repository that other people can install from GitHub.

## Scope

**In scope:**

- A Claude Code channel (an MCP server over stdio) that runs the voice pipeline in its own process.
- The local model does small talk. It sends tasks and real questions into the running session as channel events.
- Claude answers with a `reply` tool. Kokoro speaks the reply.
- Permission relay: Claude Code's tool approvals are spoken, and the user answers "yes" or "no".
- Local microphone and speaker. There is no browser.
- Packaging: a Python package that `uvx` can run from GitHub, a Claude Code plugin and a marketplace in the same repository, a `livemode setup` command, a LICENSE and a README for new users.

**Out of scope:**

- The direct Anthropic API mode. Remove `claude_advisor.py` and the `anthropic` dependency.
- The browser/WebRTC transport.
- The Agent SDK or `-p` mode. Channels do not work there.
- Languages other than English, Windows support, a Docker image and a PyPI release.
- Spoken progress while Claude works, other than the first filler.

## Approach

**Package layout** (`src/livemode/`, hatchling build, `[project.scripts] livemode = "livemode.cli:main"`):

- `cli.py`: `livemode` (start the channel server, the default) and `livemode setup`.
- `channel.py`: the MCP server. It uses the low-level `mcp` Python SDK with the experimental capabilities `claude/channel` and `claude/channel/permission`, the `reply` tool and an `instructions` string. It sends `notifications/claude/channel` and `notifications/claude/channel/permission`. It receives `notifications/claude/channel/permission_request`.
- `voice.py`: the Pipecat pipeline from the current `bot.py`. `LocalAudioTransport` replaces WebRTC. The `ask_claude_code(request)` tool replaces `ask_claude`.
- `prompts.py`: `LOCAL_PROMPT` (adapted for coding tasks) and `CHANNEL_INSTRUCTIONS`.
- `cuda_libs.py`: moves here without changes.

**Process model:** Claude Code starts `livemode` as a stdio MCP server. stdout carries only MCP messages. All logs go to stderr and to a file in `${XDG_STATE_HOME:-~/.local/state}/livemode/`. The server answers `initialize` at once. Then it starts the pipeline in a background task, in the same asyncio loop. A bridge object connects the two halves (pipeline → channel events, channel → TTS frames).

**Flow:**

1. The user speaks. Whisper sends the text to the local model.
2. Small talk: the local model replies, and Kokoro speaks the reply.
3. Task or real question: the local model calls `ask_claude_code`. The bot says "On it." The bridge sends a channel event: `content` is the rewritten request, and `meta` is `{"source": "voice"}`. The tool returns at once with `run_llm=False`.
4. Claude works in the terminal session. Then Claude calls `reply(text)`. The bridge queues `TTSSpeakFrame(text)` with `append_to_context=True`, so the local model knows what Claude said.
5. Permission request: the bridge speaks "Claude wants to <description>. Should I allow it?" The next user turn goes to a deterministic yes/no parser before the LLM. Unclear words make the bot ask again. The terminal dialog stays open too, and the first answer wins.

**`CHANNEL_INSTRUCTIONS`:** Voice messages come from the user. Always answer with `reply`. The rules for the text come from `kokoro-voice:narrate`: two or three sentences, the outcome first, no code, paths, commands, URLs or markdown, and symbols written as words.

**Audio:** `pipecat-ai[local]` (PyAudio needs the PortAudio system library). The bot mutes the user input while it speaks, with Pipecat's user mute strategy. `LIVEMODE_HEADPHONES=1` turns the mute off, so you can interrupt.

**Plugin** (in this repository):

- `.claude-plugin/marketplace.json` lists the plugin `livemode`.
- `plugins/livemode/.claude-plugin/plugin.json` has `mcpServers.livemode`: `uvx --from git+https://github.com/Sossenbinder/livemode livemode`, and `channels: [{"server": "livemode"}]`.
- The user starts it with `claude --dangerously-load-development-channels plugin:livemode@livemode`.

**`livemode setup`:**

1. Install the package into the uv cache.
2. Check that Ollama runs, and pull `OLLAMA_MODEL` through `/api/pull`.
3. Download Whisper and Kokoro.
4. Check the audio devices.
5. Run one warm-up transcription.

The first plugin start must not do these downloads, because the MCP start has a time limit.

**Dependencies:** Remove `anthropic`, `python-dotenv` and the WebRTC/runner extras. Add `mcp` and the `local` extra. Make the CUDA 12 wheels Linux x86_64 only, with environment markers. `WHISPER_DEVICE=auto` falls back to the CPU.

**Tests and tools:** Update `scripts/route_test.py` for `ask_claude_code` and coding requests. Add a test script that sends a fake `permission_request` and a `reply` through the bridge with no audio.

## Key Decisions

| Decision | Choice | Rationale |
| --- | --- | --- |
| Integration | A channel into the interactive session | The user wants voice in the real session. The Agent SDK makes a separate session. |
| Audio | Local audio, the microphone muted while the bot speaks | No browser tab. Without echo cancellation, muting stops the bot from hearing itself. |
| Who writes the spoken summary | Claude, through `reply` with the narrate rules | Claude knows the result. The 4B model would add errors and delay. |
| The `ask_claude_code` tool | Returns at once, with no wait for the reply | Claude Code tasks can take minutes. The user can continue small talk. |
| Answers to permission requests | A deterministic yes/no parser, not the LLM | A wrong approval is a security risk. The 4B model is not reliable enough. |
| Language | Python and the `mcp` SDK in one process | The pipeline is already Python. A second process would need its own IPC. |
| Distribution | `uvx` from git, and the plugin in the same repository | One repository, no PyPI account, the same pattern as `kokoro-voice`. |
| Configuration | Environment variables, no `.env` | No API key remains, so a secrets file has no use. |
| License | MIT | The same as KokoroCli. |

## Trade-offs

- Channels are a research preview. The start flag has a frightening name and can change. A Claude Code update can break the integration.
- Channels need claude.ai or Console authentication. Bedrock, Vertex and Foundry users cannot use them.
- With speakers, you cannot interrupt the bot. With headphones, you can.
- Spoken approvals: a person near the microphone can approve a tool call. The terminal dialog stays the safe path.
- New users must install `uv`, Ollama and PortAudio themselves. The rest installs automatically.

## Execution profile

- **Tier:** in-session
- **Rationale:** The work uses an experimental MCP protocol from a Python SDK that has no documented example, audio behavior that needs live tests, and a new process model. Each of these needs judgment during implementation.
- **Escalation triggers:** n/a (in-session).

## Open Questions

- Can the Python `mcp` SDK send and receive custom notification methods, or does it need a raw-message workaround? _The docs have only a TypeScript example. A short test during implementation answers this._
- Does `--dangerously-load-development-channels` accept `plugin:livemode@livemode`, or only `server:`? _The docs show only `server:` with this flag. A start test with Claude Code 2.1.319 answers this._
- How long is the MCP start limit, and does the cached `uvx` start fit in it? _This needs a measurement on the user's machine._
