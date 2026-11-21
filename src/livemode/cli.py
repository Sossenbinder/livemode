"""The `livemode` command."""

import argparse
import asyncio
import sys

from loguru import logger

from livemode import config

START_COMMAND = "claude --dangerously-load-development-channels plugin:livemode@livemode"


def _configure_logging() -> None:
    config.STATE_DIR.mkdir(parents=True, exist_ok=True)
    logger.remove()
    # stdout carries the MCP protocol. Logs go to stderr and to a file.
    logger.add(sys.stderr, level="INFO")
    logger.add(config.STATE_DIR / "livemode.log", level="DEBUG", rotation="5 MB", retention=3)


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="livemode",
        description="A local voice mode for Claude Code. Claude Code starts it as a channel.",
    )
    sub = parser.add_subparsers(dest="command")
    sub.add_parser("serve", help="Run the channel server on stdio (Claude Code does this).")
    sub.add_parser("setup", help="Download the models and check Ollama and the audio devices.")
    args = parser.parse_args()

    if args.command == "setup":
        from livemode import prepare

        sys.exit(prepare.run())

    if args.command is None and sys.stdin.isatty():
        parser.print_help()
        print(f"\nRun `livemode setup` once. Then start Claude Code with:\n  {START_COMMAND}")
        sys.exit(2)

    _configure_logging()
    from livemode import channel, voice

    try:
        asyncio.run(channel.serve(voice.run, voice.warn))
    except KeyboardInterrupt:
        pass
