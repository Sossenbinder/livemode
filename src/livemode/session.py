"""Find out whether the Claude Code session that started this server asked for the channel.

Claude Code starts a plugin's MCP server in every session. It delivers channel events
only when the session starts with --channels or --dangerously-load-development-channels
and names this channel. Without that, the voice must not start: the microphone would
listen, but nothing that the user says would reach Claude.

Claude Code does not tell the server, so this reads the command line of the claude
process above this one.
"""

import os
import re
import subprocess

_CHANNEL_FLAG = re.compile(r"--(dangerously-load-development-)?channels\b")
_THIS_CHANNEL = re.compile(r"(plugin:livemode@\S+|server:livemode)(?=[\s,]|$)")
_MAX_ANCESTORS = 10


def _parent_and_args(pid: int) -> tuple[int, str] | None:
    try:
        out = subprocess.run(
            ["ps", "-o", "ppid=,args=", "-p", str(pid)],
            capture_output=True,
            text=True,
            timeout=5,
        ).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return None
    parts = out.split(None, 1)
    if len(parts) != 2 or not parts[0].isdigit():
        return None
    return int(parts[0]), parts[1]


def channel_requested() -> bool | None:
    """Return True or False, or None if the claude process was not found."""
    pid = os.getppid()
    for _ in range(_MAX_ANCESTORS):
        if pid <= 1:
            return None
        found = _parent_and_args(pid)
        if found is None:
            return None
        parent, args = found
        if _CHANNEL_FLAG.search(args):
            return bool(_THIS_CHANNEL.search(args))
        # The native binary is "claude". An npm install runs "node .../claude-code/cli.js".
        if any("claude" in arg.lower() for arg in args.split()[:2]):
            return False
        pid = parent
    return None
