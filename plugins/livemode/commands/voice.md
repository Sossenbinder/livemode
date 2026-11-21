---
description: Turn the livemode voice on or off
argument-hint: "[on|off]"
allowed-tools: mcp__plugin_livemode_livemode__voice
---

Call the `voice` tool of the livemode MCP server. Use the state "$ARGUMENTS". If that is empty or not "on" or "off", use "toggle".

Then tell the user the new state in one short line. Do not call the reply tool for this.

If the livemode tools are not available, tell the user in one line: this session did not start with the voice channel. They must restart Claude Code with `claude --dangerously-load-development-channels plugin:livemode@livemode`.
