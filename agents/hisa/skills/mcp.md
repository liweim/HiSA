---
name: mcp
domain: all
priority: high
when_to_use: Load when the planner can call software-level APIs directly
---

# Skill: MCP

- Prefer `mcp` when the current software exposes a direct high-level operation, such as opening a settings page, querying playlist state, or operating on an active document through its application API.
- Use exactly one MCP call per step. Do not bundle multiple MCP calls together.
- Only call tools that are listed for the current domain in the system prompt.
- If the available MCP tools cannot express the needed step, do not invent new API calls.
- Before `termination`, verify the actual resulting state rather than assuming the API call succeeded.
