<img src="../../../Images/brand-kit/logo/logo-horizontal-light.png" alt="Mobius Tensors" width="200" align="right" />

# Templates

*A Mobius Tensors project.*

Starting policy files. Copy per customer, then edit. None of these is tested on a customer fleet yet.

| File | Agent | Deploy to |
| --- | --- | --- |
| `claude-code/managed-settings.json` | Claude Code, Team / Enterprise plan login | See locations below |
| `claude-code/managed-settings.bedrock.json` | Claude Code on the customer's Amazon Bedrock | See locations below |
| `codex/requirements.toml` | OpenAI Codex CLI | `/etc/codex/requirements.toml` or `%ProgramData%\OpenAI\Codex\requirements.toml` |
| `gemini-cli/settings.json` | Gemini CLI on Vertex AI | `/etc/gemini-cli/settings.json` or `C:\ProgramData\gemini-cli\settings.json` |

Claude Code managed settings locations:

- macOS: `/Library/Application Support/ClaudeCode/managed-settings.json`
- Linux / WSL: `/etc/claude-code/managed-settings.json`
- Windows: `C:\Program Files\ClaudeCode\managed-settings.json`
- or the same JSON through MDM / Group Policy / the claude.ai admin console

## Replace before deploying

- `forceLoginOrgUUID` — the customer's claude.ai organization UUID. Left as the placeholder, nobody can log in.
- `OTEL_EXPORTER_OTLP_ENDPOINT` — the customer's OpenTelemetry collector (or remove the telemetry block).
- `AWS_REGION` — the customer's Bedrock region. Also pin model versions with `ANTHROPIC_DEFAULT_OPUS_MODEL` /
  `ANTHROPIC_DEFAULT_SONNET_MODEL` / `ANTHROPIC_DEFAULT_HAIKU_MODEL` once you know which models are enabled in their account.
- Add `requiredMinimumVersion` (current stable version at rollout time).
- Add MCP servers the customer approved to `allowedMcpServers` (Claude Code) / `mcp.allowed` (Gemini CLI).
- Gemini CLI: `corp-tools` in the template is a placeholder MCP server.

The JSON files can't hold comments; the reasons for each key are in [../research.md](../research.md) §3.
