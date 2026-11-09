# Doubt

**See where Hermes is unsure, word by word, and stop commands built on a guess.**

Open models can report how likely each token they wrote was. Closed APIs mostly don't. Hermes throws
these scores away by default. Doubt is a Hermes plugin that turns them on, keeps them, and acts on them:

- **Certainty heat map.** Every reply is scored token by token. `/doubt` paints the last reply in the
  terminal. Hermes Desktop shows it in a side pane and an inline `::doubt` card. Words the model was
  unsure of, like a made-up flag, version or path, show up highlighted, with the model's next guess.
- **Re-derivation guard.** Providers return no token scores on tool-call turns. So before a guarded
  tool runs (by default `terminal`), Doubt asks the model one question, with tools off: *write the exact
  command you'd run next*. It scores that answer token by token and aligns it to the real arguments.
  Parts the model was unsure of, or wrote differently the second time, count as doubt. If anything is
  shaky, the call goes to **Hermes's own approval prompt**, or is blocked with a "verify first" message
  for the agent. With no human around to approve, Hermes's approval gate blocks the call.

A prompt or skill can't do any of this. The scores only exist inside Hermes's request pipeline.

## Install

```bash
hermes plugins install https://github.com/PinnacleCryptNG/doubt
hermes config set model.streaming false   # scores are only kept on non-streaming calls
```

Use a model that returns logprobs through an OpenAI-compatible endpoint. On OpenRouter,
`z-ai/glm-5.3-flash` and `nousresearch/hermes-4-405b` return informative logprobs on text replies.

Most providers drop logprobs from any request that carries tools. A tool-free turn keeps them, even on
free models: `hermes -z "your question" -t ""` gives a full heat map on `recursive/flint-v1.0:free`.

In Hermes Desktop, turn on the plugin's **Desktop** half under Capabilities → Plugins → Doubt to get
the side pane and the in-chat card.

## Settings

`plugins.entries.doubt.settings`, also editable in Desktop → Plugins:

| key | default | |
|---|---|---|
| `mode` | `ask` | `ask` (send to approval), `block` (make the agent verify first), `warn` (log only), `off` |
| `threshold` | `0.5` | a word is shaky below this certainty |
| `min_agreement` | `0.5` | below this match the re-derivation took another approach (e.g. a different way to search), so the command is recorded but not judged |
| `guard_tools` | `terminal` | any of `terminal, execute_code, write_file, patch` |
| `annotate` | `line` | `line` adds a one-line summary under replies, `directive` adds a Desktop heat-map card, `off` adds nothing |
| `scorer_model` / `scorer_base_url` | agent's own | model and endpoint for the guard's re-derivation |
| `DOUBT_SCORER_API_KEY` | | secret; on OpenRouter it falls back to `OPENROUTER_API_KEY` |

`/doubt` shows the last reply's heat map. `/doubt guard` lists the commands it checked.
`/doubt ask|block|warn|off` switches the mode for this process.

## How it works

| Hermes surface | Doubt uses it to |
|---|---|
| `llm_request` middleware | add `logprobs` / `top_logprobs` to chat-completions requests |
| `llm_execution` middleware | read the scores off the raw provider response before Hermes drops them |
| `pre_tool_call` hook | re-derive and score guarded arguments, then return `approve` or `block` |
| `transform_llm_output` hook | add the one-line summary or the `::doubt` card |
| plugin storage + `plugin_api.py` | keep scored replies and guard decisions for the Desktop pane |
| Desktop SDK (`panes`, `TRANSCRIPT_DIRECTIVE_AREA`) | draw the heat map |

## What it sends and stores

- **Network:** for each guarded tool call, one extra non-streaming chat request goes to the model
  endpoint the agent is already using (or `scorer_base_url` / `scorer_model` if you set them). It
  contains the latest user message and what came after it in the turn: assistant text, tool-call
  arguments and tool results, truncated. No other network calls, no telemetry.
- **Request change:** adds `logprobs` and `top_logprobs` to the agent's own model requests.
- **Storage:** reply text with its token scores, and guarded commands with their verdicts, in one
  SQLite file in the plugin's data directory (newest 500 records).
- **Approvals:** an unsure command goes to Hermes's own approval prompt; `block` mode refuses it.
  Nothing is ever auto-approved, and with no human present Hermes refuses it.
- **Credentials:** the scorer key comes from Hermes (`DOUBT_SCORER_API_KEY`, or the agent's
  OpenRouter key on openrouter.ai). No credential files are read.

## Limits

- Scores are only kept when `model.streaming` is `false`.
- Providers don't return logprobs on tool-call turns. That's why the guard re-derives the arguments.
- Some models score themselves near-100% sure of everything. Doubt is only as honest as the model's
  probabilities. GLM-5.3 and Hermes 4 were informative in testing. Small free models were not.

## Development

```bash
python3 -m venv .venv && .venv/bin/pip install "pytest>=8,<9"
PYTHONPATH=.. .venv/bin/python -m pytest -q tests
hermes plugins validate .
```

MIT licensed.
