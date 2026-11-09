"""Doubt: see where the model is unsure, word by word, and stop commands built on a guess."""
from __future__ import annotations

import logging
import threading
from collections import OrderedDict
from typing import Any

from . import analysis, render, scorer, store

log = logging.getLogger("doubt")

GUARDED = {"terminal": "command", "execute_code": "code", "write_file": "path", "patch": "path"}
DEFAULTS = {
    "mode": "ask", "threshold": 0.5, "min_agreement": 0.5, "guard_tools": "terminal", "annotate": "line",
    "top_logprobs": 3, "scorer_model": "", "scorer_base_url": "",
}


class Doubt:
    def __init__(self, ctx: Any):
        self.ctx = ctx
        self.pending: OrderedDict[str, dict[str, Any]] = OrderedDict()
        self.last_reply: dict[str, dict[str, Any]] = {}
        self.overrides: dict[str, Any] = {}
        self.lock = threading.Lock()

    def cfg(self, key: str) -> Any:
        if key in self.overrides:
            return self.overrides[key]
        value = self.ctx.get_config(key, DEFAULTS[key]) if hasattr(self.ctx, "get_config") else None
        return DEFAULTS[key] if value in (None, "") else value

    def threshold(self) -> float:
        return float(self.cfg("threshold"))

    def guarded_tools(self) -> set[str]:
        return {t.strip() for t in str(self.cfg("guard_tools")).split(",") if t.strip() in GUARDED}

    # --- llm_request: ask the provider for token logprobs ------------------------------------
    def on_request(self, **kw: Any) -> Any:
        if self.cfg("mode") == "off" or kw.get("api_mode") not in (None, "chat_completions"):
            return None
        req = dict(kw["request"])
        req["logprobs"] = True
        req["top_logprobs"] = int(self.cfg("top_logprobs"))
        return {"request": req, "source": "doubt", "reason": "token logprobs for uncertainty map"}

    # --- llm_execution: keep the scores Hermes would otherwise drop --------------------------
    def on_execution(self, **kw: Any) -> Any:
        response = kw["next_call"](kw["request"])
        if self.cfg("mode") != "off":
            try:
                self._capture(response, kw)
            except Exception:  # never break a turn over analysis
                log.exception("doubt: capture failed")
        return response

    def _capture(self, response: Any, kw: dict[str, Any]) -> None:
        choice = analysis._get(response, "choices")[0]
        message = analysis._get(choice, "message")
        tool_calls = analysis._get(message, "tool_calls") or []
        session_id = kw.get("session_id") or ""
        for tc in tool_calls:
            tc_id = analysis._get(tc, "id")
            if tc_id:
                with self.lock:
                    self.pending[tc_id] = {"messages": list(kw["request"].get("messages") or []),
                                           "model": kw.get("model") or kw["request"].get("model"),
                                           "base_url": kw.get("base_url") or "", "session_id": session_id}
                    while len(self.pending) > 64:
                        self.pending.popitem(last=False)
        content = analysis._get(message, "content")
        tokens = analysis.response_tokens(response)
        if not content or tool_calls:
            return
        low = self.threshold()
        text, probs, owner = analysis.char_probs(tokens)
        record = {
            "text": text if tokens else content, "model": kw.get("model"), "native": bool(tokens),
            "summary": analysis.summarize(tokens, low),
            "words": [w.__dict__ for w in analysis.token_words(tokens)],
        }
        record["id"] = store.add("reply", record, session_id=session_id, turn_id=kw.get("turn_id") or "")
        self.last_reply[session_id] = record
        self.last_reply[""] = record

    # --- pre_tool_call: re-derive the argument and gate on doubt -----------------------------
    def on_pre_tool_call(self, **kw: Any) -> Any:
        tool, args = kw.get("tool_name"), kw.get("args") or {}
        mode = self.cfg("mode")
        if mode == "off" or tool not in self.guarded_tools():
            return None
        field = GUARDED[tool]
        actual = args.get(field)
        with self.lock:
            ctx = self.pending.pop(kw.get("tool_call_id") or "", None)
        if not isinstance(actual, str) or not actual.strip() or ctx is None:
            return None
        verdict = self._score(tool, field, actual, ctx)
        if verdict is None:
            return None
        verdict.update(tool=tool, field=field, actual=actual, mode=mode)
        flagged = verdict["flagged"]
        verdict["decision"] = ("inconclusive" if verdict["inconclusive"] else "pass") if not flagged else {"ask": "asked", "block": "blocked"}.get(mode, "warned")
        store.add("guard", verdict, session_id=kw.get("session_id") or "", turn_id=kw.get("turn_id") or "")
        if not flagged or mode == "warn":
            return None
        why = render.shaky_list(flagged, self.threshold())
        if mode == "block":
            return {"action": "block", "message": (
                f"Doubt blocked this {tool} call: the model was unsure of {why} when it re-derived the "
                f"{field} independently (agreement {render.pct(verdict['agreement'])}). Verify those parts "
                f"first (e.g. check --help, the docs or the file on disk), then retry.")}
        return {"action": "approve", "rule_key": f"doubt:{tool}",
                "message": f"Doubt: model unsure of {why}"}

    def _score(self, tool: str, field: str, actual: str, ctx: dict[str, Any]) -> Any:
        from agent.secret_scope import get_secret
        base_url = self.cfg("scorer_base_url") or ctx["base_url"] or "https://openrouter.ai/api/v1"
        key = get_secret("DOUBT_SCORER_API_KEY") or (
            get_secret("OPENROUTER_API_KEY") if "openrouter.ai" in base_url else None)
        if not key:
            log.info("doubt: no scorer key; guard skipped")
            return None
        model = self.cfg("scorer_model") or ctx["model"]
        try:
            tokens = scorer.probe(ctx["messages"], tool, field, base_url=base_url, api_key=key, model=model)
        except Exception as exc:
            log.warning("doubt: probe failed: %s", exc)
            return None
        if not tokens:
            return None
        verdict = analysis.score_payload(actual, tokens, self.threshold(), float(self.cfg("min_agreement")))
        verdict["scorer_model"] = model
        return verdict

    # --- transform_llm_output: one honest line under the answer ------------------------------
    def on_transform(self, **kw: Any) -> Any:
        if self.cfg("annotate") == "off" or self.cfg("mode") == "off":
            return None
        rec = self.last_reply.get(kw.get("session_id") or "")
        text = kw.get("response_text") or kw.get("assistant_response") or ""
        if not rec or not rec["native"] or not text or rec["text"].strip()[:40] not in text:
            return None
        s = rec["summary"]
        if self.cfg("annotate") == "directive":
            return f"{text}\n\n::doubt{{id=\"{rec['id']}\"}}"
        shaky = render.shaky_list(rec["words"], self.threshold(), 3)
        tail = f" · shaky: {shaky}" if shaky else " · nothing shaky"
        return f"{text}\n\n— doubt: {render.pct(s['confidence'])} sure{tail}"

    # --- /doubt -------------------------------------------------------------------------------
    def command(self, raw: str, **kw: Any) -> str:
        arg = raw.strip().split()
        if arg and arg[0] in ("ask", "block", "warn", "off"):
            self.overrides["mode"] = arg[0]
            return f"Doubt mode: {arg[0]} (this process; set plugins.entries.doubt.settings.mode to keep it)"
        if arg and arg[0] == "guard":
            rows = store.recent("guard", limit=8)
            if not rows:
                return "No guarded tool calls yet."
            return "\n".join(f"[{r['decision']}] {r['tool']}: "
                             f"{render.heat(r['actual'], r['words'])}  ({render.pct(r['confidence'])} sure, "
                             f"agreement {render.pct(r['agreement'])})" for r in rows)
        rec = self.last_reply.get(kw.get("session_id") or "") or (store.recent("reply", limit=1) or [None])[0]
        if not rec:
            return "No scored replies yet. Doubt needs model.streaming: false and a model that returns logprobs."
        if not rec["native"]:
            return f"The provider for {rec['model']} returned no logprobs for the last reply."
        s = rec["summary"]
        return (f"{render.heat(rec['text'], rec['words'])}\n\n{render.legend()}\n"
                f"{render.pct(s['confidence'])} sure overall · {s['low_words']} shaky words · "
                f"perplexity {s['perplexity']} · {rec['model']}")


def register(ctx: Any) -> None:
    d = Doubt(ctx)
    ctx.register_middleware("llm_request", d.on_request)
    ctx.register_middleware("llm_execution", d.on_execution)
    ctx.register_hook("pre_tool_call", d.on_pre_tool_call)
    ctx.register_hook("transform_llm_output", d.on_transform)
    ctx.register_command("doubt", handler=d.command, description="Certainty heat map of the last reply",
                         args_hint="[guard|ask|block|warn|off]")
