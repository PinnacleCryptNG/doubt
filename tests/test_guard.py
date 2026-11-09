import math

import pytest

import doubt
from doubt import analysis


class Ctx:
    def __init__(self, **settings):
        self.settings = settings

    def get_config(self, key, default=None):
        return self.settings.get(key, default)


class Msg:
    def __init__(self, **kw):
        self.__dict__.update(kw)


def lp(token, p):
    return {"token": token, "logprob": math.log(p), "top_logprobs": []}


@pytest.fixture(autouse=True)
def memory_store(monkeypatch):
    rows = []
    monkeypatch.setattr(doubt.store, "add", lambda kind, body, **kw: rows.append((kind, body)) or str(len(rows)))
    return rows


def tool_call_response(tc_id="call_1"):
    tc = Msg(id=tc_id, function=Msg(name="terminal", arguments="{}"))
    return Msg(choices=[Msg(message=Msg(content=None, tool_calls=[tc]), logprobs=None)])


def run_turn(d, response):
    return d.on_execution(request={"messages": [{"role": "user", "content": "convert"}], "model": "m"},
                          next_call=lambda r: response, session_id="s", model="m", base_url="https://x/v1")


def guard(d, monkeypatch, probe_tokens, command):
    monkeypatch.setattr(doubt.scorer, "probe", lambda *a, **k: analysis.tokens_from_logprobs(probe_tokens))
    monkeypatch.setattr(d, "_score", lambda tool, field, actual, ctx: analysis.score_payload(
        actual, analysis.tokens_from_logprobs(probe_tokens), d.threshold()))
    run_turn(d, tool_call_response())
    return d.on_pre_tool_call(tool_name="terminal", args={"command": command}, tool_call_id="call_1", session_id="s")


SHAKY = [lp("ffmpeg", 0.99), lp(" -i", 0.99), lp(" in.mkv", 0.97), lp(" -vf", 0.95), lp(" tonemap", 0.2), lp(" out.mp4", 0.96)]


def test_shaky_command_goes_to_human_approval(monkeypatch, memory_store):
    out = guard(doubt.Doubt(Ctx()), monkeypatch, SHAKY, "ffmpeg -i in.mkv -vf tonemap=flat out.mp4")
    assert out["action"] == "approve" and "tonemap=flat" in out["message"]
    assert memory_store[-1][1]["decision"] == "asked"


def test_block_mode_tells_agent_to_verify(monkeypatch):
    out = guard(doubt.Doubt(Ctx(mode="block")), monkeypatch, SHAKY, "ffmpeg -i in.mkv -vf tonemap=flat out.mp4")
    assert out["action"] == "block" and "Verify" in out["message"]


def test_confident_command_passes(monkeypatch, memory_store):
    sure = [lp("rsync", 0.99), lp(" --version", 0.98)]
    assert guard(doubt.Doubt(Ctx()), monkeypatch, sure, "rsync --version") is None
    assert memory_store[-1][1]["decision"] == "pass"


def test_warn_mode_never_interrupts(monkeypatch, memory_store):
    assert guard(doubt.Doubt(Ctx(mode="warn")), monkeypatch, SHAKY, "ffmpeg -i in.mkv -vf tonemap=flat out.mp4") is None
    assert memory_store[-1][1]["decision"] == "warned"


def test_unguarded_tool_and_unknown_call_are_ignored():
    d = doubt.Doubt(Ctx())
    assert d.on_pre_tool_call(tool_name="read_file", args={"path": "x"}, tool_call_id="c") is None
    assert d.on_pre_tool_call(tool_name="terminal", args={"command": "ls"}, tool_call_id="never-seen") is None


def test_request_asks_for_logprobs_only_on_chat_completions():
    d = doubt.Doubt(Ctx(top_logprobs=5))
    out = d.on_request(request={"model": "m"}, api_mode="chat_completions")
    assert out["request"]["logprobs"] is True and out["request"]["top_logprobs"] == 5
    assert d.on_request(request={"model": "m"}, api_mode="anthropic_messages") is None
    assert doubt.Doubt(Ctx(mode="off")).on_request(request={}, api_mode="chat_completions") is None


def test_reply_is_scored_and_annotated(memory_store):
    d = doubt.Doubt(Ctx())
    toks = [lp("Use", 0.99), lp(" --progress", 0.95), lp(" or", 0.9), lp(" -Q", 0.2)]
    resp = Msg(choices=[Msg(message=Msg(content="Use --progress or -Q", tool_calls=None), logprobs={"content": toks})])
    run_turn(d, resp)
    assert memory_store[-1][0] == "reply"
    out = d.on_transform(response_text="Use --progress or -Q", session_id="s")
    assert "doubt:" in out and "`-Q`" in out


def test_execution_failure_in_analysis_never_breaks_the_turn():
    d = doubt.Doubt(Ctx())
    sentinel = object()
    assert run_turn(d, sentinel) is sentinel
