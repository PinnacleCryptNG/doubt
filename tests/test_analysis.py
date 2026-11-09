import math

import pytest

from doubt import analysis


def lp(token, p, alts=()):
    return {"token": token, "logprob": math.log(p),
            "top_logprobs": [{"token": token, "logprob": math.log(p)}] + [{"token": a, "logprob": math.log(q)} for a, q in alts]}


def test_tokens_drop_special_and_keep_alternatives():
    toks = analysis.tokens_from_logprobs([lp("</think>", 0.9), lp("rsync", 0.99), lp(" -P", 0.4, [(" --progress", 0.35)])])
    assert [t.text for t in toks] == ["rsync", " -P"]
    assert toks[1].alts == [(" --progress", pytest.approx(0.35, abs=1e-3))]


def test_word_takes_weakest_piece():
    toks = analysis.tokens_from_logprobs([lp("ff", 0.99), lp("mpeg", 0.98), lp(" -hdr", 0.2), lp("_strip", 0.9)])
    ws = analysis.token_words(toks)
    assert [w.text for w in ws] == ["ffmpeg", "-hdr_strip"]
    assert ws[1].conf == pytest.approx(0.2, abs=1e-3)


def test_summary_counts_shaky_words():
    toks = analysis.tokens_from_logprobs([lp("a", 0.99), lp(" b", 0.3), lp(" c", 0.95)])
    s = analysis.summarize(toks, low=0.5)
    assert s["low_words"] == 1 and s["weakest"] == ["b"]
    assert 0 < s["confidence"] < 1


def test_extract_region_from_fenced_probe():
    text = "Sure:\n```bash\n$ kubectl get pods -n prod\n```\n"
    s, e = analysis.extract_region(text)
    assert text[s:e] == "kubectl get pods -n prod"


def test_score_payload_flags_divergent_and_unsure_parts():
    probe = [lp("ffmpeg", 0.99), lp(" -i", 0.99), lp(" in.mkv", 0.97), lp(" -vf", 0.95), lp(" tonemap", 0.3, [(" zscale", 0.5)]), lp(" out.mp4", 0.96)]
    v = analysis.score_payload("ffmpeg -i in.mkv -vf tonemap=flat out.mp4", analysis.tokens_from_logprobs(probe), low=0.5)
    flagged = [w["text"] for w in v["flagged"]]
    assert flagged == ["tonemap=flat"]
    assert v["agreement"] < 1
    assert v["flagged"][0]["alts"][0][0] == " zscale"


def test_score_payload_passes_confident_identical_command():
    probe = [lp("rsync", 0.99), lp(" -av", 0.97), lp(" --progress", 0.96), lp(" src/", 0.95), lp(" dst/", 0.95)]
    v = analysis.score_payload("rsync -av --progress src/ dst/", analysis.tokens_from_logprobs(probe), low=0.5)
    assert v["flagged"] == [] and v["agreement"] == 1.0


def test_response_tokens_reads_sdk_shaped_objects():
    class O:
        def __init__(self, **kw):
            self.__dict__.update(kw)
    resp = O(choices=[O(logprobs=O(content=[O(token="hi", logprob=-0.1, top_logprobs=[])]))])
    assert [t.text for t in analysis.response_tokens(resp)] == ["hi"]
    assert analysis.response_tokens(O(choices=[O(logprobs=None)])) == []
