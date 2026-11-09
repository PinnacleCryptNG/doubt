"""Pure logprob analysis: tokens -> words -> doubt spans, and command alignment."""
from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
from difflib import SequenceMatcher
from typing import Any, Iterable, Optional

_SPECIAL = re.compile(r"^(<\|[^|]*\|>|</?think>|<turn\|>|<\|turn>|<eos>|</s>)$")
_FENCE = re.compile(r"```[a-zA-Z0-9_+-]*\n?(.*?)```", re.S)


@dataclass
class Token:
    text: str
    prob: float
    alts: list[tuple[str, float]] = field(default_factory=list)


@dataclass
class Word:
    text: str
    conf: float
    start: int
    alts: list[tuple[str, float]] = field(default_factory=list)


def _get(obj: Any, key: str, default: Any = None) -> Any:
    if isinstance(obj, dict):
        return obj.get(key, default)
    return getattr(obj, key, default)


def tokens_from_logprobs(content: Optional[Iterable[Any]]) -> list[Token]:
    """Normalize OpenAI-style ``choices[0].logprobs.content`` (dicts or SDK objects)."""
    out: list[Token] = []
    for item in content or []:
        text = _get(item, "token") or ""
        lp = _get(item, "logprob")
        if lp is None or _SPECIAL.match(text.strip()):
            continue
        alts = []
        for alt in _get(item, "top_logprobs") or []:
            at, alp = _get(alt, "token"), _get(alt, "logprob")
            if at is not None and alp is not None and at != text:
                alts.append((at, round(math.exp(alp), 4)))
        out.append(Token(text, min(1.0, math.exp(lp)), alts))
    return out


def response_tokens(response: Any) -> list[Token]:
    try:
        choice = _get(response, "choices")[0]
    except (TypeError, IndexError):
        return []
    return tokens_from_logprobs(_get(_get(choice, "logprobs"), "content"))


def char_probs(tokens: list[Token]) -> tuple[str, list[float], list[int]]:
    """Concatenated text, per-character prob, and per-character token index."""
    text, probs, owner = [], [], []
    for i, tok in enumerate(tokens):
        text.append(tok.text)
        probs.extend([tok.prob] * len(tok.text))
        owner.extend([i] * len(tok.text))
    return "".join(text), probs, owner


def words(text: str, probs: list[float], alts_at: Optional[dict[int, list]] = None) -> list[Word]:
    """Whitespace-delimited words; a word is as sure as its least sure character."""
    out = []
    for m in re.finditer(r"\S+", text):
        span = probs[m.start():m.end()]
        conf = min(span) if span else 1.0
        alts = []
        if alts_at:
            cands = [c for c in range(m.start(), m.end()) if alts_at.get(c)]
            if cands:
                alts = alts_at[min(cands, key=lambda c: probs[c])]
        out.append(Word(m.group(), round(conf, 4), m.start(), alts))
    return out


def token_words(tokens: list[Token]) -> list[Word]:
    text, probs, owner = char_probs(tokens)
    alts_at = {c: tokens[t].alts for c, t in enumerate(owner)}
    return words(text, probs, alts_at)


def summarize(tokens: list[Token], low: float) -> dict[str, Any]:
    if not tokens:
        return {"tokens": 0, "confidence": None, "perplexity": None, "low_words": 0}
    lps = [math.log(max(t.prob, 1e-9)) for t in tokens]
    ws = token_words(tokens)
    return {
        "tokens": len(tokens),
        "confidence": round(math.exp(sum(lps) / len(lps)), 4),
        "perplexity": round(math.exp(-sum(lps) / len(lps)), 3),
        "low_words": sum(1 for w in ws if w.conf < low),
        "weakest": [w.text for w in sorted(ws, key=lambda w: w.conf)[:3] if w.conf < low],
    }


def extract_region(text: str) -> tuple[int, int]:
    """Offsets of the payload in a probe answer: first fenced block, else inline code, else stripped text."""
    m = _FENCE.search(text)
    if m:
        s, e = m.start(1), m.end(1)
    else:
        m = re.search(r"`([^`\n]+)`", text)
        s, e = (m.start(1), m.end(1)) if m else (0, len(text))
    while s < e and text[s].isspace():
        s += 1
    while e > s and text[e - 1].isspace():
        e -= 1
    if text.startswith("$ ", s):
        s += 2
    return s, e


def align(actual: str, probe_text: str, probe_probs: list[float]) -> list[float]:
    """Per-character confidence for ``actual``: the probe's prob where the model re-derived the same
    characters, 0.0 where its independent re-derivation diverged."""
    out = [0.0] * len(actual)
    sm = SequenceMatcher(None, probe_text, actual, autojunk=False)
    for a, b, n in sm.get_matching_blocks():
        for k in range(n):
            out[b + k] = probe_probs[a + k]
    return out


def score_payload(actual: str, tokens: list[Token], low: float, min_agreement: float = 0.5) -> dict[str, Any]:
    """Score a tool argument against the model's independent re-derivation of it.

    Below ``min_agreement`` the re-derivation took a different approach (e.g. another way to
    search a directory), which says nothing about guessed specifics, so nothing is flagged.
    """
    text, probs, owner = char_probs(tokens)
    s, e = extract_region(text)
    probe, pprobs = text[s:e], probs[s:e]
    conf = align(actual, probe, pprobs)
    alts_at = {}
    sm = SequenceMatcher(None, probe, actual, autojunk=False)
    for a, b, n in sm.get_matching_blocks():
        for k in range(n):
            alts_at[b + k] = tokens[owner[s + a + k]].alts
    ws = words(actual, conf, alts_at)
    matched = sum(n for _, _, n in sm.get_matching_blocks())
    agreement = round(matched / max(len(actual), 1), 4)
    inconclusive = agreement < min_agreement
    return {
        "probe": probe,
        "agreement": agreement,
        "inconclusive": inconclusive,
        "words": [w.__dict__ for w in ws],
        "flagged": [] if inconclusive else [w.__dict__ for w in ws if w.conf < low],
        "confidence": round(min((w.conf for w in ws), default=1.0), 4),
    }
