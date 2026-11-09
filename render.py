"""Terminal heat map rendering."""
from __future__ import annotations

from typing import Any

_BANDS = [(0.25, "\033[1;97;41m"), (0.5, "\033[1;31m"), (0.75, "\033[33m"), (0.9, "\033[2;33m")]
_RESET = "\033[0m"


def color(conf: float) -> str:
    for edge, code in _BANDS:
        if conf < edge:
            return code
    return ""


def heat(text: str, words: list[dict[str, Any]]) -> str:
    out, pos = [], 0
    for w in words:
        out.append(text[pos:w["start"]])
        code = color(w["conf"])
        out.append(f"{code}{w['text']}{_RESET}" if code else w["text"])
        pos = w["start"] + len(w["text"])
    out.append(text[pos:])
    return "".join(out)


def pct(x: float | None) -> str:
    return "n/a" if x is None else f"{round(100 * x)}%"


def legend() -> str:
    return (f"{_BANDS[0][1]} <25% {_RESET} {_BANDS[1][1]}<50%{_RESET} "
            f"{_BANDS[2][1]}<75%{_RESET} {_BANDS[3][1]}<90%{_RESET} sure")


def shaky_list(words: list[dict[str, Any]], low: float, limit: int = 5) -> str:
    weak = sorted((w for w in words if w["conf"] < low), key=lambda w: w["conf"])[:limit]
    parts = []
    for w in weak:
        alt = f" (alt: {w['alts'][0][0].strip()!r} {pct(w['alts'][0][1])})" if w.get("alts") else ""
        parts.append(f"`{w['text']}` {pct(w['conf'])}{alt}")
    return ", ".join(parts)
