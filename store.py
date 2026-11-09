"""Doubt records: replies (token heat maps) and guard decisions, in the plugin's SQLite file."""
from __future__ import annotations

import json
import threading
import time
import uuid
from typing import Any, Optional

_LOCK = threading.Lock()
_SCHEMA = ("CREATE TABLE IF NOT EXISTS records (id TEXT PRIMARY KEY, kind TEXT, session_id TEXT, "
           "turn_id TEXT, created REAL, body TEXT)")


def _db():
    from plugins.plugin_storage import plugin_db
    conn = plugin_db("doubt", "doubt.db")
    conn.execute(_SCHEMA)
    return conn


def add(kind: str, body: dict[str, Any], *, session_id: str = "", turn_id: str = "") -> str:
    rid = uuid.uuid4().hex[:12]
    with _LOCK:
        conn = _db()
        with conn:
            conn.execute("INSERT INTO records VALUES (?,?,?,?,?,?)",
                         (rid, kind, session_id, turn_id, time.time(), json.dumps(body)))
            conn.execute("DELETE FROM records WHERE id NOT IN "
                         "(SELECT id FROM records ORDER BY created DESC LIMIT 500)")
        conn.close()
    return rid


def recent(kind: Optional[str] = None, session_id: Optional[str] = None, limit: int = 20) -> list[dict]:
    sql, args = "SELECT id, kind, session_id, turn_id, created, body FROM records WHERE 1=1", []
    if kind:
        sql, args = sql + " AND kind=?", args + [kind]
    if session_id:
        sql, args = sql + " AND session_id=?", args + [session_id]
    conn = _db()
    rows = conn.execute(sql + " ORDER BY created DESC LIMIT ?", args + [limit]).fetchall()
    conn.close()
    return [{"id": r[0], "kind": r[1], "session_id": r[2], "turn_id": r[3], "created": r[4],
             **json.loads(r[5])} for r in rows]


def get(rid: str) -> Optional[dict]:
    conn = _db()
    r = conn.execute("SELECT id, kind, session_id, turn_id, created, body FROM records WHERE id=?",
                     (rid,)).fetchone()
    conn.close()
    if not r:
        return None
    return {"id": r[0], "kind": r[1], "session_id": r[2], "turn_id": r[3], "created": r[4], **json.loads(r[5])}
