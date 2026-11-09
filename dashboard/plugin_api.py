"""Doubt backend: read-only access to scored replies and guard decisions."""
from __future__ import annotations

import importlib.util
from pathlib import Path

from fastapi import APIRouter, HTTPException

router = APIRouter()
_spec = importlib.util.spec_from_file_location("doubt_store", Path(__file__).resolve().parent.parent / "store.py")
store = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(store)


@router.get("/records")
async def records(kind: str = "", session_id: str = "", limit: int = 30):
    return {"records": store.recent(kind or None, session_id or None, max(1, min(limit, 200)))}


@router.get("/records/{rid}")
async def record(rid: str):
    rec = store.get(rid)
    if rec is None:
        raise HTTPException(status_code=404, detail="unknown record")
    return rec
