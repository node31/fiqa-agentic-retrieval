"""Append-only audit log ("who did what"), hash-chained so edits and deletions are detectable.

FORMAT: one JSON object per line in audit.jsonl
    {"seq": 7, "ts": "2026-11-24T10:15:02.113Z", "user": "alice", "tenant": "acme", "role": "user",
     "action": "search", "decision": "allowed", "status": 200, "details": {...},
     "prev": "<hash of line 6>", "hash": "<sha256 of this line without 'hash'>"}

WHY A HASH CHAIN
    Each line includes the previous line's hash, and its own hash covers that. Changing any old line
    changes its hash, which no longer matches the "prev" stored in the next line, so verify() finds the
    first broken link. Deleting a line breaks the chain the same way. (Someone who can rewrite the WHOLE
    file can rebuild the chain; production systems also ship the latest hash somewhere else, e.g. a
    write-once store, so the tail can be checked too.)

    >>> import tempfile, pathlib
    >>> log = AuditLog(pathlib.Path(tempfile.mkdtemp()) / "audit.jsonl")
    >>> _ = log.append({"user": "alice", "action": "search"}); _ = log.append({"user": "bob", "action": "upsert"})
    >>> log.verify()["ok"]
    True
    >>> lines = log.path.read_text().splitlines(); lines[0] = lines[0].replace("alice", "mallory")
    >>> _ = log.path.write_text("\\n".join(lines) + "\\n"); log.verify()
    {'ok': False, 'first_bad_seq': 1, 'reason': 'hash mismatch'}

PRIVACY
    The log is itself sensitive. Queries are stored PII-redacted (policy.py does that before calling
    append), and the log is readable only through admin endpoints.
"""
from __future__ import annotations

import hashlib
import json
import threading
from datetime import datetime, timezone
from pathlib import Path

GENESIS = "0" * 64  # the "previous hash" of the first line


def _digest(rec: dict) -> str:
    body = {k: v for k, v in rec.items() if k != "hash"}
    return hashlib.sha256(json.dumps(body, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()


class AuditLog:
    def __init__(self, path: Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._seq, self._prev = 0, GENESIS
        if self.path.exists():  # continue the chain after a restart
            for line in self.path.read_text(encoding="utf-8").splitlines():
                if line.strip():
                    rec = json.loads(line)
                    self._seq, self._prev = rec["seq"], rec["hash"]

    def append(self, event: dict) -> dict:
        """Add one event; seq, ts, prev and hash are filled in here. Returns the stored record."""
        with self._lock:
            rec = {"seq": self._seq + 1, "ts": datetime.now(timezone.utc).isoformat(timespec="milliseconds"),
                   **event, "prev": self._prev}
            rec["hash"] = _digest(rec)
            with open(self.path, "a", encoding="utf-8") as f:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            self._seq, self._prev = rec["seq"], rec["hash"]
            return rec

    def tail(self, limit: int = 50) -> list[dict]:
        if not self.path.exists():
            return []
        lines = [ln for ln in self.path.read_text(encoding="utf-8").splitlines() if ln.strip()]
        return [json.loads(ln) for ln in lines[-limit:]]

    def verify(self) -> dict:
        """Walk the chain from the start; report the first line whose hash or link doesn't check out."""
        prev, n = GENESIS, 0
        if self.path.exists():
            for line in self.path.read_text(encoding="utf-8").splitlines():
                if not line.strip():
                    continue
                rec = json.loads(line)
                n += 1
                if rec.get("prev") != prev:
                    return {"ok": False, "first_bad_seq": rec.get("seq"), "reason": "broken link"}
                if _digest(rec) != rec.get("hash"):
                    return {"ok": False, "first_bad_seq": rec.get("seq"), "reason": "hash mismatch"}
                prev = rec["hash"]
        return {"ok": True, "lines": n}
