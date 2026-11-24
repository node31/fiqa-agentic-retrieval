"""Agent memory: what the agent remembers about a conversation and about a user, and how it uses it.

TWO KINDS (the usual split in agent frameworks)
    short-term  the turns of ONE conversation (A2A ``contextId``). Lets a follow-up such as "what are the
                taxes on that?" be searched as "<previous question> what are the taxes on that?".
    long-term   facts about the USER that outlive a conversation: "I live in Canada". Every later search
                gets the fact's keywords ("Canada"), so "what retirement account should I open?" finds
                RRSP / TFSA posts instead of US 401(k) ones.

    >>> m = MemoryStore()
    >>> m.remember_fact("acme", "alice", "I live in Canada")["text"]
    'I live in Canada'
    >>> m.remember_turn("acme", "alice", "ctx-1", "what is a roth ira?", ["214174"])["kind"]
    'turn'
    >>> contextualize(m, "acme", "alice", "ctx-1", "can I withdraw from it early?")["query"]
    'what is a roth ira? can I withdraw from it early? Canada'
    >>> contextualize(m, "globex", "alice", "ctx-1", "can I withdraw from it early?")["query"]   # other tenant
    'can I withdraw from it early?'

PRIVACY RULES (the same ones the retrieval service follows, docs/topics/17_security.md)
    * Only an IDENTIFIED user has memory (a verified token's tenant + user). Anonymous callers get none:
      there is nobody to attribute it to, and nobody who could ask to see or delete it.
    * Everything is PII-redacted BEFORE it is stored. "my email is jane@x.com" is kept as
      "my email is [EMAIL]": the memory never holds what the audit log is not allowed to hold.
    * Isolation: every read filters on (tenant, user). There is no query that crosses users.
    * Expiry: every entry has a TTL (MEMORY_TTL_DAYS, default 30). Expired entries are never returned and
      are purged on the next write.
    * The user can see everything kept about them (``export``) and delete it (``forget``), one entry or all.
      Deleted rows are overwritten with zeros in the file (PRAGMA secure_delete), not only unlinked.

STORAGE
    SQLite: ":memory:" by default (tests), or a file (MEMORY_DB) so memory survives a restart. One table;
    a user has a handful of facts and a few dozen turns, so reads scan that user's rows, no index needed
    beyond (tenant, user).
"""
from __future__ import annotations

import json
import os
import re
import sqlite3
import threading
import time
import uuid

from src.security import pii

TTL_DAYS = float(os.environ.get("MEMORY_TTL_DAYS", "30"))
MAX_TURNS = 3          # how many earlier turns of the same conversation a follow-up may lean on

# Countries with enough FiQA posts to matter, and the words a passage about them tends to contain.
COUNTRIES = {
    "Canada": ["canada", "canadian", "rrsp", "tfsa", "cra"],
    "UK": ["uk", "u.k.", "britain", "british", "england", "hmrc", "isa", "sipp"],
    "India": ["india", "indian", "rupee", "rupees", "inr", "nri"],
    "Australia": ["australia", "australian", "superannuation", "ato"],
}
_ALIASES = {"canada": "Canada", "uk": "UK", "u.k.": "UK", "united kingdom": "UK", "england": "UK",
            "britain": "UK", "great britain": "UK", "scotland": "UK", "wales": "UK", "india": "India",
            "australia": "Australia"}
_LIVE = re.compile(r"\b(?:i\s+live\s+in|i'?m\s+(?:from|based\s+in|living\s+in|in)|i\s+am\s+(?:from|based\s+in|living\s+in|in)"
                   r"|i\s+moved\s+to)\s+(?:the\s+)?([a-z. ]{2,20}?)(?=[,.;!?]|\s+and\b|\s+but\b|$)", re.I)
# Words that make a question depend on an earlier one: pronouns pointing back, or "what about ...".
_FOLLOWUP = re.compile(r"\b(it|its|that|this|those|these|they|them|their|theirs|there|same|instead|also|one|ones|each)\b"
                       r"|^(and|but|so|what about|how about|what if)\b", re.I)


def extract_facts(text: str) -> list[dict]:
    """Facts worth keeping across conversations. Rule-based on purpose: predictable and free.

    >>> extract_facts("I live in Canada and want to retire early")
    [{'type': 'country', 'value': 'Canada', 'text': 'I live in Canada'}]
    >>> extract_facts("I'm from the UK, is an ISA worth it?")
    [{'type': 'country', 'value': 'UK', 'text': 'I live in UK'}]
    >>> extract_facts("what is a roth ira?")
    []
    """
    out = []
    for m in _LIVE.finditer(text):
        country = _ALIASES.get(m.group(1).strip().lower())
        if country:
            out.append({"type": "country", "value": country, "text": f"I live in {country}"})
    return out


def needs_context(question: str) -> bool:
    """Does this question lean on an earlier turn? (cheap rule; an LLM rewrite is the stronger option)

    >>> needs_context("can I withdraw from it early?"), needs_context("what about for my kids?")
    (True, True)
    >>> needs_context("how do I open a roth ira?")
    False
    """
    return bool(_FOLLOWUP.search(question.strip()))


class MemoryStore:
    """Thread-safe SQLite store. Every method takes (tenant, user) and never reads outside them."""

    def __init__(self, path: str | None = None, ttl_days: float = TTL_DAYS, now=time.time):
        self.path = path or os.environ.get("MEMORY_DB", ":memory:")
        self.ttl_s, self.now = ttl_days * 86400, now
        self.lock = threading.Lock()
        self.db = sqlite3.connect(self.path, check_same_thread=False)
        # Without this, DELETE only marks pages free and the "forgotten" text stays readable in the file
        # until SQLite reuses the page (docs/topics/22_agent_memory.md, section 3).
        self.db.execute("PRAGMA secure_delete = ON")
        self.db.execute("""CREATE TABLE IF NOT EXISTS memory (
            id TEXT PRIMARY KEY, tenant TEXT NOT NULL, user TEXT NOT NULL, kind TEXT NOT NULL,
            context_id TEXT, text TEXT NOT NULL, data TEXT, created REAL NOT NULL, expires REAL NOT NULL)""")
        self.db.execute("CREATE INDEX IF NOT EXISTS by_user ON memory (tenant, user)")
        self.db.commit()

    # --- writes --------------------------------------------------------------------------------
    def _add(self, tenant: str, user: str, kind: str, text: str, context_id: str | None = None,
             data: dict | None = None) -> dict:
        if not tenant or not user:
            raise ValueError("memory needs an identified caller (tenant and user)")
        now = self.now()
        row = {"id": uuid.uuid4().hex[:12], "kind": kind, "context_id": context_id,
               "text": pii.redact(text)[0], "data": data or {}, "created": now, "expires": now + self.ttl_s}
        with self.lock:
            self.db.execute("DELETE FROM memory WHERE expires <= ?", (now,))   # purge on write
            self.db.execute("INSERT INTO memory VALUES (?,?,?,?,?,?,?,?,?)",
                            (row["id"], tenant, user, kind, context_id, row["text"], json.dumps(row["data"]),
                             now, row["expires"]))
            self.db.commit()
        return row

    def remember_turn(self, tenant: str, user: str, context_id: str, question: str, doc_ids: list[str]) -> dict:
        return self._add(tenant, user, "turn", question, context_id, {"doc_ids": list(doc_ids)[:10]})

    def remember_fact(self, tenant: str, user: str, text: str) -> dict:
        """Store a fact. A recognised country fact replaces the user's previous country (people move)."""
        facts = extract_facts(text)
        if facts and facts[0]["type"] == "country":
            with self.lock:
                self.db.execute("DELETE FROM memory WHERE tenant=? AND user=? AND kind='fact' AND data LIKE ?",
                                (tenant, user, '%"type": "country"%'))
                self.db.commit()
            return self._add(tenant, user, "fact", facts[0]["text"], data=facts[0])
        return self._add(tenant, user, "fact", text, data={"type": "note"})

    def forget(self, tenant: str, user: str, entry_id: str | None = None) -> int:
        """Delete one entry (by id) or everything kept about this user. Returns how many rows went."""
        with self.lock:
            if entry_id:
                cur = self.db.execute("DELETE FROM memory WHERE tenant=? AND user=? AND id=?", (tenant, user, entry_id))
            else:
                cur = self.db.execute("DELETE FROM memory WHERE tenant=? AND user=?", (tenant, user))
            self.db.commit()
            return cur.rowcount

    # --- reads ---------------------------------------------------------------------------------
    def _rows(self, sql: str, args: tuple) -> list[dict]:
        with self.lock:
            cur = self.db.execute(sql, args)
            cols = [c[0] for c in cur.description]
            rows = [dict(zip(cols, r)) for r in cur.fetchall()]
        for r in rows:
            r["data"] = json.loads(r["data"] or "{}")
        return rows

    def export(self, tenant: str, user: str) -> list[dict]:
        """Everything kept about this user and not expired, oldest first (the user's right to see it)."""
        return self._rows("SELECT id, kind, context_id, text, data, created, expires FROM memory "
                          "WHERE tenant=? AND user=? AND expires > ? ORDER BY created", (tenant, user, self.now()))

    def facts(self, tenant: str, user: str) -> list[dict]:
        return [r for r in self.export(tenant, user) if r["kind"] == "fact"]

    def turns(self, tenant: str, user: str, context_id: str, n: int = MAX_TURNS) -> list[dict]:
        """The last ``n`` turns of one conversation, oldest first."""
        rows = [r for r in self.export(tenant, user) if r["kind"] == "turn" and r["context_id"] == context_id]
        return rows[-n:]


def contextualize(store: MemoryStore | None, tenant: str | None, user: str | None, context_id: str | None,
                  question: str, use_turns: bool = True, use_facts: bool = True) -> dict:
    """The query to SEARCH for ``question``, given what memory holds. The question shown to an LLM is unchanged.

    Returns {"query", "turn_used", "facts_used"} so the caller can show what memory changed.
      * follow-up (``needs_context``) in a conversation with earlier turns: prepend the LAST question;
      * a country fact, when the question names no country: append the country name.
    """
    out = {"query": question, "turn_used": None, "facts_used": []}
    if store is None or not tenant or not user:
        return out
    q = question
    if use_turns and context_id and needs_context(question):
        prev = store.turns(tenant, user, context_id, n=1)
        if prev:
            q = f"{prev[-1]['text']} {q}"
            out["turn_used"] = prev[-1]["text"]
    if use_facts:
        words = set(re.findall(r"[a-z]+", q.lower().replace("u.k.", "uk")))
        for f in store.facts(tenant, user):
            country = f["data"].get("value") if f["data"].get("type") == "country" else None
            if country and not words & set(COUNTRIES.get(country, [])):
                q = f"{q} {country}"
                out["facts_used"].append(f["text"])
    out["query"] = q
    return out
