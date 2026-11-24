"""A2A (Agent2Agent) protocol 1.0: the JSON shapes and error codes this project uses.

A2A in one paragraph: an agent publishes an AGENT CARD at ``/.well-known/agent-card.json`` saying who it
is, what it can do (skills), where to call it and how to authenticate. Another agent (the client) sends it
a MESSAGE; the server answers with a TASK that has an id, a state (submitted, working, completed, failed,
rejected, ...), and ARTIFACTS holding the result. Tasks that belong together share a ``contextId`` (one
conversation). The binding used here is JSON-RPC 2.0 over HTTP POST, methods ``SendMessage``, ``GetTask``,
``ListTasks`` and ``CancelTask``. Spec: https://a2a-protocol.org/latest/specification/ (version 1.0.0).

    >>> msg = user_message("what is a roth ira?", context_id="c1")
    >>> msg["role"], msg["parts"], msg["contextId"]
    ('ROLE_USER', [{'text': 'what is a roth ira?'}], 'c1')
    >>> text_of(msg)
    'what is a roth ira?'
    >>> rpc_error(7, TASK_NOT_FOUND, "Task not found")["error"]["code"]
    -32001

JSON conventions of A2A 1.0 (section 5.5): field names are camelCase, enum values are the protobuf names
in capitals (``TASK_STATE_COMPLETED``, ``ROLE_USER``), timestamps are ISO 8601 in UTC with a ``Z``.
"""
from __future__ import annotations

import time
import uuid

PROTOCOL_VERSION = "1.0"
DOMAIN = "a2a-protocol.org"

SUBMITTED, WORKING, COMPLETED = "TASK_STATE_SUBMITTED", "TASK_STATE_WORKING", "TASK_STATE_COMPLETED"
FAILED, CANCELED, REJECTED = "TASK_STATE_FAILED", "TASK_STATE_CANCELED", "TASK_STATE_REJECTED"
INPUT_REQUIRED, AUTH_REQUIRED = "TASK_STATE_INPUT_REQUIRED", "TASK_STATE_AUTH_REQUIRED"
STATES = (SUBMITTED, WORKING, COMPLETED, FAILED, CANCELED, REJECTED, INPUT_REQUIRED, AUTH_REQUIRED)
TERMINAL = (COMPLETED, FAILED, CANCELED, REJECTED)

# JSON-RPC 2.0 standard codes, then the A2A-specific ones (spec section 5.4).
PARSE_ERROR, INVALID_REQUEST, METHOD_NOT_FOUND, INVALID_PARAMS, INTERNAL = -32700, -32600, -32601, -32602, -32603
TASK_NOT_FOUND, TASK_NOT_CANCELABLE, PUSH_NOT_SUPPORTED, UNSUPPORTED_OPERATION = -32001, -32002, -32003, -32004
VERSION_NOT_SUPPORTED = -32009


def now_iso() -> str:
    t = time.time()
    return time.strftime("%Y-%m-%dT%H:%M:%S", time.gmtime(t)) + f".{int(t % 1 * 1000):03d}Z"


def new_id() -> str:
    return str(uuid.uuid4())


def user_message(text: str, context_id: str | None = None, task_id: str | None = None,
                 data: dict | None = None, metadata: dict | None = None) -> dict:
    """A client's message: one text part, optionally a structured ``data`` part and metadata."""
    parts = [{"text": text}] if text else []
    if data:
        parts.append({"data": data, "mediaType": "application/json"})
    msg = {"messageId": new_id(), "role": "ROLE_USER", "parts": parts}
    if context_id:
        msg["contextId"] = context_id
    if task_id:
        msg["taskId"] = task_id
    if metadata:
        msg["metadata"] = metadata
    return msg


def agent_message(text: str, context_id: str, task_id: str | None = None) -> dict:
    msg = {"messageId": new_id(), "role": "ROLE_AGENT", "parts": [{"text": text}], "contextId": context_id}
    if task_id:
        msg["taskId"] = task_id
    return msg


def text_of(message: dict) -> str:
    """All text parts of a message, joined (a Part holds exactly one of text / raw / url / data)."""
    return " ".join(p["text"].strip() for p in message.get("parts", []) if isinstance(p.get("text"), str)).strip()


def data_of(message: dict) -> dict:
    """The structured ``data`` parts of a message, merged into one dict."""
    out: dict = {}
    for p in message.get("parts", []):
        if isinstance(p.get("data"), dict):
            out.update(p["data"])
    return out


def rpc_result(req_id, result: dict) -> dict:
    return {"jsonrpc": "2.0", "id": req_id, "result": result}


def rpc_error(req_id, code: int, message: str, reason: str | None = None, **metadata) -> dict:
    """JSON-RPC error with an A2A ErrorInfo detail (spec section 9.5)."""
    err = {"code": code, "message": message}
    if reason:
        err["data"] = [{"@type": "type.googleapis.com/google.rpc.ErrorInfo", "reason": reason, "domain": DOMAIN,
                        **({"metadata": {k: str(v) for k, v in metadata.items()}} if metadata else {})}]
    return {"jsonrpc": "2.0", "id": req_id, "error": err}
