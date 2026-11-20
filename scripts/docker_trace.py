"""Prove that the router really talks to the three shard CONTAINERS over the Docker network.

WHAT
    A read-only-ish walkthrough against a running ``docker compose`` stack. ``docker_check.py``
    proves the stack *works* (searches answer, a killed container shows up as missing); this
    script shows *where every byte goes*, container by container:

    1. Inventory: 4 containers from ONE image, each with its own IP on the compose network;
       only the router publishes a port to the host, so shards are reachable ONLY via Docker.
    2. DNS: inside the router container, the names in SHARD_URLS (shard0/1/2) resolve to the
       shard containers' IPs (Docker's embedded DNS at 127.0.0.11).
    3. Writes: N new docs go to the ROUTER (host port 8000). We predict the owner of each one
       with ``crc32(doc_id) % 3`` and then ask EVERY shard container directly (``docker exec``
       -> its own localhost:8101) which ids it physically holds. Exactly the predicted owner
       must have each doc; the other two must not; each shard's doc count grows by exactly
       its share.
    4. Reads: one search through the router returns all N docs, i.e. it merged hits that live
       in three different containers.
    5. Wire evidence: each shard container's kernel TCP table (/proc/net/tcp) lists established
       connections on port 8101 whose remote end is the ROUTER container's IP.
    6. Optional ``--failover``: stop one shard container. Searches come back partial (missing
       that shard), writes owned by it fail with 503, writes owned by the others still succeed.
       Start it again: its acked docs are back (WAL replay) and searches are complete again.
    7. Cleanup: the test docs are deleted (tombstoned) through the router and counts return.

WHY
    Passing end-to-end checks could in principle be explained by something else (one container
    doing everything, a stale cache, the host process). Reading each container's own state is
    direct evidence that partitioning, scatter-gather and failure isolation happen across
    separate processes on separate network addresses.

RUN (stack must be up, e.g. ``scripts/bootstrap.ps1`` / ``scripts/bootstrap.sh``)
    python -m scripts.docker_trace              # steps 1-5 + cleanup  (~5 s)
    python -m scripts.docker_trace --failover   # + stop/start one shard container (~1 min)

TOY ANALOGY
    Three filing cabinets in three different rooms (shards) and one receptionist (router).
    You hand the receptionist 6 letters; afterwards you walk into each room yourself and look
    in the cabinet. If the letters are spread 2/2/2 exactly as the filing rule (crc32 % 3)
    says, the receptionist really is walking between rooms.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.index.sharding import shard_of  # noqa: E402  (same routing rule the router uses)

URL = "http://127.0.0.1:8000"
SHARD_PORT = 8101
FAILS: list[str] = []

# Runs INSIDE a shard container: POST to that container's own localhost:8101.
IN_SHARD_RPC = (
    "import sys,urllib.request;"
    "r=urllib.request.Request('http://127.0.0.1:%d/'+sys.argv[1],data=sys.argv[2].encode(),"
    "headers={'Content-Type':'application/json'});"
    "print(urllib.request.urlopen(r,timeout=20).read().decode())" % SHARD_PORT
)
# Runs INSIDE a shard container: established TCP connections whose LOCAL port is 8101.
# /proc/net/tcp is per network namespace, i.e. it shows this container's sockets only.
IN_SHARD_TCP = (
    "import socket,struct,json;out=[]\n"
    "for l in open('/proc/net/tcp').readlines()[1:]:\n"
    "    p=l.split();lp=int(p[1].split(':')[1],16)\n"
    "    if lp==%d and p[3]=='01':\n"  # 01 = TCP_ESTABLISHED
    "        h,pt=p[2].split(':');out.append(socket.inet_ntoa(struct.pack('<I',int(h,16))))\n"
    "print(json.dumps(out))" % SHARD_PORT
)


def docker() -> str:
    found = shutil.which("docker")
    if found:
        return found
    cand = Path(os.environ.get("LOCALAPPDATA", "")) / "Programs" / "DockerDesktop" / "resources" / "bin" / "docker.exe"
    if cand.exists():
        return str(cand)
    sys.exit("docker CLI not found")


def sh(*args: str, check: bool = True) -> str:
    env = {**os.environ, "PATH": str(Path(docker()).parent) + os.pathsep + os.environ.get("PATH", "")}
    p = subprocess.run([docker(), *args], cwd=ROOT, env=env, capture_output=True, text=True)
    if check and p.returncode:
        sys.exit(f"docker {' '.join(args)} failed: {p.stderr.strip()}")
    return p.stdout.strip()


def http(method: str, path: str, body: dict | None = None, timeout: float = 30) -> tuple[int, dict]:
    data = json.dumps(body).encode() if body is not None else None
    req = urllib.request.Request(URL + path, data=data, method=method, headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return r.status, json.loads(r.read() or b"{}")
    except urllib.error.HTTPError as e:
        return e.code, json.loads(e.read() or b"{}")


def check(name: str, ok: bool, detail: str = "") -> None:
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}" + (f"  ({detail})" if detail else ""), flush=True)
    if not ok:
        FAILS.append(name)


def section(title: str) -> None:
    print(f"\n== {title}", flush=True)


class Stack:
    """Container ids/IPs of the running compose project."""

    def __init__(self) -> None:
        self.cid = {s: sh("compose", "ps", "-q", s) for s in ("router", "shard0", "shard1", "shard2")}
        missing = [s for s, c in self.cid.items() if not c]
        if missing:
            sys.exit(f"not running: {missing}. Start the stack first (scripts/bootstrap.*).")

    def ip(self, svc: str) -> str:
        return sh("inspect", "-f", "{{range .NetworkSettings.Networks}}{{.IPAddress}}{{end}}", self.cid[svc])

    def shard_rpc(self, i: int, op: str, payload: dict) -> dict:
        """Call shard i from INSIDE its own container (bypasses the router entirely)."""
        return json.loads(sh("exec", self.cid[f"shard{i}"], "python", "-c", IN_SHARD_RPC, op, json.dumps(payload)))

    def shard_conns(self, i: int) -> list[str]:
        return json.loads(sh("exec", self.cid[f"shard{i}"], "python", "-c", IN_SHARD_TCP))


def holders(stack: Stack, ids: list[str]) -> dict[int, set[str]]:
    """Ask every shard container which of ``ids`` it physically stores."""
    return {i: set(stack.shard_rpc(i, "docs", {"ids": ids})) for i in range(3)}


def counts(stack: Stack) -> dict[int, int]:
    return {i: stack.shard_rpc(i, "health", {})["n_docs"] for i in range(3)}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--per-shard", type=int, default=2, help="test docs per shard (default 2)")
    ap.add_argument("--failover", action="store_true", help="also stop/start one shard container")
    args = ap.parse_args()

    stack = Stack()

    # 1 -------------------------------------------------------------------------------------
    section("1. Containers: one image, four processes, four network addresses")
    ips = {s: stack.ip(s) for s in stack.cid}
    images = {s: sh("inspect", "-f", "{{.Config.Image}}", c) for s, c in stack.cid.items()}
    ports = {s: sh("port", c) for s, c in stack.cid.items()}
    print(f"  {'service':8} {'container':14} {'ip':15} {'image':22} published-to-host")
    for s in stack.cid:
        print(f"  {s:8} {stack.cid[s][:12]:14} {ips[s]:15} {images[s]:22} {ports[s].splitlines()[0] if ports[s] else '-'}")
    check("4 distinct containers", len(set(stack.cid.values())) == 4)
    check("4 distinct IPs on the compose network", len(set(ips.values())) == 4)
    check("all from the same image", len(set(images.values())) == 1, next(iter(images.values())))
    check("only the router publishes a port; shards are reachable only inside Docker",
          bool(ports["router"]) and not any(ports[f"shard{i}"] for i in range(3)))

    # 2 -------------------------------------------------------------------------------------
    section("2. Service discovery inside the router container")
    env = sh("exec", stack.cid["router"], "printenv", "SHARD_URLS")
    print(f"  router SHARD_URLS = {env}")
    resolved = json.loads(sh("exec", stack.cid["router"], "python", "-c",
                             "import socket,json;print(json.dumps({h:socket.gethostbyname(h) for h in ['shard0','shard1','shard2']}))"))
    for h, ip in resolved.items():
        print(f"  router resolves {h:7} -> {ip}   (container ip {ips[h]})")
    check("router's DNS names map to the shard containers", all(resolved[h] == ips[h] for h in resolved))

    # 3 -------------------------------------------------------------------------------------
    section("3. Writes through the router land in the predicted container")
    tag = f"trace{int(time.time())}"
    token = f"zq{tag}"  # a made-up word so a BM25 search for it matches only our docs
    ids: list[str] = []
    k = 0
    while any(sum(shard_of(d, 3) == i for d in ids) < args.per_shard for i in range(3)):
        d = f"{tag}-{k}"
        if sum(shard_of(x, 3) == shard_of(d, 3) for x in ids) < args.per_shard:
            ids.append(d)
        k += 1
    before = counts(stack)
    for d in ids:
        st, r = http("POST", "/upsert", {"doc_id": d, "version": 1, "text": f"{token} docker trace document {d}"})
        print(f"  POST router/upsert {d:18} predicted owner shard{shard_of(d, 3)}  -> HTTP {st} {r}")
    http("POST", "/admin/flush")  # no-op in sync mode; waits for the applier in async mode
    have = holders(stack, ids)
    after = counts(stack)
    print(f"  {'doc_id':18} {'predicted':10} shard0 shard1 shard2   (asked each container directly)")
    for d in ids:
        print(f"  {d:18} shard{shard_of(d, 3):<5} " + " ".join(f"{'  yes ' if d in have[i] else '   -  '}" for i in range(3)))
    check("every doc is stored in exactly its predicted container",
          all(have[i] == {d for d in ids if shard_of(d, 3) == i} for i in range(3)))
    check("per-container doc counts grew by exactly their share",
          all(after[i] - before[i] == args.per_shard for i in range(3)),
          ", ".join(f"shard{i}: {before[i]} -> {after[i]}" for i in range(3)))

    # 4 -------------------------------------------------------------------------------------
    section("4. One search through the router merges hits from all three containers")
    for strategy in ("bm25", "hybrid"):
        st, r = http("POST", "/search", {"query": token, "k": 10, "strategy": strategy})
        got = [h["doc_id"] for h in r.get("results", [])]
        mine = [d for d in got if d in ids]
        owners = sorted({shard_of(d, 3) for d in mine})
        print(f"  {strategy:6} -> {len(mine)}/{len(ids)} trace docs, from shards {owners}, partial={r.get('partial')}, "
              f"{r.get('latency_ms')} ms")
        check(f"{strategy}: all {len(ids)} docs found, spread over 3 containers, not partial",
              set(mine) == set(ids) and owners == [0, 1, 2] and not r.get("partial"))

    # 5 -------------------------------------------------------------------------------------
    section("5. Wire evidence: TCP connections into each shard come from the router's IP")
    for i in range(3):
        peers = stack.shard_conns(i)
        labels = sorted({("router" if p == ips["router"] else "localhost(healthcheck)" if p.startswith("127.") else p)
                         for p in peers})
        print(f"  shard{i} :{SHARD_PORT} established from: {labels}  ({len(peers)} sockets)")
        check(f"shard{i} has live connections from the router container ({ips['router']})", ips["router"] in peers)

    # 6 -------------------------------------------------------------------------------------
    if args.failover:
        victim = 2
        section(f"6. Failure isolation: stop the shard{victim} container")
        sh("compose", "stop", f"shard{victim}")
        st, r = http("POST", "/search", {"query": token, "k": 10, "strategy": "bm25"})
        got = {h["doc_id"] for h in r.get("results", [])} & set(ids)
        print(f"  search -> partial={r.get('partial')} missing={r.get('missing_shards')} trace docs={len(got)}")
        check(f"search still answers, flagged partial, missing [{victim}]",
              st == 200 and r.get("partial") and r.get("missing_shards") == [victim])
        check(f"only shard{victim}'s docs are missing from the answer",
              got == {d for d in ids if shard_of(d, 3) != victim})
        dead = next(d for d in (f"{tag}-dead-{j}" for j in range(500)) if shard_of(d, 3) == victim)
        live = next(d for d in (f"{tag}-live-{j}" for j in range(500)) if shard_of(d, 3) != victim)
        sd, _ = http("POST", "/upsert", {"doc_id": dead, "version": 1, "text": f"{token} x"})
        sl, _ = http("POST", "/upsert", {"doc_id": live, "version": 1, "text": f"{token} x"})
        print(f"  write owned by shard{victim} -> HTTP {sd};  write owned by shard{shard_of(live, 3)} -> HTTP {sl}")
        check("write to the stopped container fails loudly (503), others still accept writes", sd == 503 and sl == 200)
        ids.append(live)
        t0 = time.time()
        sh("compose", "start", f"shard{victim}")
        while time.time() - t0 < 300:
            st, s = http("GET", "/admin/status")
            if st == 200 and not s.get("missing_shards") and all(v.get("healthy") for v in s["shards"].values()):
                break
            time.sleep(2)
        st, r = http("POST", "/search", {"query": token, "k": 10, "strategy": "bm25"})
        got = {h["doc_id"] for h in r.get("results", [])} & set(ids)
        print(f"  restarted in {time.time() - t0:.0f} s -> partial={r.get('partial')} trace docs={len(got)}/{len(ids)}")
        check(f"shard{victim} back: its acked docs reloaded from its WAL, search complete",
              not r.get("partial") and got == set(ids))

    # 7 -------------------------------------------------------------------------------------
    section("7. Cleanup: delete the trace docs through the router")
    for d in ids:
        http("POST", "/delete", {"doc_id": d, "version": 2})
    http("POST", "/admin/flush")
    have = holders(stack, ids)
    check("trace docs gone from every container", not any(have.values()))

    print(f"\n{'ALL PASSED' if not FAILS else f'{len(FAILS)} FAILED: {FAILS}'}")
    sys.exit(1 if FAILS else 0)


if __name__ == "__main__":
    main()
