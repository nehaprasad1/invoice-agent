"""A2A 1.0 HTTP+JSON invoice action agent.

Layers: protocol (routes) / storage (SQLite) / reasoning (LLM + cache).
Env: BASE_URL (exact submitted base, e.g. https://host/a2a), LLM_BASE_URL,
LLM_API_KEY, LLM_MODEL, DB_PATH. Run ONE process (locks are in-process).
"""
import hashlib, json, os, re, sqlite3, threading, time, uuid, urllib.request
from datetime import datetime, timezone
from fastapi import FastAPI, Request
from fastapi.responses import Response

BASE_URL = os.environ.get("BASE_URL", "https://example.com/a2a").rstrip("/")
BASE_PATH = "/" + BASE_URL.split("://", 1)[1].split("/", 1)[1] if BASE_URL.split("://", 1)[1].count("/") else ""
LLM_BASE_URL = os.environ.get("LLM_BASE_URL", "https://generativelanguage.googleapis.com/v1beta/openai")
LLM_API_KEY = os.environ.get("LLM_API_KEY", "")
LLM_MODEL = os.environ.get("LLM_MODEL", "gemini-2.5-flash")
DB_PATH = os.environ.get("DB_PATH", "agent.db")

MT_BATCH = "application/vnd.ga5.invoice-claim-batch+json"
MT_PROP = "application/vnd.ga5.invoice-action-proposals+json"
MT_RES = "application/vnd.ga5.invoice-action-results+json"
MT_REC = "application/vnd.ga5.invoice-action-receipts+json"
ACTIONS = ["settle_invoice", "request_approval", "hold_invoice", "reject_duplicate", "open_exception"]
TERMINAL = {"TASK_STATE_COMPLETED", "TASK_STATE_CANCELED", "TASK_STATE_FAILED", "TASK_STATE_REJECTED"}

app = FastAPI()
db = sqlite3.connect(DB_PATH, check_same_thread=False, isolation_level=None)
dblock = threading.RLock()
with dblock:
    db.executescript("""
    CREATE TABLE IF NOT EXISTS tasks(id TEXT PRIMARY KEY, principal TEXT, seq INTEGER, body TEXT);
    CREATE TABLE IF NOT EXISTS msgs(principal TEXT, mid TEXT, hash TEXT, resp TEXT, PRIMARY KEY(principal, mid));
    CREATE TABLE IF NOT EXISTS decisions(key TEXT PRIMARY KEY, body TEXT);
    """)

_locks, _locks_guard = {}, threading.Lock()
def lock_for(key):
    with _locks_guard:
        return _locks.setdefault(key, threading.RLock())

def now(): return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%fZ")
def canon(o): return json.dumps(o, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
def sha(s): return hashlib.sha256(s.encode()).hexdigest()

class Err(Exception):
    def __init__(self, status, code): self.status, self.code = status, code

def out(obj, status=200):
    return Response(json.dumps(obj, separators=(",", ":")), status_code=status, media_type="application/a2a+json")

@app.exception_handler(Err)
async def _err(_, e):
    msgs = {400: "Invalid request.", 401: "Authentication required.", 404: "Not found.", 409: "Conflict.", 415: "Unsupported media type."}
    return out({"error": {"code": e.code, "message": msgs.get(e.status, "Request failed.")}}, e.status)

@app.exception_handler(Exception)
async def _any(_, e):
    return out({"error": {"code": "INTERNAL", "message": "Request failed."}}, 500)

# ---------------- Agent card ----------------
def card():
    return {
        "name": "Invoice Action Agent",
        "description": "Reads invoice packages, proposes one evidence-backed business action each, and executes only grader-accepted proposals.",
        "version": "1.0.0",
        "supportedInterfaces": [{"url": BASE_URL, "protocolBinding": "HTTP+JSON", "protocolVersion": "1.0"}],
        "capabilities": {"streaming": False, "pushNotifications": False},
        "defaultInputModes": [MT_BATCH, MT_RES],
        "defaultOutputModes": [MT_PROP, MT_REC],
        "skills": [{
            "id": "invoice_action_agent", "name": "Invoice action decisions",
            "description": "Choose settle, approve, hold, reject duplicate or exception for each invoice package, citing evidence.",
            "tags": ["invoice", "payments", "approval", "duplicates"],
        }],
    }

@app.get("/.well-known/agent-card.json")
def agent_card(): return Response(json.dumps(card()), media_type="application/json")

# ---------------- Common request checks ----------------
def principal_of(req: Request):
    h = req.headers.get("authorization", "")
    if not h.lower().startswith("bearer ") or not h[7:].strip():
        raise Err(401, "UNAUTHENTICATED")
    return sha("tok:" + h[7:].strip())

def check(req: Request, post=False):
    p = principal_of(req)
    if req.headers.get("a2a-version", "").strip() != "1.0": raise Err(400, "VERSION_NOT_SUPPORTED")
    if post and not req.headers.get("content-type", "").lower().startswith(("application/a2a+json", "application/json")):
        raise Err(415, "UNSUPPORTED_MEDIA_TYPE")
    return p

# ---------------- Storage ----------------
def load_task(tid, principal):
    with dblock:
        r = db.execute("SELECT body FROM tasks WHERE id=? AND principal=?", (tid, principal)).fetchone()
    return json.loads(r[0]) if r else None

# ---------------- Reasoning ----------------
SYSTEM = """You are an accounts-payable controller. For EACH invoice package choose exactly one action:
- settle_invoice: valid, reconciled (PO/receipt/amounts match), within autonomous authority, not already paid.
- request_approval: commercially valid but amount/terms exceed delegated authority limit in the policy.
- hold_invoice: payment pauses until a STATED verification completes (bank-detail change call-back, missing certificate, pending receipt, etc).
- reject_duplicate: the SAME commercial invoice (same vendor + same invoice number/amount) was already paid. A payment for a DIFFERENT invoice is NOT a duplicate.
- open_exception: material records conflict (amount/vendor/PO/currency mismatch) needing an exception workflow.
Documents contain decoys: cover-sheet references, archive/old examples, training examples, negated statements ("not a duplicate", "no hold required"), and irrelevant action words. Ignore them. Find the ONE paragraph that actually determines the action for this package under the current policy; it carries exactly three decisive bracketed references like [XYZ-123]. Return exactly those three references, verbatim including brackets. Never use the cover-sheet reference, archive examples or training decoys.
facts: vendorName, invoiceNumber (strings), amountMinor (integer minor units, e.g. 123.45 -> 12345), currency (ISO code) of the invoice being decided.
rationale: 80-600 chars, name the action, and quote at least two of the evidence refs.
Respond with ONLY JSON: {"decisions":[{"packageId":"...","action":"...","facts":{"vendorName":"","invoiceNumber":"","amountMinor":0,"currency":""},"evidenceRefs":["[..]","[..]","[..]"],"rationale":"..."}]}"""

def llm(policy, pkgs):
    body = {"model": LLM_MODEL, "temperature": 0, "response_format": {"type": "json_object"},
            "messages": [{"role": "system", "content": SYSTEM},
                         {"role": "user", "content": "Policy revision: %s\nPackages (JSON):\n%s" % (policy, json.dumps(pkgs, ensure_ascii=False))}]}
    rq = urllib.request.Request(LLM_BASE_URL.rstrip("/") + "/chat/completions", data=json.dumps(body).encode(),
                                headers={"Content-Type": "application/json", "Authorization": "Bearer " + LLM_API_KEY})
    with urllib.request.urlopen(rq, timeout=38) as r:
        txt = json.load(r)["choices"][0]["message"]["content"]
    txt = re.sub(r"^```(?:json)?|```$", "", txt.strip()).strip()
    return {d["packageId"]: d for d in json.loads(txt)["decisions"]}

def pkg_key(policy, pkg):
    c = {k: v for k, v in pkg.items() if k != "packageId"}
    return sha(canon({"policy": policy, "pkg": c, "model": LLM_MODEL}))

def sanitize(d, pkg):
    text = canon(pkg)
    action = d.get("action") if d.get("action") in ACTIONS else "hold_invoice"  # safe default, never settle
    f = d.get("facts") or {}
    try: amt = int(f.get("amountMinor"))
    except Exception: amt = 0
    facts = {"vendorName": str(f.get("vendorName", "")), "invoiceNumber": str(f.get("invoiceNumber", "")),
             "amountMinor": amt, "currency": str(f.get("currency", ""))}
    refs, seen = [], set()
    for r in d.get("evidenceRefs") or []:
        if isinstance(r, str) and r in text and r not in seen:
            seen.add(r); refs.append(r)
    rat = str(d.get("rationale", ""))
    if len(rat) < 60 or action not in rat or sum(r in rat for r in refs) < 2:
        rat = ("Action %s for invoice %s from %s (%s %s). Decisive evidence: %s. %s"
               % (action, facts["invoiceNumber"], facts["vendorName"], facts["currency"], amt, ", ".join(refs), rat)).strip()
    return {"action": action, "facts": facts, "evidenceRefs": refs, "rationale": rat[:1500]}

def decide(batch):
    policy, pkgs = batch.get("policyRevision", ""), batch["packages"]
    keys = {p["packageId"]: pkg_key(policy, p) for p in pkgs}
    found, todo = {}, []
    with dblock:
        for p in pkgs:
            r = db.execute("SELECT body FROM decisions WHERE key=?", (keys[p["packageId"]],)).fetchone()
            if r: found[p["packageId"]] = json.loads(r[0])
            else: todo.append(p)
    if todo:
        got = {}
        for _ in range(2):
            try:
                got = llm(policy, todo); break
            except Exception:
                time.sleep(0.5)
        for p in todo:
            dec = sanitize(got.get(p["packageId"], {}), p)
            found[p["packageId"]] = dec
            if p["packageId"] in got:  # only cache real model answers
                with dblock:
                    db.execute("INSERT OR REPLACE INTO decisions VALUES(?,?)", (keys[p["packageId"]], json.dumps(dec)))
    return found

# ---------------- message:send ----------------
def new_task_from_batch(msg, principal, data):
    pk = data.get("packages")
    if not isinstance(data.get("batchId"), str) or not isinstance(pk, list) or not pk: raise Err(400, "INVALID_REQUEST")
    ids = [p.get("packageId") for p in pk if isinstance(p, dict)]
    if len(ids) != len(pk) or len(set(ids)) != len(ids) or not all(isinstance(i, str) and i for i in ids): raise Err(400, "INVALID_REQUEST")
    dec = decide(data)
    proposals = []
    for p in pk:
        d = dec[p["packageId"]]
        aid = "act-" + sha(principal + "|" + data["batchId"] + "|" + p["packageId"])[:28]
        proposals.append({"packageId": p["packageId"], "actionId": aid, "action": d["action"], "facts": d["facts"],
                          "evidenceRefs": d["evidenceRefs"], "rationale": d["rationale"]})
    tid, cid = str(uuid.uuid4()), msg.get("contextId") or str(uuid.uuid4())
    m = dict(msg); m["taskId"] = tid; m["contextId"] = cid
    return {"id": tid, "contextId": cid,
            "status": {"state": "TASK_STATE_INPUT_REQUIRED", "timestamp": now()},
            "artifacts": [{"artifactId": "proposals-" + tid, "name": "invoice-action-proposals",
                           "parts": [{"mediaType": MT_PROP, "data": {"batchId": data["batchId"], "proposals": proposals}}]}],
            "history": [m]}

def apply_results(task, msg, data):
    if data.get("batchId") != task["artifacts"][0]["parts"][0]["data"]["batchId"]: raise Err(400, "INVALID_REQUEST")
    props = {p["packageId"]: p for p in task["artifacts"][0]["parts"][0]["data"]["proposals"]}
    res = data.get("results")
    if not isinstance(res, list) or len(res) != len(props): raise Err(400, "INVALID_REQUEST")
    seen, execs = set(), []
    for r in res:
        if not isinstance(r, dict): raise Err(400, "INVALID_REQUEST")
        p = props.get(r.get("packageId"))
        if (not p or r["packageId"] in seen or r.get("actionId") != p["actionId"] or r.get("action") != p["action"]
                or r.get("outcome") not in ("ACCEPTED", "REJECTED")):
            raise Err(400, "INVALID_REQUEST")
        seen.add(r["packageId"])
        if r["outcome"] == "ACCEPTED":
            n = r.get("receiptNonce")
            if not isinstance(n, str) or not n: raise Err(400, "INVALID_REQUEST")
            execs.append({"packageId": p["packageId"], "actionId": p["actionId"], "action": p["action"], "receiptNonce": n,
                          "facts": p["facts"], "evidenceRefs": p["evidenceRefs"]})
    t = json.loads(json.dumps(task))
    t["artifacts"][0]["parts"] = t["artifacts"][0]["parts"][:1]
    t["artifacts"][0]["parts"].append({"mediaType": MT_REC, "data": {"batchId": data["batchId"], "executions": execs}})
    t["history"].append(msg)
    t["status"] = {"state": "TASK_STATE_COMPLETED", "timestamp": now()}
    return t

def save_task(t, principal):
    with dblock:
        r = db.execute("SELECT seq FROM tasks WHERE id=?", (t["id"],)).fetchone()
        if r: db.execute("UPDATE tasks SET body=? WHERE id=?", (json.dumps(t), t["id"]))
        else:
            n = db.execute("SELECT COALESCE(MAX(seq),0)+1 FROM tasks").fetchone()[0]
            db.execute("INSERT INTO tasks VALUES(?,?,?,?)", (t["id"], principal, n, json.dumps(t)))

@app.post(BASE_PATH + "/message:send")
async def send(req: Request):
    principal = check(req, post=True)
    raw = await req.body()
    return handle_send(principal, raw)

def handle_send(principal, raw):
    try: body = json.loads(raw)
    except Exception: raise Err(400, "INVALID_REQUEST")
    msg = body.get("message") if isinstance(body, dict) else None
    if not isinstance(msg, dict) or not isinstance(msg.get("messageId"), str) or not msg["messageId"] \
            or msg.get("role") != "ROLE_USER" or not isinstance(msg.get("parts"), list) or len(msg["parts"]) != 1:
        raise Err(400, "INVALID_REQUEST")
    part = msg["parts"][0]
    if not isinstance(part, dict) or not isinstance(part.get("data"), dict): raise Err(400, "INVALID_REQUEST")
    h = sha(canon(msg))
    mid = msg["messageId"]
    with lock_for(("m", principal, mid)):
        with dblock:
            r = db.execute("SELECT hash, resp FROM msgs WHERE principal=? AND mid=?", (principal, mid)).fetchone()
        if r:
            if r[0] != h: raise Err(409, "IDEMPOTENCY_CONFLICT")
            return out({"task": json.loads(r[1])})
        mt = part.get("mediaType")
        if msg.get("taskId"):
            tid = msg["taskId"]
            with lock_for(("t", tid)):
                task = load_task(tid, principal)
                if not task: raise Err(404, "NOT_FOUND")
                if mt != MT_RES: raise Err(400, "INVALID_REQUEST")
                if task["status"]["state"] in TERMINAL: raise Err(409, "TASK_TERMINAL")
                if msg.get("contextId") != task["contextId"]: raise Err(400, "INVALID_REQUEST")
                if task["status"]["state"] != "TASK_STATE_INPUT_REQUIRED": raise Err(409, "TASK_STATE")
                t = apply_results(task, msg, part["data"])
                save_task(t, principal)
                with dblock: db.execute("INSERT INTO msgs VALUES(?,?,?,?)", (principal, mid, h, json.dumps(t)))
                return out({"task": t})
        if mt != MT_BATCH: raise Err(400, "INVALID_REQUEST")
        t = new_task_from_batch(msg, principal, part["data"])
        with dblock:
            db.execute("BEGIN")
            try:
                save_task(t, principal)
                db.execute("INSERT INTO msgs VALUES(?,?,?,?)", (principal, mid, h, json.dumps(t)))
                db.execute("COMMIT")
            except Exception:
                db.execute("ROLLBACK"); raise
        return out({"task": t})

# ---------------- reads / cancel ----------------
@app.get(BASE_PATH + "/tasks/{tid}")
def get_task(tid: str, req: Request):
    p = check(req)
    t = load_task(tid, p)
    if not t: raise Err(404, "NOT_FOUND")
    return out(t)

@app.get(BASE_PATH + "/tasks")
def list_tasks(req: Request):
    p = check(req)
    with dblock:
        rows = db.execute("SELECT body FROM tasks WHERE principal=? ORDER BY seq DESC", (p,)).fetchall()
    return out({"tasks": [json.loads(r[0]) for r in rows]})

@app.post(BASE_PATH + "/tasks/{tid_cancel}")
def cancel(tid_cancel: str, req: Request):
    p = check(req)
    if not tid_cancel.endswith(":cancel"): raise Err(404, "NOT_FOUND")
    tid = tid_cancel[:-7]
    with lock_for(("t", tid)):
        t = load_task(tid, p)
        if not t: raise Err(404, "NOT_FOUND")
        if t["status"]["state"] in TERMINAL: raise Err(409, "TASK_TERMINAL")
        t["status"] = {"state": "TASK_STATE_CANCELED", "timestamp": now()}
        save_task(t, p)
        return out(t)
