from pathlib import Path
import json
import os
import queue
import subprocess
import sys
import threading
import time
from urllib.parse import urlparse

import requests as _requests
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.responses import FileResponse, Response, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
load_dotenv(ROOT / ".env")

from app.providers.ollama_provider import OllamaProvider, OllamaProviderError
from app.rag import retrieve, verify_grounding, get_active_flags
from app.query_log import log_query, read_recent

api = FastAPI(title="Enterprise RAG AI")
app = api  # uvicorn app:app

_STATIC_DIR = ROOT / "static"

# Serve JS/CSS with no-cache so browsers always pick up code changes
@app.get("/static/app.js")
def serve_app_js():
    path = _STATIC_DIR / "app.js"
    return FileResponse(str(path), media_type="application/javascript",
                        headers={"Cache-Control": "no-cache, no-store, must-revalidate",
                                 "Pragma": "no-cache", "Expires": "0"})

@app.get("/static/style.css")
def serve_style_css():
    path = _STATIC_DIR / "style.css"
    return FileResponse(str(path), media_type="text/css",
                        headers={"Cache-Control": "no-cache, no-store, must-revalidate",
                                 "Pragma": "no-cache", "Expires": "0"})

if _STATIC_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(_STATIC_DIR)), name="static")


class ChatRequest(BaseModel):
    message: str
    company_code: str = ""


@app.get("/")
def index():
    index_file = ROOT / "static" / "index.html"
    if index_file.exists():
        return FileResponse(str(index_file))
    return {"status": "ok", "service": "enterprise-rag-ai"}


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/rag/flags")
def rag_flags():
    """Show which RAG feature flags are currently active."""
    return get_active_flags()


_KB_LABELS: dict[str, str] = {
    "locktonwattana": "Lockton Wattana",
    "deves": "Deves Insurance",
    "mgcars": "MG Cars Thailand",
}


@app.get("/kb")
def list_kb():
    """Return available KB tenants with doc/faq counts via PostgREST REST API."""
    url = os.getenv("SUPABASE_URL", "")
    key = os.getenv("SUPABASE_KEY", "")
    default_code = os.getenv("COMPANY_CODE", "")

    count_hdr = {
        "apikey": key,
        "Authorization": f"Bearer {key}",
        "Prefer": "count=exact",
    }

    # รวม company_codes จาก _KB_LABELS + COMPANY_CODE env + ฐานข้อมูลจริง (แบบไดนามิก)
    known = list(_KB_LABELS.keys())
    if default_code and default_code not in known:
        known.append(default_code)
    try:
        rd = _requests.get(
            f"{url}/rest/v1/documents?select=company_code&company_code=not.is.null&limit=1000",
            headers={"apikey": key, "Authorization": f"Bearer {key}"}, timeout=5,
        )
        rf = _requests.get(
            f"{url}/rest/v1/faq_items?select=company_code&company_code=not.is.null&limit=1000",
            headers={"apikey": key, "Authorization": f"Bearer {key}"}, timeout=5,
        )
        for resp in (rd, rf):
            if resp.ok:
                for row in resp.json():
                    code = (row.get("company_code") or "").strip()
                    if code and code not in known:
                        known.append(code)
    except Exception:
        pass

    rows: list[dict] = []
    for code in known:
        try:
            rd = _requests.get(
                f"{url}/rest/v1/documents?company_code=eq.{code}&select=id&limit=1",
                headers=count_hdr, timeout=5,
            )
            rf = _requests.get(
                f"{url}/rest/v1/faq_items?company_code=eq.{code}&select=id&limit=1",
                headers=count_hdr, timeout=5,
            )
            doc_count = int(rd.headers.get("content-range", "0/0").split("/")[-1]) if rd.ok else 0
            faq_count = int(rf.headers.get("content-range", "0/0").split("/")[-1]) if rf.ok else 0
        except Exception:
            doc_count = faq_count = 0
        rows.append({"company_code": code, "docs": doc_count, "faqs": faq_count})

    return {
        "default": default_code,
        "items": [
            {
                "code": row["company_code"],
                "label": _KB_LABELS.get(row["company_code"], row["company_code"]),
                "docs": int(row.get("docs") or 0),
                "faqs": int(row.get("faqs") or 0),
            }
            for row in rows
        ],
    }


@app.get("/provider")
def provider():
    p = OllamaProvider()
    return {
        "provider": p.config.provider,
        "rag_mode": os.getenv("RAG_MODE", "full"),
        "base_url": p.config.base_url,
        "chat_model": p.config.chat_model,
        "generate_model": p.config.generate_model,
        "embed_model": p.config.embed_model,
    }


_COMPANY_HINTS: dict[str, str] = {
    "mgcars": (
        "MG Cars Thailand ไม่มีสำนักงานใหญ่เดียว แต่มีโชว์รูมและตัวแทนจำหน่ายทั่วประเทศ "
        "หากถูกถามเรื่องที่อยู่หรือสถานที่ตั้ง ให้แนะนำให้ลูกค้าค้นหาโชว์รูมใกล้บ้านได้ที่ "
        "https://www.mgcars.com/th/find-showroom "
        "สำหรับข้อมูลติดต่อบริษัท ดูได้ที่เว็บไซต์หลัก https://www.mgcars.com "
    ),
}


def _build_system_prompt(company_code: str) -> str:
    label = _KB_LABELS.get(company_code, company_code or "บริษัท")
    hint = _COMPANY_HINTS.get(company_code, "")
    return (
        f"คุณเป็นผู้ช่วยตอบคำถามของ {label} "
        f"(company_code: {company_code}) "
        "ตอบโดยอิงจากข้อมูลบริบทที่ให้มาเท่านั้น "
        "เมื่อถูกถามเรื่องชื่อบริษัท ที่อยู่ เบอร์โทร หรือข้อมูลติดต่อ ให้ใช้ข้อมูลจากบริบทด้านล่างได้เลย "
        + (f"{hint}" if hint else "")
        + "หากข้อมูลในบริบทไม่เพียงพอ ให้ตอบตรงๆ ว่าไม่มีข้อมูลนั้นในระบบและแนะนำช่องทางที่เกี่ยวข้อง "
        "ตอบเป็นภาษาเดียวกับคำถาม"
    )


def _build_sources(docs: list[dict], faqs: list[dict]) -> list[dict]:
    seen: set[str] = set()
    sources: list[dict] = []

    for d in docs:
        url = (d.get("url") or "").strip()
        if not url or url in seen:
            continue
        seen.add(url)
        sources.append({"url": url, "title": url, "similarity": round(float(d.get("similarity", 0)), 3)})

    for f in faqs:
        url = (f.get("source_url") or "").strip()
        if not url or url in seen:
            continue
        seen.add(url)
        sources.append({"url": url, "title": url, "similarity": round(float(f.get("similarity", 0)), 3)})

    sources.sort(key=lambda s: s["similarity"], reverse=True)
    return sources[:5]


@app.post("/chat")
def chat(req: ChatRequest):
    _t0 = time.time()
    try:
        p = OllamaProvider()
        context, docs, faqs = retrieve(req.message, p, company_code=req.company_code)

        resolved_code = req.company_code or os.getenv("COMPANY_CODE", "")
        system_prompt = _build_system_prompt(resolved_code)

        if context:
            system = f"{system_prompt}\n\n{context}"
            messages = [
                {"role": "system", "content": system},
                {"role": "user", "content": req.message},
            ]
        else:
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": req.message},
            ]

        answer = p.chat(messages)
        answer = verify_grounding(req.message, answer, context, p)
        sources = _build_sources(docs, faqs)
        log_query(resolved_code, req.message, answer, sources,
                  docs_found=len(docs), faqs_found=len(faqs),
                  status="ok", duration_ms=int((time.time() - _t0) * 1000))
        return {"answer": answer, "sources": sources}
    except OllamaProviderError as ex:
        log_query(os.getenv("COMPANY_CODE", ""), req.message, "", [],
                  status="error", error_detail=str(ex),
                  duration_ms=int((time.time() - _t0) * 1000))
        raise HTTPException(status_code=500, detail=str(ex))


@app.post("/chat/stream")
def chat_stream(req: ChatRequest):
    """สตรีมความคืบหน้าของแต่ละ node เป็น NDJSON แล้วตามด้วยคำตอบทีละ token"""
    def _event(obj: dict) -> str:
        return json.dumps(obj, ensure_ascii=False) + "\n"

    def generate():
        _t0 = time.time()
        node_timings: dict = {}
        try:
            p = OllamaProvider()
        except Exception as ex:
            log_query(os.getenv("COMPANY_CODE", ""), req.message, "", [],
                      status="error", error_detail=f"provider: {ex}",
                      duration_ms=int((time.time() - _t0) * 1000))
            yield _event({"type": "error", "detail": str(ex)})
            return

        # retrieve() รันในเธรดแยก แล้วดันสถานะทุก node ออกทางคิว
        q: queue.Queue = queue.Queue()
        outcome: dict = {}

        def progress(step: str, label: str) -> None:
            node_timings[f"{step}_at_s"] = round(time.time() - _t0, 2)
            q.put(_event({"type": "status", "step": step, "label": label}))

        def _run_retrieve():
            try:
                outcome["result"] = retrieve(req.message, p, company_code=req.company_code, progress=progress)
            except Exception as ex:
                outcome["error"] = str(ex)
            q.put(None)

        threading.Thread(target=_run_retrieve, daemon=True).start()
        while True:
            item = q.get()
            if item is None:
                break
            yield item

        if "error" in outcome:
            log_query(req.company_code or os.getenv("COMPANY_CODE", ""), req.message, "", [],
                      status="error", error_detail=f"retrieve: {outcome['error']}",
                      node_timings=node_timings,
                      duration_ms=int((time.time() - _t0) * 1000))
            yield _event({"type": "error", "detail": outcome["error"]})
            return

        context, docs, faqs = outcome["result"]
        node_timings["docs_found"] = len(docs)
        node_timings["faqs_found"] = len(faqs)
        if not context:
            yield _event({"type": "status", "step": "nocontext",
                          "label": "ไม่พบข้อมูลที่เกี่ยวข้องในฐานความรู้ — โมเดลจะตอบจากความรู้ทั่วไป"})

        resolved_code = req.company_code or os.getenv("COMPANY_CODE", "")
        system_prompt = _build_system_prompt(resolved_code)
        if context:
            system = f"{system_prompt}\n\n{context}"
            messages = [
                {"role": "system", "content": system},
                {"role": "user", "content": req.message},
            ]
        else:
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": req.message},
            ]

        yield _event({"type": "status", "step": "llm",
                      "label": f"Ollama ({p.config.chat_model}): กำลังเขียนคำตอบ"})

        parts: list = []
        try:
            for token in p.chat_stream(messages):
                parts.append(token)
                yield _event({"type": "token", "text": token})
        except OllamaProviderError:
            answer = p.chat(messages)
            parts = [answer]
            if answer:
                yield _event({"type": "token", "text": answer})

        answer = "".join(parts)
        answer = verify_grounding(req.message, answer, context, p)
        sources = _build_sources(docs, faqs)
        node_timings["llm_done_at_s"] = round(time.time() - _t0, 2)
        log_query(resolved_code, req.message, answer, sources,
                  docs_found=len(docs), faqs_found=len(faqs),
                  node_timings=node_timings, status="ok",
                  duration_ms=int((time.time() - _t0) * 1000))
        yield _event({"type": "done", "answer": answer, "sources": sources})

    return StreamingResponse(
        generate(),
        media_type="application/x-ndjson",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@app.get("/logs")
def list_logs(limit: int = 50):
    """ประวัติคำถาม-คำตอบล่าสุด (JSON) — ใช้วิเคราะห์ว่าอะไรตอบได้/ไม่ได้"""
    return {"count": limit, "items": read_recent(limit)}


@app.get("/logs/view")
def logs_view(limit: int = 50):
    """หน้าตารางดู log ง่าย ๆ ในเบราว์เซอร์ + แผงเทรน KB ใหม่จากเว็บไซต์"""
    rows = read_recent(limit)
    body = ""
    for r in reversed(rows):  # ใหม่สุดอยู่บน
        timings = r.get("node_timings") or {}
        docs = r.get("docs_found", 0)
        ok = r.get("status") == "ok"
        color = "#86efac" if ok and docs else "#fca5a5"
        tim = ", ".join(f"{k}={v}" for k, v in timings.items()) or "-"
        ans = (r.get("answer") or "").replace("<", "&lt;").replace("\n", " ")[:200]
        q = (r.get("question") or "").replace("<", "&lt;")
        srcs = ", ".join(s.get("url", "") for s in (r.get("sources") or [])[:3])
        body += f"""
        <tr>
          <td>{r.get('created_at','')}</td>
          <td>{r.get('company_code','')}</td>
          <td>{q}</td>
          <td style="color:{color}">{docs}</td>
          <td>{tim}</td>
          <td>{r.get('duration_ms','')} ms</td>
          <td>{ans}</td>
          <td>{srcs}</td>
        </tr>"""
    html = f"""<!doctype html><html lang="th"><head><meta charset="utf-8">
<title>Query Logs + Train</title><style>
body{{font-family:sans-serif;background:#0f172a;color:#e2e8f0;padding:20px}}
table{{border-collapse:collapse;width:100%;font-size:13px}}
td,th{{border:1px solid #334155;padding:6px 8px;text-align:left;vertical-align:top}}
th{{background:#1e293b}} tr:nth-child(even){{background:#1e293b55}}
.train-panel{{background:#1e293b;border:1px solid #334155;border-radius:12px;padding:14px 16px;margin-bottom:18px}}
.train-panel h2{{margin:0 0 10px;font-size:16px}}
.train-row{{display:flex;gap:8px;flex-wrap:wrap;align-items:center}}
.train-row input{{background:#0f172a;border:1px solid #334155;color:#e2e8f0;border-radius:8px;padding:8px 10px}}
#trainBtn{{background:#0284c7;border:none;color:#fff;border-radius:8px;padding:8px 16px;cursor:pointer;font-weight:600}}
#trainBtn:disabled{{opacity:.5;cursor:not-allowed}}
.train-status{{margin-top:10px;font-size:12px;color:#94a3b8;white-space:pre-wrap;font-family:monospace;max-height:280px;overflow:auto;background:#0b1222;border-radius:8px;padding:10px}}
.train-status.ok{{color:#86efac}}.train-status.err{{color:#fca5a5}}
.hint{{font-size:12px;color:#64748b;margin-top:6px}}
</style></head><body>
<div class="train-panel">
  <h2>🚀 เทรน Knowledge Base ใหม่ — ใส่เว็บไซต์แล้วกดปุ่ม (ระบบรัน pipeline เองทั้งหมด)</h2>
  <div class="train-row">
    <input id="trainUrl" placeholder="https://เว็บของคุณ.com" size="42">
    <input id="trainCode" placeholder="ชื่อ KB (เว้นว่าง = ตั้งจากโดเมนอัตโนมัติ)" size="32">
    <input id="trainPages" type="number" value="15" min="1" max="500" style="width:80px" title="จำนวนหน้าสูงสุด (เพดาน ไม่ใช่เป้าหมาย)">
    <span style="font-size:12px;color:#64748b">หน้าสูงสุด</span>
    <button id="trainBtn">เริ่มเทรน</button>
  </div>
  <div id="trainStatus" class="train-status">พร้อมเริ่มเทรน — ใส่ URL แล้วกด "เริ่มเทรน"
ระบบจะคลานเว็บ → AI ทำความสะอาด → AI ตัดย่อหน้า → ฝังเวกเตอร์ → ยัด Supabase ให้เอง
ใช้เวลา ~1-2 นาที/หน้าบนเครื่องนี้ (ปิดหน้านี้ได้ เทรนทำงานต่อในเซิร์ฟเวอร์)</div>
  <div class="hint">เพดานหน้า = หยุดก่อนถ้าหน้าจริงน้อยกว่า · คลานเฉพาะโดเมนเดียวกัน · เทรนได้ทีละ 1 KB</div>
</div>
<h2>📋 Query Logs ({len(rows)} รายการล่าสุด) — <a href="/" style="color:#7dd3fc">กลับหน้าแชท</a></h2>
<table><tr><th>เวลา</th><th>KB</th><th>คำถาม</th><th>เอกสารเจอ</th><th>node timings (วิ)</th><th>รวม</th><th>คำตอบ</th><th>แหล่งอ้างอิง</th></tr>{body}</table>
<p style="color:#94a3b8">เขียว = ค้นเจอเอกสาร (ตอบจากฐานความรู้) · แดง = ไม่เจอ (โมเดลตอบเดา ควรเทรนเพิ่ม)</p>
<script src="/static/logs.js"></script>
</body></html>"""
    return Response(content=html, media_type="text/html; charset=utf-8")


class BuildRequest(BaseModel):
    url: str
    company_code: str = ""
    max_pages: int = 15


_BUILD_JOB: dict = {
    "running": False, "url": "", "code": "", "max_pages": 0,
    "started_at": "", "finished_at": "", "returncode": None,
    "lines": [], "error": "",
}


def _run_build(url: str, code: str, max_pages: int) -> None:
    """รัน build_kb.py เป็นโปรเซสลูก แล้วดัน output ทีละบรรทัดเข้า _BUILD_JOB"""
    try:
        env = dict(os.environ)
        env["PYTHONIOENCODING"] = "utf-8"
        proc = subprocess.Popen(
            [sys.executable, str(ROOT / "scripts" / "build_kb.py"),
             url, code, "--max-pages", str(max_pages)],
            cwd=str(ROOT / "scripts"),
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            text=True, encoding="utf-8", errors="replace", env=env,
        )
        for line in proc.stdout:
            _BUILD_JOB["lines"].append(line.rstrip())
            if len(_BUILD_JOB["lines"]) > 500:
                _BUILD_JOB["lines"] = _BUILD_JOB["lines"][-400:]
        proc.wait()
        _BUILD_JOB["returncode"] = proc.returncode
        if proc.returncode != 0:
            _BUILD_JOB["error"] = f"build_kb exit {proc.returncode}"
    except Exception as ex:
        _BUILD_JOB["error"] = str(ex)
    finally:
        _BUILD_JOB["finished_at"] = time.strftime("%Y-%m-%dT%H:%M:%S")
        _BUILD_JOB["running"] = False


@app.post("/kb/build")
def start_build(req: BuildRequest):
    """เริ่มเทรน KB ใหม่จาก URL — รันเป็นพื้นหลัง กลับมาดูสถานะที่ /logs/view ได้ตลอด"""
    if _BUILD_JOB["running"]:
        raise HTTPException(status_code=409,
                            detail=f"มีการเทรนอยู่แล้ว (KB '{_BUILD_JOB['code']}') — รอจบก่อนหรือดูสถานะที่ /logs/view")
    url = req.url.strip()
    if not url.startswith(("http://", "https://")):
        raise HTTPException(status_code=400, detail="URL ต้องขึ้นต้นด้วย http:// หรือ https://")
    code = req.company_code.strip().lower().replace(" ", "_")
    if not code:
        host = urlparse(url).hostname or ""
        code = host.replace("www.", "").replace(".", "_").lower() or f"kb{time.strftime('%H%M%S')}"
    if not code.replace("_", "").isalnum():
        raise HTTPException(status_code=400, detail="ชื่อ KB ใช้ได้เฉพาะ a-z 0-9 _")
    if not (1 <= req.max_pages <= 500):
        raise HTTPException(status_code=400, detail="จำนวนหน้าต้องอยู่ระหว่าง 1-500")

    _BUILD_JOB.update({
        "running": True, "url": url, "code": code, "max_pages": req.max_pages,
        "started_at": time.strftime("%Y-%m-%dT%H:%M:%S"), "finished_at": "",
        "returncode": None, "lines": [], "error": "",
    })
    threading.Thread(target=_run_build, args=(url, code, req.max_pages), daemon=True).start()
    return {
        "ok": True, "company_code": code, "url": url, "max_pages": req.max_pages,
        "hint": "เทรนทำงานต่อในเซิร์ฟเวอร์แม้ปิดหน้านี้ — ดูความคืบหน้าที่ /logs/view",
    }


@app.get("/kb/build/status")
def build_status():
    return {**_BUILD_JOB, "lines": _BUILD_JOB["lines"][-40:]}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="127.0.0.1", port=8080, reload=False)
