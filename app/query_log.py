"""
บันทึกประวัติคำถาม-คำตอบทุกครั้ง เพื่อเอาไว้วิเคราะห์ว่า:
- คำถามไหนตอบได้/ตอบไม่ได้
- ค้นหาเจอเอกสารกี่ชิ้น (ความเกี่ยวข้องเท่าไร)
- แต่ละ node ใช้เวลากี่วินาที (จุดไหนช้า)

เก็บ 2 ที่:
1. logs/query_logs.jsonl  (เสมอ — ไฟล์ local อ่านง่าย)
2. Supabase ตาราง query_logs (ถ้าสร้างตารางไว้แล้ว — ถ้ายังไม่มีตารางจะข้ามเงียบ ๆ)
"""
import json
import os
import threading
import time
from pathlib import Path

import requests

BASE_DIR = Path(__file__).resolve().parent.parent
LOG_FILE = BASE_DIR / "logs" / "query_logs.jsonl"


def log_query(
    company_code: str,
    question: str,
    answer: str,
    sources: list | None = None,
    docs_found: int = 0,
    faqs_found: int = 0,
    node_timings: dict | None = None,
    status: str = "ok",
    error_detail: str = "",
    duration_ms: int = 0,
) -> None:
    """บันทึก 1 รายการ — ไม่เด้ง error กลับไปที่ผู้ใช้เด็ดขาด (log ล้มเหลว = ข้าม)"""
    record = {
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "company_code": company_code,
        "question": question,
        "answer": (answer or "")[:4000],
        "sources": sources or [],
        "docs_found": docs_found,
        "faqs_found": faqs_found,
        "node_timings": node_timings or {},
        "status": status,
        "error_detail": (error_detail or "")[:500],
        "duration_ms": duration_ms,
    }
    threading.Thread(target=_write_local, args=(record,), daemon=True).start()
    threading.Thread(target=_write_supabase, args=(record,), daemon=True).start()


def _write_local(record: dict) -> None:
    try:
        LOG_FILE.parent.mkdir(parents=True, exist_ok=True)
        with open(LOG_FILE, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
    except Exception:
        pass


def _write_supabase(record: dict) -> None:
    url = os.getenv("SUPABASE_URL", "")
    key = os.getenv("SUPABASE_KEY", "")
    if not url or not key:
        return
    try:
        requests.post(
            f"{url}/rest/v1/query_logs",
            headers={"apikey": key, "Authorization": f"Bearer {key}", "Content-Type": "application/json",
                     "Prefer": "return=minimal"},
            json=record,
            timeout=10,
        )
        # ถ้ายังไม่ได้สร้างตาราง query_logs จะได้ 404 — เงียบ ๆ ได้ (มีไฟล์ local อยู่แล้ว)
    except Exception:
        pass


def read_recent(limit: int = 50) -> list:
    """อ่าน log ล่าสุด N รายการ (จากไฟล์ local) เก่าสุดก่อน"""
    try:
        if not LOG_FILE.exists():
            return []
        lines = LOG_FILE.read_text(encoding="utf-8").strip().splitlines()
        rows = []
        for line in lines[-limit:]:
            try:
                rows.append(json.loads(line))
            except Exception:
                continue
        return rows
    except Exception:
        return []
