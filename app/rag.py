"""
RAG retrieval pipeline with configurable feature flags.

Pipeline per request:
  1. [ENABLE_AI_INTENT]        detect intent (rule-based or AI)
  2.                           [always] intent=fact  → search company_facts  (company_code)
                               [always] intent=staff → search staff_contacts (company_code)
  3. [ENABLE_QUERY_REWRITE]    rewrite query with insurance synonyms
  4.                           embed query (bge-m3, NaN chunk-avg fallback)
  5.                           search documents + FAQ (company_code filter)
  6. [ENABLE_METADATA_SEARCH]  text search metadata_items
  7. [ENABLE_KNOWLEDGE_SEARCH] text search knowledge_items
  8. [ENABLE_RERANKER]         AI rerank retrieved docs
  9.                           build context string
  10.[ENABLE_GROUNDING]        verify answer is grounded  ← called from main.py

Note: each enabled AI step (3, 8, 10) adds ~10-30s latency.
Steps 2 (company_facts / staff_contacts) are always active — no LLM call needed.
"""
from __future__ import annotations

import json
import math
import os
import re
from typing import TYPE_CHECKING, Optional

import requests
from dotenv import load_dotenv

load_dotenv()

if TYPE_CHECKING:
    from app.providers.ollama_provider import OllamaProvider

# ---------------------------------------------------------------------------
# Feature flags (read once at import time)
# ---------------------------------------------------------------------------
_RAG_MODE              = os.getenv("RAG_MODE", "full").lower()
_ENABLE_AI_INTENT      = os.getenv("ENABLE_AI_INTENT", "false").lower() == "true"
_ENABLE_QUERY_REWRITE  = os.getenv("ENABLE_QUERY_REWRITE", "false").lower() == "true"
_ENABLE_RERANKER       = os.getenv("ENABLE_RERANKER", "false").lower() == "true"
_ENABLE_GROUNDING      = os.getenv("ENABLE_GROUNDING", "false").lower() == "true"
_ENABLE_METADATA       = os.getenv("ENABLE_METADATA_SEARCH", "false").lower() == "true"
_ENABLE_KNOWLEDGE      = os.getenv("ENABLE_KNOWLEDGE_SEARCH", "false").lower() == "true"
_ENABLE_RELATIONSHIP   = os.getenv("ENABLE_RELATIONSHIP_SEARCH", "false").lower() == "true"

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
_CHUNK_SIZE       = 150
_SIM_THRESHOLD    = 0.40
_MAX_CONTENT_CHARS = 600

_STAFF_KEYWORDS = ["ใครดูแล","ผู้ดูแล","เจ้าหน้าที่","contact person","ผู้ติดต่อ","ทีม","แผนก",
                   "manager","broker","consultant","ติดต่อใคร"]
_FACT_KEYWORDS  = ["ชื่อบริษัท","company","เบอร์","โทร","phone","tel","email","อีเมล",
                   "ที่อยู่","address","website","เว็บ","สาขา"]
_CLAIM_KEYWORDS = ["เคลม","claim","สินไหม","ชดเชย","แจ้งอุบัติเหตุ","รายงานความเสียหาย"]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _supabase_headers() -> dict:
    key = os.getenv("SUPABASE_KEY", "")
    return {"apikey": key, "Authorization": f"Bearer {key}", "Content-Type": "application/json"}


def _is_valid_vec(vec: list[float]) -> bool:
    return bool(vec) and all(math.isfinite(v) for v in vec)


def _extract_json_object(text: str) -> Optional[dict]:
    try:
        return json.loads(text)
    except Exception:
        pass
    m = re.search(r"\{.*\}", text or "", re.DOTALL)
    if m:
        try:
            return json.loads(m.group(0))
        except Exception:
            pass
    return None


def _extract_json_array(text: str) -> list:
    try:
        data = json.loads(text)
        if isinstance(data, list):
            return data
        if isinstance(data, dict) and "items" in data:
            return data["items"]
    except Exception:
        pass
    m = re.search(r"\[\s*\{.*?\}\s*\]", text or "", re.DOTALL)
    if m:
        try:
            return json.loads(m.group(0))
        except Exception:
            pass
    return []


# ---------------------------------------------------------------------------
# Step 1 — Intent detection
# ---------------------------------------------------------------------------
def _detect_intent_rule(query: str) -> str:
    q = query.lower()
    if any(k.lower() in q for k in _STAFF_KEYWORDS):
        return "staff"
    if any(k.lower() in q for k in _FACT_KEYWORDS):
        return "fact"
    if any(k.lower() in q for k in _CLAIM_KEYWORDS):
        return "claim"
    return "knowledge"


def _detect_intent(query: str, provider: "OllamaProvider") -> str:
    if not _ENABLE_AI_INTENT:
        return _detect_intent_rule(query)
    prompt = (
        "Classify intent for enterprise insurance chatbot.\n"
        'Return JSON only: {"intent":"fact/staff/faq/knowledge/claim/comparison","reason":"..."}\n\n'
        f"Question: {query}"
    )
    try:
        raw = provider.generate(prompt, options={"temperature": 0.0, "num_predict": 200})
        data = _extract_json_object(raw)
        return data.get("intent", "knowledge") if data else _detect_intent_rule(query)
    except Exception:
        return _detect_intent_rule(query)


# ---------------------------------------------------------------------------
# Step 2 — Query rewrite
# ---------------------------------------------------------------------------
def _rewrite_query(query: str, provider: "OllamaProvider") -> str:
    if not _ENABLE_QUERY_REWRITE:
        return query
    prompt = (
        "Rewrite this question for enterprise insurance RAG search.\n"
        "Add Thai/English insurance synonyms. Keep meaning. Output search query only.\n\n"
        f"Question: {query}\n\nSearch Query:"
    )
    try:
        result = provider.generate(prompt, options={"temperature": 0.0, "num_predict": 200})
        result = result.strip().strip('"').strip()
        return result if result else query
    except Exception:
        return query


# ---------------------------------------------------------------------------
# Step 3 — Embed
# ---------------------------------------------------------------------------
def _embed_query(text: str, provider: "OllamaProvider") -> Optional[list[float]]:
    """Embed with chunk-averaging fallback for bge-m3 NaN bug."""
    try:
        vec = provider.embed(text)
        if _is_valid_vec(vec):
            return vec
    except Exception:
        pass

    chunks = [text[i:i + _CHUNK_SIZE] for i in range(0, len(text), _CHUNK_SIZE)
              if text[i:i + _CHUNK_SIZE].strip()]
    valid_vecs: list[list[float]] = []
    for chunk in chunks:
        try:
            vec = provider.embed(chunk)
            if _is_valid_vec(vec):
                valid_vecs.append(vec)
        except Exception:
            continue

    if not valid_vecs:
        return None

    dim = len(valid_vecs[0])
    avg = [sum(v[i] for v in valid_vecs) / len(valid_vecs) for i in range(dim)]
    mag = math.sqrt(sum(x * x for x in avg)) or 1.0
    return [x / mag for x in avg]


# ---------------------------------------------------------------------------
# Step 4 — Vector search (documents + FAQ)
# ---------------------------------------------------------------------------
def _search_documents(vec: list[float], company_code: str, k: int = 6) -> list[dict]:
    url = os.getenv("SUPABASE_URL", "")
    rpc_filter = {"company_code": company_code} if company_code else {}
    try:
        r = requests.post(
            f"{url}/rest/v1/rpc/match_documents",
            headers=_supabase_headers(),
            json={"query_embedding": vec, "match_count": k, "filter": rpc_filter},
            timeout=15,
        )
        data = r.json()
        if isinstance(data, list):
            return [d for d in data if float(d.get("similarity", 0)) >= _SIM_THRESHOLD]
    except Exception:
        pass
    return []


def _search_faq(vec: list[float], company_code: str, k: int = 4) -> list[dict]:
    url = os.getenv("SUPABASE_URL", "")
    rpc_filter = {"company_code": company_code} if company_code else {}
    try:
        r = requests.post(
            f"{url}/rest/v1/rpc/match_faq",
            headers=_supabase_headers(),
            json={"query_embedding": vec, "match_count": k, "filter": rpc_filter},
            timeout=15,
        )
        data = r.json()
        if isinstance(data, list):
            return [d for d in data if float(d.get("similarity", 0)) >= _SIM_THRESHOLD]
    except Exception:
        pass
    return []


# ---------------------------------------------------------------------------
# Steps 2a-2b — Company facts & staff contacts (always active, by intent)
# ---------------------------------------------------------------------------

def _fact_type_from_query(query: str) -> Optional[str]:
    """Map query keywords → fact_type column value."""
    q = query.lower()
    if any(k in q for k in ["โทร", "phone", "tel", "เบอร์", "ติดต่อ"]):
        return "phone"
    if any(k in q for k in ["email", "อีเมล", "อีเมล์"]):
        return "email"
    if any(k in q for k in ["ที่อยู่", "address", "สำนักงาน", "ตั้งอยู่"]):
        return "address"
    if any(k in q for k in ["ชื่อบริษัท", "company name", "บริษัทชื่อ"]):
        return "company_name"
    if any(k in q for k in ["เว็บ", "website", "เว็บไซต์"]):
        return "website"
    return None  # return all facts


def _pg_query(sql: str) -> list[dict]:
    """Execute raw SQL via Supabase pg/query endpoint (bypasses RLS)."""
    url = os.getenv("SUPABASE_URL", "")
    try:
        r = requests.post(f"{url}/pg/query", headers=_supabase_headers(),
                          json={"query": sql}, timeout=10)
        data = r.json()
        return data if isinstance(data, list) else []
    except Exception:
        return []


def _safe(value: str) -> str:
    """Escape single quotes for SQL string literals."""
    return (value or "").replace("'", "''")


def _search_company_facts(query: str, company_code: str) -> list[dict]:
    """Search company_facts via pg/query (REST is blocked by RLS)."""
    fact_type = _fact_type_from_query(query)
    code_clause = f"AND company_code = '{_safe(company_code)}'" if company_code else ""
    type_clause = f"AND fact_type = '{_safe(fact_type)}'" if fact_type else ""
    sql = f"""
        SELECT fact_type, fact_value, language, source_url
        FROM company_facts
        WHERE 1=1 {code_clause} {type_clause}
        ORDER BY confidence DESC
        LIMIT 20;
    """
    results = _pg_query(sql)
    # Fallback 1: specific fact_type not found — return ALL facts for this company so the
    # LLM can construct a helpful answer (e.g., redirect to website when address is missing)
    if not results and fact_type and company_code:
        sql_no_type = f"""
            SELECT fact_type, fact_value, language, source_url
            FROM company_facts
            WHERE 1=1 {code_clause}
            ORDER BY confidence DESC
            LIMIT 20;
        """
        results = _pg_query(sql_no_type)
    # Fallback 2: drop company_code filter if still empty
    if not results and company_code:
        sql_fallback = f"""
            SELECT fact_type, fact_value, language, source_url
            FROM company_facts
            WHERE 1=1 {type_clause}
            ORDER BY confidence DESC
            LIMIT 20;
        """
        results = _pg_query(sql_fallback)
    return results


def _search_staff_contacts(query: str, company_code: str) -> list[dict]:
    """Search staff_contacts via pg/query (REST is blocked by RLS)."""
    code_clause = f"AND company_code = '{_safe(company_code)}'" if company_code else ""
    sql = f"""
        SELECT name, position, department, email, phone, source_url
        FROM staff_contacts
        WHERE 1=1 {code_clause}
        ORDER BY confidence DESC
        LIMIT 20;
    """
    results = _pg_query(sql)
    if not results and company_code:
        results = _pg_query("SELECT name,position,department,email,phone,source_url FROM staff_contacts ORDER BY confidence DESC LIMIT 20;")
    return results


# ---------------------------------------------------------------------------
# Steps 5-6 — Optional supplementary text searches
# ---------------------------------------------------------------------------
def _search_metadata(query: str, company_code: str) -> list[dict]:
    if not _ENABLE_METADATA:
        return []
    kw = _safe(query[:80])
    return _pg_query(
        f"SELECT title,heading,language,category,service,keywords,intent_tags "
        f"FROM metadata_items WHERE title ILIKE '%{kw}%' OR heading ILIKE '%{kw}%' LIMIT 5;"
    )


def _search_knowledge(query: str, company_code: str) -> list[dict]:
    if not _ENABLE_KNOWLEDGE:
        return []
    kw = _safe(query[:80])
    return _pg_query(
        f"SELECT title,heading,question,answer,knowledge_type,language FROM knowledge_items "
        f"WHERE question ILIKE '%{kw}%' OR answer ILIKE '%{kw}%' OR title ILIKE '%{kw}%' LIMIT 5;"
    )


def _search_relationships(query: str) -> list[dict]:
    if not _ENABLE_RELATIONSHIP:
        return []
    kw = _safe(query[:80])
    return _pg_query(
        f"SELECT subject,predicate,object,subject_type,object_type FROM relationships "
        f"WHERE subject ILIKE '%{kw}%' OR object ILIKE '%{kw}%' LIMIT 5;"
    )


# ---------------------------------------------------------------------------
# Step 7 — AI Reranker
# ---------------------------------------------------------------------------
def _rerank_docs(query: str, docs: list[dict], provider: "OllamaProvider") -> list[dict]:
    if not _ENABLE_RERANKER or len(docs) <= 1:
        return docs
    candidates = [
        {"index": i + 1, "title": d.get("title", ""), "heading": d.get("heading", ""),
         "content": (d.get("content") or "")[:600]}
        for i, d in enumerate(docs)
    ]
    prompt = (
        "Rerank these documents for answering the question. Return JSON array only:\n"
        '[{"index":1,"score":0.95}]\n\n'
        f"Question: {query}\n\n"
        f"Candidates:\n{json.dumps(candidates, ensure_ascii=False)}"
    )
    try:
        raw = provider.generate(prompt, options={"temperature": 0.0, "num_predict": 500})
        ranks = _extract_json_array(raw)
        ordered: list[dict] = []
        seen: set[int] = set()
        for r in sorted(ranks, key=lambda x: float(x.get("score", 0)), reverse=True):
            idx = int(r.get("index", 0)) - 1
            if 0 <= idx < len(docs) and idx not in seen:
                seen.add(idx)
                docs[idx]["_rerank_score"] = float(r.get("score", 0))
                ordered.append(docs[idx])
        # append any docs that didn't appear in reranker output
        for i, d in enumerate(docs):
            if i not in seen:
                ordered.append(d)
        return ordered
    except Exception:
        return docs


# ---------------------------------------------------------------------------
# Step 8 — Build context string
# ---------------------------------------------------------------------------
def _build_context(
    docs: list[dict],
    faqs: list[dict],
    facts: list[dict],
    staff: list[dict],
    knowledge: list[dict],
    relationships: list[dict],
) -> str:
    parts: list[str] = []

    if facts:
        parts.append("## ข้อมูลบริษัท")
        for f in facts:
            ft = (f.get("fact_type") or "").strip()
            fv = (f.get("fact_value") or "").strip()
            if ft and fv:
                parts.append(f"{ft}: {fv}")

    if staff:
        parts.append("## ผู้ติดต่อ / เจ้าหน้าที่")
        for s in staff:
            line = " | ".join(filter(None, [
                s.get("name"), s.get("position"), s.get("department"),
                s.get("email"), s.get("phone"),
            ]))
            if line:
                parts.append(line)

    if faqs:
        parts.append("## คำถามที่พบบ่อย")
        for f in faqs:
            q = (f.get("question") or "").strip()
            a = (f.get("answer") or "").strip()
            if q and a:
                parts.append(f"Q: {q}\nA: {a}")

    if docs:
        parts.append("## ข้อมูลอ้างอิง")
        seen: set[str] = set()
        for d in docs:
            title = d.get("title", "")
            heading = d.get("heading", "")
            content = (d.get("content") or "").strip()
            label = f"{title} — {heading}" if heading else title
            if label in seen or not content:
                continue
            seen.add(label)
            excerpt = content[:_MAX_CONTENT_CHARS]
            if len(content) > _MAX_CONTENT_CHARS:
                excerpt += "…"
            parts.append(f"[{label}]\n{excerpt}")

    if knowledge:
        parts.append("## ความรู้เพิ่มเติม")
        for k in knowledge:
            q = (k.get("question") or "").strip()
            a = (k.get("answer") or "").strip()
            if q and a:
                parts.append(f"Q: {q}\nA: {a}")

    if relationships:
        parts.append("## ความสัมพันธ์")
        for r in relationships:
            parts.append(f"{r.get('subject')} → {r.get('predicate')} → {r.get('object')}")

    return "\n\n".join(parts)


# ---------------------------------------------------------------------------
# Step 9 — Grounding verification (exported, called from main.py)
# ---------------------------------------------------------------------------
def verify_grounding(
    question: str,
    answer: str,
    context: str,
    provider: "OllamaProvider",
) -> str:
    """Return verified (possibly corrected) answer. Call after LLM generates answer."""
    if not _ENABLE_GROUNDING or not context:
        return answer
    prompt = (
        "Verify whether the answer is grounded in the context below.\n"
        'Return JSON only: {"grounded":true,"score":0.9,"fixed_answer":"..."}\n\n'
        f"Question: {question}\nAnswer: {answer}\nContext: {context[:4000]}"
    )
    try:
        raw = provider.generate(prompt, options={"temperature": 0.0, "num_predict": 800})
        data = _extract_json_object(raw)
        if data and not data.get("grounded", True):
            fixed = (data.get("fixed_answer") or "").strip()
            if fixed:
                return fixed
    except Exception:
        pass
    return answer


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------
def get_active_flags() -> dict:
    return {
        "rag_mode": _RAG_MODE,
        "ai_intent": _ENABLE_AI_INTENT,
        "query_rewrite": _ENABLE_QUERY_REWRITE,
        "reranker": _ENABLE_RERANKER,
        "grounding": _ENABLE_GROUNDING,
        "metadata_search": _ENABLE_METADATA,
        "knowledge_search": _ENABLE_KNOWLEDGE,
        "relationship_search": _ENABLE_RELATIONSHIP,
    }


def retrieve(
    query: str,
    provider: "OllamaProvider",
    company_code: str = "",
    progress=None,
) -> tuple[str, list[dict], list[dict]]:
    """
    Run the full RAG pipeline.
    Returns (context_text, docs, faqs).
    company_code is resolved from arg → env var COMPANY_CODE → "" (no filter).
    progress: optional callback(step, label) called at each pipeline node.
    """
    def _emit(step: str, label: str) -> None:
        if progress:
            try:
                progress(step, label)
            except Exception:
                pass

    resolved_code = company_code or os.getenv("COMPANY_CODE", "")

    # 1. Intent detection (rule-based by default; AI if ENABLE_AI_INTENT=true)
    _emit("intent", "โหนดจับใจความ: วิเคราะห์ประเภทคำถาม")
    intent = _detect_intent(query, provider)

    # 2. Company facts / staff contacts — always active, no LLM call
    facts: list[dict] = []
    staff: list[dict] = []
    if intent in ("fact", "claim", "knowledge"):
        _emit("facts", "โหนดข้อมูลบริษัท: ค้นหาข้อเท็จจริงของบริษัท")
        facts = _search_company_facts(query, resolved_code)
    if intent == "staff":
        _emit("facts", "โหนดข้อมูลบริษัท: ค้นหาผู้ติดต่อ")
        staff = _search_staff_contacts(query, resolved_code)

    # 3. Query rewrite (optional, adds latency)
    search_query = _rewrite_query(query, provider)

    # 4. Embed
    _emit("embed", f"โหนดฝังเวกเตอร์: Ollama ({provider.config.embed_model}) กำลังแปลงคำถาม")
    vec = _embed_query(search_query, provider)
    if not vec:
        context = _build_context([], [], facts, staff, [], [])
        return context, [], []

    # 5. Vector search
    _emit("search", "โหนดค้นหาฐานความรู้: Supabase pgvector กำลังค้นแบบเวกเตอร์")
    docs = _search_documents(vec, resolved_code)
    faqs = _search_faq(vec, resolved_code)

    # 6. Supplementary text searches (optional)
    knowledge = _search_knowledge(query, resolved_code)
    relationships = _search_relationships(query)
    meta_docs = _search_metadata(query, resolved_code)

    # 7. Rerank docs (optional, adds latency)
    docs = _rerank_docs(query, docs + meta_docs, provider)

    _emit("found", f"โหนดค้นหาฐานความรู้: พบเอกสาร {len(docs)} ชิ้น · FAQ {len(faqs)} รายการ")

    # 8. Build context
    context = _build_context(docs, faqs, facts, staff, knowledge, relationships)
    return context, docs, faqs
