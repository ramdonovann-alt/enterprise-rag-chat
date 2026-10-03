from _common import CRAWL_DIR, clean_text, read_jsonl, write_jsonl, call_ollama, extract_json_object, detect_language_rule, detect_category_rule, md5_id
import re, json, time, os

INPUT = f"{CRAWL_DIR}/clean_sections.jsonl"
OUTPUT = f"{CRAWL_DIR}/chunks.jsonl"
REPORT = f"{CRAWL_DIR}/chunk_report.json"
MAX_CHUNK = 1200
OVERLAP = 180

def ai_metadata(row):
    content = clean_text(row.get("content",""))[:3500]
    prompt = f"""
วิเคราะห์ section สำหรับ Enterprise RAG แล้วตอบ JSON เท่านั้น:
{{
 "clean_content":"...",
 "language":"th/en",
 "category":"motor_insurance/marine_insurance/health_insurance/property_insurance/liability_insurance/employee_benefits/claims/contact/about/general",
 "service":"motor/marine/health/property/liability/claims/general",
 "topic":"...",
 "page_type":"knowledge/contact/about/news/service/faq/general",
 "keywords":["..."],
 "suggested_questions":["..."]
}}

ห้ามแต่งข้อมูลใหม่ ห้ามแปลเนื้อหาหลัก

Title: {row.get('title')}
Heading: {row.get('heading')}
Content:
{content}
"""
    try:
        return extract_json_object(call_ollama(prompt, num_predict=2200)) or {}
    except Exception:
        return {}

def split_semantic(text):
    text = clean_text(text)
    if len(text) <= MAX_CHUNK: return [text] if text else []
    parts = re.split(r"\n\s*\n|(?<=\. )|(?<=\? )|(?<=! )", text)
    chunks, cur = [], ""
    for p in [clean_text(x) for x in parts if clean_text(x)]:
        if len(cur) + len(p) + 2 <= MAX_CHUNK:
            cur = (cur + "\n\n" + p).strip()
        else:
            if cur: chunks.append(cur)
            cur = (cur[-OVERLAP:] + "\n\n" + p).strip() if len(cur) > OVERLAP else p
    if cur: chunks.append(cur)
    return [c for c in chunks if len(c) >= 80]

def build_chunks(row, meta):
    title, heading, url = row.get("title",""), row.get("heading",""), row.get("url","")
    content = clean_text(meta.get("clean_content") or row.get("content",""))
    lang = meta.get("language") or row.get("language") or detect_language_rule(content)
    cat = meta.get("category") or row.get("category") or detect_category_rule(title, heading, content)
    service = meta.get("service") or cat
    topic = meta.get("topic") or heading or title
    page_type = meta.get("page_type") or row.get("page_type") or "general"
    keywords = meta.get("keywords") if isinstance(meta.get("keywords"), list) else []
    suggested = meta.get("suggested_questions") if isinstance(meta.get("suggested_questions"), list) else []
    out = []
    for idx, chunk in enumerate(split_semantic(content), 1):
        text_for_embedding = f"Title: {title}\nHeading: {heading}\nTopic: {topic}\nCategory: {cat}\nService: {service}\nLanguage: {lang}\nKeywords: {', '.join(keywords)}\nURL: {url}\n\nContent:\n{chunk}"
        out.append({
            "id": md5_id(os.getenv("COMPANY_CODE", ""), url, heading, idx, chunk),  # ใส่ company_code กัน KB ต่างเว็บเดียวกันชนกัน
            "url": url, "title": title, "heading": heading,
            "heading_path": [x for x in [title, heading] if x],
            "category": cat, "service": service, "topic": topic, "page_type": page_type,
            "language": lang, "chunk_index": idx, "chunk_size": len(chunk),
            "keywords": keywords, "suggested_questions": suggested,
            "content": chunk, "text_for_embedding": text_for_embedding
        })
    return out

def main():
    rows = read_jsonl(INPUT)
    out, ok = [], 0
    for i,r in enumerate(rows,1):
        print(f"[{i}/{len(rows)}] {r.get('heading')}")
        meta = ai_metadata(r)
        if meta: ok += 1
        out.extend(build_chunks(r, meta))
        time.sleep(0.2)
    write_jsonl(OUTPUT, out)
    report = {"input":INPUT,"output":OUTPUT,"sections":len(rows),"chunks":len(out),"ai_metadata_ok":ok}
    open(REPORT,"w",encoding="utf-8").write(json.dumps(report,ensure_ascii=False,indent=2))
    print("DONE", report)

if __name__ == "__main__":
    main()
