"""
สร้าง Knowledge Base ใหม่จากลิงก์เว็บไซต์ — คำสั่งเดียวจบ
รัน pipeline 01→05 อัตโนมัติ: คลานเว็บ → แตกหัวข้อ → AI ทำความสะอาด
→ AI ตัดย่อหน้า → ฝังเวกเตอร์ → ยัดลง Supabase

ใช้งาน:
  python build_kb.py https://เว็บคุณ.com ชื่อ_kb
  python build_kb.py https://เว็บคุณ.com ชื่อ_kb --max-pages 30
  python build_kb.py https://เว็บคุณ.com ชื่อ_kb --full   # + facts/FAQ/knowledge (นานมาก)

หลังรันเสร็จ หน้าเว็บแชทจะเลือก KB ใหม่ได้ทันที (แถบซ้าย) — รีเฟรชหน้าก่อน 1 ครั้ง
"""
import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path

import requests
from dotenv import load_dotenv

BASE_DIR = Path(__file__).resolve().parent.parent
load_dotenv(BASE_DIR / ".env")

CORE_STEPS = [
    "01_crawl_website.py",
    "02_extract_sections.py",
    "03_clean_sections_ai.py",
    "04_semantic_chunking_ai.py",
    "05_embed_documents.py",
]
FULL_EXTRA_STEPS = [
    "06_extract_facts.py",
    "07_import_facts.py",
    "08_generate_faq.py",
    "09_embed_faq.py",
    "10_extract_metadata.py",
    "11_extract_knowledge.py",
    "12_build_relationships.py",
]


def run_step(script: str, env: dict) -> None:
    print(f"\n{'=' * 60}\n===== {script}\n{'=' * 60}", flush=True)
    result = subprocess.run(
        [sys.executable, str(Path(__file__).parent / script)],
        env=env,
        cwd=str(Path(__file__).parent),
    )
    if result.returncode != 0:
        sys.exit(f"[ABORT] {script} ล้มเหลว (exit {result.returncode}) — แก้แล้วรันใหม่ได้")


def supabase_count(table: str, company_code: str) -> int:
    url = os.getenv("SUPABASE_URL", "")
    key = os.getenv("SUPABASE_KEY", "")
    try:
        r = requests.get(
            f"{url}/rest/v1/{table}?company_code=eq.{company_code}&select=id&limit=1",
            headers={"apikey": key, "Authorization": f"Bearer {key}", "Prefer": "count=exact"},
            timeout=15,
        )
        return int(r.headers.get("content-range", "0/0").split("/")[-1])
    except Exception:
        return -1


def main() -> None:
    parser = argparse.ArgumentParser(description="สร้าง Knowledge Base จากลิงก์เว็บไซต์ (คำสั่งเดียว)")
    parser.add_argument("start_url", help="URL เว็บไซต์ที่จะให้บอทเรียนรู้ เช่น https://example.com")
    parser.add_argument("company_code", help="ชื่อ KB (a-z, 0-9, _ เช่น my_shop) — ใช้แยกแต่ละเรื่อง")
    parser.add_argument("--max-pages", type=int, default=15, help="จำนวนหน้าสูงสุดที่คลาน (default 15)")
    parser.add_argument("--full", action="store_true",
                        help="รันต่อ 06-12: ข้อเท็จจริงบริษัท, FAQ อัตโนมัติ, metadata, knowledge (AI เยอะมาก ใช้เวลานาน)")
    args = parser.parse_args()

    if not args.start_url.startswith("http"):
        sys.exit("start_url ต้องขึ้นต้นด้วย http:// หรือ https://")
    code = args.company_code.strip().lower().replace(" ", "_")
    if not code.replace("_", "").isalnum():
        sys.exit("company_code ใช้ได้เฉพาะ a-z 0-9 _ เท่านั้น")

    crawl_dir = str((BASE_DIR.parent / "crawl-output" / code).resolve())
    try:
        if Path(crawl_dir).exists():
            shutil.rmtree(crawl_dir)  # เคลียร์ของเก่าถ้าเคยสร้าง KB นี้ไว้ ให้เริ่มใหม่สะอาด ๆ
    except PermissionError:
        print("⚠ ลบโฟลเดอร์เก่าไม่ได้ (มีโปรแกรมอื่นเปิดค้างอยู่) — เดินต่อด้วยการเขียนทับไฟล์เดิม")

    env = dict(os.environ)
    env.update({
        "START_URL": args.start_url,
        "COMPANY_CODE": code,
        "CRAWL_DIR": crawl_dir,
        "MAX_PAGES": str(args.max_pages),
    })

    steps = CORE_STEPS + (FULL_EXTRA_STEPS if args.full else [])
    print(f"สร้าง KB '{code}' จาก {args.start_url}")
    print(f"จำนวนหน้าสูงสุด: {args.max_pages} | ขั้นตอน: {len(steps)} สคริปต์ | บันทึกที่: {crawl_dir}")

    for script in steps:
        run_step(script, env)

    docs = supabase_count("documents", code)
    faqs = supabase_count("faq_items", code)
    print(f"\n{'=' * 60}\nเสร็จสิ้น! KB '{code}'")
    print(f"  เอกสารใน Supabase: {docs} chunks")
    print(f"  FAQ ใน Supabase:   {faqs} รายการ")
    print(f"  เปิดหน้าเว็บแชท http://127.0.0.1:8080 แล้วรีเฟรช จะเห็น KB ใหม่ในแถบซ้าย")
    if docs == 0:
        print("  ⚠ ไม่มีเอกสารถูกบันทึก — เช็กว่าเว็บต้นทางเข้าถึงได้และมีเนื้อหาข้อความ")


if __name__ == "__main__":
    main()
