from bs4 import BeautifulSoup
from _common import CRAWL_DIR, clean_text, write_jsonl, read_jsonl, detect_language_rule
import os, re, json

INPUT = f"{CRAWL_DIR}/raw_pages.jsonl"
OUTPUT = f"{CRAWL_DIR}/all_sections.jsonl"

REMOVE = "nav,header,footer,aside,script,style,noscript,form,iframe,svg,.menu,.navbar,.sidebar,.breadcrumb,.cookies,.cookie,.ads,.social,.comments,.newsletter,#menu,#navbar,#sidebar,#footer,#header"

def remove_noise(soup):
    for tag in soup.select(REMOVE):
        tag.decompose()
    return soup

def content_root(soup):
    for sel in ["article",".entry-content",".post-content",".wp-block-post-content",".elementor-widget-theme-post-content",".elementor-location-single","main","#content","#main",".content"]:
        found = soup.select_one(sel)
        if found and len(clean_text(found.get_text(" ", strip=True))) > 80:
            return found
    return soup.body or soup

def extract_sections(row):
    soup = BeautifulSoup(row.get("html",""), "html.parser")
    remove_noise(soup)
    root = content_root(soup)
    title = clean_text(row.get("title",""))
    url = row.get("url","")
    sections = []
    current = {"heading": title, "content": []}
    for el in root.find_all(["h1","h2","h3","h4","h5","h6","p","li","blockquote","table"]):
        tag = el.name.lower()
        txt = clean_text(el.get_text(" ", strip=True))
        if not txt: continue
        if tag.startswith("h"):
            if current["content"]:
                content = "\n\n".join(current["content"])
                sections.append({"url":url,"title":title,"heading":current["heading"],"content":content,"language":detect_language_rule(content)})
            current = {"heading":txt, "content":[]}
        elif tag == "li":
            current["content"].append("- " + txt)
        else:
            current["content"].append(txt)
    if current["content"]:
        content = "\n\n".join(current["content"])
        sections.append({"url":url,"title":title,"heading":current["heading"],"content":content,"language":detect_language_rule(content)})
    # fallback: หน้าที่เขียนด้วย div ธรรมดา (ไม่มี p/h* ให้วน) — เก็บทั้งหน้าเป็น 1 หัวข้อ
    if not sections:
        whole = clean_text(root.get_text(" ", strip=True))
        if len(whole) >= 50:
            sections.append({"url":url,"title":title,"heading":title or "(main content)","content":whole,"language":detect_language_rule(whole)})
    return [s for s in sections if len(clean_text(s["content"])) >= 50]

def main():
    rows = read_jsonl(INPUT)
    out = []
    for i, r in enumerate(rows, 1):
        out.extend(extract_sections(r))
        print(f"[{i}/{len(rows)}] sections={len(out)}")
    write_jsonl(OUTPUT, out)
    print("DONE", len(out), OUTPUT)

if __name__ == "__main__":
    main()
