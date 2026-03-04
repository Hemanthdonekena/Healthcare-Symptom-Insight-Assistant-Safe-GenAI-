import re
import io
import csv
import time
import requests
from typing import List, Dict, Optional, Set
from lxml import etree as ET
from bs4 import BeautifulSoup

MEDLINEPLUS_WS = "https://wsearch.nlm.nih.gov/ws/query"
HEADERS = {"User-Agent": "capstone_system_assistant/1.0"}

# -----------------------------
# MedlinePlus search (XML)
# -----------------------------
def strip_html(text: str) -> str:
    if not text:
        return ""
    return re.sub(r"<[^>]+>", "", text).strip()

def search_medlineplus_topics(term: str, top_k: int = 5, rettype: str = "brief") -> List[Dict]:
    params = {
        "db": "healthTopics",
        "term": term,
        "retmax": str(top_k),
        "rettype": rettype,
        "tool": "capstone_system_assistant",
    }
    r = requests.get(MEDLINEPLUS_WS, params=params, headers=HEADERS, timeout=30)
    r.raise_for_status()

    tree = ET.parse(io.BytesIO(r.content))
    root = tree.getroot()

    documents = root.xpath(".//*[local-name()='document']")
    results: List[Dict] = []

    for doc in documents:
        url = doc.get("url")
        title = snippet = None

        for c in doc.xpath("./*[local-name()='content']"):
            name = (c.get("name") or "").lower()
            text = "".join(c.itertext()).strip()
            if name == "title":
                title = strip_html(text)
            elif name == "snippet":
                snippet = strip_html(text)

        results.append({"url": url, "title": title, "snippet": snippet})

    return results

# -----------------------------
# Fetch + clean (HTML)
# -----------------------------
def fetch_page_soup(url: str) -> BeautifulSoup:
    r = requests.get(url, headers=HEADERS, timeout=30)
    r.raise_for_status()
    return BeautifulSoup(r.text, "lxml")

def clean_text(text: str) -> str:
    text = re.sub(r"\s+", " ", text)
    return text.strip()

def extract_meta_description(soup: BeautifulSoup) -> Optional[str]:
    tag = soup.find("meta", attrs={"name": "description"})
    if tag and tag.get("content"):
        return clean_text(tag["content"])
    return None

def extract_section_by_heading_keywords(soup: BeautifulSoup, keywords: List[str]) -> Optional[str]:
    keywords = [k.lower() for k in keywords]
    headings = soup.find_all(["h2", "h3"])

    target = None
    for h in headings:
        t = clean_text(h.get_text()).lower()
        if any(k in t for k in keywords):
            target = h
            break

    if not target:
        return None

    chunks = []
    for sib in target.find_all_next():
        if sib.name in {"h2", "h3"} and sib is not target:
            break
        if sib.name in {"p", "li"}:
            txt = clean_text(sib.get_text(" ", strip=True))
            if txt and len(txt) > 20:
                chunks.append(txt)

    return "\n".join(chunks).strip() if chunks else None

def extract_symptoms_bullets(soup: BeautifulSoup) -> List[str]:
    symptoms_heading = None
    for h in soup.find_all(["h2", "h3"]):
        t = clean_text(h.get_text()).lower()
        if "symptom" in t:
            symptoms_heading = h
            break
    if not symptoms_heading:
        return []

    ul = symptoms_heading.find_next(["ul", "ol"])
    if not ul:
        return []

    bullets = []
    for li in ul.find_all("li"):
        txt = clean_text(li.get_text(" ", strip=True))
        if txt:
            bullets.append(txt)
    return bullets

def symptoms_from_snippet(snippet: str, max_items: int = 8) -> List[str]:
    if not snippet:
        return []
    snippet = re.sub(r"<[^>]+>", "", snippet)  # remove <span> etc.
    snippet = re.sub(r"\s+", " ", snippet).strip()

    m = re.search(r"include[s]?:\s*(.+)", snippet, flags=re.IGNORECASE)
    text = m.group(1) if m else snippet

    parts = re.split(r",|;|\band\b|\bor\b", text)
    cleaned = []
    for p in parts:
        p = p.strip(" .:-")
        if 3 <= len(p) <= 40:
            cleaned.append(p.lower())
        if len(cleaned) >= max_items:
            break
    return cleaned

def symptoms_from_overview(overview: str, max_items: int = 8) -> List[str]:
    if not overview:
        return []
    overview_l = overview.lower()
    m = re.search(r"(symptoms include|signs and symptoms include|symptoms are)\s*(.*)", overview_l)
    if not m:
        return []
    tail = m.group(2)
    parts = re.split(r",|;|\band\b|\bor\b|\.", tail)
    cleaned = []
    for p in parts:
        p = p.strip(" :-")
        if 3 <= len(p) <= 40:
            cleaned.append(p)
        if len(cleaned) >= max_items:
            break
    return cleaned

def extract_when_to_seek_care(soup: BeautifulSoup) -> str:
    keywords = [
        "when to call", "when to see", "call your doctor", "call a doctor",
        "seek medical", "get medical help", "emergency", "when to get help",
        "when should i call"
    ]
    text = extract_section_by_heading_keywords(soup, keywords)
    if not text:
        return ("Seek medical care if symptoms are severe, worsening, or if you have "
                "emergency warning signs such as trouble breathing, chest pain, confusion, "
                "or persistent high fever.")
    return text

def extract_topic_record(url: str, title: str, snippet: str, collected_from_query: str) -> Dict:
    soup = fetch_page_soup(url)

    overview = (
        extract_section_by_heading_keywords(soup, ["summary", "overview"])
        or extract_meta_description(soup)
        or ""
    )

    # symptoms: bullets -> snippet -> overview fallback
    symptoms_list = extract_symptoms_bullets(soup)
    if not symptoms_list:
        symptoms_list = symptoms_from_snippet(snippet)
    if not symptoms_list:
        symptoms_list = symptoms_from_overview(overview)

    seek_care = extract_when_to_seek_care(soup)

    return {
        "condition": title or "",
        "overview": overview,
        "common_symptoms": "; ".join(symptoms_list),
        "when_to_seek_care": seek_care,
        "source_url": url,
        "collected_from_query": collected_from_query
    }

# -----------------------------
# CSV helpers
# -----------------------------
FIELDNAMES = [
    "condition",
    "overview",
    "common_symptoms",
    "when_to_seek_care",
    "source_url",
    "collected_from_query"
]

def load_existing_urls(csv_path: str) -> Set[str]:
    urls = set()
    try:
        with open(csv_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            if reader.fieldnames and "source_url" in reader.fieldnames:
                for r in reader:
                    u = (r.get("source_url") or "").strip()
                    if u:
                        urls.add(u)
    except FileNotFoundError:
        pass
    return urls

def write_records(csv_path: str, records: List[Dict], append: bool) -> None:
    mode = "a" if append else "w"
    with open(csv_path, mode, newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        if not append:
            writer.writeheader()
        for r in records:
            writer.writerow(r)

# -----------------------------
# Main: multi-query collection
# -----------------------------
if __name__ == "__main__":
    # ✅ Edit this list to match your capstone scope (start with 6–10 queries)
    QUERIES = [
        "fever nausea",
        "cough cold",
        "sore throat fever",
        "headache nausea",
        "diarrhea stomach cramps",
        "rash fever",
        "shortness of breath wheezing",
        "dizziness fainting",
        "acne eczema"
    ]

    TOP_K_PER_QUERY = 5
    OUTPUT_CSV = "medical_kb_combined.csv"

    # If file exists, we append and dedupe; otherwise create new
    existing_urls = load_existing_urls(OUTPUT_CSV)
    append_mode = len(existing_urls) > 0

    print(f"Output file: {OUTPUT_CSV}")
    print(f"Existing URLs loaded: {len(existing_urls)} (append_mode={append_mode})")

    new_records: List[Dict] = []
    total_topics_seen = 0
    total_pages_fetched = 0
    total_skipped_dupe = 0

    for q in QUERIES:
        print(f"\n=== Query: {q} ===")
        try:
            topics = search_medlineplus_topics(q, top_k=TOP_K_PER_QUERY, rettype="brief")
        except Exception as e:
            print(f"  ❌ Search failed for query '{q}': {e}")
            continue

        print(f"  Topics returned: {len(topics)}")
        total_topics_seen += len(topics)

        for t in topics:
            url = (t.get("url") or "").strip()
            title = (t.get("title") or "").strip()
            snippet = t.get("snippet") or ""

            if not url:
                continue

            if url in existing_urls:
                total_skipped_dupe += 1
                continue

            try:
                rec = extract_topic_record(url, title, snippet, collected_from_query=q)
                new_records.append(rec)
                existing_urls.add(url)
                total_pages_fetched += 1
                print(f"  ✅ Added: {title} | {url}")
            except Exception as e:
                print(f"  ❌ Fetch/extract failed: {url} | {e}")

            # polite delay (avoid hammering servers)
            time.sleep(0.5)

    if not new_records:
        print("\nNo new records collected (all duplicates or failures).")
    else:
        # if file existed, append; else write new
        write_records(OUTPUT_CSV, new_records, append=append_mode)
        print("\n========================")
        print(f"✅ New records saved: {len(new_records)}")
        print(f"✅ Total topics seen: {total_topics_seen}")
        print(f"✅ Pages fetched: {total_pages_fetched}")
        print(f"✅ Skipped duplicates: {total_skipped_dupe}")
        print("========================")