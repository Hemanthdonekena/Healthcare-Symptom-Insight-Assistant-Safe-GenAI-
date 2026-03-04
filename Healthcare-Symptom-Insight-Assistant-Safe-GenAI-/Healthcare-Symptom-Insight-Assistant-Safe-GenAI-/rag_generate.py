import os
import csv
import re
from typing import List, Dict, Tuple

from openai import OpenAI


import os
from dotenv import load_dotenv

load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY # Ensure it's in os.environ for tools/LLMs

CSV_PATH = "medical_kb_combined.csv"

STOPWORDS = {
    "and","the","of","a","an","in","on","to","for","with","or",
    "is","are","was","were","it","this","that","as","by","from",
    "at","be","can","may","when"
}

def strip_html(text: str) -> str:
    if not text:
        return ""
    return re.sub(r"<[^>]+>", "", text).strip()

def load_kb(csv_path: str) -> List[Dict]:
    rows = []
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for r in reader:
            # clean any leftover HTML tags from earlier saves
            r["condition"] = strip_html(r.get("condition", ""))
            r["overview"] = strip_html(r.get("overview", ""))
            r["common_symptoms"] = strip_html(r.get("common_symptoms", ""))
            r["when_to_seek_care"] = strip_html(r.get("when_to_seek_care", ""))
            rows.append(r)
    return rows

def tokens(text: str) -> List[str]:
    toks = re.findall(r"[a-z]+", (text or "").lower())
    return [t for t in toks if t not in STOPWORDS and len(t) > 2]

def count_matches(query_tokens: List[str], text: str) -> int:
    tset = set(tokens(text))
    return sum(1 for q in query_tokens if q in tset)

def weighted_score(query: str, rec: Dict) -> float:
    q = tokens(query)

    m_title = count_matches(q, rec.get("condition",""))
    m_sym   = count_matches(q, rec.get("common_symptoms",""))
    m_over  = count_matches(q, rec.get("overview",""))

    score = (3.0 * m_sym) + (2.0 * m_title) + (1.0 * m_over)

    if (rec.get("common_symptoms") or "").strip():
        score += 0.2

    return score

def is_non_condition(rec: Dict) -> bool:
    t = (rec.get("condition") or "").lower()
    bad = ["medicine", "medicines", "shot", "vaccine", "over-the-counter"]
    return any(b in t for b in bad)

def retrieve_top_k(query: str, kb: List[Dict], k: int = 5) -> List[Tuple[float, Dict]]:
    scored = []
    for rec in kb:
        if is_non_condition(rec):
            continue
        s = weighted_score(query, rec)
        scored.append((s, rec))

    scored.sort(key=lambda x: x[0], reverse=True)
    return [x for x in scored[:k] if x[0] > 0]

def build_context(top_k: List[Tuple[float, Dict]], max_chars: int = 700) -> str:
    blocks = []
    for i, (score, rec) in enumerate(top_k, start=1):
        condition = (rec.get("condition") or "").strip()
        overview = (rec.get("overview") or "").strip()
        symptoms = (rec.get("common_symptoms") or "").strip()
        seek = (rec.get("when_to_seek_care") or "").strip()
        url = (rec.get("source_url") or "").strip()

        if len(overview) > max_chars:
            overview = overview[:max_chars] + "..."
        if len(seek) > max_chars:
            seek = seek[:max_chars] + "..."

        blocks.append(
            f"[Doc {i}] {condition}\n"
            f"Overview: {overview}\n"
            f"Common symptoms: {symptoms}\n"
            f"When to seek care: {seek}\n"
            f"Source: {url}"
        )
    return "\n\n".join(blocks)

def generate_answer(user_query: str, context: str) -> str:
    client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

    system_prompt = (
        "You are a safety-focused health information assistant. "
        "You do NOT diagnose or prescribe. "
        "You must use ONLY the provided context. "
        "If the context is insufficient, say so and recommend seeking professional medical advice. "
        "Always include an urgent warning if symptoms could be severe or worsening."
    )

    user_prompt = f"""
User symptoms/question: {user_query}

Context (use only this):
{context}

Write an answer in this exact structure:

1) Related topics (list 2–5 topic titles)
2) What these symptoms could relate to (informational, no diagnosis)
3) General self-care (safe, non-prescriptive)
4) When to seek care (include red flags)
5) Sources (list URLs used)
"""

    resp = client.chat.completions.create(
        model="gpt-4.1-mini",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0.2,
    )
    return resp.choices[0].message.content

if __name__ == "__main__":
    kb = load_kb(CSV_PATH)
    print(f"Loaded KB rows: {len(kb)}")

    q = input("Describe your symptoms: ").strip()

    top = retrieve_top_k(q, kb, k=5)
    if not top:
        print("No relevant topics found in KB.")
        raise SystemExit(0)

    ctx = build_context(top)

    print("\n=== Context being sent to LLM ===\n")
    print(ctx)

    print("\n\n=== LLM Answer ===\n")
    answer = generate_answer(q, ctx)
    print(answer)