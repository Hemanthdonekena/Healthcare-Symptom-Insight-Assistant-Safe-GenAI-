#new
import csv
import re
from typing import List, Dict, Tuple
from context_builder import build_context


CSV_PATH = "medical_kb_combined.csv"

STOPWORDS = {
    "and","the","of","a","an","in","on","to","for","with","or",
    "is","are","was","were","it","this","that","as","by","from",
    "at","be","can","may","when"
}

def load_kb(csv_path: str) -> List[Dict]:
    rows = []
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for r in reader:
            rows.append(r)
    return rows

def tokens(text: str) -> List[str]:
    toks = re.findall(r"[a-z]+", (text or "").lower())
    return [t for t in toks if t not in STOPWORDS and len(t) > 2]

def count_matches(query_tokens: List[str], text: str) -> int:
    """
    Count how many query tokens appear in the text (bag-of-words style).
    This gives better tie-breaking than set intersection.
    """
    t = tokens(text)
    tset = set(t)
    return sum(1 for q in query_tokens if q in tset)

def weighted_score(query: str, rec: Dict) -> float:
    q = tokens(query)

    # Field matches
    m_title = count_matches(q, rec.get("condition",""))
    m_sym   = count_matches(q, rec.get("common_symptoms",""))
    m_over  = count_matches(q, rec.get("overview",""))

    # Weighted: symptoms > title > overview
    score = (3.0 * m_sym) + (2.0 * m_title) + (1.0 * m_over)

    # Tie-breakers: prefer records with non-empty symptoms
    if (rec.get("common_symptoms") or "").strip():
        score += 0.2

    return score

def retrieve_top_k(query: str, kb: List[Dict], k: int = 5) -> List[Tuple[float, Dict]]:
    scored = []
    for rec in kb:
        s = weighted_score(query, rec)
        scored.append((s, rec))

    scored.sort(key=lambda x: x[0], reverse=True)

    # Keep only scores > 0
    top = [item for item in scored[:k] if item[0] > 0]
    return top

import re

def clean_symptom_field(symptoms: str) -> str:
    """
    Convert messy symptom text into a clean ';'-separated list.
    """
    if not symptoms:
        return ""
    s = symptoms.lower()

    # remove obvious noise words
    s = re.sub(r"\b(symptoms?|include|includes|usually|often|may|might|can)\b", "", s)
    s = re.sub(r"\s+", " ", s).strip()

    # split into chunks based on separators
    parts = re.split(r";|,|\band\b|\bor\b|\.", s)
    cleaned = []
    for p in parts:
        p = p.strip(" :-")
        if 3 <= len(p) <= 40:
            cleaned.append(p)

    # dedupe while preserving order
    seen = set()
    deduped = []
    for item in cleaned:
        if item not in seen:
            seen.add(item)
            deduped.append(item)

    return "; ".join(deduped[:12])

if __name__ == "__main__":
    kb = load_kb(CSV_PATH)
    print(f"Loaded KB rows: {len(kb)}")

    user_query = input("Describe your symptoms: ").strip()

    results = retrieve_top_k(user_query, kb, k=5)

    print("\n=== Top-K Retrieved Topics (Weighted) ===")
    if not results:
        print("No matches found.")
    else:
        for rank, (score, rec) in enumerate(results, start=1):
            print(f"\nRank {rank} | Score: {score:.2f}")
            print("Condition:", rec["condition"])
            print("Source:", rec["source_url"])
            print("Symptoms:", (rec.get("common_symptoms","")[:180] + "...") if len(rec.get("common_symptoms","")) > 180 else rec.get("common_symptoms",""))
            # ... after results retrieval
            
    context = build_context(results)
    
    print("\n=== CONTEXT SENT TO LLM ===\n")
    print(context)