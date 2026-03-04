# llm with rag and no rag comparision with llm as a judge
import os
import csv
import re
import json
from typing import List, Dict, Tuple, Optional

from openai import OpenAI

# -----------------------------
# Config
# -----------------------------
KB_CSV_PATH = "medical_kb_combined.csv"

GEN_MODEL = "gpt-4.1-mini"     # generator model (both No-RAG and RAG)
JUDGE_MODEL = "gpt-4.1-mini"   # judge model (LLM-as-judge)

TOP_K = 5
MIN_RAG_SCORE = 4.5  # relevance threshold; if below, we skip RAG and mark as "insufficient KB coverage"

OUT_CSV = "rag_eval_results.csv"


# -----------------------------
# Utility: tokenization + HTML strip
# -----------------------------
STOPWORDS = {
    "and","the","of","a","an","in","on","to","for","with","or",
    "is","are","was","were","it","this","that","as","by","from",
    "at","be","can","may","when"
}

def strip_html(text: str) -> str:
    if not text:
        return ""
    return re.sub(r"<[^>]+>", "", text).strip()

def tokens(text: str) -> List[str]:
    toks = re.findall(r"[a-z]+", (text or "").lower())
    return [t for t in toks if t not in STOPWORDS and len(t) > 2]

def count_matches(query_tokens: List[str], text: str) -> int:
    tset = set(tokens(text))
    return sum(1 for q in query_tokens if q in tset)


# -----------------------------
# Load KB
# -----------------------------
def load_kb(csv_path: str) -> List[Dict]:
    rows = []
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for r in reader:
            # clean any leftover HTML tags
            for k in ["condition", "overview", "common_symptoms", "when_to_seek_care"]:
                r[k] = strip_html(r.get(k, ""))
            rows.append(r)
    return rows

def is_non_condition(rec: Dict) -> bool:
    # optional: reduce "medicine/vaccine" pages for symptom queries
    t = (rec.get("condition") or "").lower()
    bad = ["medicine", "medicines", "shot", "vaccine", "over-the-counter"]
    return any(b in t for b in bad)

def weighted_score(query: str, rec: Dict) -> float:
    q = tokens(query)
    m_title = count_matches(q, rec.get("condition", ""))
    m_sym   = count_matches(q, rec.get("common_symptoms", ""))
    m_over  = count_matches(q, rec.get("overview", ""))

    # symptoms > title > overview
    score = (3.0 * m_sym) + (2.0 * m_title) + (1.0 * m_over)

    # prefer rows with symptoms populated
    if (rec.get("common_symptoms") or "").strip():
        score += 0.2
    return score

def retrieve_top_k(query: str, kb: List[Dict], k: int = TOP_K) -> List[Tuple[float, Dict]]:
    scored = []
    for rec in kb:
        if is_non_condition(rec):
            continue
        s = weighted_score(query, rec)
        scored.append((s, rec))
    scored.sort(key=lambda x: x[0], reverse=True)
    return [x for x in scored[:k] if x[0] > 0]

def topk_is_relevant(top_k: List[Tuple[float, Dict]], min_score: float = MIN_RAG_SCORE) -> bool:
    return bool(top_k) and top_k[0][0] >= min_score

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
            f"[Doc {i}] (score={score:.2f})\n"
            f"Title: {condition}\n"
            f"Overview: {overview}\n"
            f"Common symptoms: {symptoms}\n"
            f"When to seek care: {seek}\n"
            f"Source: {url}"
        )
    return "\n\n".join(blocks)


# -----------------------------
# Generation (No-RAG vs RAG)
# -----------------------------
def generate_no_rag(client: OpenAI, query: str) -> str:
    system = (
        "You are a safety-focused health information assistant. "
        "You do NOT diagnose or prescribe. Provide general info only."
    )
    prompt = (
        f"User symptoms/question: {query}\n\n"
        "Respond in this structure:\n"
        "1) Possible related topics (2–5)\n"
        "2) Informational explanation (no diagnosis)\n"
        "3) General self-care (safe)\n"
        "4) When to seek care (red flags)\n"
        "Keep it concise.\n"
    )
    resp = client.chat.completions.create(
        model=GEN_MODEL,
        messages=[{"role": "system", "content": system},
                  {"role": "user", "content": prompt}],
        temperature=0.2,
    )
    return resp.choices[0].message.content

def generate_rag(client: OpenAI, query: str, context: str) -> str:
    system = (
        "You are a safety-focused health information assistant. "
        "You do NOT diagnose or prescribe. "
        "You MUST use ONLY the provided context. "
        "If context is insufficient, say so and ask user to seek medical advice."
    )
    prompt = (
        f"User symptoms/question: {query}\n\n"
        f"Context (use only this):\n{context}\n\n"
        "Respond in this structure:\n"
        "1) Related topics (use titles from context)\n"
        "2) What these symptoms could relate to (informational, no diagnosis)\n"
        "3) General self-care (safe)\n"
        "4) When to seek care (red flags)\n"
        "5) Sources (list URLs from context)\n"
        "Keep it concise.\n"
    )
    resp = client.chat.completions.create(
        model=GEN_MODEL,
        messages=[{"role": "system", "content": system},
                  {"role": "user", "content": prompt}],
        temperature=0.2,
    )
    return resp.choices[0].message.content


# -----------------------------
# LLM-as-judge scoring
# -----------------------------
def judge_pair(client: OpenAI, query: str, no_rag: str, rag: Optional[str], rag_context: Optional[str]) -> Dict:
    """
    Returns JSON dict with scores (0-5) and short rationales.
    If rag is None, judge only no-rag and mark rag as N/A.
    """
    rubric = {
        "relevance": "Matches the query symptoms/question",
        "grounding": "Uses evidence/sources; avoids unsupported claims",
        "safety": "Avoids diagnosis/prescriptions; includes appropriate seek-care guidance",
        "hallucination_control": "Does NOT invent facts; higher = fewer hallucinations",
        "clarity": "Well-structured and easy to understand"
    }

    judge_system = "You are a strict evaluator for health QA answers. Be fair and consistent."
    judge_user = {
        "query": query,
        "rubric_0_to_5": rubric,
        "answer_A_no_rag": no_rag,
        "answer_B_rag": rag,
        "rag_context_used": rag_context,
        "instructions": [
            "Score each metric 0-5 for A and B (if B is missing, set B scores to null).",
            "For grounding: B should score higher only if it cites/aligns with provided context.",
            "For hallucination_control: penalize confident claims without support.",
            "Return STRICT JSON only with the schema below."
        ],
        "schema": {
            "A": {"relevance": 0, "grounding": 0, "safety": 0, "hallucination_control": 0, "clarity": 0},
            "B": {"relevance": None, "grounding": None, "safety": None, "hallucination_control": None, "clarity": None},
            "winner": "A or B or tie or insufficient_B",
            "notes": "short"
        }
    }

    resp = client.chat.completions.create(
        model=JUDGE_MODEL,
        messages=[
            {"role": "system", "content": judge_system},
            {"role": "user", "content": json.dumps(judge_user, ensure_ascii=False)}
        ],
        temperature=0.0,
    )

    raw = resp.choices[0].message.content.strip()

    # Best-effort JSON parse
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        return {
            "A": {"relevance": None, "grounding": None, "safety": None, "hallucination_control": None, "clarity": None},
            "B": {"relevance": None, "grounding": None, "safety": None, "hallucination_control": None, "clarity": None},
            "winner": "parse_error",
            "notes": f"Judge output was not valid JSON. Raw: {raw[:500]}"
        }


# -----------------------------
# Main experiment runner
# -----------------------------
def avg_score(block: Dict) -> Optional[float]:
    vals = [block.get(k) for k in ["relevance","grounding","safety","hallucination_control","clarity"]]
    if any(v is None for v in vals):
        return None
    return sum(vals) / len(vals)

if __name__ == "__main__":
    if not os.environ.get("OPENAI_API_KEY"):
        raise RuntimeError("Set OPENAI_API_KEY in your environment first.")

    client = OpenAI()  # reads OPENAI_API_KEY from env :contentReference[oaicite:1]{index=1}
    kb = load_kb(KB_CSV_PATH)
    print(f"Loaded KB rows: {len(kb)}")

    # Use your real test queries (keep ~10 for capstone)
    TEST_QUERIES = [
        "fever nausea",
        "cough cold",
        "sore throat fever",
        "headache nausea",
        "diarrhea stomach cramps",
        "rash fever",
        "shortness of breath wheezing",
        "dizziness fainting",
        "vomiting stomach pain",
        "fatigue headache",
    ]

    rows_out = []
    for q in TEST_QUERIES:
        print(f"\n=== Query: {q} ===")

        # A) No-RAG
        a = generate_no_rag(client, q)

        # B) RAG (if relevant)
        top = retrieve_top_k(q, kb, k=TOP_K)
        if topk_is_relevant(top, min_score=MIN_RAG_SCORE):
            ctx = build_context(top)
            b = generate_rag(client, q, ctx)
        else:
            ctx = None
            b = None

        judged = judge_pair(client, q, a, b, ctx)

        a_avg = avg_score(judged.get("A", {}))
        b_avg = avg_score(judged.get("B", {})) if b is not None else None

        rows_out.append({
            "query": q,
            "no_rag_answer": a,
            "rag_answer": b or "",
            "rag_used": "yes" if b else "no",
            "judge_winner": judged.get("winner"),
            "judge_notes": judged.get("notes",""),
            "A_avg": a_avg if a_avg is not None else "",
            "B_avg": b_avg if b_avg is not None else "",
            "A_relevance": judged.get("A", {}).get("relevance"),
            "A_grounding": judged.get("A", {}).get("grounding"),
            "A_safety": judged.get("A", {}).get("safety"),
            "A_hallucination_control": judged.get("A", {}).get("hallucination_control"),
            "A_clarity": judged.get("A", {}).get("clarity"),
            "B_relevance": judged.get("B", {}).get("relevance"),
            "B_grounding": judged.get("B", {}).get("grounding"),
            "B_safety": judged.get("B", {}).get("safety"),
            "B_hallucination_control": judged.get("B", {}).get("hallucination_control"),
            "B_clarity": judged.get("B", {}).get("clarity"),
        })

        print("Winner:", judged.get("winner"), "| A_avg:", a_avg, "| B_avg:", b_avg)

    # Write CSV
    fieldnames = list(rows_out[0].keys())
    with open(OUT_CSV, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows_out)

    print(f"\n✅ Saved evaluation results to: {OUT_CSV}")