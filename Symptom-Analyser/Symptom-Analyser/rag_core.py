import os
import csv
import re
from typing import List, Dict, Tuple, Optional

from openai import OpenAI

CSV_PATH_DEFAULT = "medical_kb_combined.csv"

STOPWORDS = {
    "and","the","of","a","an","in","on","to","for","with","or",
    "is","are","was","were","it","this","that","as","by","from",
    "at","be","can","may","when"
}

RED_FLAG_TERMS = [
    "chest pain", "trouble breathing", "difficulty breathing", "shortness of breath",
    "confusion", "fainting", "passed out", "seizure", "severe bleeding",
    "face swelling", "throat swelling", "blue lips", "suicidal"
]

FOLLOWUP_TRIGGERS = [
    "treat this", "treat it", "what should i do", "next steps",
    "is it serious", "what does it mean", "how can i treat",
    "how to treat", "how do i treat", "what can i take",
    "what medication", "what medicine", "help me", "what now"
]

def is_followup_question(text: str) -> bool:
    t = (text or "").lower().strip()
    return any(p in t for p in FOLLOWUP_TRIGGERS) and len(t.split()) <= 10

def strip_html(text: str) -> str:
    if not text:
        return ""
    return re.sub(r"<[^>]+>", "", text).strip()

def load_kb(csv_path: str = CSV_PATH_DEFAULT) -> List[Dict]:
    rows = []
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for r in reader:
            for k in ["condition", "overview", "common_symptoms", "when_to_seek_care"]:
                r[k] = strip_html(r.get(k, ""))
            rows.append(r)
    return rows

def tokens(text: str) -> List[str]:
    toks = re.findall(r"[a-z]+", (text or "").lower())
    return [t for t in toks if t not in STOPWORDS and len(t) > 2]

def count_matches(query_tokens: List[str], text: str) -> int:
    tset = set(tokens(text))
    return sum(1 for q in query_tokens if q in tset)

def is_non_condition(rec: Dict) -> bool:
    # Optional filter to reduce medicine/vaccine pages for symptom queries
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

def retrieve_top_k(query: str, kb: List[Dict], k: int = 5) -> List[Tuple[float, Dict]]:
    scored = []
    for rec in kb:
        if is_non_condition(rec):
            continue
        s = weighted_score(query, rec)
        if s > 0:
            scored.append((s, rec))
    scored.sort(key=lambda x: x[0], reverse=True)
    return scored[:k]

def topk_is_relevant(top_k: List[Tuple[float, Dict]], min_score: float = 4.5) -> bool:
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

def detect_red_flags(text: str) -> List[str]:
    t = (text or "").lower()
    hits = [rf for rf in RED_FLAG_TERMS if rf in t]
    return hits

def last_user_memory(history: List[Dict], max_turns: int = 4) -> str:
    """
    Returns last few user messages as memory.
    We avoid using assistant messages as 'facts' to keep grounding clean.
    """
    msgs = [m["content"] for m in history if m["role"] == "user"][-max_turns:]
    return "\n".join([f"- {m}" for m in msgs]) if msgs else ""

def make_chat_query(history: List[Dict], new_user_msg: str, max_turns: int = 4) -> str:
    """
    Builds a compact retrieval query using user history.
    If the message is a vague follow-up, anchor on the last meaningful user message.
    """
    if is_followup_question(new_user_msg):
        for m in reversed(history):
            if m["role"] == "user":
                prev = (m["content"] or "").strip()
                if len(prev.split()) >= 3 and not is_followup_question(prev):
                    return f"{prev} | follow-up: {new_user_msg}"
        return f"follow-up: {new_user_msg}"

    user_msgs = [m["content"] for m in history if m["role"] == "user"][-max_turns:]
    user_msgs.append(new_user_msg)
    return " | ".join(user_msgs)
FOLLOWUP_TRIGGERS = [
    "treat this", "treat it", "what should i do", "next steps",
    "is it serious", "what does it mean", "how can i treat",
    "how to treat", "how do i treat", "what can i take"
]

def is_followup_question(text: str) -> bool:
    t = (text or "").lower().strip()
    return any(p in t for p in FOLLOWUP_TRIGGERS) and len(t.split()) <= 8

def generate_rag_answer(
    user_msg: str,
    history: List[Dict],
    kb: List[Dict],
    model: str = "gpt-4.1-mini",
    top_k: int = 5,
    min_score: float = 4.5,
) -> Tuple[str, List[Tuple[float, Dict]], Optional[str]]:
    """
    Returns: (assistant_answer, top_k_results, context_or_none)
    """
    if not os.environ.get("OPENAI_API_KEY"):
        raise RuntimeError("OPENAI_API_KEY is not set in your environment.")

    client = OpenAI()

    retrieval_query = make_chat_query(history, user_msg)
    top = retrieve_top_k(retrieval_query, kb, k=top_k)

    # Red flag banner (UI can display this)
    red_flags = detect_red_flags(user_msg)

    if not topk_is_relevant(top, min_score=min_score):
        # If KB coverage is weak, be honest and ask for more detail
        msg = (
            "I couldn’t find strong matches in my medical knowledge base for your message.\n\n"
            "Try:\n"
            "- adding more symptoms (e.g., rash, cough, headache)\n"
            "- adding duration (e.g., 2 days)\n"
            "- describing severity (mild/moderate/severe)\n\n"
            "If symptoms feel severe or are worsening, please seek medical care."
        )
        if red_flags:
            msg = (
                "⚠️ **Possible urgent symptoms detected**: "
                + ", ".join(red_flags)
                + "\n\nIf you have severe symptoms or feel unsafe, seek urgent medical care.\n\n"
                + msg
            )
        return msg, top, None

    context = build_context(top)

    system = (
        "You are a safety-focused health information assistant. "
        "You do NOT diagnose or prescribe. "
        "Use ONLY the provided context. "
        "If the context is insufficient, say so. "
        "Be concise and practical."
    )
    memory = last_user_memory(history, max_turns=4)
    user_prompt = f"""
Conversation memory (previous user messages, for context only):
{memory if memory else "- (none)"}

Current user message:
{user_msg}

Context (authoritative, use ONLY this for medical facts):
{context}

Rules:
- Use conversation memory only to understand what "this" refers to.
- Do NOT use memory as medical evidence.
- Do NOT diagnose or prescribe.
- If the context does not support an answer, say so.

Write an answer in this structure:

1) Related topics (2–5 titles from context)
2) What this could relate to (informational, no diagnosis)
3) General self-care (safe, non-prescriptive)
4) When to seek care (include red flags)
5) Sources (URLs)

Keep it concise.
"""
    resp = client.chat.completions.create(
        model=model,
        messages=[{"role": "system", "content": system},
                  {"role": "user", "content": user_prompt}],
        temperature=0.2,
    )

    answer = resp.choices[0].message.content

    # If red flags, prepend a warning
    if red_flags:
        answer = (
            "⚠️ **Possible urgent symptoms detected**: "
            + ", ".join(red_flags)
            + "\n\nIf you have severe symptoms or feel unsafe, seek urgent medical care.\n\n"
            + answer
        )

    return answer, top, context