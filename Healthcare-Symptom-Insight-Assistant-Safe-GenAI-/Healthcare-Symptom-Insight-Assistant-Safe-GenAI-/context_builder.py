#llm work
from typing import List, Dict, Tuple

def build_context(top_k: List[Tuple[float, Dict]], max_chars_per_field: int = 700) -> str:
    """
    Build a compact evidence context from Top-K retrieved records.
    This is what you will pass to the LLM as 'grounding context'.
    """
    blocks = []
    for i, (score, rec) in enumerate(top_k, start=1):
        condition = (rec.get("condition") or "").strip()
        overview = (rec.get("overview") or "").strip()
        symptoms = (rec.get("common_symptoms") or "").strip()
        seek = (rec.get("when_to_seek_care") or "").strip()
        url = (rec.get("source_url") or "").strip()

        # Trim fields so prompt doesn't explode
        if len(overview) > max_chars_per_field:
            overview = overview[:max_chars_per_field] + "..."
        if len(seek) > max_chars_per_field:
            seek = seek[:max_chars_per_field] + "..."

        block = (
            f"[Doc {i}] (score={score:.2f})\n"
            f"Title: {condition}\n"
            f"Overview: {overview}\n"
            f"Common symptoms: {symptoms}\n"
            f"When to seek care: {seek}\n"
            f"Source: {url}"
        )
        blocks.append(block)

    return "\n\n".join(blocks)