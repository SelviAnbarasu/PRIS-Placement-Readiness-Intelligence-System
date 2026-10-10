"""Local lightweight RAG retriever for PRIS using TF-IDF + cosine similarity."""
from pathlib import Path
from typing import List, Dict
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

KNOWLEDGE_DIR = Path(__file__).resolve().parent / "rag_knowledge"

def load_knowledge_base() -> List[Dict[str, str]]:
    docs = []
    for path in sorted(KNOWLEDGE_DIR.glob("*.md")) if KNOWLEDGE_DIR.exists() else []:
        text = path.read_text(encoding="utf-8").strip()
        if text:
            docs.append({"source": path.name, "title": text.splitlines()[0].lstrip("# ").strip(), "text": text})
    return docs

def retrieve_rag_context(query: str, top_k: int = 4):
    docs = load_knowledge_base()
    if not docs or not query.strip():
        return []
    vectorizer = TfidfVectorizer(lowercase=True, stop_words="english", ngram_range=(1, 2), sublinear_tf=True)
    matrix = vectorizer.fit_transform([d["text"] for d in docs])
    scores = cosine_similarity(vectorizer.transform([query]), matrix)[0]
    ranked = scores.argsort()[::-1][:top_k]
    return [{**docs[i], "score": round(float(scores[i]), 4)} for i in ranked if scores[i] > 0]

def build_rag_context(jd_text: str, missing_skills=None, target_role: str = "", top_k: int = 4):
    missing_skills = missing_skills or []
    query = "\n".join(x for x in [target_role, jd_text, " ".join(map(str, missing_skills))] if x.strip())
    results = retrieve_rag_context(query, top_k)
    context = "\n\n".join(f"[SOURCE: {x['title']}]\n{x['text']}" for x in results)
    return context, results
