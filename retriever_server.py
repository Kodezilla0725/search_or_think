"""BM25 retrieval server over a Pyserini Lucene index. Runs in its own venv (Java 21 + pyserini).

POST /search {"queries": [...], "k": 3} -> {"results": [[{docid, title, text, score}, ...], ...]}
"""
import json
import os
from typing import List

import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel

os.environ.setdefault("OPENAI_API_KEY", "unused")  # pyserini builds an OpenAI client at import time; BM25 never calls it
from pyserini.search.lucene import LuceneSearcher  # noqa: E402

searcher = LuceneSearcher(os.environ["INDEX_DIR"])
THREADS = int(os.environ.get("THREADS", os.cpu_count() or 4))
app = FastAPI()


class SearchRequest(BaseModel):
    queries: List[str]
    k: int = 3


def parse_doc(docid):
    doc = searcher.doc(docid)
    contents = doc.raw() if doc is not None else ""
    try:
        contents = json.loads(contents).get("contents", contents)
    except (ValueError, AttributeError):
        pass
    title, sep, text = contents.partition("\n")  # DPR wiki passages are '"Title"\ntext'
    if not sep:
        title, text = "", contents
    return {"docid": docid, "title": title.strip().strip('"'), "text": text.strip()}


@app.get("/health")
def health():
    return {"ok": True, "num_docs": getattr(searcher, "num_docs", None)}


@app.post("/search")
async def search(req: SearchRequest):  # async: runs on the main thread that owns the JVM
    results = [[] for _ in req.queries]
    idx = [i for i, q in enumerate(req.queries) if q and q.strip()]
    if idx:
        hits = searcher.batch_search([req.queries[i].strip() for i in idx], [str(i) for i in idx],
                                     k=req.k, threads=THREADS)
        for i in idx:
            results[i] = [{**parse_doc(h.docid), "score": float(h.score)} for h in hits.get(str(i), [])]
    return {"results": results}


if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=int(os.environ.get("PORT", 8000)), log_level="warning")