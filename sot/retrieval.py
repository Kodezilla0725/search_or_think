"""Client for the local BM25 retrieval server (retriever_server.py)."""
import requests


class Retriever:
    def __init__(self, url="http://127.0.0.1:8000", k=3, batch=200, timeout=900):
        self.url, self.k, self.batch, self.timeout = url, k, batch, timeout

    def health(self):
        return requests.get(f"{self.url}/health", timeout=10).json()

    def __call__(self, queries):
        out = []
        for i in range(0, len(queries), self.batch):
            r = requests.post(f"{self.url}/search", json={"queries": queries[i:i + self.batch], "k": self.k},
                              timeout=self.timeout)
            r.raise_for_status()
            out += r.json()["results"]
        return out