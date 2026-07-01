"""
retriever.py
TF-IDF semantic search over the catalog.
"""
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np


class CatalogRetriever:
    def __init__(self, catalog):
        self.catalog = catalog
        self.corpus = [
            f"{c['name']} {c['name']} {c['name']} {c['description']} "
            f"{' '.join(c['keys'])} {' '.join(c['job_levels'])}"
            for c in catalog
        ]
        self.vectorizer = TfidfVectorizer(stop_words="english", max_features=8000, ngram_range=(1, 2))
        self.matrix = self.vectorizer.fit_transform(self.corpus)

    def search(self, query, top_k=15, keys_filter=None, job_level_filter=None):
        if not query.strip():
            return []
        qvec = self.vectorizer.transform([query])
        sims = cosine_similarity(qvec, self.matrix).flatten()
        order = np.argsort(-sims)

        results = []
        for idx in order:
            if sims[idx] <= 0:
                break
            rec = self.catalog[idx]
            if keys_filter and not (set(keys_filter) & set(rec["keys"])):
                continue
            if job_level_filter and not (set(job_level_filter) & set(rec["job_levels"])):
                continue
            results.append((rec, float(sims[idx])))
            if len(results) >= top_k:
                break
        return results

    def find_by_name(self, fragment):
        f = fragment.lower()
        return [c for c in self.catalog if f in c["name"].lower()]

    def get(self, url):
        return next((c for c in self.catalog if c["url"] == url), None)