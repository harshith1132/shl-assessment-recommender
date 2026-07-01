import json
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CACHE_PATH = os.path.join(BASE_DIR, "catalog_cache.json")
OVERRIDES_PATH = os.path.join(BASE_DIR, "catalog_overrides.json")

def load_catalog():
    if not os.path.exists(CACHE_PATH):
        raise FileNotFoundError(
            f"catalog_cache.json not found at {CACHE_PATH}. "
            "Make sure it's committed to the repo (not gitignored) "
            "or run `python scraper.py` before starting the server."
        )
    with open(CACHE_PATH, encoding="utf-8") as f:
        catalog = json.load(f)

    if os.path.exists(OVERRIDES_PATH):
        with open(OVERRIDES_PATH, encoding="utf-8") as f:
            overrides = {o["url"]: o for o in json.load(f)}
        for rec in catalog:
            if rec["url"] in overrides:
                rec.update(overrides[rec["url"]])

    return catalog

def index_by_url(catalog):
    return {c["url"]: c for c in catalog}

def find_by_name(catalog, fragment):
    f = fragment.lower()
    return [c for c in catalog if f in c["name"].lower()]