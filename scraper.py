"""
scraper.py
Fetches the SHL catalog JSON feed and normalizes it into catalog_cache.json.
Run standalone to (re)build the cache:
    python scraper.py
"""
import json
import requests

CATALOG_URL = "https://tcp-us-prod-rnd.shl.com/voiceRater/shl-ai-hiring/shl_product_catalog.json"
OUTPUT_PATH = "catalog_cache.json"

LABEL = {
    "Ability & Aptitude": "A",
    "Biodata & Situational Judgment": "B",
    "Competencies": "C",
    "Development & 360": "D",
    "Assessment Exercises": "E",
    "Knowledge & Skills": "K",
    "Personality & Behavior": "P",
    "Simulations": "S",
}

def test_type_code(keys):
    return ",".join(LABEL.get(k, "?") for k in keys) if keys else ""

def build_catalog():
    r = requests.get(CATALOG_URL, timeout=30)
    r.raise_for_status()
    raw = json.loads(r.text, strict=False)

    catalog = []
    for item in raw:
        if item.get("status") and item["status"] != "ok":
            continue
        catalog.append({
            "id": item.get("entity_id"),
            "name": item.get("name", "").strip(),
            "url": item.get("link", "").strip(),
            "description": item.get("description", "") or "",
            "keys": item.get("keys", []) or [],
            "test_type": test_type_code(item.get("keys", [])),
            "job_levels": item.get("job_levels", []) or [],
            "languages": item.get("languages", []) or [],
            "duration": item.get("duration", "") or "",
            "remote": item.get("remote") == "yes",
            "adaptive": item.get("adaptive") == "yes",
        })
    catalog = [c for c in catalog if c["name"] and c["url"]]
    return catalog

if __name__ == "__main__":
    catalog = build_catalog()
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(catalog, f, indent=2, ensure_ascii=False)
    print(f"Wrote {len(catalog)} items to {OUTPUT_PATH}")