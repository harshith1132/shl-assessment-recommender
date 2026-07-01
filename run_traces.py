"""
tests/run_traces.py
Replays each trace turn-by-turn against your running local server and
reports overlap between expected and returned recommendation URLs.
Run the server first: uvicorn main:app --reload
Then: python tests/run_traces.py
"""
import re
import glob
import requests
import time
import argparse

API = "http://localhost:8000/chat"


def parse_trace(path):
    text = open(path, encoding="utf-8").read()
    chunks = re.split(r"### Turn \d+", text)[1:]
    turns = []
    for chunk in chunks:
        user_m = re.search(r"\*\*User\*\*\s*\n*>\s*(.+?)\n\n\*\*Agent\*\*", chunk, re.S)
        user_text = user_m.group(1).strip().lstrip(">").strip() if user_m else None
        urls = re.findall(r"<(https://\S+?)>", chunk)
        eoc = bool(re.search(r"end_of_conversation.*?\*\*true\*\*", chunk, re.S))
        turns.append({"user": user_text, "expected_urls": urls, "eoc": eoc})
    return turns


def run_trace(path):
    turns = parse_trace(path)
    history = []
    print(f"\n=== {path} ===")
    for i, t in enumerate(turns, 1):
        if not t["user"]:
            continue
        history.append({"role": "user", "content": t["user"]})
        time.sleep(5)  # stay under free-tier rate limits
        resp = requests.post(API, json={"messages": history}, timeout=60).json()
        got_urls = {r["url"] for r in resp["recommendations"]}
        exp_urls = set(t["expected_urls"])
        overlap = got_urls & exp_urls
        print(f" turn {i}: expected={len(exp_urls)} got={len(got_urls)} "
              f"overlap={len(overlap)} eoc_expected={t['eoc']} eoc_got={resp['end_of_conversation']}")
        print(f"    user:  {t['user'][:150]}")
        print(f"    reply: {resp.get('reply', '')[:250]}")
        print(f"    got_urls: {sorted(got_urls)}")
        print(f"    exp_urls: {sorted(exp_urls)}")
        history.append({"role": "assistant", "content": resp["reply"]})


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--traces-dir", default="traces")
    parser.add_argument("--trace", default=None, help="Run a single trace file, e.g. C1.md")
    args = parser.parse_args()

    if args.trace:
        pattern = f"{args.traces_dir}/{args.trace}"
    else:
        pattern = f"{args.traces_dir}/C*.md"

    for path in sorted(glob.glob(pattern)):
        run_trace(path)
        time.sleep(5)