"""
main.py
FastAPI service: GET /health, POST /chat
"""
from dotenv import load_dotenv
load_dotenv()  # must run before anthropic client is created in conversation.py

from fastapi import FastAPI
from schemas import ChatRequest, ChatResponse, Recommendation
from catalog import load_catalog, index_by_url
from retriever import CatalogRetriever
from conversation import run_agent, to_recommendations

app = FastAPI(title="SHL Assessment Recommender")

CATALOG = load_catalog()
CATALOGBYURL = index_by_url(CATALOG)
RETRIEVER = CatalogRetriever(CATALOG)


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest):
    try:
        parsed, candidate_records = run_agent(req.messages, RETRIEVER)
        urls = parsed.get("recommendation_urls") or []
        recs = to_recommendations(urls, candidate_records, CATALOGBYURL)
        return ChatResponse(
            reply=parsed.get("reply", ""),
            recommendations=[Recommendation(**r) for r in recs],
            end_of_conversation=bool(parsed.get("end_of_conversation", False)),
        )
    except Exception as e:
        import traceback
        print("CHAT ERROR:", repr(e))
        traceback.print_exc()
        return ChatResponse(
            reply="I ran into an issue processing that -- could you try again?",
            recommendations=[],
            end_of_conversation=False,
        )