#!/usr/bin/env python3
'''command run:- uvicorn server:app --host 0.0.0.0 --port $PORT'''
import os
import asyncio
import traceback
from pathlib import Path
from typing import List, Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel

from improved_query_engine import ImprovedSHLQueryEngine

app = FastAPI(
    title="SHL Assessment Recommendation API",
    description="AI-powered recommendation system for SHL assessments",
    version="2.0.0",
    docs_url="/api/docs",
    redoc_url="/api/redoc",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

query_engine: Optional[ImprovedSHLQueryEngine] = None
_engine_initializing = False
_engine_init_error: Optional[str] = None


class RecommendationRequest(BaseModel):
    query: str


class AssessmentResponse(BaseModel):
    url: str
    name: str
    adaptive_support: str
    description: str
    duration: int
    remote_support: str
    test_type: List[str]


class RecommendationResponse(BaseModel):
    recommended_assessments: List[AssessmentResponse]


class HealthResponse(BaseModel):
    status: str


class SystemInfoResponse(BaseModel):
    version: str
    total_assessments: int
    features: List[str]
    performance: dict


ROOT_INDEX = Path("index.html")
STATIC_DIR = Path("static")
STATIC_DIR.mkdir(exist_ok=True)
app.mount("/static", StaticFiles(directory="static"), name="static")


@app.get("/", response_class=FileResponse)
async def serve_frontend():
    if ROOT_INDEX.exists():
        return ROOT_INDEX
    raise HTTPException(status_code=404, detail="Frontend index.html not found")


@app.get("/health", response_model=HealthResponse)
async def health_check():
    if _engine_initializing:
        return HealthResponse(status="initializing")
    if query_engine:
        return HealthResponse(status="healthy")
    if _engine_init_error:
        return HealthResponse(status="unhealthy")
    return HealthResponse(status="uninitialized")


@app.get("/api/info", response_model=SystemInfoResponse)
async def system_info():
    total = len(query_engine.products_meta) if query_engine else 0
    return SystemInfoResponse(
        version="2.0.0",
        total_assessments=total,
        features=[
            "AI-Powered Recommendations",
            "Duration-Aware Filtering",
            "Pattern Learning",
            "Test Type Balancing",
            "LLM Re-ranking",
        ],
        performance={
            "vector_search": "optimized" if query_engine else "unavailable",
            "llm_integration": "enabled" if query_engine and getattr(query_engine, "llm_client", None) else "disabled",
            "pattern_learning": "enabled" if query_engine and getattr(query_engine, "patterns", None) else "disabled",
        },
    )


@app.get("/api/test/examples")
async def get_test_examples():
    return {
        "examples": [
            {"query": "Looking to hire mid-level professionals proficient in Python, SQL, JavaScript. Max 60 mins."},
            {"query": "Hiring an analyst. Need Cognitive + Personality tests within 45 mins."},
            {"query": "Content Writer with SEO expertise."},
            {"query": "Product manager with SDLC, Jira, Confluence experience."},
            {"query": "Java developer with collaboration skills, 40 mins."},
        ]
    }


async def _init_query_engine_background(
    vector_store_dir="vector_store",
    train_data_path="data/Gen_AI Dataset.xlsx"
):
    global query_engine, _engine_initializing, _engine_init_error

    if query_engine or _engine_initializing:
        return

    _engine_initializing = True
    _engine_init_error = None

    try:
        def init_sync():
            return ImprovedSHLQueryEngine(
                vector_store_dir=vector_store_dir,
                train_data_path=train_data_path
            )

        engine = await asyncio.to_thread(init_sync)
        query_engine = engine

    except Exception as e:
        _engine_init_error = f"{type(e).__name__}: {str(e)}"
        traceback.print_exc()

    finally:
        _engine_initializing = False


@app.on_event("startup")
async def startup_event():
    asyncio.create_task(_init_query_engine_background())


async def _get_recommendations_for_query(query_text, use_reranking=True, top_k=10, use_expansion=True):
    if _engine_initializing:
        raise HTTPException(status_code=503, detail="Query engine is initializing")
    if not query_engine:
        raise HTTPException(status_code=503, detail=f"Engine unavailable: {_engine_init_error}")

    try:
        recs = query_engine.recommend(
            query=query_text,
            top_k=top_k,
            use_reranking=use_reranking,
            balance_types=True,
            apply_patterns=True,
            use_expansion=use_expansion,
        )

        if not recs:
            return {"recommended_assessments": []}

        final = []
        for r in recs[:top_k]:
            final.append({
                "url": r.get("url", ""),
                "name": r.get("name", ""),
                "adaptive_support": r.get("adaptive_support", "No"),
                "description": r.get("description", ""),
                "duration": int(r.get("duration", 0) or 0),
                "remote_support": r.get("remote_support", "No"),
                "test_type": r.get("test_type", []) or [],
            })
        return {"recommended_assessments": final}

    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Recommendation failed: {str(e)}")


@app.post("/recommend", response_class=JSONResponse)
async def recommend_root(request: RecommendationRequest):
    return await _get_recommendations_for_query(request.query)


@app.post("/api/recommend", response_model=RecommendationResponse)
async def recommend_api(request: RecommendationRequest):
    return await _get_recommendations_for_query(request.query)


@app.post("/api/recommend/basic", response_model=RecommendationResponse)
async def recommend_basic(request: RecommendationRequest):
    return await _get_recommendations_for_query(
        request.query, use_reranking=False, top_k=10, use_expansion=False
    )


@app.get("/api/products/count")
async def get_products_count():
    if not query_engine:
        raise HTTPException(status_code=503, detail="Engine still initializing")
    return {"total_products": len(query_engine.products_meta)}


@app.get("/api/products/sample")
async def get_sample_products(limit: int = 5):
    if not query_engine:
        raise HTTPException(status_code=503, detail="Engine still initializing")
    return {"sample_products": list(query_engine.products_meta.values())[:limit]}


@app.get("/api/patterns/stats")
async def get_pattern_stats():
    if not query_engine:
        raise HTTPException(status_code=503, detail="Engine still initializing")
    return {
        "role_mappings": len(query_engine.patterns.get("role_mappings", {})),
        "skill_mappings": len(query_engine.patterns.get("skill_mappings", {})),
        "duration_preferences": len(query_engine.patterns.get("duration_preferences", {})),
        "sample_roles": list(query_engine.patterns.get("role_mappings", {}))[:5],
        "sample_skills": list(query_engine.patterns.get("skill_mappings", {}))[:5],
    }


if __name__ == "__main__":
    port = int(os.getenv("PORT", "8000"))
    import uvicorn as _uvicorn
    _uvicorn.run(
        "server:app",
        host="0.0.0.0",
        port=port,
        log_level="info",
    )