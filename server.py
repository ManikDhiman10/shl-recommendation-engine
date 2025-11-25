#!/usr/bin/env python3
"""
backend.py:- FastAPI backend for SHL Assessment Recommendation
#run command:- python backend.py
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel
import uvicorn
from typing import List
from pathlib import Path
import traceback

from improved_query_engine import ImprovedSHLQueryEngine

app = FastAPI(
    title="SHL Assessment Recommendation API",
    description="AI-powered recommendation system for SHL assessments",
    version="2.0.0",
    docs_url="/api/docs",
    redoc_url="/api/redoc"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

query_engine = None
try:
    query_engine = ImprovedSHLQueryEngine(
        vector_store_dir="vector_store",
        train_data_path="data/Gen_AI Dataset.xlsx"
    )
    print("✅ Improved query engine initialized successfully")
    print(f"✅ Loaded {len(query_engine.products_meta)} products")
except Exception as e:
    print(f"❌ Failed to initialize query engine: {e}")
    traceback.print_exc()
    query_engine = None

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
    model_loaded: bool
    products_count: int
    patterns_learned: bool

class SystemInfoResponse(BaseModel):
    version: str
    total_assessments: int
    features: List[str]
    performance: dict

STATIC_DIR = Path("static")
STATIC_DIR.mkdir(exist_ok=True)
TARGET_INDEX = STATIC_DIR / "index.html"

if TARGET_INDEX.exists():
    print(f"Serving UI from: {TARGET_INDEX}")
else:
    print("ERROR: static/index.html not found.")

app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

@app.get("/", response_class=FileResponse)
async def serve_frontend():
    if TARGET_INDEX.exists():
        return TARGET_INDEX
    possible = STATIC_DIR / "index.html"
    if possible.exists():
        return possible
    raise HTTPException(status_code=404, detail="Frontend not found.")

@app.get("/health", response_class=JSONResponse)
async def health_check():
    if query_engine:
        return JSONResponse(status_code=200, content={"status": "healthy"})
    return JSONResponse(status_code=503, content={"status": "unhealthy"})

@app.get("/api/info", response_model=SystemInfoResponse)
async def system_info():
    return SystemInfoResponse(
        version="2.0.0",
        total_assessments=len(query_engine.products_meta) if query_engine else 0,
        features=[
            "AI-Powered Recommendations",
            "Duration-Aware Filtering",
            "Pattern Learning",
            "Test Type Balancing",
            "LLM Re-ranking"
        ],
        performance={
            "vector_search": "optimized",
            "llm_integration": "enabled" if query_engine and query_engine.llm_client else "disabled",
            "pattern_learning": "enabled" if query_engine and query_engine.patterns else "disabled"
        }
    )

@app.get("/api/test/examples")
async def get_test_examples():
    examples = [
        {
            "query": "Looking to hire mid-level professionals proficient in Python, SQL, JavaScript. Max 60 mins.",
            "description": "Multi-skill technical assessment"
        },
        {
            "query": "Hiring an analyst. Need Cognitive + Personality tests within 45 mins.",
            "description": "Analyst cognitive focus"
        },
        {
            "query": "Content Writer with SEO expertise.",
            "description": "Content writer assessment"
        },
        {
            "query": "Product manager with SDLC, Jira, Confluence experience.",
            "description": "PM tools assessment"
        },
        {
            "query": "Java developer with collaboration skills, 40 mins.",
            "description": "Java + soft skills"
        }
    ]
    return {"examples": examples}

async def _get_recommendations_for_query(query_text: str, use_reranking=True, top_k=10, use_expansion=True):
    if not query_engine:
        raise HTTPException(status_code=503, detail="Query engine not initialized")
    try:
        recs = query_engine.recommend(
            query=query_text,
            top_k=top_k,
            use_reranking=use_reranking,
            balance_types=True,
            apply_patterns=True,
            use_expansion=use_expansion
        )
        if not recs:
            fallback = None
            if query_engine and getattr(query_engine, "products_meta", {}):
                first_meta = None
                for _, v in query_engine.products_meta.items():
                    first_meta = v
                    break
                if first_meta:
                    fallback = {
                        "url": first_meta.get("url", "") or "",
                        "name": first_meta.get("name", "SHL Assessment"),
                        "adaptive_support": first_meta.get("adaptive_support", "No"),
                        "description": first_meta.get("description", "") or "Recommended assessment (fallback).",
                        "duration": int(first_meta.get("duration", 0) or 0),
                        "remote_support": first_meta.get("remote_support", "No"),
                        "test_type": first_meta.get("test_type_names", []) or first_meta.get("test_type", []) or []
                    }
            if not fallback:
                fallback = {
                    "url": "",
                    "name": "SHL Assessment (Fallback)",
                    "adaptive_support": "No",
                    "description": "No matching assessments found for the query; returning fallback recommendation.",
                    "duration": 0,
                    "remote_support": "No",
                    "test_type": []
                }
            return {"recommended_assessments": [fallback]}
        final = []
        for r in recs[:top_k]:
            final.append({
                "url": r.get("url", ""),
                "name": r.get("name", ""),
                "adaptive_support": r.get("adaptive_support", "No"),
                "description": r.get("description", ""),
                "duration": int(r.get("duration", 0) or 0),
                "remote_support": r.get("remote_support", "No"),
                "test_type": r.get("test_type", []) or []
            })
        return {"recommended_assessments": final}
    except HTTPException:
        raise
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
        request.query,
        use_reranking=False,
        top_k=10,
        use_expansion=False
    )

@app.get("/api/products/count")
async def get_products_count():
    if not query_engine:
        raise HTTPException(status_code=503)
    return {"total_products": len(query_engine.products_meta)}

@app.get("/api/products/sample")
async def get_sample_products(limit: int = 5):
    if not query_engine:
        raise HTTPException(status_code=503)
    return {"sample_products": list(query_engine.products_meta.values())[:limit]}

@app.get("/api/patterns/stats")
async def get_pattern_stats():
    if not query_engine or not query_engine.patterns:
        raise HTTPException(status_code=404)
    return {
        "role_mappings": len(query_engine.patterns.get('role_mappings', {})),
        "skill_mappings": len(query_engine.patterns.get('skill_mappings', {})),
        "duration_preferences": len(query_engine.patterns.get('duration_preferences', {})),
        "sample_roles": list(query_engine.patterns.get('role_mappings', {}).keys())[:5],
        "sample_skills": list(query_engine.patterns.get('skill_mappings', {}).keys())[:5]
    }

if __name__ == "__main__":
    print("🚀 Starting SHL Recommendation Backend...")
    print("📊 Docs: http://localhost:8000/api/docs")
    print("🌐 UI:   http://localhost:8000")
    print("❤️ Health: http://localhost:8000/health")

    uvicorn.run(
        "backend:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )
