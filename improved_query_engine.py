#!/usr/bin/env python3
"""
improved_query_engine.py:- SHL recommendation engine
run command:- python improved_query_engine.py --query "Java Developer Beginner Level"
"""

import argparse
import json
import os
import re
import pandas as pd
from pathlib import Path
from typing import List, Dict, Any
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv
import pickle

# ---------- Optional LLM ----------
try:
    import google.genai as genai
    GENAI_AVAILABLE = True
except ImportError:
    GENAI_AVAILABLE = False
    print("Warning: google-genai not installed. LLM re-ranking and expansion will be disabled.")

# ---------- Env ----------
load_dotenv()

# ---------- BM25 Tokenizer ----------
_WORD_RE = re.compile(r"[a-zA-Z0-9]+(?:'[a-zA-Z0-9]+)?")

def _tokenize_for_bm25(text: str) -> List[str]:
    if not text:
        return []
    text = text.lower()
    return _WORD_RE.findall(text)

# ---------- Engine Class ----------
class ImprovedSHLQueryEngine:
    def __init__(self, vector_store_dir: str = "vector_store", train_data_path: str = "data/Gen_AI Dataset.xlsx"):
        self.vector_store_dir = Path(vector_store_dir)
        self.model = None
        self.short_index = None
        self.chunks_index = None
        self.products_meta = {}
        self.prod_map = {}
        self.chunks_meta = []
        self.llm_client = None
        self.patterns = {}
        self.short_bm25 = None
        self.short_tokenized = None
        self.chunk_bm25 = None
        self.chunk_tokenized = None
        
        self._load_indices()
        self._init_llm()
        
        if train_data_path and os.path.exists(train_data_path):
            self._learn_from_training_data(train_data_path)
    
    # ---------- Index Loading ----------
    def _load_indices(self):
        print("Loading FAISS indices and metadata...")
        
        model_name_path = self.vector_store_dir / "model_name.txt"
        if model_name_path.exists():
            with open(model_name_path, "r") as f:
                model_name = f.read().strip()
        else:
            model_name = "all-MiniLM-L6-v2"
        
        self.model = SentenceTransformer(model_name)
        
        short_index_path = self.vector_store_dir / "short.index"
        chunks_index_path = self.vector_store_dir / "chunks.index"
        
        if short_index_path.exists():
            self.short_index = faiss.read_index(str(short_index_path))
            try:
                print(f"Loaded short index with {self.short_index.ntotal} vectors")
            except Exception:
                print("Loaded short index")
        else:
            raise FileNotFoundError(f"Short index not found: {short_index_path}")
        
        if chunks_index_path.exists():
            self.chunks_index = faiss.read_index(str(chunks_index_path))
            try:
                print(f"Loaded chunks index with {self.chunks_index.ntotal} vectors")
            except Exception:
                print("Loaded chunks index")
        else:
            raise FileNotFoundError(f"Chunks index not found: {chunks_index_path}")
        
        products_meta_path = self.vector_store_dir / "products_meta.json"
        prod_map_path = self.vector_store_dir / "prod_map.json"
        chunks_meta_path = self.vector_store_dir / "chunks_meta.json"
        
        if products_meta_path.exists():
            with open(products_meta_path, "r", encoding="utf-8") as f:
                self.products_meta = json.load(f)
        
        if prod_map_path.exists():
            with open(prod_map_path, "r", encoding="utf-8") as f:
                self.prod_map = json.load(f)
        
        if chunks_meta_path.exists():
            with open(chunks_meta_path, "r", encoding="utf-8") as f:
                self.chunks_meta = json.load(f)
        
        print(f"Loaded {len(self.products_meta)} products, {len(self.chunks_meta)} chunks")
        
        short_bm25_path = self.vector_store_dir / "short_texts.bm25.pkl"
        chunk_bm25_path = self.vector_store_dir / "chunks.bm25.pkl"
        try:
            if short_bm25_path.exists():
                with open(short_bm25_path, "rb") as f:
                    short_data = pickle.load(f)
                    self.short_tokenized = short_data.get("tokenized")
                    self.short_bm25 = short_data.get("bm25")
                print("Loaded BM25 for short texts")
            if chunk_bm25_path.exists():
                with open(chunk_bm25_path, "rb") as f:
                    chunk_data = pickle.load(f)
                    self.chunk_tokenized = chunk_data.get("tokenized")
                    self.chunk_bm25 = chunk_data.get("bm25")
                print("Loaded BM25 for chunk texts")
        except Exception as e:
            print(f"Warning: Could not load BM25 pickles: {e}")
            self.short_bm25 = None
            self.chunk_bm25 = None
    
    # ---------- LLM Init ----------
    def _init_llm(self):
        if not GENAI_AVAILABLE:
            print("WARNING: google-genai not available. LLM re-ranking and expansion will be disabled.")
            return
            
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            print("WARNING: GEMINI_API_KEY not found in environment. LLM re-ranking and expansion will be disabled.")
            return
        
        try:
            self.llm_client = genai.Client(api_key=api_key)
            print("Gemini LLM client initialized successfully")
        except Exception as e:
            print(f"WARNING: Failed to initialize Gemini client: {e}")
    
    # ---------- Training Patterns ----------
    def _learn_from_training_data(self, train_data_path: str):
        print("Learning patterns from training data...")
        
        try:
            df = pd.read_excel(train_data_path, sheet_name='Train-Set')
            
            query_groups = df.groupby('Query')
            
            self.patterns = {
                'role_mappings': {},
                'skill_mappings': {},
                'duration_preferences': {},
                'test_type_patterns': {}
            }
            
            for query, group in query_groups:
                duration_match = re.search(r'(\d+)\s*min', query, re.IGNORECASE)
                if duration_match:
                    max_duration = int(duration_match.group(1))
                    self.patterns['duration_preferences'][query] = max_duration
                
                roles = self._extract_roles(query)
                for role in roles:
                    if role not in self.patterns['role_mappings']:
                        self.patterns['role_mappings'][role] = set()
                    self.patterns['role_mappings'][role].update(group['Assessment_url'].tolist())
                
                skills = self._extract_skills(query)
                for skill in skills:
                    if skill not in self.patterns['skill_mappings']:
                        self.patterns['skill_mappings'][skill] = set()
                    self.patterns['skill_mappings'][skill].update(group['Assessment_url'].tolist())
            
            print(f"Learned patterns for {len(self.patterns['role_mappings'])} roles and {len(self.patterns['skill_mappings'])} skills")
            
        except Exception as e:
            print(f"Warning: Could not load training data: {e}")
    
    # ---------- Role/Skill Extractors ----------
    def _extract_roles(self, query: str) -> List[str]:
        roles = []
        role_keywords = [
            'java developer', 'sales role', 'sales', 'developer', 'analyst', 
            'COO', 'chief operating officer', 'content writer', 'QA engineer',
            'quality assurance', 'admin', 'administrative', 'marketing manager',
            'consultant', 'data analyst', 'research engineer', 'presales',
            'customer support', 'product manager', 'finance analyst'
        ]
        
        query_lower = query.lower()
        for role in role_keywords:
            if role in query_lower:
                roles.append(role)
        
        return roles
    
    def _extract_skills(self, query: str) -> List[str]:
        skills = []
        skill_keywords = [
            'python', 'sql', 'javascript', 'java', 'collaboration', 'communication',
            'cognitive', 'personality', 'behavioral', 'technical', 'selenium',
            'html', 'css', 'excel', 'jira', 'confluence', 'sdlc', 'english',
            'seo', 'ai', 'ml', 'machine learning', 'nlp', 'computer vision',
            'tensorflow', 'pytorch', 'financial', 'analytical'
        ]
        
        query_lower = query.lower()
        for skill in skill_keywords:
            if skill in query_lower:
                skills.append(skill)
        
        return skills
    
    # ---------- Duration Extraction ----------
    def _extract_duration_constraint(self, query: str) -> int:
        patterns = [
            r'(\d+)\s*min', r'(\d+)\s*minute', r'(\d+)\s*m',
            r'(\d+)-(\d+)\s*min', r'about an hour', r'1 hour', r'2 hour'
        ]
        
        query_lower = query.lower()
        
        for pattern in patterns[:3]:
            match = re.search(pattern, query_lower)
            if match:
                return int(match.group(1))
        
        range_match = re.search(r'(\d+)-(\d+)\s*min', query_lower)
        if range_match:
            return int(range_match.group(2))
        
        if 'about an hour' in query_lower or '1 hour' in query_lower:
            return 60
        if '2 hour' in query_lower or '1-2 hour' in query_lower:
            return 120
        
        return None
    
    # ---------- Pattern Boosting ----------
    def _apply_pattern_based_boost(self, candidates: List[Dict], query: str) -> List[Dict]:
        if not self.patterns:
            return candidates
        
        query_lower = query.lower()
        boosted_candidates = []
        
        for candidate in candidates:
            candidate_url = candidate["metadata"].get("url", "")
            boost_score = 1.0
            
            for role, urls in self.patterns['role_mappings'].items():
                if role in query_lower and candidate_url in urls:
                    boost_score *= 1.5
            
            for skill, urls in self.patterns['skill_mappings'].items():
                if skill in query_lower and candidate_url in urls:
                    boost_score *= 1.3
            
            candidate["score"] *= boost_score
            boosted_candidates.append(candidate)
        
        boosted_candidates.sort(key=lambda x: x["score"], reverse=True)
        return boosted_candidates
    
    # ---------- Duration Filter ----------
    def _filter_by_duration(self, candidates: List[Dict], max_duration: int) -> List[Dict]:
        if not max_duration:
            return candidates
        
        filtered = []
        for candidate in candidates:
            candidate_duration = candidate["metadata"].get("duration", 0)
            if not candidate_duration or candidate_duration <= max_duration:
                filtered.append(candidate)
        
        return filtered
    
    # ---------- BM25 Heuristic ----------
    def should_use_bm25(self, query: str) -> bool:
        if not (self.short_bm25 or self.chunk_bm25):
            return False
        
        q = query.strip()
        tokens = _WORD_RE.findall(q)
        if len(tokens) <= 6:
            return True
        
        cap_count = sum(1 for t in q.split() if re.match(r'^[A-Z][a-zA-Z0-9\+\#\-\&]+$', t))
        if cap_count >= 2:
            return True
        
        tech_keywords = ["java", "python", "sql", "c++", "javascript", "react", "node", "excel", "selenium"]
        q_low = q.lower()
        for tk in tech_keywords:
            if tk in q_low:
                return True
        
        return False
    
    # ---------- Score Normalization ----------
    def _normalize_scores(self, scores: List[float]) -> np.ndarray:
        arr = np.array(scores, dtype=float)
        if arr.size == 0:
            return arr
        mn = arr.min()
        mx = arr.max()
        if mx - mn < 1e-12:
            return np.ones_like(arr)
        return (arr - mn) / (mx - mn)
    
    # ---------- Query Expansion (LLM / fallback) ----------
    def expand_query_with_llm(self, query: str, max_expansions: int = 5) -> List[str]:
        expansions = [query.strip()]
        if not self.llm_client:
            roles = self._extract_roles(query)
            skills = self._extract_skills(query)
            extras = []
            for r in roles[:2]:
                extras.append(f"{r} hiring assessment")
                for s in skills[:2]:
                    extras.append(f"{r} with {s} skills assessment")
            for s in skills[:4]:
                extras.append(f"{query} {s}")
            for e in extras:
                if len(expansions) >= max_expansions:
                    break
                e = e.strip()
                if e and e not in expansions:
                    expansions.append(e)
            i = 0
            while len(expansions) < min(3, max_expansions):
                parap = f"{query} (paraphrase {i+1})"
                expansions.append(parap)
                i += 1
            return expansions[:max_expansions]
        
        try:
            prompt = f"""You are a search-quality assistant. Given this user query, produce between 3 and {max_expansions} concise search queries or paraphrases that preserve user intent but broaden the lexical/technical coverage to improve recall for an information retrieval system.
Original query: "{query}"
if and only if the query is multilingual translate it in english and also try to add more language keywords for better search.
if there is some typo correct it and then expand.
if there is some law compliance mentioning add more keywords for better search results.
if user provide link then open it and search the job description and then expand the query using the collected data.
Return the queries as newline-separated lines only, shortest reasonable phrasing, without numbering or explanation.
"""
            response = self.llm_client.models.generate_content(
                model="gemini-2.5-flash",
                contents=prompt
            )
            text = ""
            if hasattr(response, "text"):
                text = response.text
            elif isinstance(response, dict) and "candidates" in response:
                text = response["candidates"][0].get("content", "")
            else:
                text = str(response)
            lines = [l.strip() for l in re.split(r'[\r\n]+', text) if l.strip()]
            for l in lines:
                if len(expansions) >= max_expansions:
                    break
                if l not in expansions:
                    expansions.append(l)
            i = 0
            while len(expansions) < min(3, max_expansions):
                parap = f"{query} (paraphrase {i+1})"
                expansions.append(parap)
                i += 1
            return expansions[:max_expansions]
        except Exception as e:
            print(f"LLM expansion failed: {e}. Using heuristic expansions.")
            roles = self._extract_roles(query)
            skills = self._extract_skills(query)
            extras = []
            for r in roles[:2]:
                extras.append(f"{r} hiring assessment")
                for s in skills[:2]:
                    extras.append(f"{r} with {s} skills assessment")
            for s in skills[:4]:
                extras.append(f"{query} {s}")
            for e in extras:
                if len(expansions) >= max_expansions:
                    break
                e = e.strip()
                if e and e not in expansions:
                    expansions.append(e)
            i = 0
            while len(expansions) < min(3, max_expansions):
                parap = f"{query} (paraphrase {i+1})"
                expansions.append(parap)
                i += 1
            return expansions[:max_expansions]
    
    # ---------- Dedupe Helper ----------
    def _dedupe_preserve_order(self, candidates: List[Dict]) -> List[Dict]:
        seen = set()
        out = []
        for c in candidates:
            pid = c.get("product_id")
            if pid is None:
                out.append(c)
                continue
            if pid in seen:
                continue
            seen.add(pid)
            out.append(c)
        return out

    # ---------- Vector + BM25 Hybrid Search ----------
    def search_vectors(self, query: str, top_k: int = 20) -> List[Dict[str, Any]]:
        query_embedding = self.model.encode([query], convert_to_numpy=True, show_progress_bar=False)
        faiss.normalize_L2(query_embedding)
        
        short_scores, short_indices = self.short_index.search(query_embedding, top_k)
        chunk_scores, chunk_indices = self.chunks_index.search(query_embedding, top_k * 2)
        
        product_scores = {}
        for score, idx in zip(short_scores[0], short_indices[0]):
            if idx < len(self.prod_map.get("ids", [])):
                product_id = self.prod_map["ids"][idx]
                product_scores.setdefault(product_id, []).append(float(score))
        for score, idx in zip(chunk_scores[0], chunk_indices[0]):
            if idx < len(self.chunks_meta):
                chunk_meta = self.chunks_meta[idx]
                product_id = chunk_meta["product_id"]
                product_scores.setdefault(product_id, []).append(float(score))
        
        vector_candidates = []
        for pid, scores_list in product_scores.items():
            agg_score = float(max(scores_list))
            meta = self.products_meta.get(pid, {})
            vector_candidates.append({
                "product_id": pid,
                "score_vector": agg_score,
                "metadata": meta
            })
        
        bm25_candidates = {}
        use_bm25_strong = self.should_use_bm25(query)
        if self.short_bm25 is not None or self.chunk_bm25 is not None:
            q_tokens = _tokenize_for_bm25(query)
            if self.short_bm25 is not None:
                try:
                    short_bm25_scores = self.short_bm25.get_scores(q_tokens)
                    topn = min(len(short_bm25_scores), top_k * 4)
                    topn_idx = np.argsort(short_bm25_scores)[::-1][:topn]
                    for idx in topn_idx:
                        if idx < len(self.prod_map.get("ids", [])):
                            pid = self.prod_map["ids"][int(idx)]
                            bm25_candidates.setdefault(pid, []).append(float(short_bm25_scores[int(idx)]))
                except Exception as e:
                    print("Warning: short BM25 scoring failed:", e)
            if self.chunk_bm25 is not None:
                try:
                    chunk_bm25_scores = self.chunk_bm25.get_scores(q_tokens)
                    topn = min(len(chunk_bm25_scores), top_k * 8)
                    topn_idx = np.argsort(chunk_bm25_scores)[::-1][:topn]
                    for idx in topn_idx:
                        if idx < len(self.chunks_meta):
                            chunk_meta = self.chunks_meta[int(idx)]
                            pid = chunk_meta["product_id"]
                            bm25_candidates.setdefault(pid, []).append(float(chunk_bm25_scores[int(idx)]))
                except Exception as e:
                    print("Warning: chunk BM25 scoring failed:", e)
        
        bm25_list = []
        for pid, scores_list in bm25_candidates.items():
            bm25_list.append({
                "product_id": pid,
                "score_bm25": float(max(scores_list)),
                "metadata": self.products_meta.get(pid, {})
            })
        
        fused = {}
        vec_scores = [c["score_vector"] for c in vector_candidates] if vector_candidates else []
        bm_scores = [c["score_bm25"] for c in bm25_list] if bm25_list else []
        vec_norm = self._normalize_scores(vec_scores) if vec_scores else np.array([])
        bm_norm = self._normalize_scores(bm_scores) if bm_scores else np.array([])
        pid_to_vec = {c["product_id"]: c for c in vector_candidates}
        pid_to_bm = {c["product_id"]: c for c in bm25_list}
        
        if use_bm25_strong:
            w_bm = 0.7
            w_vec = 0.3
        else:
            w_bm = 0.35
            w_vec = 0.65
        
        vec_pid_list = [c["product_id"] for c in vector_candidates]
        bm_pid_list = [c["product_id"] for c in bm25_list]
        vec_norm_map = {pid: vec_norm[i] for i, pid in enumerate(vec_pid_list)} if vec_scores else {}
        bm_norm_map = {pid: bm_norm[i] for i, pid in enumerate(bm_pid_list)} if bm_scores else {}
        
        all_pids = set(vec_pid_list) | set(bm_pid_list)
        for pid in all_pids:
            vscore = float(vec_norm_map.get(pid, 0.0))
            bscore = float(bm_norm_map.get(pid, 0.0))
            fused_score = w_vec * vscore + w_bm * bscore
            metadata = self.products_meta.get(pid, {})
            fused[pid] = {
                "product_id": pid,
                "score": fused_score,
                "score_vector": vscore,
                "score_bm25": bscore,
                "metadata": metadata
            }
        
        if not fused:
            for c in vector_candidates:
                pid = c["product_id"]
                fused[pid] = {
                    "product_id": pid,
                    "score": c["score_vector"],
                    "score_vector": c["score_vector"],
                    "score_bm25": 0.0,
                    "metadata": c["metadata"]
                }
        
        fused_list = list(fused.values())
        fused_list.sort(key=lambda x: x["score"], reverse=True)
        
        results = []
        for item in fused_list:
            results.append({
                "product_id": item["product_id"],
                "score": float(item["score"]),
                "source": "hybrid",
                "metadata": item.get("metadata", {})
            })
        
        return results[:max(top_k * 2, 50)]
    
    # ---------- Combine + Dedupe ----------
    def _combine_and_deduplicate(self, short_results: List[Dict], chunk_results: List[Dict], top_k: int) -> List[Dict]:
        seen_products = set()
        combined = []
        
        for result in short_results:
            pid = result["product_id"]
            if pid not in seen_products:
                seen_products.add(pid)
                combined.append(result)
        
        for result in chunk_results:
            pid = result["product_id"]
            if pid not in seen_products and len(combined) < top_k * 2:
                seen_products.add(pid)
                combined.append(result)
        
        combined.sort(key=lambda x: x["score"], reverse=True)
        
        return combined[:top_k * 2]
    
    # ---------- LLM Re-rank ----------
    def llm_rerank(self, query: str, candidates: List[Dict], top_k: int = 10) -> List[Dict]:
        if not self.llm_client or len(candidates) == 0:
            return candidates[:top_k]
        
        try:
            duration_constraint = self._extract_duration_constraint(query)
            duration_context = f" Maximum duration: {duration_constraint} minutes." if duration_constraint else ""
            
            candidate_info = []
            for i, candidate in enumerate(candidates[:20]):
                metadata = candidate["metadata"]
                candidate_info.append({
                    "id": i,
                    "name": metadata.get("name", ""),
                    "description": metadata.get("description", "")[:500],
                    "test_type": metadata.get("test_type_names", []),
                    "duration": metadata.get("duration"),
                    "skills": metadata.get("skills", [])
                })
            
            prompt = self._create_enhanced_reranking_prompt(query, candidate_info, top_k, duration_context)
            
            response = self.llm_client.models.generate_content(
                model="gemini-2.5-flash",
                contents=prompt
            )
            
            resp_text = ""
            if hasattr(response, "text"):
                resp_text = response.text
            elif isinstance(response, dict) and "candidates" in response:
                resp_text = response["candidates"][0].get("content", "")
            else:
                resp_text = str(response)
            reranked_indices = self._parse_llm_reranking_response(resp_text, len(candidate_info))
            
            reranked_candidates = []
            for idx in reranked_indices:
                if idx < len(candidates):
                    reranked_candidates.append(candidates[idx])
            
            if not reranked_candidates:
                print("LLM re-ranking returned no valid results, using original order")
                return candidates[:top_k]
            
            return reranked_candidates[:top_k]
            
        except Exception as e:
            print(f"LLM re-ranking failed: {e}. Falling back to hybrid search results.")
            return candidates[:top_k]
    
    # ---------- Rerank Prompt / Parse ----------
    def _create_enhanced_reranking_prompt(self, query: str, candidates: List[Dict], top_k: int, duration_context: str) -> str:
        candidate_str = ""
        for cand in candidates:
            candidate_str += f"ID: {cand['id']}\n"
            candidate_str += f"Name: {cand['name']}\n"
            candidate_str += f"Description: {cand['description'][:300]}...\n" if len(cand['description']) > 300 else f"Description: {cand['description']}\n"
            candidate_str += f"Test Types: {', '.join(cand['test_type'])}\n"
            candidate_str += f"Duration: {cand['duration']} minutes\n" if cand['duration'] else "Duration: Not specified\n"
            candidate_str += "---\n"
        
        prompt = f"""
        You are an expert HR consultant helping to recommend SHL assessments for job roles.
        
        QUERY: "{query}"
        {duration_context}
        
        CANDIDATE ASSESSMENTS:
        {candidate_str}
        
        TASK:
        1. Select the top {top_k} most relevant assessments for the query minimum 5 assessments and max 10
        2. Ensure a balanced mix of test types (behavioral, technical, cognitive, etc.)
        3. Consider both hard skills and soft skills mentioned in the query
        4. Prioritize assessments that directly match the job requirements
        5. Respect duration constraints mentioned in the query
        6. Consider the role level (entry-level, mid-level, senior) if mentioned
        7. Also Most importantly if a technical role is there we need to balance the results with other soft skills.
        8. Other most important thing we need if a test mentions about must adaptive or must remote the results should always adhere to it.
        
        OUTPUT FORMAT:

        Return ONLY a comma-separated list of candidate IDs in order of relevance (most relevant first).
        Example: "3, 1, 5, 2, 0, 4"
        
        Your ranked list:
        """
        
        return prompt
    
    def _parse_llm_reranking_response(self, response: str, max_index: int) -> List[int]:
        numbers = re.findall(r'\d+', response)
        indices = []
        
        for num in numbers:
            idx = int(num)
            if idx < max_index and idx not in indices:
                indices.append(idx)
        
        return indices
    
    # ---------- Balance Test Types ----------
    def balance_test_types(self, candidates: List[Dict], top_k: int = 10) -> List[Dict]:
        if not candidates:
            return candidates
        
        candidates = self._dedupe_preserve_order(candidates)
        
        test_type_groups: Dict[str, List[Dict]] = {}
        for cand in candidates:
            test_types = cand["metadata"].get("test_type_names", []) or ["Unknown"]
            if isinstance(test_types, list) and test_types:
                primary = test_types[0]
            else:
                primary = "Unknown"
            test_type_groups.setdefault(primary, []).append(cand)
        
        for k, group in test_type_groups.items():
            group.sort(key=lambda x: x.get("score", 0), reverse=True)
        
        balanced: List[Dict] = []
        for group in test_type_groups.values():
            if group:
                balanced.append(group.pop(0))
        
        while len(balanced) < top_k:
            added = False
            for group in list(test_type_groups.values()):
                if group and len(balanced) < top_k:
                    balanced.append(group.pop(0))
                    added = True
            if not added:
                break
        
        final = self._dedupe_preserve_order(balanced)[:top_k]
        return final
    
    # ---------- Recommend API ----------
    def recommend(self, query: str, top_k: int = 10, use_reranking: bool = True, 
                  balance_types: bool = True, apply_patterns: bool = True,
                  use_expansion: bool = True) -> List[Dict[str, Any]]:
        print(f"Processing query: '{query}'")
        
        if use_expansion:
            expansions = self.expand_query_with_llm(query, max_expansions=5)
            print(f"Query expansions enabled. Expanded queries: {expansions}")
        else:
            expansions = [query]
            print("Query expansion disabled (--no-expand). Using original query only.")
        decay_weights = [1.0, 0.85, 0.7, 0.55, 0.4][:len(expansions)]
        
        aggregated = {}
        for q_i, qtext in enumerate(expansions):
            weight = decay_weights[q_i]
            try:
                results_i = self.search_vectors(qtext, top_k=20)
            except Exception as e:
                print(f"Warning: search_vectors failed for expansion '{qtext}': {e}")
                results_i = []
            for r in results_i:
                pid = r["product_id"]
                sc = float(r.get("score", 0.0)) * weight
                meta = r.get("metadata", {})
                if pid not in aggregated:
                    aggregated[pid] = {"score": 0.0, "metadata": meta}
                aggregated[pid]["score"] += sc
        
        agg_list = []
        for pid, data in aggregated.items():
            agg_list.append({
                "product_id": pid,
                "score": data["score"],
                "metadata": data["metadata"]
            })
        agg_list.sort(key=lambda x: x["score"], reverse=True)
        
        vector_results = [{"product_id": a["product_id"], "score": a["score"], "metadata": a["metadata"]} for a in agg_list][:max(top_k * 2, 50)]
        print(f"Aggregated candidates after expansion: {len(vector_results)}")
        
        vector_results = self._dedupe_preserve_order(vector_results)
        
        max_duration = self._extract_duration_constraint(query)
        if max_duration:
            print(f"Detected duration constraint: {max_duration} minutes")
        
        if apply_patterns and self.patterns:
            vector_results = self._apply_pattern_based_boost(vector_results, query)
            print("Applied pattern-based boosting")
            vector_results.sort(key=lambda x: x.get("score", 0), reverse=True)
        
        if max_duration:
            before_filter = len(vector_results)
            vector_results = self._filter_by_duration(vector_results, max_duration)
            after_filter = len(vector_results)
            print(f"Duration filtering: {before_filter} -> {after_filter} candidates")
        
        if use_reranking and self.llm_client:
            print("Applying LLM re-ranking...")
            reranked_results = self.llm_rerank(query, vector_results, top_k=top_k)
            print(f"LLM re-ranking returned {len(reranked_results)} candidates")
        else:
            reranked_results = vector_results[:top_k]
            if not use_reranking:
                print("LLM re-ranking disabled, using hybrid search results")
            else:
                print("LLM client not available, using hybrid search results")
        
        reranked_results = self._dedupe_preserve_order(reranked_results)
        print(f"After deduplication: {len(reranked_results)} unique candidates")
        
        if balance_types:
            final_results = self.balance_test_types(reranked_results, top_k=top_k)
            print(f"Final balanced results: {len(final_results)} assessments")
        else:
            final_results = reranked_results[:top_k]
            print(f"Final results: {len(final_results)} assessments")
        
        formatted_results = []
        for result in final_results:
            metadata = result["metadata"]
            description = metadata.get("description", "")
            if len(description) > 500:
                description = description[:500] + "..."
            
            formatted_results.append({
                "url": metadata.get("url", ""),
                "name": metadata.get("name", ""),
                "adaptive_support": metadata.get("adaptive_support", "No"),
                "description": description,
                "duration": metadata.get("duration", 0) or 0,
                "remote_support": metadata.get("remote_support", "No"),
                "test_type": metadata.get("test_type_names", [])
            })
        
        return formatted_results

# ---------- CLI ----------
def main():
    parser = argparse.ArgumentParser(description="Enhanced SHL Assessment Recommendation Engine")
    parser.add_argument("--query", required=True, help="Job description or query")
    parser.add_argument("--top-k", type=int, default=10, help="Number of recommendations (default: 10)")
    parser.add_argument("--vector-store", default="vector_store", help="Vector store directory (default: vector_store)")
    parser.add_argument("--train-data", help="Path to training data Excel file")
    parser.add_argument("--no-rerank", action="store_true", help="Disable LLM re-ranking")
    parser.add_argument("--no-balance", action="store_true", help="Disable test type balancing")
    parser.add_argument("--no-patterns", action="store_true", help="Disable pattern learning")
    parser.add_argument("--no-expand", action="store_true", help="Disable LLM query expansion")
    
    args = parser.parse_args()
    
    try:
        engine = ImprovedSHLQueryEngine(
            vector_store_dir=args.vector_store,
            train_data_path=args.train_data
        )
    except Exception as e:
        print(f"Failed to initialize query engine: {e}")
        return
    
    recommendations = engine.recommend(
        query=args.query,
        top_k=args.top_k,
        use_reranking=not args.no_rerank,
        balance_types=not args.no_balance,
        apply_patterns=not args.no_patterns,
        use_expansion=not args.no_expand,
    )
    
    # print("\n" + "="*80)
    # print("JSON OUTPUT:")
    # print("="*80)
    print(json.dumps({"recommended_assessments": recommendations}, indent=2))

if __name__ == "__main__":
    main()
