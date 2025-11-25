# **SHL Assessment Recommendation System (AI-Powered RAG Pipeline)**

An AI-driven retrieval and recommendation engine that suggests the most relevant **SHL Assessments** based on a job description or hiring query.
The system uses **vector search**, **BM25**, **LLM-powered re-ranking**, **pattern learning from training data**, and a **FastAPI backend** with a lightweight frontend.

---

## 🚀 **Key Features**

### 🔍 **Hybrid Retrieval (Vectors + BM25)**

* FAISS vector search over product metadata & description chunks
* BM25 lexical search for keyword-heavy or technical queries
* Intelligent hybrid fusion with dynamic weighting

### 🧠 **LLM-Powered Enhancements**

* Query expansion (Gemini LLM + heuristics fallback)
* LLM-based re-ranking with HR-aware rules (skills, roles, duration, soft skills balance)
* Multilingual support & typo correction

### 📊 **Pattern Learning from Training Data**

* Learns role–assessment mappings
* Learns skill–assessment mappings
* Extracts duration constraints
* Automatically boosts historically relevant assessments

### ⚙️ **Scraper + Index Builder**

* Async Playwright scraper for SHL catalog
* Cleans & parses 250+ assessment pages
* Builds FAISS + BM25 indices
* Generates chunk embeddings with overlap

### 🌐 **FastAPI Backend**

* `/recommend` and `/api/recommend` endpoints
* JSON response matching required schema
* Health, sample products, and pattern stats endpoints
* Static frontend support

---

## 🛠️ **Tech Stack**

### **Backend**

* Python 3
* FastAPI
* Uvicorn

### **NLP / Retrieval**

* SentenceTransformers (BAAI/bge-base-en-v1.5)
* FAISS (IndexFlatIP + HNSW)
* Rank-BM25
* Numpy, Pandas

### **LLM**

* Google Gemini 2.5 Flash
* google-genai SDK
* Fallback heuristics when LLM unavailable

### **Scraping**

* Playwright (async)
* BeautifulSoup4

### **Other**

* TQDM
* pydantic
* dotenv
* asyncio

---

## 📂 **Project Structure**

```
project/
│── backend.py                 # FastAPI backend
│── improved_query_engine.py   # Core recommendation engine
│── scrape_shl.py              # SHL catalog scraper
│── build_embeddings.py        # Vector + BM25 index builder
│── vector_store/              # FAISS + BM25 indices
│── static/index.html          # Frontend UI
│── data/Gen_AI Dataset.xlsx   # Training dataset
│── README.md
```

---

## 🧩 **How the System Works**

### **1. Scraping & Cleaning**

`playwright` → loads SHL catalog pages → extracts:

* name
* description
* job levels
* languages
* duration
* test types
* adaptive/remote support

Output → `catalog_clean.json`.

---

### **2. Building Vector & BM25 Index**

`build_embeddings.py` performs:

* Short-field embeddings (name + search text)
* Chunking long descriptions with overlap
* Chunk embeddings
* FAISS IndexFlatIP & HNSW index creation
* BM25 indices for both short text & chunks

---

### **3. Query Processing Pipeline**

For each user query:

#### **a. Query Expansion**

LLM → expands into 3–5 variants
Fallback logic → skills + roles-based expansions

#### **b. Hybrid Retrieval**

* Vector search (FAISS)
* BM25 search
* Hybrid fusion using normalized weighted scores

#### **c. Pattern-based Boosting**

Boost relevance using learned patterns from training data.

#### **d. Duration Filtering**

Drop tests exceeding query duration constraint.

#### **e. LLM Re-ranking**

Gemini re-ranks 10–20 candidates considering:

* skills
* soft/technical balance
* duration
* must-have adaptive/remote conditions

#### **f. Test Type Balancing**

Ensures final results contain:

* Technical
* Cognitive
* Behavioral
* Knowledge tests
  …based on role relevance.

#### **g. Final Formatting**

Returns top 5–10 assessments in required JSON schema.

---

## 🔧 **Setup Instructions**

### **1. Install dependencies**

```
pip install -r requirements.txt
```

### **2. Scrape SHL Assessments**

```
python scrape_shl.py --output catalog_clean.json --concurrency 6 --delay 0.25
```

### **3. Build vector store**

```
OMP_NUM_THREADS=1 python build_embeddings.py \
  --input catalog_clean.json \
  --outdir vector_store \
  --chunk-tokens 512 \
  --overlap-tokens 160 \
  --batch 64
```

### **4. Start backend**

```
python backend.py
```

### **5. Access UI**

```
http://localhost:8000
```

### **6. Test recommendation**

```
curl -X POST http://localhost:8000/recommend \
  -H "Content-Type: application/json" \
  -d '{"query":"Java developer with SQL and teamwork, 45 minutes"}'
```

---

## 📘 **API Endpoints**

### **POST /recommend**

Returns top recommended assessments.

### **GET /health**

Basic engine health.

### **GET /api/products/count**

Total products scraped.

### **GET /api/patterns/stats**

Shows learned patterns from training dataset.

### **GET /api/info**

Engine version, features, LLM status.

---

## 📄 **Submission CSV Generator**

Use:

```
python create_submission_csv.py --input queries.txt --output submission.csv
```

Output schema:

```
Query,Assessment_url
Query 1,URL_1
Query 1,URL_2
Query 1,URL_3
...
```

---

## 🤝 **Contributing**

Pull requests are welcome for:

* Improving ranking logic
* Better LLM prompting
* Adding new retrieval signals
* Improving UI

---

## 📜 **License**

This project is licensed under MIT License.

---

If you want, I can also generate:
✅ Architecture diagram
✅ Mermaid flowchart
✅ GIF demo for README
Just tell me!
