# Information Retrieval System

A comprehensive Information Retrieval (IR) system implementation for **LIS5033 - Automatic Classification and Indexing** at National Taiwan University. This project implements traditional IR techniques based on the textbook "Introduction to Information Retrieval" by Manning, Raghavan, and Schütze.

> **完整中文文檔**: See [docs/README.md](docs/README.md) for complete documentation in Traditional Chinese.

## Portfolio News Search Demo

This repository now includes a Flask-based **News Information Retrieval System** for searching a unified Taiwanese news corpus. The demo focuses on explainable IR rather than a black-box search box:

- Unified search across BM25, TF-IDF, Boolean, Hybrid RRF, LM, fuzzy, and CSoundex modes.
- Faceted metadata filtering by source, category, taxonomy, content type, date, and tags.
- Search result explanations with matched terms, field boosts, component scores, snippets, and highlights.
- Document detail enrichment with summary, KWIC, keywords, taxonomy metadata, and related documents.
- Model comparison for BM25 / TF-IDF / Hybrid / LM on the same query.
- Demo evaluation dashboard with Precision@K, Recall@K, MAP, MRR, nDCG, PR curves, per-query breakdown, and clearly labeled small demo qrels.

### Demo Flow

```bash
pip install -r requirements.txt
IR_ENABLE_HEAVY_MODELS=false python app.py
```

Open `http://localhost:5001` and try:

1. Search `半導體` with BM25 or Hybrid.
2. Open a document detail modal and inspect Summary, Keywords, KWIC, and Related News.
3. Visit `/compare` and compare `人工智慧` across BM25 / TF-IDF / Hybrid / LM.
4. Visit `/evaluation`, choose `News Demo Qrels`, and run the evaluation dashboard.

### Demo Media

The UI verification script writes screenshots and a short browser recording to `docs/assets/evaluation/`:

```bash
python scripts/verify_ui_playwright.py
```

Artifacts:

- [Search Results](docs/assets/evaluation/search-results.png)
- [Document Detail](docs/assets/evaluation/document-detail.png)
- [Model Comparison](docs/assets/evaluation/model-compare.png)
- [Evaluation Dashboard](docs/assets/evaluation/evaluation-dashboard.png)
- [Demo Video](docs/assets/evaluation/cnirs-demo.webm)

### Runtime Configuration

| Variable | Default | Purpose |
| --- | --- | --- |
| `IR_DATASET_PATH` | first available unified/CNA corpus | Active JSONL corpus |
| `IR_FALLBACK_DATASET_PATH` | `datasets/mini/ir_documents.json` | Small fallback fixture |
| `IR_INDEX_DIR` | `data/indexes` | Persistent lexical index cache |
| `IR_MAX_DOCUMENTS` | `25000` | Startup corpus cap; empty or `0` means no cap |
| `IR_TOKENIZER_ENGINE` | `jieba` | Lightweight tokenizer engine |
| `IR_ENABLE_HEAVY_MODELS` | `false` | Keeps CKIP/BERT-style optional models disabled on low-resource hosts |
| `IR_HOST` / `IR_PORT` | `0.0.0.0` / `5001` | Flask bind address |

Heavy semantic features such as BERT, KeyBERT, BERTopic, CKIP transformers, sentence-transformers, and FAISS are treated as optional. Missing optional dependencies should return structured unavailable responses instead of crashing startup.

## Features

### Core Modules

- **M1: Boolean Retrieval** - Inverted index with AND/OR/NOT operators, phrase queries
- **M2: CSoundex** - Chinese phonetic encoding for fuzzy matching and tolerant retrieval
- **M3: Vector Space Model** - TF-IDF weighting with cosine similarity ranking
- **M4: Evaluation Metrics** - Precision, Recall, MAP, nDCG
- **M5: Query Expansion** - Rocchio algorithm with pseudo-relevance feedback
- **M6: Clustering** - K-means, hierarchical clustering (single-link, complete-link)
- **M7: Summarization** - Lead-k, key sentence extraction, KWIC (KeyWord In Context)

### Special Features

- **CSoundex Encoding**: Novel Chinese phonetic encoding system for:
  - Homophone matching (同音字搜尋)
  - Variant character tolerance (異體字處理)
  - Fuzzy name search with configurable similarity thresholds

- **Multilingual Support**: Handles mixed Chinese/English text with punctuation normalization

## Quick Start

### Prerequisites

- Python 3.10+
- Conda environment (recommended: `ai_env`)

### Installation

```bash
# Clone the repository
git clone <repository-url>
cd information-retrieval

# Activate conda environment
conda activate ai_env

# Install dependencies
pip install -r requirements.txt

# Verify installation
pytest tests/ -v
```

### Basic Usage

#### CSoundex Encoding

```bash
# Encode Chinese text
python scripts/csoundex_encode.py --text "三聚氰胺"

# Encode from file
python scripts/csoundex_encode.py --file input.txt --output encoded.txt
```

#### Boolean Search

```bash
# Simple query
python scripts/boolean_search.py --query "information AND retrieval"

# Phrase query
python scripts/boolean_search.py --query "\"information retrieval\"" --index data/index.pkl
```

#### Vector Space Model Search

```bash
# TF-IDF ranking
python scripts/vsm_search.py --query "machine learning" --topk 10
```

#### Evaluation

```bash
# Run evaluation metrics
python scripts/eval_run.py --results results.json --qrels qrels.txt --metrics MAP,nDCG,P@10
```

## Project Structure

```
.
├── src/                           # Source code modules
│   └── ir/                        # IR core implementations
│       ├── text/                  # Text processing (CSoundex)
│       ├── index/                 # Indexing (inverted, positional)
│       ├── retrieval/             # Retrieval models (Boolean, VSM)
│       ├── eval/                  # Evaluation metrics
│       ├── ranking/               # Ranking algorithms (Rocchio)
│       ├── cluster/               # Clustering algorithms
│       └── summarize/             # Summarization
├── scripts/                       # CLI tools
├── tests/                         # Pytest unit tests
├── configs/                       # YAML configuration files
├── datasets/                      # Sample datasets
│   ├── mini/                      # Small test datasets
│   └── lexicon/                   # Pinyin dictionaries
├── docs/                          # Documentation (Chinese)
│   ├── exams/                     # Exam materials
│   ├── hw/                        # Homework reports
│   ├── project/                   # Final project docs
│   └── guides/                    # Implementation guides
└── logs/                          # Application logs
```

## Documentation

- **[docs/README.md](docs/README.md)** - Complete project documentation in Traditional Chinese
- **[docs/CHANGELOG.md](docs/CHANGELOG.md)** - Version history and change log
- **[docs/guides/CSOUNDEX_DESIGN.md](docs/guides/CSOUNDEX_DESIGN.md)** - CSoundex technical design document
- **[docs/guides/IMPLEMENTATION.md](docs/guides/IMPLEMENTATION.md)** - Implementation guides for all modules
- **[docs/PROJECT_ROADMAP.md](docs/PROJECT_ROADMAP.md)** - Development roadmap

## Testing

```bash
# Run all tests
pytest tests/

# Run specific module tests
pytest tests/test_csoundex.py -v

# Run with coverage report
pytest tests/ --cov=src/ir --cov-report=html

# Open coverage report
# Open htmlcov/index.html in browser

# Run only unit tests
pytest tests/ -m unit

# Run excluding slow tests
pytest tests/ -m "not slow"
```

## Development

### Code Quality

```bash
# Format code with Black
black src/ tests/ scripts/

# Type checking with mypy
mypy src/

# Linting with pylint
pylint src/ir/

# Code style with flake8
flake8 src/ tests/
```

### Configuration

- **configs/csoundex.yaml** - CSoundex phonetic grouping rules
- **configs/logging.yaml** - Logging configuration
- **pytest.ini** - Testing framework configuration

## API Examples

### CSoundex API

```python
from ir.text.csoundex import CSoundex

encoder = CSoundex()

# Basic encoding
code = encoder.encode("張")  # Returns: "Z811"

# Batch encoding
codes = encoder.encode_batch(["張", "章", "彰"])  # All return "Z811"

# Similarity calculation
sim = encoder.similarity("張三", "章三")  # Returns: 0.75
```

### Inverted Index API

```python
from ir.index.inverted_index import InvertedIndex

index = InvertedIndex()
index.build(documents)

# Boolean query
results = index.query("information AND retrieval")

# Phrase query
results = index.phrase_query("information retrieval")
```

### Vector Space Model API

```python
from ir.retrieval.vsm import VectorSpaceModel

vsm = VectorSpaceModel()
vsm.build_index(documents)

# Search with TF-IDF
results = vsm.search("machine learning", topk=10)
```

## Performance

- **Indexing**: O(T) where T is total number of tokens
- **Boolean Query**: O(k) where k is result size
- **VSM Search**: O(V + k log k) where V is vocabulary size
- **CSoundex Encoding**: O(1) per character with LRU cache

## Roadmap

- [x] **v0.1.0** - Project infrastructure setup
- [ ] **v0.2.0** - CSoundex module implementation
- [ ] **v0.3.0** - Boolean retrieval system
- [ ] **v0.4.0** - Vector space model
- [ ] **v0.5.0** - Evaluation metrics
- [ ] **v0.6.0** - Query expansion (Rocchio)
- [ ] **v0.7.0** - Clustering algorithms
- [ ] **v0.8.0** - Summarization
- [ ] **v1.0.0** - Final project integration

See [docs/PROJECT_ROADMAP.md](docs/PROJECT_ROADMAP.md) for detailed timeline.

## License

This project is for educational purposes as part of LIS5033 coursework at National Taiwan University.

## References

- Manning, C. D., Raghavan, P., & Schütze, H. (2008). *Introduction to Information Retrieval*. Cambridge University Press.
- Course materials from LIS5033 - Automatic Classification and Indexing

## Contact

For questions about this project, please refer to the course materials or contact the teaching staff at National Taiwan University.

---

**Note**: This is an academic project. For production use, consider using established IR frameworks like Apache Lucene, Elasticsearch, or Whoosh.
# news-information-retrieval-system
# information-retrieval
# information-retrieval
