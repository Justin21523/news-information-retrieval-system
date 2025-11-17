# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

---

## Project Overview

This repository contains implementations for **LIS5033 - Automatic Classification and Indexing**, an Information Retrieval course at National Taiwan University. The project implements traditional IR techniques based on the textbook "Introduction to Information Retrieval" including indexing, retrieval models, evaluation metrics, query expansion, clustering, and summarization.

**Reference Textbook**: *Introduction to Information Retrieval* (Manning, Raghavan, Schütze)

---

## Critical Development Policies

### File Management Policy (STRICTLY ENFORCE)
- **NEVER** create new documentation files for similar content - always UPDATE existing files
- **NEVER** keep old versions when updating (no `file_v2.md`, `file_copy.md`, `file_final.md`)
- **NEVER** create temporary files in `tmp/` or `playground/` directories
- When refactoring or renaming files, **DELETE** the old file and document changes in `docs/CHANGELOG.md`
- Same-topic content MUST be consolidated into a single, well-structured document

### Language Policy
- **Documentation & Explanations**: Traditional Chinese (繁體中文)
- **Technical Terms**: Provide bilingual format (中文 *English Term*)
- **Code, Comments, Docstrings**: English only
- **Commit Messages**: English only (Conventional Commits format)
- **File Names**: English only

### Code Documentation Standards
- Every function/class must have detailed English docstrings
- Include complexity analysis (Time: O(?), Space: O(?))
- Inline comments for non-trivial logic
- Code must be immediately executable with clear examples

---

## Project Structure

```
.
├── docs/                          # Documentation (Chinese with EN terms)
│   ├── exams/                     # Exam materials
│   │   └── midterm/              # Midterm exam preparation
│   │       ├── OUTLINE.md        # Exam outline
│   │       ├── DRAFT.md          # Full draft answers
│   │       └── FIGS/             # Diagrams (SVG format)
│   ├── hw/                        # Homework reports (fixed naming)
│   ├── project/                   # Final project documentation
│   │   ├── PROPOSAL.md
│   │   └── REPORT.md
│   ├── guides/                    # Implementation guides
│   ├── CHANGELOG.md               # Change log (update on every change)
│   └── README.md                  # Project overview (Chinese)
├── src/                           # Source code modules
│   └── ir/                        # IR core implementations
│       ├── text/                  # Text processing
│       │   └── csoundex.py       # Chinese Soundex encoding
│       ├── index/                 # Indexing (inverted index, positional)
│       ├── retrieval/             # Retrieval models (Boolean, VSM)
│       ├── eval/                  # Evaluation metrics
│       │   └── metrics.py        # Precision, Recall, MAP, nDCG
│       ├── ranking/               # Ranking algorithms
│       │   └── rocchio.py        # Rocchio & Query Expansion
│       ├── cluster/               # Clustering algorithms
│       │   ├── term_cluster.py
│       │   └── doc_cluster.py
│       └── summarize/             # Summarization
│           ├── static.py          # Lead-k, key sentence extraction
│           └── dynamic.py         # KWIC with caching
├── scripts/                       # CLI tools (argparse)
│   ├── csoundex_encode.py        # CSoundex CLI
│   ├── eval_run.py               # Run evaluation metrics
│   ├── expand_query.py           # Query expansion
│   └── format_to_docx.py         # Export reports to DOCX
├── tests/                         # pytest unit tests
│   └── test_csoundex.py
├── configs/                       # YAML configuration files
│   └── csoundex.yaml             # Pinyin grouping rules
├── datasets/                      # Sample datasets
│   ├── mini/                      # Small test datasets
│   └── lexicon/
│       └── basic_pinyin.tsv      # Basic pinyin dictionary
├── requirements.txt               # Python dependencies
└── README.md                      # English brief overview
```

---

## Development Workflow

### 1. Before Implementation
- Read relevant docs in `docs/guides/` and `docs/hw/`
- Check configuration files in `configs/`
- Review existing module structure in `src/`

### 2. During Implementation
- Write code in `src/` with comprehensive English comments
- Create CLI tools in `scripts/` with `--help` support
- Add tests in `tests/` (minimum 3 cases: normal, boundary, edge)
- Use small datasets from `datasets/mini/` for testing

### 3. After Implementation
- **Update (don't create new)** the relevant documentation in `docs/`
- Update `docs/CHANGELOG.md` with changes
- Ensure all tests pass: `pytest tests/`
- Verify CLI tools work: `python scripts/<tool>.py --help`

---

## Common Commands

### Testing
```bash
# Run all tests
pytest tests/

# Run specific test file
pytest tests/test_csoundex.py -v

# Run with coverage
pytest tests/ --cov=src/ir --cov-report=html
```

### CSoundex (Chinese Soundex Encoding)
```bash
# Encode text
python scripts/csoundex_encode.py --text "三聚氰胺"

# Encode from file
python scripts/csoundex_encode.py --file input.txt --output encoded.txt
```

### Evaluation Metrics
```bash
# Run evaluation on results
python scripts/eval_run.py --results results.json --qrels qrels.txt --metrics MAP,nDCG,P@10
```

### Query Expansion (Rocchio)
```bash
# Pseudo-relevance feedback
python scripts/expand_query.py --query "information retrieval" --mode pseudo --topk 10

# Explicit relevance feedback
python scripts/expand_query.py --query "IR" --mode explicit --relevant rel_docs.txt
```

---

## Key Implementation Modules

### M1: Boolean Retrieval (布林檢索 *Boolean Retrieval*)
- **Location**: `src/ir/index/inverted_index.py`, `src/ir/retrieval/boolean.py`
- **Features**: Inverted index, AND/OR/NOT operators, phrase queries
- **CLI**: `scripts/boolean_search.py`

### M2: Dictionary & Tolerant Retrieval (字典與容錯檢索)
- **Location**: `src/ir/text/`, `src/ir/index/dictionary.py`
- **Features**: Wildcard queries, spell correction, edit distance
- **CSoundex**: Chinese phonetic encoding for fuzzy matching

### M3: TF-IDF & Vector Space Model (向量空間模型)
- **Location**: `src/ir/retrieval/vsm.py`
- **Features**: TF-IDF weighting, cosine similarity, document ranking
- **Complexity**: O(T) indexing, O(k log k) for top-k retrieval

### M4: Evaluation (評估指標 *Evaluation Metrics*)
- **Location**: `src/ir/eval/metrics.py`
- **Metrics**:
  - Precision, Recall, F-measure
  - Average Precision (AP), Mean Average Precision (MAP)
  - Normalized Discounted Cumulative Gain (nDCG)

### M5: Query Expansion (查詢擴展 *Query Expansion*)
- **Location**: `src/ir/ranking/rocchio.py`
- **Techniques**:
  - Rocchio Algorithm
  - Pseudo-Relevance Feedback
  - Explicit Relevance Feedback

### M6: Clustering (分群 *Clustering*)
- **Location**: `src/ir/cluster/`
- **Algorithms**:
  - Hierarchical: Complete-link, Single-link
  - Flat: K-means, Star clustering
  - String clustering for terms

### M7: Summarization (摘要 *Summarization*)
- **Location**: `src/ir/summarize/`
- **Types**:
  - Static: Lead-k, key sentence extraction
  - Dynamic: KWIC (KeyWord In Context) with windowing

---

## Report & Exam Guidelines

### Midterm Exam Format
- **Location**: `docs/exams/midterm/`
- **Required Files**:
  - `OUTLINE.md`: Structured outline with key points
  - `DRAFT.md`: Full answers (Chinese, bilingual terms)
  - `FIGS/`: Diagrams in SVG format
- **Structure**: Use 4-paragraph format (觀點→例證→反例→小結)
- **Topics**:
  - Embedded search (Facebook/Blog) vs General search engines
  - Searching vs Browsing (information access strategies)
  - Self-defined topics from course modules

### Report Template (docs/hw/HWxx/REPORT.md)
```markdown
# [Assignment Title - English / 中文]

## 1. Problem & Goal (題目與目標)
[Describe the task with bilingual technical terms]

## 2. Theoretical Background (理論背景)
[Key concepts, equations, algorithms]

## 3. Method Design (方法設計)
[Architecture, data flow diagrams]

## 4. Implementation Details (實作細節)
- **Modules**: [List modules and APIs]
- **Complexity**: Time O(?), Space O(?)
- **Parameters**: [Configuration details]

## 5. Experimental Design (實驗設計)
- **Dataset**: [Description, size, format]
- **Metrics**: [Evaluation criteria]
- **Baselines**: [Comparison methods]

## 6. Results & Analysis (結果與分析)
[Tables, charts (SVG), error analysis]

## 7. Limitations & Future Work (限制與未來工作)
[Discuss shortcomings and improvements]

## 8. Reproduction Steps (如何重現)
\`\`\`bash
# Step-by-step commands
\`\`\`
```

### Final Project Requirements
- **Title**: Build a small academic search engine (or chosen domain)
- **Core Features**:
  - Full-text indexing with positional index
  - VSM ranking (TF-IDF + cosine similarity)
  - Boolean + phrase + wildcard queries
  - Field-based search (title/author/year)
  - Faceted browsing or clustering
- **Deliverables**:
  - `docs/project/PROPOSAL.md`
  - `docs/project/REPORT.md` (follow report template)
  - `scripts/run_demo.sh` (executable demo)

---

## CSoundex Specification

### Purpose
Chinese phonetic encoding for fuzzy matching (similar to English Soundex), supporting tolerance for:
- Homophone matching (同音字)
- Variant characters (異形字)
- Mixed punctuation and English

### Algorithm Design
1. Convert Chinese characters to Pinyin romanization
2. Normalize: remove tones, lowercase
3. Group consonants by phonetic similarity:
   - 0 = vowels (a/e/i/o/u)
   - 1 = b/p, 2 = f, 3 = m
   - 4 = d/t, 5 = n/l, 6 = g/k/h
   - 7 = j/q/x, 8 = zh/ch/sh/r, 9 = z/c/s
4. Output format: `[First_Letter][3 digits]` (e.g., "Z800" for 張/章/彰)

### Configuration
- **File**: `configs/csoundex.yaml`
- **No external downloads**: Use embedded lexicon at `datasets/lexicon/basic_pinyin.tsv`

### Testing Requirements
- Homophone grouping (張/章 → Z800)
- Variant handling (裡/裏)
- Punctuation tolerance
- Mixed English/Chinese text

---

## Code Quality Standards

### Docstring Template
```python
def build_inverted_index(docs: list[str]) -> dict:
    """
    Build an inverted index from a list of documents.

    Args:
        docs: A list of raw text documents.

    Returns:
        A dictionary mapping term -> list of (doc_id, term_freq).

    Complexity:
        Time: O(T) where T is the total number of tokens.
        Space: O(U + P) where U is unique terms, P is postings size.

    Example:
        >>> docs = ["hello world", "world peace"]
        >>> index = build_inverted_index(docs)
        >>> index["world"]
        [(0, 1), (1, 1)]
    """
    pass
```

### CLI Standards
- Use `argparse` with comprehensive `--help`
- Required args: `--input`, `--output`
- Optional args: `--topk`, `--mode`, `--threshold`
- Example:
  ```bash
  python scripts/tool.py --help
  # Should display clear usage instructions
  ```

### Testing Standards
```python
def test_csoundex_homophone():
    """Test that homophones map to same code."""
    assert encode("張") == encode("章") == "Z800"

def test_csoundex_boundary():
    """Test edge cases like empty string, single char."""
    assert encode("") == ""
    assert len(encode("我")) == 4

def test_csoundex_mixed():
    """Test mixed Chinese/English/punctuation."""
    text = "三聚氰胺(melamine)是化學物"
    result = encode(text)
    assert "S" in result  # 三 -> S
```

---

## Document Export & Formatting

### Export to DOCX/PDF
```bash
# Using pandoc (if available)
pandoc docs/exams/midterm/DRAFT.md -o ACI2025MidTerm<ID>.docx --reference-doc=template.docx
pandoc ACI2025MidTerm<ID>.docx -o ACI2025MidTerm<ID>.pdf

# Alternative: Use scripts/format_to_docx.py
python scripts/format_to_docx.py --input docs/exams/midterm/DRAFT.md --output ACI2025MidTerm<ID>.docx
```

### Format Requirements (from 期末報告格式.docx)
- Margins: 2.54 cm (all sides)
- Line spacing: 1.5
- Font sizes: Title (larger), headings, body text
- Figures: Centered with captions below
- Tables: Centered with captions above
- References: Listed at end

---

## Course Context: Information Access

### Key Concepts (from lecture slides)

**Information Retrieval vs Data Retrieval**:
- IR: Partial match, probabilistic, natural language, incomplete query
- DR: Exact match, deterministic, artificial language