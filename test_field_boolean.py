"""
Test script for field-based Boolean queries
"""

import logging
logging.basicConfig(level=logging.INFO)

from src.ir.index.inverted_index import InvertedIndex
from src.ir.index.positional_index import PositionalIndex
from src.ir.index.field_indexer import FieldIndexer
from src.ir.retrieval.boolean import BooleanQueryEngine
from src.ir.text.chinese_tokenizer import ChineseTokenizer

print("=" * 60)
print("Field-Based Boolean Query Testing")
print("=" * 60)

# Initialize CKIP tokenizer
print("\n1. Initializing CKIP tokenizer...")
tokenizer = ChineseTokenizer(engine='ckip', mode='default', use_pos=False, device=-1)
print("✅ CKIP tokenizer initialized")

# Sample documents with metadata
documents = [
    {
        'title': '台灣經濟發展快速',
        'content': '台灣的經濟發展在過去幾年持續快速成長，科技產業是主要動力。',
        'category': 'economy',
        'category_name': '財經',
        'author': '記者張三',
        'tags': ['台灣', '經濟', '發展'],
        'published_date': '2025-11-10',
        'source': '中央社'
    },
    {
        'title': '美國科技產業分析',
        'content': '美國科技產業在人工智慧領域持續領先全球，投資金額創新高。',
        'category': 'technology',
        'category_name': '科技',
        'author': '記者李四',
        'tags': ['美國', '科技', 'AI'],
        'published_date': '2025-11-11',
        'source': '中央社'
    },
    {
        'title': '台灣政治新聞',
        'content': '台灣政壇最近動態頻繁，各政黨積極備戰下次選舉。',
        'category': 'politics',
        'category_name': '政治',
        'author': '記者王五',
        'tags': ['台灣', '政治'],
        'published_date': '2025-11-12',
        'source': '中央社'
    },
    {
        'title': '科技創新推動經濟',
        'content': '全球科技創新持續推動經濟發展，尤其在AI和半導體領域。',
        'category': 'technology',
        'category_name': '科技',
        'author': '記者趙六',
        'tags': ['科技', '經濟', 'AI'],
        'published_date': '2025-11-13',
        'source': '聯合報'
    }
]

# Extract content for inverted/positional indices
doc_contents = [doc['content'] for doc in documents]

# Build inverted index
print("\n2. Building inverted index...")
inverted_index = InvertedIndex(tokenizer=tokenizer.tokenize)
inverted_index.build(doc_contents)
print(f"✅ Inverted index built: {len(inverted_index.vocabulary)} terms")

# Build positional index
print("\n3. Building positional index...")
positional_index = PositionalIndex(tokenizer=tokenizer.tokenize)
positional_index.build(doc_contents)
print(f"✅ Positional index built")

# Build field indexer
print("\n4. Building field indexer...")
field_indexer = FieldIndexer(tokenizer=tokenizer.tokenize)
field_indexer.build(documents)
stats = field_indexer.get_stats()
print(f"✅ Field indexer built: {stats['total_fields']} fields, {stats['total_terms']} terms")

# Create Boolean query engine
print("\n5. Creating Boolean query engine...")
boolean_engine = BooleanQueryEngine(
    inverted_index=inverted_index,
    positional_index=positional_index,
    field_indexer=field_indexer
)
print("✅ Boolean query engine created")

# Test queries
print("\n" + "=" * 60)
print("Testing Field-Based Queries")
print("=" * 60)

test_queries = [
    # Field queries
    ("title:台灣", "Search for '台灣' in title field"),
    ("category:politics", "Search for politics category"),
    ("source:中央社", "Search for 中央社 source"),

    # Combined queries
    ("title:台灣 AND category:politics", "Taiwan news in politics category"),
    ("category:technology OR category:economy", "Technology or Economy articles"),
    ("tags:AI AND source:中央社", "AI tags from CNA"),

    # Date range queries
    ("published_date:[2025-11-10 TO 2025-11-11]", "Date range Nov 10-11"),
    ("published_date:[2025-11-12 TO 2025-11-13]", "Date range Nov 12-13"),

    # Complex queries
    ("(title:台灣 OR title:美國) AND category:technology", "Taiwan or US in tech category"),
    ("category:economy AND published_date:[2025-11-10 TO 2025-11-13]", "Economy news in date range"),
]

for idx, (query, description) in enumerate(test_queries, 1):
    print(f"\nTest {idx}: {description}")
    print(f"Query: {query}")
    try:
        result = boolean_engine.query(query)
        print(f"✅ Results: {result.doc_ids} ({result.num_results} docs)")

        # Show matching titles
        if result.doc_ids:
            print(f"   Titles:")
            for doc_id in result.doc_ids:
                print(f"   - [{doc_id}] {documents[doc_id]['title']}")
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()

# Test regular content search (backward compatibility)
print("\n" + "=" * 60)
print("Testing Regular Content Search (Backward Compatibility)")
print("=" * 60)

regular_queries = [
    ("台灣", "Search for '台灣' in content"),
    ("科技 AND 經濟", "Search for '科技' and '經濟'"),
    ("台灣 OR 美國", "Search for '台灣' or '美國'"),
]

for idx, (query, description) in enumerate(regular_queries, 1):
    print(f"\nTest {idx}: {description}")
    print(f"Query: {query}")
    try:
        result = boolean_engine.query(query)
        print(f"✅ Results: {result.doc_ids} ({result.num_results} docs)")

        if result.doc_ids:
            print(f"   Titles:")
            for doc_id in result.doc_ids:
                print(f"   - [{doc_id}] {documents[doc_id]['title']}")
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()

print("\n" + "=" * 60)
print("✅ All tests completed!")
print("=" * 60)
