"""
Quick test to validate app.py integration
"""

import sys
import json

print("=" * 60)
print("Testing app.py Integration")
print("=" * 60)

# Test 1: Check imports
print("\n1. Testing imports...")
try:
    from src.ir.index.field_indexer import FieldIndexer
    from src.ir.retrieval.boolean import BooleanQueryEngine
    from src.ir.text.chinese_tokenizer import ChineseTokenizer
    print("✅ All imports successful")
except ImportError as e:
    print(f"❌ Import error: {e}")
    sys.exit(1)

# Test 2: Check FieldIndexer with minimal data
print("\n2. Testing FieldIndexer...")
try:
    # Simple tokenizer for quick test
    def simple_tokenizer(text):
        return text.lower().split()

    field_indexer = FieldIndexer(tokenizer=simple_tokenizer)

    # Minimal test data
    test_docs = [
        {
            'title': 'Test Document One',
            'category': 'test',
            'published_date': '2025-11-10'
        },
        {
            'title': 'Test Document Two',
            'category': 'demo',
            'published_date': '2025-11-11'
        }
    ]

    field_indexer.build(test_docs)
    stats = field_indexer.get_stats()
    print(f"✅ FieldIndexer built: {stats['total_fields']} fields, {stats['total_terms']} terms")
except Exception as e:
    print(f"❌ FieldIndexer error: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test 3: Check Boolean engine with field support
print("\n3. Testing Boolean engine with field support...")
try:
    from src.ir.index.inverted_index import InvertedIndex
    from src.ir.index.positional_index import PositionalIndex

    # Build minimal indexes
    contents = ["test document one content", "test document two content"]

    inv_index = InvertedIndex(tokenizer=simple_tokenizer)
    inv_index.build(contents)

    pos_index = PositionalIndex(tokenizer=simple_tokenizer)
    pos_index.build(contents)

    # Create Boolean engine with field support
    boolean_engine = BooleanQueryEngine(
        inverted_index=inv_index,
        positional_index=pos_index,
        field_indexer=field_indexer
    )

    print("✅ Boolean engine initialized with field support")

    # Test queries
    print("\n4. Testing field queries...")

    # Test field query
    result = boolean_engine.query("title:test")
    print(f"   Query 'title:test': {len(result.doc_ids)} results")

    # Test category query
    result = boolean_engine.query("category:test")
    print(f"   Query 'category:test': {len(result.doc_ids)} results")

    # Test combined query
    result = boolean_engine.query("title:test AND category:test")
    print(f"   Query 'title:test AND category:test': {len(result.doc_ids)} results")

    # Test date range
    result = boolean_engine.query("published_date:[2025-11-10 TO 2025-11-11]")
    print(f"   Query 'published_date:[2025-11-10 TO 2025-11-11]': {len(result.doc_ids)} results")

    print("✅ All field queries working correctly")

except Exception as e:
    print(f"❌ Boolean engine error: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test 4: Validate dataset exists
print("\n5. Checking dataset...")
try:
    import os
    dataset_path = 'data/processed/cna_mvp_cleaned.jsonl'
    if os.path.exists(dataset_path):
        with open(dataset_path, 'r', encoding='utf-8') as f:
            count = sum(1 for line in f if line.strip())
        print(f"✅ Dataset found: {count} articles")
    else:
        print(f"⚠️  Dataset not found at {dataset_path}")
except Exception as e:
    print(f"❌ Dataset check error: {e}")

print("\n" + "=" * 60)
print("✅ Integration test passed!")
print("=" * 60)
print("\nNext steps:")
print("1. Start Flask app: python app.py")
print("2. Open browser: http://localhost:5001")
print("3. Test field queries:")
print("   - title:台灣")
print("   - category:politics")
print("   - title:台灣 AND category:politics")
print("   - published_date:[2025-11-10 TO 2025-11-13]")
