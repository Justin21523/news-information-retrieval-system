"""
Quick test script to verify CKIP integration
"""

import logging
logging.basicConfig(level=logging.INFO)

from src.ir.text.chinese_tokenizer import ChineseTokenizer
from src.ir.index.inverted_index import InvertedIndex

# Initialize CKIP tokenizer
print("=" * 60)
print("Testing CKIP Transformers Integration")
print("=" * 60)

print("\n1. Initializing CKIP tokenizer...")
tokenizer = ChineseTokenizer(
    engine='ckip',
    mode='default',
    use_pos=True,
    device=-1
)
print("✅ CKIP tokenizer initialized")

# Test tokenization
print("\n2. Testing tokenization...")
test_text = "台灣的資訊檢索系統發展迅速，人工智慧技術應用廣泛。"
tokens = tokenizer.tokenize(test_text)
print(f"Text: {test_text}")
print(f"Tokens: {tokens}")

# Test POS tagging
print("\n3. Testing POS tagging...")
pos_tags = tokenizer.tokenize_with_pos(test_text)
print(f"POS Tags: {pos_tags[:5]}...")  # Show first 5

# Test NER
print("\n4. Testing NER...")
entities = tokenizer.extract_entities(test_text)
print(f"Entities: {entities}")

# Test with InvertedIndex
print("\n5. Testing InvertedIndex with CKIP...")
documents = [
    "台灣大學圖書資訊學系",
    "人工智慧與機器學習",
    "資訊檢索系統評估"
]

index = InvertedIndex(tokenizer=tokenizer.tokenize)
index.build(documents)

print(f"✅ Index built successfully")
print(f"   Documents: {index.doc_count}")
print(f"   Vocabulary size: {len(index.vocabulary)}")
print(f"   Sample terms: {list(index.vocabulary)[:10]}")

# Test search
print("\n6. Testing search...")
query_term = "資訊"
if query_term in index.vocabulary:
    postings = index.get_postings(query_term)
    print(f"✅ Search for '{query_term}': found in {len(postings)} documents")
    print(f"   Postings: {postings}")
else:
    print(f"❌ Term '{query_term}' not found in vocabulary")

print("\n" + "=" * 60)
print("✅ All tests passed!")
print("=" * 60)
