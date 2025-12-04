#!/usr/bin/env python3
"""VSM Search CLI Tool"""

import sys
import argparse
import json
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.ir.retrieval.vsm import VectorSpaceModel


def load_documents(filepath):
    """Load documents from JSON file."""
    with open(filepath, 'r', encoding='utf-8') as f:
        data = json.load(f)
    documents = [item['text'] for item in data]
    metadata = [{k: v for k, v in item.items() if k != 'text'} for item in data]
    return documents, metadata


def build_index(input_file, output_file):
    """Build VSM index."""
    print(f"Loading documents from {input_file}...")
    documents, metadata = load_documents(input_file)
    
    vsm = VectorSpaceModel()
    vsm.build_index(documents, metadata)
    
    print(f"Index built: {len(documents)} documents")
    print(f"Vocabulary: {len(vsm.inverted_index.vocabulary)} terms")
    
    # Save index
    vsm.inverted_index.save(output_file.replace('.json', '_inv.json'))
    print(f"Index saved to {output_file}")


def search(query, index_file, topk):
    """Execute VSM search."""
    from src.ir.index.inverted_index import InvertedIndex
    
    # Load index
    inv_index = InvertedIndex()
    inv_index.load(index_file.replace('.json', '_inv.json'))
    
    # Build VSM
    vsm = VectorSpaceModel(inv_index)
    vsm.term_weighting.build_from_index(inv_index)
    vsm._compute_document_vectors()
    
    # Search
    result = vsm.search(query, topk=topk)
    
    print(f"Query: {query}")
    print(f"Found {result.num_results} results\n")
    
    for i, doc_id in enumerate(result.doc_ids, 1):
        score = result.scores[doc_id]
        print(f"{i}. Document {doc_id} (score: {score:.4f})")
        if doc_id in inv_index.doc_metadata:
            meta = inv_index.doc_metadata[doc_id]
            if 'title' in meta:
                print(f"   Title: {meta['title']}")


def main():
    parser = argparse.ArgumentParser(description='VSM Search')
    parser.add_argument('--build', action='store_true')
    parser.add_argument('--search', type=str)
    parser.add_argument('--input', type=str)
    parser.add_argument('--index', type=str, required=True)
    parser.add_argument('--topk', type=int, default=10)
    
    args = parser.parse_args()
    
    if args.build:
        build_index(args.input, args.index)
    elif args.search:
        search(args.search, args.index, args.topk)


if __name__ == '__main__':
    main()
