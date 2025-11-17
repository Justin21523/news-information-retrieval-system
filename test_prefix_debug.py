#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Debug script for PAT-tree prefix search."""

from src.ir.index.pat_tree import PatriciaTree

# Create a simple tree
tree = PatriciaTree()

# Add some test terms
test_terms = ["台灣", "台北", "台中", "台南", "高雄", "中國", "中山", "美國"]

for term in test_terms:
    tree.insert(term, doc_id=0)

print("=== Tree Statistics ===")
stats = tree.get_statistics()
for key, value in stats.items():
    print(f"{key}: {value}")

print("\n=== Test Prefix Search ===")
test_prefixes = ["台", "台灣", "中", "美"]

for prefix in test_prefixes:
    print(f"\nSearching for prefix: '{prefix}'")
    results = tree.starts_with(prefix)
    print(f"Found {len(results)} matches:")
    for term, node in results[:5]:
        print(f"  - {term} (freq: {node.frequency})")

print("\n=== Test Visualization ===")
viz_result = tree.visualize_tree(max_nodes=20, prefix="台")
print(f"Success: {viz_result.get('success', True)}")
print(f"Prefix: {viz_result['prefix']}")
print(f"Tree is None: {viz_result['tree'] is None}")
if viz_result['tree']:
    print(f"Children count: {len(viz_result['tree'].get('children', []))}")
    for child in viz_result['tree'].get('children', [])[:5]:
        print(f"  - {child['label']}")
