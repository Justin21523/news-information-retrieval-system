"""
PAT-tree (Patricia Trie) Implementation for Keyword Extraction and Visualization.

A Patricia Trie is a space-optimized trie where each node with only one child
is merged with its parent. This implementation includes:
- Complete Patricia Trie data structure
- Keyword extraction from documents
- Post-processing (filtering, ranking, scoring)
- Statistics and visualization support

Time Complexity: O(k) for insertion and search, where k is key length
Space Complexity: O(ALPHABET_SIZE * N * M) where N is number of keys, M is max key length
"""

from typing import List, Dict, Tuple, Optional, Set
from dataclasses import dataclass
from collections import defaultdict
import math


@dataclass
class PatNode:
    """
    Node in Patricia Trie.

    Attributes:
        label: Edge label (can be multi-character for compressed paths)
        children: Dictionary of child nodes
        is_terminal: Whether this node marks end of a key
        frequency: Number of occurrences of this term
        doc_ids: Set of document IDs containing this term
        metadata: Additional information (position, context, etc.)
    """
    label: str = ""
    children: Dict[str, 'PatNode'] = None
    is_terminal: bool = False
    frequency: int = 0
    doc_ids: Set[str] = None
    metadata: Dict = None

    def __post_init__(self) -> None:
        """Initialize mutable default fields safely."""
        if self.children is None:
            self.children = {}
        if self.doc_ids is None:
            self.doc_ids = set()
        if self.metadata is None:
            self.metadata = {}


class PatriciaTree:
    """
    Patricia Trie (Practical Algorithm To Retrieve Information Coded In Alphanumeric).

    Features:
    - Space-efficient compressed trie
    - Support for keyword extraction
    - Document frequency tracking
    - Post-processing and ranking
    """

    def __init__(self):
        """Initialize an empty Patricia Trie."""
        self.root = PatNode(label="ROOT")
        self.total_terms = 0
        self.unique_terms = 0
        self.term_stats = {}  # term -> {freq, doc_freq, tfidf, etc.}

    def insert(self, key: str, doc_id: str = None, metadata: Dict = None) -> None:
        """
        Insert a key into the Patricia Trie.

        Args:
            key: The term to insert
            doc_id: Optional document ID containing this term
            metadata: Optional metadata (position, context, etc.)

        Complexity: O(k) where k is length of key
        """
        if not key:
            return

        self._insert_helper(self.root, key, doc_id, metadata)
        self.total_terms += 1

        # Update term statistics
        if key not in self.term_stats:
            self.term_stats[key] = {
                'frequency': 0,
                'doc_frequency': 0,
                'doc_ids': set()
            }
            self.unique_terms += 1

        self.term_stats[key]['frequency'] += 1
        if doc_id:
            self.term_stats[key]['doc_ids'].add(doc_id)
            self.term_stats[key]['doc_frequency'] = len(self.term_stats[key]['doc_ids'])

    def _insert_helper(self, node: PatNode, key: str, doc_id: str, metadata: Dict) -> None:
        """
        Recursive helper for insertion with path compression.

        Args:
            node: Current node
            key: Remaining key to insert
            doc_id: Document ID
            metadata: Metadata
        """
        if not key:
            node.is_terminal = True
            node.frequency += 1
            if doc_id:
                node.doc_ids.add(doc_id)
            if metadata:
                if 'occurrences' not in node.metadata:
                    node.metadata['occurrences'] = []
                node.metadata['occurrences'].append(metadata)
            return

        # Find matching child
        first_char = key[0]

        # Check if there's a child starting with this character
        matching_child = None
        for child_label, child_node in node.children.items():
            if child_label and child_label[0] == first_char:
                matching_child = child_label
                break

        if matching_child is None:
            # No matching child, create new node with full remaining key
            new_node = PatNode(label=key)
            node.children[key] = new_node
            self._insert_helper(new_node, "", doc_id, metadata)
        else:
            # Found matching child, find common prefix
            child_node = node.children[matching_child]
            common_prefix_len = self._common_prefix_length(matching_child, key)

            if common_prefix_len == len(matching_child):
                # Child label is prefix of key, recurse
                remaining_key = key[common_prefix_len:]
                self._insert_helper(child_node, remaining_key, doc_id, metadata)
            elif common_prefix_len == len(key):
                # Key is prefix of child label, split child
                common_prefix = key[:common_prefix_len]
                remaining_child_label = matching_child[common_prefix_len:]

                # Create new intermediate node
                new_node = PatNode(label=common_prefix)
                new_node.is_terminal = True
                new_node.frequency = 1
                if doc_id:
                    new_node.doc_ids.add(doc_id)
                if metadata:
                    new_node.metadata['occurrences'] = [metadata]

                # Update child label
                child_node.label = remaining_child_label

                # Rewire connections
                del node.children[matching_child]
                node.children[common_prefix] = new_node
                new_node.children[remaining_child_label] = child_node
            else:
                # Partial match, split both
                common_prefix = key[:common_prefix_len]
                remaining_key = key[common_prefix_len:]
                remaining_child_label = matching_child[common_prefix_len:]

                # Create intermediate node
                intermediate_node = PatNode(label=common_prefix)

                # Update existing child
                child_node.label = remaining_child_label

                # Create new leaf for remaining key
                new_leaf = PatNode(label=remaining_key)

                # Rewire connections
                del node.children[matching_child]
                node.children[common_prefix] = intermediate_node
                intermediate_node.children[remaining_child_label] = child_node
                intermediate_node.children[remaining_key] = new_leaf

                # Mark new leaf as terminal
                self._insert_helper(new_leaf, "", doc_id, metadata)

    def _common_prefix_length(self, s1: str, s2: str) -> int:
        """Find length of common prefix between two strings."""
        min_len = min(len(s1), len(s2))
        for i in range(min_len):
            if s1[i] != s2[i]:
                return i
        return min_len

    def search(self, key: str) -> Optional[PatNode]:
        """
        Search for a key in the Patricia Trie.

        Args:
            key: The term to search for

        Returns:
            PatNode if found and is terminal, None otherwise

        Complexity: O(k) where k is length of key
        """
        node = self._search_helper(self.root, key)
        return node if node and node.is_terminal else None

    def _search_helper(self, node: PatNode, key: str) -> Optional[PatNode]:
        """Recursive helper for search."""
        if not key:
            return node

        first_char = key[0]

        for child_label, child_node in node.children.items():
            if child_label and child_label[0] == first_char:
                if key.startswith(child_label):
                    remaining_key = key[len(child_label):]
                    return self._search_helper(child_node, remaining_key)
                else:
                    return None

        return None

    def starts_with(self, prefix: str) -> List[Tuple[str, PatNode]]:
        """
        Find all keys that start with given prefix.

        Handles compressed edges - if prefix ends in the middle of an edge,
        we still collect all descendants of that edge.

        Args:
            prefix: The prefix to search for

        Returns:
            List of (key, node) tuples

        Complexity: O(k + m) where k is prefix length, m is number of matches
        """
        results = []
        self._starts_with_helper(self.root, "", prefix, results)
        return results

    def _starts_with_helper(self, node: PatNode, current_key: str,
                           prefix: str, results: List) -> None:
        """
        Helper for prefix search that handles compressed edges.

        Args:
            node: Current node
            current_key: The key accumulated so far
            prefix: The prefix we're searching for
            results: List to accumulate (key, node) tuples
        """
        # If we've matched the entire prefix, collect all descendants
        if not prefix:
            # If current node is terminal, include it
            if node.is_terminal:
                results.append((current_key, node))
            # Collect all descendants
            self._collect_all_keys(node, current_key, results)
            return

        # Try to match prefix with child edges
        for child_label, child_node in node.children.items():
            if not child_label:
                continue

            # Check if prefix starts with this edge label (or vice versa)
            common_len = self._common_prefix_length(child_label, prefix)

            if common_len == 0:
                # No match with this edge
                continue

            new_key = current_key + child_label

            if common_len == len(prefix):
                # Prefix is fully consumed within or at end of this edge
                # Collect this node and all descendants
                if child_node.is_terminal:
                    results.append((new_key, child_node))
                self._collect_all_keys(child_node, new_key, results)
            elif common_len == len(child_label):
                # Edge label is fully consumed, continue with remaining prefix
                remaining_prefix = prefix[common_len:]
                self._starts_with_helper(child_node, new_key, remaining_prefix, results)

    def _collect_all_keys(self, node: PatNode, current_key: str, results: List) -> None:
        """Recursively collect all keys from a node."""
        for child_label, child_node in node.children.items():
            full_key = current_key + child_label
            if child_node.is_terminal:
                results.append((full_key, child_node))
            self._collect_all_keys(child_node, full_key, results)

    def extract_keywords(self, top_k: int = 20, min_freq: int = 2,
                        min_doc_freq: int = 1, method: str = 'tfidf') -> List[Dict]:
        """
        Extract keywords from the Patricia Trie with post-processing.

        Args:
            top_k: Number of top keywords to return
            min_freq: Minimum term frequency threshold
            min_doc_freq: Minimum document frequency threshold
            method: Scoring method ('tfidf', 'frequency', 'doc_frequency', 'combined')

        Returns:
            List of keyword dictionaries with scores and metadata

        Post-processing steps:
        1. Filter by minimum frequency and document frequency
        2. Calculate scores based on selected method
        3. Rank and select top-k
        4. Enrich with additional statistics
        """
        candidates = []

        # Step 1: Filter terms by thresholds
        for term, stats in self.term_stats.items():
            freq = stats['frequency']
            doc_freq = stats['doc_frequency']

            if freq >= min_freq and doc_freq >= min_doc_freq:
                candidates.append({
                    'term': term,
                    'frequency': freq,
                    'doc_frequency': doc_freq,
                    'doc_ids': stats['doc_ids']
                })

        if not candidates:
            return []

        # Step 2: Calculate scores
        total_docs = len(set().union(*[c['doc_ids'] for c in candidates]))

        for candidate in candidates:
            freq = candidate['frequency']
            doc_freq = candidate['doc_frequency']

            # TF component (normalized)
            tf = freq / self.total_terms if self.total_terms > 0 else 0

            # IDF component
            idf = math.log((total_docs + 1) / (doc_freq + 1)) + 1

            # Different scoring methods
            if method == 'tfidf':
                score = tf * idf
            elif method == 'frequency':
                score = freq
            elif method == 'doc_frequency':
                score = doc_freq
            elif method == 'combined':
                # Combine multiple signals
                score = (tf * idf) * (1 + math.log(doc_freq + 1))
            else:
                score = tf * idf

            candidate['score'] = score
            candidate['tf'] = tf
            candidate['idf'] = idf

        # Step 3: Rank and select top-k
        ranked = sorted(candidates, key=lambda x: x['score'], reverse=True)
        top_keywords = ranked[:top_k]

        # Step 4: Enrich with additional statistics
        for kw in top_keywords:
            kw['rank'] = ranked.index(kw) + 1
            kw['percentile'] = (1 - kw['rank'] / len(ranked)) * 100
            # Remove doc_ids from output (too large)
            kw['doc_count'] = len(kw['doc_ids'])
            del kw['doc_ids']

        return top_keywords

    def get_statistics(self) -> Dict:
        """
        Get comprehensive statistics about the Patricia Trie.

        Returns:
            Dictionary with various statistics
        """
        node_count = self._count_nodes(self.root)
        max_depth = self._max_depth(self.root)

        # Calculate compression ratio
        # (nodes in uncompressed trie) / (nodes in Patricia trie)
        total_chars = sum(len(term) for term in self.term_stats.keys())
        compression_ratio = total_chars / node_count if node_count > 0 else 0

        return {
            'total_terms': self.total_terms,
            'unique_terms': self.unique_terms,
            'total_nodes': node_count,
            'max_depth': max_depth,
            'compression_ratio': compression_ratio,
            'avg_term_frequency': self.total_terms / self.unique_terms if self.unique_terms > 0 else 0
        }

    def _count_nodes(self, node: PatNode) -> int:
        """Count total number of nodes in trie."""
        count = 1
        for child in node.children.values():
            count += self._count_nodes(child)
        return count

    def _max_depth(self, node: PatNode, current_depth: int = 0) -> int:
        """Find maximum depth of trie."""
        if not node.children:
            return current_depth
        return max(self._max_depth(child, current_depth + 1)
                  for child in node.children.values())

    def visualize_tree(self, max_nodes: int = 100, prefix: str = "") -> Dict:
        """
        Generate tree structure for visualization.

        Args:
            max_nodes: Maximum number of nodes to include
            prefix: Optional prefix to filter terms

        Returns:
            Dictionary representing tree structure for frontend visualization
        """
        if prefix:
            # For prefix search, use starts_with to find all matching terms
            # Then build a filtered view of the tree
            matches = self.starts_with(prefix)

            if not matches:
                return {
                    'tree': None,
                    'statistics': self.get_statistics(),
                    'prefix': prefix,
                    'error': 'No terms found with this prefix'
                }

            # Build a simplified tree structure from matches
            tree_data = self._build_prefix_tree(matches, prefix, max_nodes)
        else:
            # Use mutable counter for proper node counting across recursion
            counter = {'count': 0}
            tree_data = self._build_tree_data(self.root, "", max_nodes, counter)

        return {
            'tree': tree_data,
            'statistics': self.get_statistics(),
            'prefix': prefix
        }

    def _build_prefix_tree(self, matches: List[Tuple[str, PatNode]],
                          prefix: str, max_nodes: int) -> Dict:
        """
        Build a simplified tree structure from prefix search matches.

        Args:
            matches: List of (term, node) tuples from starts_with()
            prefix: The search prefix
            max_nodes: Maximum nodes to include

        Returns:
            Dictionary representing filtered tree structure
        """
        # Create a virtual root for the prefix results
        root_data = {
            'label': 'ROOT',
            'key': '',
            'terminal': False,
            'frequency': 0,
            'doc_count': 0,
            'children': []
        }

        count = 0
        for term, node in matches[:max_nodes]:
            count += 1
            if count > max_nodes:
                break

            # Create a leaf node for each match
            child_data = {
                'label': term,
                'key': term,
                'terminal': node.is_terminal,
                'frequency': node.frequency,
                'doc_count': len(node.doc_ids),
                'children': []
            }
            root_data['children'].append(child_data)

        return root_data

    def _find_prefix_node(self, node: PatNode, prefix: str) -> Optional[PatNode]:
        """
        Find the node that best matches the given prefix.

        Unlike _search_helper which requires exact match, this returns
        the deepest node along the prefix path. If the prefix ends in the
        middle of an edge, returns the parent node (since we can't split edges
        for visualization).

        Args:
            node: Starting node
            prefix: Prefix to find

        Returns:
            PatNode at or before the end of prefix path, or None if not found
        """
        if not prefix:
            return node

        first_char = prefix[0]

        for child_label, child_node in node.children.items():
            if child_label and child_label[0] == first_char:
                # Check how much of child_label matches prefix
                common_len = self._common_prefix_length(child_label, prefix)

                if common_len == 0:
                    # No match despite first char matching (shouldn't happen)
                    continue

                if common_len == len(child_label):
                    # Child label is fully consumed, continue with remaining prefix
                    remaining_prefix = prefix[common_len:]
                    return self._find_prefix_node(child_node, remaining_prefix)
                elif common_len == len(prefix):
                    # Prefix is fully consumed within this edge
                    # Since the prefix matches, we should show this subtree
                    # Return the child node (not current node) to show all terms with this prefix
                    return child_node
                else:
                    # Partial match but neither fully consumed
                    # This means prefix doesn't correspond to any path in tree
                    return None

        return None

    def _build_tree_data(self, node: PatNode, current_key: str,
                        max_nodes: int, counter: Dict) -> Optional[Dict]:
        """
        Recursively build tree data for visualization.

        Args:
            node: Current node
            current_key: Accumulated key string
            max_nodes: Maximum nodes to include
            counter: Mutable dictionary with 'count' key for tracking nodes

        Returns:
            Dictionary with 'label', 'key', 'terminal', 'frequency', 'children'
            or None if max_nodes reached
        """
        if counter['count'] >= max_nodes:
            return None

        node_data = {
            'label': node.label,
            'key': current_key,
            'terminal': node.is_terminal,
            'frequency': node.frequency,
            'doc_count': len(node.doc_ids),
            'children': []
        }

        counter['count'] += 1

        for child_label, child_node in sorted(node.children.items()):
            if counter['count'] >= max_nodes:
                break

            child_key = current_key + child_label
            child_data = self._build_tree_data(child_node, child_key, max_nodes, counter)

            if child_data:
                node_data['children'].append(child_data)

        return node_data


def build_pat_tree_from_documents(documents: List[Dict], tokenizer) -> PatriciaTree:
    """
    Build a Patricia Trie from a collection of documents.

    Args:
        documents: List of document dictionaries with 'doc_id' and 'content'
        tokenizer: Tokenizer instance with tokenize() method

    Returns:
        Populated PatriciaTree instance

    Example:
        >>> from src.ir.text.chinese_tokenizer import ChineseTokenizer
        >>> docs = [
        ...     {'doc_id': 'doc1', 'content': '台灣是一個美麗的島嶼'},
        ...     {'doc_id': 'doc2', 'content': '台北是台灣的首都'}
        ... ]
        >>> tokenizer = ChineseTokenizer()
        >>> tree = build_pat_tree_from_documents(docs, tokenizer)
        >>> keywords = tree.extract_keywords(top_k=5)
    """
    tree = PatriciaTree()

    for doc in documents:
        doc_id = doc.get('doc_id', '')
        content = doc.get('content', '')

        if not content:
            continue

        tokens = tokenizer.tokenize(content)

        for position, token in enumerate(tokens):
            metadata = {
                'position': position,
                'doc_id': doc_id
            }
            tree.insert(token, doc_id=doc_id, metadata=metadata)

    return tree
