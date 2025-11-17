"""
Syntax Module for Information Retrieval System

This module provides syntactic parsing and analysis tools for Chinese text,
including dependency parsing and SVO (Subject-Verb-Object) extraction.

Components:
    - DependencyParser: Dependency parsing using SuPar
    - SVOExtractor: Subject-Verb-Object triple extraction
    - SyntaxAnalyzer: High-level syntax analysis interface

Author: Information Retrieval System
License: Educational Use
"""

from .parser import DependencyParser, SVOExtractor, SyntaxAnalyzer

__all__ = [
    'DependencyParser',
    'SVOExtractor',
    'SyntaxAnalyzer'
]
