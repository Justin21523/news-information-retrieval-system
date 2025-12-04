"""
Database Module

PostgreSQL database integration for storing and managing news articles.

Key Features:
    - PostgreSQL connection management
    - News article schema and CRUD operations
    - JSONL bulk import
    - Full-text search support
    - Deduplication with content hashing

Author: Information Retrieval System
License: Educational Use
"""

from .postgres_manager import PostgresManager
from .jsonl_importer import JSONLImporter

__all__ = ['PostgresManager', 'JSONLImporter']
