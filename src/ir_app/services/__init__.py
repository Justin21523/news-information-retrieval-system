"""Service layer for the Flask IR application."""

from .document_service import DocumentService
from .search_service import FeatureUnavailableError, SearchService

__all__ = ["DocumentService", "FeatureUnavailableError", "SearchService"]
