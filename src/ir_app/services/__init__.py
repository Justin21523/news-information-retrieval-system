"""Service layer for the Flask IR application."""

from .document_service import DocumentService
from .cluster_topic_service import ClusterTopicService
from .corpus_audit_service import CorpusAuditService
from .document_detail_service import DocumentDetailService
from .evaluation_cache_service import EvaluationCacheService
from .evaluation_job_service import EvaluationJobService
from .evaluation_service import EvaluationService
from .feedback_analytics_service import FeedbackAnalyticsService
from .feedback_service import FeedbackService
from .learning_to_rank_feature_service import LearningToRankFeatureService
from .learning_to_rank_training_service import LearningToRankTrainingService
from .ranking_diagnostics_service import RankingDiagnosticsService
from .retrieval_orchestrator import RetrievalOrchestrator
from .search_log_service import SearchLogService
from .search_service import FeatureUnavailableError, SearchService

__all__ = [
    "CorpusAuditService",
    "ClusterTopicService",
    "DocumentDetailService",
    "DocumentService",
    "EvaluationCacheService",
    "EvaluationJobService",
    "EvaluationService",
    "FeedbackAnalyticsService",
    "FeedbackService",
    "FeatureUnavailableError",
    "LearningToRankFeatureService",
    "LearningToRankTrainingService",
    "RankingDiagnosticsService",
    "RetrievalOrchestrator",
    "SearchLogService",
    "SearchService",
]
