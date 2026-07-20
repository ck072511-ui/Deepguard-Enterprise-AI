"""
DeepGuard — datasets/__init__.py

Top-level public API for the datasets package.
Provides the DatasetFactory as the primary entry point for consumers.
"""

from datasets.downloader import DatasetDownloader
from datasets.factory import DatasetFactory
from datasets.splitter import StratifiedSplitter, SubjectAwareSplitter
from datasets.statistics import DatasetStatistics
from datasets.validator import DatasetValidator
from datasets.versioning import DatasetVersioner

__all__ = [
    "DatasetFactory",
    "DatasetDownloader",
    "DatasetValidator",
    "DatasetStatistics",
    "DatasetVersioner",
    "StratifiedSplitter",
    "SubjectAwareSplitter",
]
