"""
Agricultural Intelligence Framework (AAIF)
==========================================
Agentic AI pipeline that transforms heterogeneous agricultural literature and
datasets into the Universal Agricultural Machine Learning Schema (UAMS), trains
yield models, and produces fuzzy fertilizer recommendations.
"""

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("agri-ai-agent")
except PackageNotFoundError:  # not installed (e.g. running from a source checkout)
    __version__ = "2.0.0"

__author__ = "AAIF contributors"
