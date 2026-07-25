"""UAMS ontology package: registry loader/validator and namespace helpers.

The machine-readable registry lives in spec/uams_ontology.yaml and is the single
source of ontology truth for the framework.
"""

from agri_ai_agent.ontology.namespace import UAMS_BASE, uams_base, uams_term
from agri_ai_agent.ontology.registry import (
    DEFAULT_REGISTRY_PATH,
    Registry,
    load_registry,
)

__all__ = [
    "UAMS_BASE",
    "uams_base",
    "uams_term",
    "Registry",
    "load_registry",
    "DEFAULT_REGISTRY_PATH",
]
