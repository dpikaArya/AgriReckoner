"""Literature connectors.

Each module implements one or more :class:`LiteratureConnector` subclasses
against the provider's official API.  Modules are discovered dynamically by
:mod:`agri_ai_agent.literature.registry`.
"""

from agri_ai_agent.literature.connectors.agris import AgrisConnector
from agri_ai_agent.literature.connectors.arxiv import ArxivConnector
from agri_ai_agent.literature.connectors.crossref import CrossrefConnector
from agri_ai_agent.literature.connectors.datacite import DataCiteConnector
from agri_ai_agent.literature.connectors.dimensions import DimensionsConnector
from agri_ai_agent.literature.connectors.doaj import DoajConnector
from agri_ai_agent.literature.connectors.europepmc import EuropePmcConnector
from agri_ai_agent.literature.connectors.figshare import FigshareConnector
from agri_ai_agent.literature.connectors.mendeley import MendeleyDataConnector
from agri_ai_agent.literature.connectors.openalex import OpenAlexConnector
from agri_ai_agent.literature.connectors.pubmed import PubMedConnector
from agri_ai_agent.literature.connectors.scopus import ScopusConnector
from agri_ai_agent.literature.connectors.semanticscholar import SemanticScholarConnector
from agri_ai_agent.literature.connectors.wos import WosConnector
from agri_ai_agent.literature.connectors.zenodo import ZenodoConnector

__all__ = [
    "AgrisConnector",
    "ArxivConnector",
    "CrossrefConnector",
    "DataCiteConnector",
    "DimensionsConnector",
    "DoajConnector",
    "EuropePmcConnector",
    "FigshareConnector",
    "MendeleyDataConnector",
    "OpenAlexConnector",
    "PubMedConnector",
    "ScopusConnector",
    "SemanticScholarConnector",
    "WosConnector",
    "ZenodoConnector",
]
