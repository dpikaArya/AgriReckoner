"""Agricultural data connector implementations (official sources)."""

from agri_ai_agent.literature.agri.connectors.cgiar_hf import CgiarHuggingFaceConnector
from agri_ai_agent.literature.agri.connectors.chirps import ChirpsConnector
from agri_ai_agent.literature.agri.connectors.dataverse_agri import (
    CimmytConnector,
    HarvestChoiceConnector,
    IcrisatConnector,
    IfpriConnector,
    IrriConnector,
)
from agri_ai_agent.literature.agri.connectors.faostat import FaostatConnector
from agri_ai_agent.literature.agri.connectors.gbif import GbifConnector
from agri_ai_agent.literature.agri.connectors.gee import GoogleEarthEngineConnector
from agri_ai_agent.literature.agri.connectors.geoglam import GeoglamConnector
from agri_ai_agent.literature.agri.connectors.gyga import GygaConnector
from agri_ai_agent.literature.agri.connectors.mapspam import MapSpamConnector
from agri_ai_agent.literature.agri.connectors.nass import NassConnector
from agri_ai_agent.literature.agri.connectors.power import NasaPowerConnector
from agri_ai_agent.literature.agri.connectors.soilgrids import SoilGridsConnector
from agri_ai_agent.literature.agri.connectors.worldbank import WorldBankConnector

__all__ = [
    "CgiarHuggingFaceConnector",
    "ChirpsConnector",
    "CimmytConnector",
    "FaostatConnector",
    "GbifConnector",
    "GeoglamConnector",
    "GoogleEarthEngineConnector",
    "GygaConnector",
    "HarvestChoiceConnector",
    "IcrisatConnector",
    "IfpriConnector",
    "IrriConnector",
    "MapSpamConnector",
    "NassConnector",
    "NasaPowerConnector",
    "SoilGridsConnector",
    "WorldBankConnector",
]
