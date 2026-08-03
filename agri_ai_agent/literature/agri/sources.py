"""Registry of all agricultural data connector classes.

The :class:`AgriculturalDataManager` discovers connector classes by scanning
this module's namespace, so adding a new connector here makes it available to
the pipeline.
"""

from agri_ai_agent.literature.agri.connectors import (
    CgiarHuggingFaceConnector,
    ChirpsConnector,
    CimmytConnector,
    FaostatConnector,
    GbifConnector,
    GeoglamConnector,
    GoogleEarthEngineConnector,
    GygaConnector,
    HarvestChoiceConnector,
    IcrisatConnector,
    IfpriConnector,
    IrriConnector,
    MapSpamConnector,
    NasaPowerConnector,
    NassConnector,
    SoilGridsConnector,
    WorldBankConnector,
)
from agri_ai_agent.literature.agri.connectors.dataverse_agri import MapSpamDataverseConnector

# CDS-based connectors (Copernicus) — imported lazily so the pipeline does not
# require cdsapi when they are disabled.
try:
    from agri_ai_agent.literature.agri.connectors.cds_agri import (
        AgEra5Connector,
        CdsConnector,
        Era5Connector,
    )
except Exception:  # pragma: no cover - optional dependency guard
    CdsConnector = None
    Era5Connector = None
    AgEra5Connector = None

__all__ = [
    "CdsConnector",
    "Era5Connector",
    "AgEra5Connector",
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
    "MapSpamDataverseConnector",
    "NassConnector",
    "NasaPowerConnector",
    "SoilGridsConnector",
    "WorldBankConnector",
]
