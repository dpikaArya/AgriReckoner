"""Institutional agricultural Dataverse connectors.

Subclasses of :class:`DataverseConnector` for the official CGIAR/IFPRI
repositories.  ICRISAT and CIMMYT dataverses require an API key; without one
those connectors are disabled (probe-verified: HTTP 401).
"""

from __future__ import annotations

from agri_ai_agent.literature.agri.dataverse import DataverseConnector
from agri_ai_agent.literature.config import ConnectorAuth


class IfpriConnector(DataverseConnector):
    source_name = "IFPRI"
    display_name = "IFPRI Dataverse"
    instance_url = "https://dataverse.harvard.edu"
    subtree = "IFPRI"

    auth = ConnectorAuth(env_vars=("DATAVERSE_API_KEY", "AGRI_DATAVERSE_API_KEY"), required=False)


class HarvestChoiceConnector(DataverseConnector):
    source_name = "HarvestChoice"
    display_name = "IFPRI HarvestChoice"
    instance_url = "https://dataverse.harvard.edu"
    subtree = "HarvestChoice"

    auth = ConnectorAuth(env_vars=("DATAVERSE_API_KEY", "AGRI_DATAVERSE_API_KEY"), required=False)


class IcrisatConnector(DataverseConnector):
    source_name = "ICRISAT"
    display_name = "ICRISAT Dataverse"
    instance_url = "https://dataverse.icrisat.org"
    subtree = None

    auth = ConnectorAuth(env_vars=("DATAVERSE_API_KEY", "AGRI_DATAVERSE_API_KEY"), required=True)


class IrriConnector(DataverseConnector):
    source_name = "IRRI"
    display_name = "IRRI Dataverse"
    instance_url = "https://dataverse.irri.org"
    subtree = None

    auth = ConnectorAuth(env_vars=("DATAVERSE_API_KEY", "AGRI_DATAVERSE_API_KEY"), required=False)


class CimmytConnector(DataverseConnector):
    source_name = "CIMMYT"
    display_name = "CIMMYT Dataverse"
    instance_url = "https://dataverse.cimmyt.org"
    subtree = None

    auth = ConnectorAuth(env_vars=("DATAVERSE_API_KEY", "AGRI_DATAVERSE_API_KEY"), required=True)


class MapSpamDataverseConnector(DataverseConnector):
    """MapSPAM data on the IFPRI dataverse (the canonical distributed copy)."""

    source_name = "MapSPAM_Dataverse"
    display_name = "MapSPAM (IFPRI dataverse)"
    instance_url = "https://dataverse.harvard.edu"
    subtree = "spam"

    auth = ConnectorAuth(env_vars=("DATAVERSE_API_KEY", "AGRI_DATAVERSE_API_KEY"), required=False)
