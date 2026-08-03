"""Copernicus CDS connectors (re-exported from ``agri_ai_agent.literature.agri.cds``).

Kept in the connectors package so the sources registry can discover them
uniformly.  The ``cdsapi`` package is optional; these connectors disable
themselves when it is not installed.
"""

from agri_ai_agent.literature.agri.cds import AgERA5Connector, CDSConnector, ERA5Connector

# Aliases used by the sources registry.
CdsConnector = CDSConnector
Era5Connector = ERA5Connector
AgEra5Connector = AgERA5Connector

__all__ = [
    "CDSConnector",
    "ERA5Connector",
    "AgERA5Connector",
    "CdsConnector",
    "Era5Connector",
    "AgEra5Connector",
]
