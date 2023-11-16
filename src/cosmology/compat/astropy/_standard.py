"""The Cosmology API compatability wrapper for Astropy."""

from __future__ import annotations

from dataclasses import dataclass

from astropy.cosmology import FLRW  # noqa: TCH002

from cosmology.compat.astropy._components import (
    BaryonComponent,
    CurvatureComponent,
    DarkEnergyComponent,
    DarkMatterComponent,
    MatterComponent,
    NeutrinoComponent,
    PhotonComponent,
    TotalComponent,
)
from cosmology.compat.astropy._core import CosmologyWrapper
from cosmology.compat.astropy._distances import DistanceMeasures
from cosmology.compat.astropy._extras import CriticalDensity, HubbleParameter

__all__: list[str] = []


################################################################################


@dataclass(frozen=True)
class StandardCosmologyWrapper(
    CosmologyWrapper,
    # Mix in the components
    NeutrinoComponent,
    BaryonComponent,
    PhotonComponent,
    DarkMatterComponent,
    MatterComponent,
    DarkEnergyComponent,
    CurvatureComponent,
    TotalComponent,
    HubbleParameter,
    CriticalDensity,
    DistanceMeasures,
):
    """The Cosmology API wrapper for :class:`~astropy.cosmology.FLRW`."""

    cosmo: FLRW
    """The underlying :class:`~astropy.cosmology.FLRW` instance."""
