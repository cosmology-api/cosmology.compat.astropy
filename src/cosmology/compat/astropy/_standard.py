"""The Cosmology API compatability wrapper for Astropy."""

from __future__ import annotations

from dataclasses import dataclass

from astropy.cosmology import FLRW  # noqa: TCH002

from cosmology.compat.astropy._components import (
    HasBaryonComponent,
    HasDarkEnergyComponent,
    HasDarkMatterComponent,
    HasGlobalCurvatureComponent,
    HasMatterComponent,
    HasNeutrinoComponent,
    HasPhotonComponent,
    HasTotalComponent,
)
from cosmology.compat.astropy._core import CosmologyWrapper
from cosmology.compat.astropy._distances import HasDistanceMeasures
from cosmology.compat.astropy._extras import HasCriticalDensity, HasHubbleParameter

__all__: list[str] = []


################################################################################


@dataclass(frozen=True)
class StandardCosmologyWrapper(
    HasNeutrinoComponent,
    HasBaryonComponent,
    HasPhotonComponent,
    HasDarkMatterComponent,
    HasMatterComponent,
    HasDarkEnergyComponent,
    HasGlobalCurvatureComponent,
    HasTotalComponent,
    HasHubbleParameter,
    HasCriticalDensity,
    HasDistanceMeasures,
    CosmologyWrapper,
):
    """The Cosmology API wrapper for :mod:`astropy.cosmology.Cosmology`."""

    cosmo: FLRW
