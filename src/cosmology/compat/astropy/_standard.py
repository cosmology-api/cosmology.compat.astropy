"""The Cosmology API compatability wrapper for Astropy."""

from __future__ import annotations

from dataclasses import dataclass

from astropy.cosmology import FLRW  # noqa: TCH002

from cosmology.compat.astropy._components import (
    AstropyBaryonComponent,
    AstropyCurvatureComponent,
    AstropyDarkEnergyComponent,
    AstropyDarkMatterComponent,
    AstropyMatterComponent,
    AstropyNeutrinoComponent,
    AstropyPhotonComponent,
    AstropyTotalComponent,
)
from cosmology.compat.astropy._core import AstropyCosmology
from cosmology.compat.astropy._distances import AstropyDistanceMeasures
from cosmology.compat.astropy._extras import (
    AstropyCriticalDensity,
    AstropyHubbleParameter,
)

__all__: list[str] = []


################################################################################


@dataclass(frozen=True)
class AstropyStandardCosmology(
    AstropyNeutrinoComponent,
    AstropyBaryonComponent,
    AstropyPhotonComponent,
    AstropyDarkMatterComponent,
    AstropyMatterComponent,
    AstropyDarkEnergyComponent,
    AstropyCurvatureComponent,
    AstropyTotalComponent,
    AstropyHubbleParameter,
    AstropyCriticalDensity,
    AstropyDistanceMeasures,
    AstropyCosmology,
):
    """The Cosmology API wrapper for :mod:`astropy.cosmology.Cosmology`."""

    cosmo: FLRW
