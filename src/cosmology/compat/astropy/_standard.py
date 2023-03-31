"""The Cosmology API compatability wrapper for Astropy."""

from __future__ import annotations

from dataclasses import dataclass

from astropy.cosmology import FLRW  # noqa: TCH002

from cosmology.compat.astropy._components import (
    AstropyHasBaryonComponent,
    AstropyHasDarkEnergyComponent,
    AstropyHasDarkMatterComponent,
    AstropyHasGlobalCurvatureComponent,
    AstropyHasMatterComponent,
    AstropyHasNeutrinoComponent,
    AstropyHasPhotonComponent,
    AstropyHasTotalComponent,
)
from cosmology.compat.astropy._core import AstropyCosmology
from cosmology.compat.astropy._distances import AstropyHasDistanceMeasures
from cosmology.compat.astropy._extras import (
    AstropyHasCriticalDensity,
    AstropyHasHubbleParameter,
)

__all__: list[str] = []


################################################################################


@dataclass(frozen=True)
class AstropyStandardCosmology(
    AstropyHasNeutrinoComponent,
    AstropyHasBaryonComponent,
    AstropyHasPhotonComponent,
    AstropyHasDarkMatterComponent,
    AstropyHasMatterComponent,
    AstropyHasDarkEnergyComponent,
    AstropyHasGlobalCurvatureComponent,
    AstropyHasTotalComponent,
    AstropyHasHubbleParameter,
    AstropyHasCriticalDensity,
    AstropyHasDistanceMeasures,
    AstropyCosmology,
):
    """The Cosmology API wrapper for :mod:`astropy.cosmology.Cosmology`."""

    cosmo: FLRW
