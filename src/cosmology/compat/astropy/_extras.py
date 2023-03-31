"""The cosmology API."""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol

import astropy.units as u
import numpy as np
from astropy.units import Quantity

from cosmology.api import HasCriticalDensity, HasHubbleParameter
from cosmology.compat.astropy._core import InputT, NDFloating

__all__: list[str] = []

if TYPE_CHECKING:
    from astropy.cosmology import FLRW


################################################################################
# PARAMETERS

_H0_UNITS = u.km / u.s / u.Mpc
_RHO_UNITS = u.solMass / u.Mpc**3


################################################################################


class AstropyHasCriticalDensity(HasCriticalDensity[Quantity, InputT], Protocol):
    cosmo: FLRW

    @property
    def critical_density0(self) -> Quantity:
        return self.cosmo.critical_density0.to(_RHO_UNITS)

    def critical_density(self, z: Quantity | NDFloating | float, /) -> Quantity:
        return self.cosmo.critical_density(z).to(_RHO_UNITS)


class AstropyHasHubbleParameter(HasHubbleParameter[Quantity, InputT], Protocol):
    cosmo: FLRW

    @property
    def H0(self) -> Quantity:
        return self.cosmo.H0.to(_H0_UNITS)

    @property
    def hubble_distance(self) -> Quantity:
        return self.cosmo.hubble_distance.to(u.Mpc)

    @property
    def hubble_time(self) -> Quantity:
        return self.cosmo.hubble_time.to(u.Gyr)

    def H(self, z: Quantity | NDFloating | float, /) -> Quantity:
        return self.cosmo.H(z).to(_H0_UNITS)

    def h_over_h0(self, z: Quantity | NDFloating | float, /) -> Quantity:
        return np.asarray(self.cosmo.efunc(z)) << u.one
