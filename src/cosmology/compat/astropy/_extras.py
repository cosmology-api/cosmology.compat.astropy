"""The cosmology API."""

from __future__ import annotations

from typing import TYPE_CHECKING

import astropy.units as u
import numpy as np

__all__: list[str] = []

if TYPE_CHECKING:
    from astropy.cosmology import FLRW
    from astropy.units import Quantity

    from cosmology.compat.astropy._core import NDFloating


################################################################################
# PARAMETERS

_H0_UNITS = u.km / u.s / u.Mpc
_RHO_UNITS = u.solMass / u.Mpc**3


################################################################################


class CriticalDensity:
    """The cosmology has methods for the critical density."""

    cosmo: FLRW

    @property
    def critical_density0(self) -> Quantity:
        """Critical density at z = 0 in Msol Mpc-3."""
        return self.cosmo.critical_density0.to(_RHO_UNITS)

    def critical_density(self, z: Quantity | NDFloating | float, /) -> Quantity:
        """Redshift-dependent critical density in Msol Mpc-3."""
        return self.cosmo.critical_density(z).to(_RHO_UNITS)


class HubbleParameter:
    r"""The cosmology has methods to retrieve the Hubble parameter :math:`H`."""

    cosmo: FLRW

    @property
    def H0(self) -> Quantity:
        """Hubble parameter at redshift 0 in km s-1 Mpc-1."""
        return self.cosmo.H0.to(_H0_UNITS)

    @property
    def hubble_distance(self) -> Quantity:
        """Hubble distance in Mpc."""
        return self.cosmo.hubble_distance.to(u.Mpc)

    @property
    def hubble_time(self) -> Quantity:
        """Hubble time in Gyr."""
        return self.cosmo.hubble_time.to(u.Gyr)

    def H(self, z: Quantity | NDFloating | float, /) -> Quantity:
        """Hubble parameter :math:`H(z)` in km s-1 Mpc-1.

        Parameters
        ----------
        z : Array
            The redshift(s) at which to evaluate the Hubble parameter.

        Returns
        -------
        Array

        """
        return self.cosmo.H(z).to(_H0_UNITS)

    def H_over_H0(self, z: Quantity | NDFloating | float, /) -> Quantity:
        """Standardised Hubble function :math:`E(z) = H(z)/H_0`.

        Parameters
        ----------
        z : Array
            The redshift(s) at which to evaluate.

        Returns
        -------
        Array

        """
        return np.asarray(self.cosmo.efunc(z)) << u.one
