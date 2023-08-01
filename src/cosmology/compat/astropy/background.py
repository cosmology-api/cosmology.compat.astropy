"""The Cosmology API compatability wrapper for Astropy."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import astropy.units as u
import numpy as np

from cosmology.api import BackgroundCosmologyWrapper
from cosmology.compat.astropy.core import AstropyCosmology

if TYPE_CHECKING:
    import astropy.cosmology as astropy_cosmology
    from astropy.units import Quantity

    from cosmology.compat.astropy.core import NDFloating

__all__: list[str] = []


################################################################################
# PARAMETERS

_RHO_UNITS = u.solMass / u.Mpc**3
_MPC3_UNITS = u.Mpc**3
_MPC3_SR_UNITS = _MPC3_UNITS / u.sr


################################################################################


@dataclass(frozen=True)
class AstropyBackgroundCosmology(
    AstropyCosmology,
    BackgroundCosmologyWrapper["NDFloating"],
):
    """The Cosmology API wrapper for :mod:`astropy.cosmology.Cosmology`."""

    cosmo: astropy_cosmology.FLRW

    # ==============================================================
    # Derived properties

    @property
    def scale_factor0(self) -> NDFloating:
        """Scale factor at z=0."""
        return np.asarray(1.0)

    @property
    def Otot0(self) -> NDFloating:
        """Omega total; the total density/critical density at z=0."""
        return np.asarray(self.cosmo.Otot0)

    @property
    def critical_density0(self) -> Quantity:
        """Critical density at z = 0 in Msol Mpc-3."""
        return self.cosmo.critical_density0.to(_RHO_UNITS)

    # ==============================================================
    # Methods

    def scale_factor(self, z: NDFloating | Quantity | float, /) -> NDFloating:
        """Redshift-dependenct scale factor :math:`a = a_0 / (1 + z)`."""
        return np.asarray(self.cosmo.scale_factor(z))

    def Otot(self, z: NDFloating | Quantity | float, /) -> NDFloating:
        """Redshift-dependent total density parameter."""
        return np.asarray(self.cosmo.Otot(z))

    def critical_density(self, z: NDFloating | Quantity | float, /) -> Quantity:
        """Redshift-dependent critical density in Msol Mpc-3."""
        return self.cosmo.critical_density(z).to(_RHO_UNITS)

    # ----------------------------------------------
    # Time

    def age(self, z: NDFloating | Quantity | float, /) -> Quantity:
        """Age of the universe in Gyr at redshift ``z``."""
        return self.cosmo.age(z).to(u.Gyr)

    def lookback_time(self, z: NDFloating | Quantity | float, /) -> Quantity:
        """Lookback time to redshift ``z`` in Gyr.

        The lookback time is the difference between the age of the Universe now
        and the age at redshift ``z``.
        """
        return self.cosmo.lookback_time(z).to(u.Gyr)

    # ----------------------------------------------
    # Comoving distance

    def comoving_distance(self, z: NDFloating | Quantity | float, /) -> Quantity:
        r"""Comoving line-of-sight distance :math:`d_c(z)` in Mpc.

        The comoving distance along the line-of-sight between two objects
        remains constant with time for objects in the Hubble flow.
        """
        return self.cosmo.comoving_distance(z).to(u.Mpc)

    def comoving_transverse_distance(
        self,
        z: NDFloating | Quantity | float,
        /,
    ) -> Quantity:
        r"""Transverse comoving distance :math:`d_M(z)` in Mpc.

        This value is the transverse comoving distance at redshift ``z``
        corresponding to an angular separation of 1 radian. This is the same as
        the comoving distance if :math:`\Omega_k` is zero (as in the current
        concordance Lambda-CDM model).
        """
        return self.cosmo.comoving_transverse_distance(z).to(u.Mpc)

    def comoving_volume(self, z: NDFloating | Quantity | float, /) -> Quantity:
        r"""Comoving volume in cubic Mpc.

        This is the volume of the universe encompassed by redshifts less than
        ``z``. For the case of :math:`\Omega_k = 0` it is a sphere of radius
        `comoving_distance` but it is less intuitive if :math:`\Omega_k` is not.
        """
        return self.cosmo.comoving_volume(z).to(_MPC3_UNITS)

    def differential_comoving_volume(
        self,
        z: NDFloating | Quantity | float,
        /,
    ) -> Quantity:
        r"""Differential comoving volume in cubic Mpc per steradian.

        If :math:`V_c` is the comoving volume of a redshift slice with solid
        angle :math:`\Omega`, this function returns

        .. math::

            \mathtt{dvc(z)}
            = \frac{1}{d_H^3} \, \frac{dV_c}{d\Omega \, dz}
            = \frac{x_M^2(z)}{E(z)}
            = \frac{\mathtt{xm(z)^2}}{\mathtt{ef(z)}} \;.

        """
        return self.cosmo.differential_comoving_volume(z).to(_MPC3_SR_UNITS)

    # ----------------------------------------------
    # Angular diameter distance

    def angular_diameter_distance(
        self,
        z: NDFloating | Quantity | float,
        /,
    ) -> Quantity:
        """Angular diameter distance :math:`d_A(z)` in Mpc.

        This gives the proper (sometimes called 'physical') transverse
        distance corresponding to an angle of 1 radian for an object
        at redshift ``z`` ([1]_, [2]_, [3]_).

        References
        ----------
        .. [1] Weinberg, 1972, pp 420-424; Weedman, 1986, pp 421-424.
        .. [2] Weedman, D. (1986). Quasar astronomy, pp 65-67.
        .. [3] Peebles, P. (1993). Principles of Physical Cosmology, pp 325-327.
        """
        return self.cosmo.angular_diameter_distance(z).to(u.Mpc)

    # ----------------------------------------------
    # Luminosity distance

    def luminosity_distance(self, z: NDFloating | Quantity | float, /) -> Quantity:
        """Redshift-dependent luminosity distance in Mpc.

        This is the distance to use when converting between the bolometric flux
        from an object at redshift ``z`` and its bolometric luminosity [1]_.

        References
        ----------
        .. [1] Weinberg, 1972, pp 420-424; Weedman, 1986, pp 60-62.
        """
        return self.cosmo.luminosity_distance(z).to(u.Mpc)
