"""The Cosmology API compatability wrapper for Astropy."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np
from cosmology.api import FLRWAPIConformantWrapper
from cosmology.compat.astropy.core import AstropyCosmology
from typing_extensions import TypeAlias

import astropy.cosmology as astropy_cosmology
import astropy.units as u
from astropy.units import Quantity

if TYPE_CHECKING:
    from numpy import floating
    from numpy.typing import NDArray

    NDFloating: TypeAlias = NDArray[floating[Any]]


__all__: list[str] = []


_H0_UNITS = u.km / u.s / u.Mpc
_RHO_UNITS = u.solMass / u.Mpc**3
_MPC3_UNITS = u.Mpc**3
_MPC3_SR_UNITS = _MPC3_UNITS / u.sr


@dataclass(frozen=True)
class AstropyFLRW(AstropyCosmology, FLRWAPIConformantWrapper):
    """The Cosmology API wrapper for :mod:`astropy.cosmology.Cosmology`."""

    cosmo: astropy_cosmology.FLRW

    @property
    def H0(self) -> Quantity:
        """Hubble function at redshift 0 in km s-1 Mpc-1."""
        return self.cosmo.H0.to(_H0_UNITS)

    @property
    def Om0(self) -> NDFloating:
        """Matter density parameter at redshift 0."""
        return np.asarray(self.cosmo.Om0)

    @property
    def Ode0(self) -> NDFloating:
        """Dark energy density parameter at redshift 0."""
        return np.asarray(self.cosmo.Ode0)

    @property
    def Tcmb0(self) -> Quantity:
        """Temperature of the CMB at redshift 0 in Kelvin."""
        return self.cosmo.Tcmb0.to(u.K)

    @property
    def Neff(self) -> NDFloating:
        """Neutrino mass in eV."""
        return np.asarray(self.cosmo.Neff)

    @property
    def m_nu(self) -> tuple[Quantity, ...]:
        """Neutrino mass in eV."""
        return tuple(self.cosmo.m_nu.to(u.eV))

    @property
    def Ob0(self) -> NDFloating:
        """Baryon density parameter at redshift 0."""
        return np.asarray(self.cosmo.Ob0)

    # ==============================================================
    # Derived properties

    @property
    def scale_factor0(self) -> NDFloating:
        """Scale factor at z=0."""
        return np.asarray(1.0)

    # ----------------------------------------------
    # Hubble

    @property
    def h(self) -> NDFloating:
        """Dimensionless Hubble constant at z=0."""
        return np.asarray(self.cosmo.h)

    @property
    def hubble_distance(self) -> Quantity:
        """Hubble distance at z=0."""
        return self.cosmo.hubble_distance.to(u.Mpc)

    @property
    def hubble_time(self) -> Quantity:
        """Hubble time in Gyr."""
        return self.cosmo.hubble_time.to(u.Gyr)

    # ----------------------------------------------
    # Omega

    @property
    def Otot0(self) -> NDFloating:
        r"""Omega total; the total density/critical density at z=0.

        .. math::

            \Omega_{\rm tot} = \Omega_{\rm m} + \Omega_{\rm \gamma} +
            \Omega_{\rm \nu} + \Omega_{\rm de} + \Omega_{\rm k}
        """
        return np.asarray(self.cosmo.Otot0)

    @property
    def Odm0(self) -> NDFloating:
        """Omega dark matter; dark matter density/critical density at z=0."""
        return np.asarray(self.cosmo.Odm0)

    @property
    def Ok0(self) -> NDFloating:
        """Omega curvature; the effective curvature density/critical density at z=0."""
        return np.asarray(self.cosmo.Ok0)

    @property
    def Ogamma0(self) -> NDFloating:
        """Omega gamma; the density/critical density of photons at z=0."""
        return np.asarray(self.cosmo.Ogamma0)

    @property
    def Onu0(self) -> NDFloating:
        """Omega nu; the density/critical density of neutrinos at z=0."""
        return np.asarray(self.cosmo.Onu0)

    # ----------------------------------------------
    # Density

    @property
    def rho_critical0(self) -> Quantity:
        """Critical density at z = 0 in Msol Mpc-3."""
        return self.cosmo.critical_density0.to(_RHO_UNITS)

    @property
    def rho_tot0(self) -> Quantity:
        """Total density at z = 0 in Msol Mpc-3."""
        return self.rho_critical0 * self.Otot0

    @property
    def rho_m0(self) -> Quantity:
        """Matter density at z = 0 in Msol Mpc-3."""
        return self.rho_critical0 * self.Om0

    @property
    def rho_de0(self) -> Quantity:
        """Dark energy density at z = 0 in Msol Mpc-3."""
        return self.rho_critical0 * self.Ode0

    @property
    def rho_b0(self) -> Quantity:
        """Baryon density at z = 0 in Msol Mpc-3."""
        return self.rho_critical0 * self.Ob0

    @property
    def rho_dm0(self) -> Quantity:
        """Dark matter density at z = 0 in Msol Mpc-3."""
        return self.rho_critical0 * self.Odm0

    @property
    def rho_k0(self) -> Quantity:
        """Curvature density at z = 0 in Msol Mpc-3."""
        return self.rho_critical0 * self.Ok0

    @property
    def rho_gamma0(self) -> Quantity:
        """Radiation density at z = 0 in Msol Mpc-3."""
        return self.rho_critical0 * self.Ogamma0

    @property
    def rho_nu0(self) -> Quantity:
        """Neutrino density at z = 0 in Msol Mpc-3."""
        return self.rho_critical0 * self.Onu0

    # ==============================================================
    # Methods

    def scale_factor(self, z: NDFloating | Quantity, /) -> NDFloating:
        """Redshift-dependenct scale factor :math:`a = a_0 / (1 + z)`."""
        return np.asarray(self.cosmo.scale_factor(z))

    # ----------------------------------------------
    # Hubble

    def H(self, z: NDFloating | Quantity, /) -> Quantity:
        """Hubble function :math:`H(z)` in km s-1 Mpc-1."""  # noqa: D402
        return self.cosmo.H(z).to(_H0_UNITS)

    def efunc(self, z: NDFloating | Quantity, /) -> NDFloating:
        """Standardised Hubble function :math:`E(z) = H(z)/H_0`."""
        return np.asarray(self.cosmo.efunc(z))

    def inv_efunc(self, z: NDFloating | Quantity, /) -> NDFloating:
        """Inverse of ``efunc``."""
        return np.asarray(self.cosmo.inv_efunc(z))

    # ----------------------------------------------
    # Omega

    def Otot(self, z: NDFloating | Quantity, /) -> NDFloating:
        r"""Redshift-dependent total density parameter.

        This is the sum of the matter, radiation, neutrino, dark energy, and
        curvature density parameters.

        .. math::

            \Omega_{\rm tot} = \Omega_{\rm m} + \Omega_{\rm \gamma} +
            \Omega_{\rm \nu} + \Omega_{\rm de} + \Omega_{\rm k}
        """
        return np.asarray(self.cosmo.Otot(z))

    def Om(self, z: NDFloating | Quantity, /) -> NDFloating:
        """Redshift-dependent non-relativistic matter density parameter.

        Notes
        -----
        This does not include neutrinos, even if non-relativistic at the
        redshift of interest; see `Onu`.
        """
        return np.asarray(self.cosmo.Om(z))

    def Ob(self, z: NDFloating | Quantity, /) -> NDFloating:
        """Redshift-dependent baryon density parameter.

        Raises
        ------
        ValueError
            If ``Ob0`` is `None`.
        """
        return np.asarray(self.cosmo.Ob(z))

    def Odm(self, z: NDFloating | Quantity, /) -> NDFloating:
        """Redshift-dependent dark matter density parameter.

        Raises
        ------
        ValueError
            If ``Ob0`` is `None`.

        Notes
        -----
        This does not include neutrinos, even if non-relativistic at the
        redshift of interest.
        """
        return np.asarray(self.cosmo.Odm(z))

    def Ok(self, z: NDFloating | Quantity, /) -> NDFloating:
        """Redshift-dependent curvature density parameter."""
        return np.asarray(self.cosmo.Ok(z))

    def Ode(self, z: NDFloating | Quantity, /) -> NDFloating:
        """Redshift-dependent dark energy density parameter."""
        return np.asarray(self.cosmo.Ode(z))

    def Ogamma(self, z: NDFloating | Quantity, /) -> NDFloating:
        """Redshift-dependent photon density parameter."""
        return np.asarray(self.cosmo.Ogamma(z))

    def Onu(self, z: NDFloating | Quantity, /) -> NDFloating:
        r"""Redshift-dependent neutrino density parameter.

        The energy density of neutrinos relative to the critical density at each
        redshift. Note that this includes their kinetic energy (if
        they have mass), so it is not equal to the commonly used :math:`\sum
        \frac{m_{\nu}}{94 eV}`, which does not include kinetic energy.
        Returns `float` if the input is scalar.
        """
        return np.asarray(self.cosmo.Onu(z))

    # ----------------------------------------------
    # Rho

    def rho_critical(self, z: NDFloating | Quantity, /) -> Quantity:
        """Redshift-dependent critical density in Msol Mpc-3."""
        return self.cosmo.critical_density(z).to(_RHO_UNITS)

    def rho_tot(self, z: NDFloating | Quantity, /) -> Quantity:
        """Redshift-dependent total density in Msol Mpc-3."""
        return self.rho_critical(z) * self.Otot(z)

    def rho_m(self, z: NDFloating | Quantity, /) -> Quantity:
        """Redshift-dependent matter density in Msol Mpc-3."""
        return self.rho_critical(z) * self.Om(z)

    def rho_de(self, z: NDFloating | Quantity, /) -> Quantity:
        """Redshift-dependent dark energy density in Msol Mpc-3."""
        return self.rho_critical(z) * self.Ode(z)

    def rho_k(self, z: NDFloating | Quantity, /) -> Quantity:
        """Redshift-dependent curvature density in Msol Mpc-3."""
        return self.rho_critical(z) * self.Ok(z)

    def rho_dm(self, z: NDFloating | Quantity, /) -> Quantity:
        """Redshift-dependent dark matter density in Msol Mpc-3."""
        return self.rho_critical(z) * self.Odm(z)

    def rho_b(self, z: NDFloating | Quantity, /) -> Quantity:
        """Redshift-dependent baryon density in Msol Mpc-3."""
        return self.rho_critical(z) * self.Ob(z)

    def rho_gamma(self, z: NDFloating | Quantity, /) -> Quantity:
        """Redshift-dependent photon density in Msol Mpc-3."""
        return self.rho_critical(z) * self.Ogamma(z)

    def rho_nu(self, z: NDFloating | Quantity, /) -> Quantity:
        """Redshift-dependent neutrino density in Msol Mpc-3."""
        return self.rho_critical(z) * self.Onu(z)

    # ----------------------------------------------
    # Time

    def age(self, z: NDFloating | Quantity, /) -> Quantity:
        """Age of the universe in Gyr at redshift ``z``."""
        return self.cosmo.age(z).to(u.Gyr)

    def lookback_time(self, z: NDFloating | Quantity, /) -> Quantity:
        """Lookback time to redshift ``z`` in Gyr.

        The lookback time is the difference between the age of the Universe now
        and the age at redshift ``z``.
        """
        return self.cosmo.lookback_time(z).to(u.Gyr)

    # ----------------------------------------------
    # Comoving distance

    def comoving_distance(self, z: NDFloating | Quantity, /) -> Quantity:
        r"""Comoving line-of-sight distance :math:`d_c(z)` in Mpc.

        The comoving distance along the line-of-sight between two objects
        remains constant with time for objects in the Hubble flow.
        """
        return self.cosmo.comoving_distance(z).to(u.Mpc)

    def comoving_transverse_distance(self, z: NDFloating | Quantity, /) -> Quantity:
        r"""Transverse comoving distance :math:`d_M(z)` in Mpc.

        This value is the transverse comoving distance at redshift ``z``
        corresponding to an angular separation of 1 radian. This is the same as
        the comoving distance if :math:`\Omega_k` is zero (as in the current
        concordance Lambda-CDM model).
        """
        return self.cosmo.comoving_transverse_distance(z).to(u.Mpc)

    def comoving_volume(self, z: NDFloating | Quantity, /) -> Quantity:
        r"""Comoving volume in cubic Mpc.

        This is the volume of the universe encompassed by redshifts less than
        ``z``. For the case of :math:`\Omega_k = 0` it is a sphere of radius
        `comoving_distance` but it is less intuitive if :math:`\Omega_k` is not.
        """
        return self.cosmo.comoving_volume(z).to(_MPC3_UNITS)

    def differential_comoving_volume(self, z: NDFloating | Quantity, /) -> Quantity:
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

    def angular_diameter_distance(self, z: NDFloating | Quantity, /) -> Quantity:
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

    def luminosity_distance(self, z: NDFloating | Quantity, /) -> Quantity:
        """Redshift-dependent luminosity distance in Mpc.

        This is the distance to use when converting between the bolometric flux
        from an object at redshift ``z`` and its bolometric luminosity [1]_.

        Parameters
        ----------
        z : Array
            Input redshift.

        Returns
        -------
        Array

        References
        ----------
        .. [1] Weinberg, 1972, pp 420-424; Weedman, 1986, pp 60-62.
        """
        return self.cosmo.luminosity_distance(z).to(u.Mpc)
