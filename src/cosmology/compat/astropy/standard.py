"""The Cosmology API compatability wrapper for Astropy."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import astropy.units as u
import numpy as np
from astropy.units import Quantity

from cosmology.api import BackgroundCosmologyWrapperAPI
from cosmology.compat.astropy.background import AstropyBackgroundCosmology

if TYPE_CHECKING:
    import astropy.cosmology as astropy_cosmology

    from cosmology.compat.astropy.core import NDFloating

__all__: list[str] = []


################################################################################
# PARAMETERS

_H0_UNITS = u.km / u.s / u.Mpc
_RHO_UNITS = u.solMass / u.Mpc**3
_MPC3_UNITS = u.Mpc**3
_MPC3_SR_UNITS = _MPC3_UNITS / u.sr


################################################################################


@dataclass(frozen=True)
class AstropyStandardCosmology(
    AstropyBackgroundCosmology,
    BackgroundCosmologyWrapperAPI["NDFloating"],
):
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
        # TODO: cache
        return (
            np.asarray(self.cosmo.Ob0) if self.cosmo.Ob0 is not None else np.zeros(())
        )

    # ==============================================================
    # Derived properties

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
        return np.asarray(
            self.cosmo.Odm0 if self.cosmo.Odm0 is not None else self.cosmo.Om0,
        )

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

    # ==============================================================
    # Methods

    def Tcmb(self, z: NDFloating | Quantity, /) -> Quantity:
        """CMB background temperature in K."""
        return self.cosmo.Tcmb(z).to(u.K)

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
        """Redshift-dependent baryon density parameter."""
        try:
            return np.asarray(self.cosmo.Ob(z))
        except ValueError:
            return np.asarray(np.zeros_like(z))

    def Odm(self, z: NDFloating | Quantity, /) -> NDFloating:
        """Redshift-dependent dark matter density parameter.

        Notes
        -----
        This does not include neutrinos, even if non-relativistic at the
        redshift of interest.
        """
        try:
            return np.asarray(self.cosmo.Odm(z))
        except ValueError:
            return np.asarray(self.cosmo.Om(z))

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
        """
        return np.asarray(self.cosmo.Onu(z))

    # ----------------------------------------------
    # Rho

    def critical_density(self, z: NDFloating | Quantity, /) -> Quantity:
        """Redshift-dependent critical density in Msol Mpc-3."""
        return self.cosmo.critical_density(z).to(_RHO_UNITS)
