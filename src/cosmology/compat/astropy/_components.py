"""The cosmology API."""

from __future__ import annotations

from typing import TYPE_CHECKING

import astropy.units as u
import numpy as np

__all__: list[str] = []

if TYPE_CHECKING:
    from astropy.cosmology import FLRW
    from astropy.units import Quantity

    from cosmology.compat.astropy._core import InputT


class TotalComponent:
    r"""The cosmology contains a total density, described by :math:`Omega_{\rm tot}`."""

    cosmo: FLRW

    @property
    def Omega_tot0(self) -> Quantity:
        r"""Omega total; the total density/critical density at z=0."""
        return np.asarray(self.cosmo.Otot0) << u.one

    def Omega_tot(self, z: InputT, /) -> Quantity:
        r"""Omega total; the total density/critical density at z=0."""
        return np.asarray(self.cosmo.Otot(z)) << u.one


class CurvatureComponent:
    r"""The cosmology contains global curvature, described by :math:`Omega_k`."""

    cosmo: FLRW

    @property
    def Omega_k0(self) -> Quantity:
        """Omega curvature; the effective curvature density/critical density at z=0."""
        return np.asarray(self.cosmo.Ok0) << u.one

    def Omega_k(self, z: InputT, /) -> Quantity:
        """Redshift-dependent curvature density parameter.

        Parameters
        ----------
        z : Array, positional-only
            Input redshift.

        Returns
        -------
        Array
        """
        return np.asarray(self.cosmo.Ok(z)) << u.one


class MatterComponent:
    r"""The cosmology contains matter, described by :math:`Omega_m`."""

    cosmo: FLRW

    @property
    def Omega_m0(self) -> Quantity:
        """Omega matter; matter density/critical density at z=0."""
        return np.asarray(self.cosmo.Om0) << u.one

    def Omega_m(self, z: InputT, /) -> Quantity:
        """Redshift-dependent non-relativistic matter density parameter.

        Parameters
        ----------
        z : Array, positional-only
            Input redshift.

        Returns
        -------
        Array

        Notes
        -----
        This does not include neutrinos, even if non-relativistic at the
        redshift of interest; see `Omega_nu`.
        """
        return np.asarray(self.cosmo.Om(z)) << u.one


class BaryonComponent:
    r"""The cosmology contains baryons, described by :math:`Omega_b`."""

    cosmo: FLRW

    @property
    def Omega_b0(self) -> Quantity:
        """Omega baryon; baryon density/critical density at z=0."""
        # TODO: cache
        return (
            np.asarray(self.cosmo.Ob0) if self.cosmo.Ob0 is not None else np.zeros(())
        ) << u.one

    def Omega_b(self, z: InputT, /) -> Quantity:
        """Redshift-dependent baryon density parameter.

        Parameters
        ----------
        z : Array, positional-only
            Input redshift.

        Returns
        -------
        Array
        """
        try:
            return np.asarray(self.cosmo.Ob(z)) << u.one
        except ValueError:
            return np.asarray(np.zeros_like(z)) << u.one


class NeutrinoComponent:
    r"""The cosmology contains neutrinos, described by :math:`Omega_\nu`."""

    cosmo: FLRW

    @property
    def Omega_nu0(self) -> Quantity:
        """Omega nu; the density/critical density of neutrinos at z=0."""
        return np.asarray(self.cosmo.Onu0) << u.one

    @property
    def Neff(self) -> Quantity:
        """Effective number of neutrino species."""
        return np.asarray(self.cosmo.Neff) << u.one

    @property
    def m_nu(self) -> tuple[Quantity, ...]:
        """Neutrino mass in eV."""
        return tuple(self.cosmo.m_nu.to(u.eV))

    def Omega_nu(self, z: InputT, /) -> Quantity:
        r"""Redshift-dependent neutrino density parameter.

        Parameters
        ----------
        z : Array, positional-only
            Input redshift.

        Returns
        -------
        Array
        """
        return np.asarray(self.cosmo.Onu(z)) << u.one


class DarkEnergyComponent:
    r"""The cosmology contains photons, described by :math:`Omega_{\rm de}`."""

    cosmo: FLRW

    @property
    def Omega_de0(self) -> Quantity:
        """Omega dark energy; dark energy density/critical density at z=0."""
        return np.asarray(self.cosmo.Ode0) << u.one

    def Omega_de(self, z: InputT, /) -> Quantity:
        """Redshift-dependent dark energy density parameter.

        Parameters
        ----------
        z : Array, positional-only
            Input redshift.

        Returns
        -------
        Array
        """
        return np.asarray(self.cosmo.Ode(z)) << u.one


class DarkMatterComponent:
    r"""The cosmology contains cold dark matter, described by :math:`Omega_dm`."""

    cosmo: FLRW

    @property
    def Omega_dm0(self) -> Quantity:
        """Omega dark matter; dark matter density/critical density at z=0."""
        # TODO: cache
        return (
            np.asarray(
                self.cosmo.Odm0 if self.cosmo.Odm0 is not None else self.cosmo.Om0,
            )
            << u.one
        )

    def Omega_dm(self, z: InputT, /) -> Quantity:
        """Redshift-dependent dark matter density parameter.

        Parameters
        ----------
        z : Array, positional-only
            Input redshift.

        Returns
        -------
        Array

        Notes
        -----
        This does not include neutrinos, even if non-relativistic at the
        redshift of interest.
        """
        try:
            return np.asarray(self.cosmo.Odm(z)) << u.one
        except ValueError:
            return np.asarray(self.cosmo.Om(z)) << u.one


class PhotonComponent:
    r"""The cosmology contains photons, described by :math:`Omega_\gamma`."""

    cosmo: FLRW

    @property
    def Omega_gamma0(self) -> Quantity:
        """Omega gamma; the density/critical density of photons at z=0."""
        return np.asarray(self.cosmo.Ogamma0) << u.one

    def Omega_gamma(self, z: InputT, /) -> Quantity:
        """Redshift-dependent photon density parameter.

        Parameters
        ----------
        z : Array, positional-only
            Input redshift.

        Returns
        -------
        Array
        """
        return np.asarray(self.cosmo.Ogamma(z)) << u.one
