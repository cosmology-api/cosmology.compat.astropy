"""Test the Cosmology API compat library."""

from __future__ import annotations

import astropy.units as u
import numpy as np
from hypothesis import given

from .conftest import z_arr_st
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

################################################################################
# TESTS
################################################################################


class AstropyTotalComponent_Test:
    def test_wrapper_is_compliant(self, wrapper):
        """Test that AstropyCosmology is a BackgroundCosmologyWrapper."""
        if hasattr(super(), "test_wrapper_is_compliant"):
            super().test_wrapper_is_compliant(wrapper)

        assert isinstance(wrapper, AstropyTotalComponent)

    def test_Omega_tot0(self, wrapper, cosmo):
        """Test that the wrapper has the same Otot0 as the wrapped object."""
        assert wrapper.Omega_tot0 == cosmo.Otot0
        assert isinstance(wrapper.Omega_tot0, np.ndarray)
        assert isinstance(wrapper.Omega_tot0, u.Quantity)

    @given(z_arr_st())
    def test_Otot(self, wrapper, cosmo, z):
        """Test that the wrapper's Otot is the same as the wrapped object's."""
        omega = wrapper.Omega_tot(z)
        assert np.array_equal(omega, cosmo.Otot(z))
        assert isinstance(omega, np.ndarray)
        assert isinstance(omega, u.Quantity)


class AstropyCurvatureComponent_Test:
    def test_wrapper_is_compliant(self, wrapper):
        """Test that AstropyCosmology is a BackgroundCosmologyWrapper."""
        if hasattr(super(), "test_wrapper_is_compliant"):
            super().test_wrapper_is_compliant(wrapper)

        assert isinstance(wrapper, AstropyCurvatureComponent)

    def test_Omega_k0(self, wrapper, cosmo):
        """Test that the wrapper has the same Omega_k0 as the wrapped object."""
        assert wrapper.Omega_k0 == cosmo.Ok0
        assert isinstance(wrapper.Omega_k0, np.ndarray)
        assert isinstance(wrapper.Omega_k0, u.Quantity)

    @given(z_arr_st())
    def test_Omega_k(self, wrapper, cosmo, z):
        """Test that the wrapper's Omega_k is the same as the wrapped object's."""
        omega = wrapper.Omega_k(z)
        assert np.array_equal(omega, cosmo.Ok(z))
        assert isinstance(omega, np.ndarray)
        assert isinstance(omega, u.Quantity)


class AstropyMatterComponent_Test:
    def test_wrapper_is_compliant(self, wrapper):
        """Test that AstropyCosmology is a BackgroundCosmologyWrapper."""
        if hasattr(super(), "test_wrapper_is_compliant"):
            super().test_wrapper_is_compliant(wrapper)

        assert isinstance(wrapper, AstropyMatterComponent)

    def test_Omega_m0(self, wrapper, cosmo):
        """Test that the wrapper has the same Om0 as the wrapped object."""
        assert wrapper.Omega_m0 == cosmo.Om0
        assert isinstance(wrapper.Omega_m0, np.ndarray)
        assert isinstance(wrapper.Omega_m0, u.Quantity)

    # TODO: why do these fail for z-> inf?
    @given(z_arr_st(max_value=1e9))
    def test_Omega_m(self, wrapper, cosmo, z):
        """Test that the wrapper's Om is the same as the wrapped object's."""
        omega = wrapper.Omega_m(z)
        assert np.array_equal(omega, cosmo.Om(z))
        assert isinstance(omega, np.ndarray)
        assert isinstance(omega, u.Quantity)


class AstropyBaryonComponent_Test:
    def test_wrapper_is_compliant(self, wrapper):
        """Test that AstropyCosmology is a BackgroundCosmologyWrapper."""
        if hasattr(super(), "test_wrapper_is_compliant"):
            super().test_wrapper_is_compliant(wrapper)

        assert isinstance(wrapper, AstropyBaryonComponent)

    def test_Omega_b0(self, wrapper, cosmo):
        """Test that the wrapper has the same Omega_b0 as the wrapped object."""
        assert wrapper.Omega_b0 == cosmo.Ob0
        assert isinstance(wrapper.Omega_b0, np.ndarray)
        assert isinstance(wrapper.Omega_b0, u.Quantity)

    @given(z_arr_st(max_value=1e9))
    def test_Omega_b(self, wrapper, cosmo, z):
        """Test that the wrapper's Omega_b is the same as the wrapped object's."""
        omega = wrapper.Omega_b(z)
        assert np.array_equal(omega, cosmo.Ob(z))
        assert isinstance(omega, np.ndarray)
        assert isinstance(omega, u.Quantity)


class AstropyNeutrinoComponent_Test:
    def test_wrapper_is_compliant(self, wrapper):
        """Test that AstropyCosmology is a BackgroundCosmologyWrapper."""
        if hasattr(super(), "test_wrapper_is_compliant"):
            super().test_wrapper_is_compliant(wrapper)

        assert isinstance(wrapper, AstropyNeutrinoComponent)

    def test_Omega_nu0(self, wrapper, cosmo):
        """Test that the wrapper has the same Omega_nu0 as the wrapped object."""
        assert wrapper.Omega_nu0 == cosmo.Onu0
        assert isinstance(wrapper.Omega_nu0, np.ndarray)
        assert isinstance(wrapper.Omega_nu0, u.Quantity)

    def test_Neff(self, wrapper, cosmo):
        """Test that the wrapper has the same Neff as the wrapped object."""
        assert wrapper.Neff == cosmo.Neff
        assert isinstance(wrapper.Neff, np.ndarray)
        assert isinstance(wrapper.Neff, u.Quantity)

    def test_m_nu(self, wrapper, cosmo):
        """Test that the wrapper has the same m_nu as the wrapped object."""
        assert all(np.equal(w, c) for w, c in zip(wrapper.m_nu, tuple(cosmo.m_nu)))
        assert isinstance(wrapper.m_nu, tuple)
        assert all(isinstance(m, u.Quantity) for m in wrapper.m_nu)
        assert all(m.unit == u.Unit("eV") for m in wrapper.m_nu)

    @given(z_arr_st(max_value=1e9))
    def test_Omega_nu(self, wrapper, cosmo, z):
        """Test that the wrapper's Omega_nu is the same as the wrapped object's."""
        omega = wrapper.Omega_nu(z)
        assert np.array_equal(omega, cosmo.Onu(z))
        assert isinstance(omega, np.ndarray)
        assert isinstance(omega, u.Quantity)


class AstropyDarkEnergyComponent_Test:
    def test_wrapper_is_compliant(self, wrapper):
        """Test that AstropyCosmology is a BackgroundCosmologyWrapper."""
        if hasattr(super(), "test_wrapper_is_compliant"):
            super().test_wrapper_is_compliant(wrapper)

        assert isinstance(wrapper, AstropyDarkEnergyComponent)

    def test_Omega_de0(self, wrapper, cosmo):
        """Test that the wrapper has the same Omega_de0 as the wrapped object."""
        assert wrapper.Omega_de0 == cosmo.Ode0
        assert isinstance(wrapper.Omega_de0, np.ndarray)
        assert isinstance(wrapper.Omega_de0, u.Quantity)

    @given(z_arr_st())
    def test_Omega_de(self, wrapper, cosmo, z):
        """Test that the wrapper's Omega_de is the same as the wrapped object's."""
        omega = wrapper.Omega_de(z)
        assert np.array_equal(omega, cosmo.Ode(z))
        assert isinstance(omega, np.ndarray)
        assert isinstance(omega, u.Quantity)


class AstropyDarkMatterComponent_Test:
    def test_wrapper_is_compliant(self, wrapper):
        """Test that AstropyCosmology is a BackgroundCosmologyWrapper."""
        if hasattr(super(), "test_wrapper_is_compliant"):
            super().test_wrapper_is_compliant(wrapper)

        assert isinstance(wrapper, AstropyDarkMatterComponent)

    def test_Omega_dm0(self, wrapper, cosmo):
        """Test that the wrapper has the same Omega_dm0 as the wrapped object."""
        assert wrapper.Omega_dm0 == cosmo.Odm0
        assert isinstance(wrapper.Omega_dm0, np.ndarray)
        assert isinstance(wrapper.Omega_dm0, u.Quantity)

    @given(z_arr_st(max_value=1e9))
    def test_Omega_dm(self, wrapper, cosmo, z):
        """Test that the wrapper's Omega_dm is the same as the wrapped object's."""
        omega = wrapper.Omega_dm(z)
        assert np.array_equal(omega, cosmo.Odm(z))
        assert isinstance(omega, np.ndarray)
        assert isinstance(omega, u.Quantity)


class AstropyPhotonComponent_Test:
    def test_wrapper_is_compliant(self, wrapper):
        """Test that AstropyCosmology is a BackgroundCosmologyWrapper."""
        if hasattr(super(), "test_wrapper_is_compliant"):
            super().test_wrapper_is_compliant(wrapper)

        assert isinstance(wrapper, AstropyPhotonComponent)

    def test_Omega_gamma0(self, wrapper, cosmo):
        """Test that the wrapper has the same Omega_gamma0 as the wrapped object."""
        assert wrapper.Omega_gamma0 == cosmo.Ogamma0
        assert isinstance(wrapper.Omega_gamma0, np.ndarray)
        assert isinstance(wrapper.Omega_gamma0, u.Quantity)

    @given(z_arr_st(max_value=1e9))
    def test_Omega_gamma(self, wrapper, cosmo, z):
        """Test that the wrapper's Omega_gamma is the same as the wrapped object's."""
        omega = wrapper.Omega_gamma(z)
        assert np.array_equal(omega, cosmo.Ogamma(z))
        assert isinstance(omega, np.ndarray)
        assert isinstance(omega, u.Quantity)
