"""Test the Cosmology API compat library."""

from __future__ import annotations

import astropy.units as u
import numpy as np
from hypothesis import given

from cosmology.api import CriticalDensity, HubbleParameter

from .conftest import z_arr_st

################################################################################
# TESTS
################################################################################


class CriticalDensity_Test:
    def test_wrapper_is_compliant(self, wrapper):
        """Test that AstropyCosmology is a BackgroundCosmologyWrapper."""
        if hasattr(super(), "test_wrapper_is_compliant"):
            super().test_wrapper_is_compliant(wrapper)

        assert isinstance(wrapper, CriticalDensity)

    def test_critical_density0(self, wrapper, cosmo):
        """
        Test that the wrapper's critical_density0 is the same as
        critical_density0.
        """
        assert wrapper.critical_density0 == cosmo.critical_density0
        assert isinstance(wrapper.critical_density0, u.Quantity)
        assert wrapper.critical_density0.unit == u.Unit("Msun / Mpc3")

    @given(z_arr_st(max_value=1e9))
    def test_critical_density(self, wrapper, cosmo, z):
        r"""Test that the wrapper's critical_density is critical_density."""
        rho = wrapper.critical_density(z)
        assert np.array_equal(rho, cosmo.critical_density(z))
        assert isinstance(rho, u.Quantity)
        assert rho.unit == u.Unit("Msun / Mpc3")


class HubbleParameter_Test:
    def test_wrapper_is_compliant(self, wrapper):
        """Test that AstropyCosmology is a BackgroundCosmologyWrapper."""
        if hasattr(super(), "test_wrapper_is_compliant"):
            super().test_wrapper_is_compliant(wrapper)

        assert isinstance(wrapper, HubbleParameter)

    def test_H0(self, wrapper, cosmo):
        """Test that the wrapper has the same H0 as the wrapped object."""
        assert wrapper.H0 == cosmo.H0
        assert isinstance(wrapper.H0, u.Quantity)
        assert wrapper.H0.unit == u.Unit("km / (Mpc s)")

    def test_hubble_distance(self, wrapper, cosmo):
        """Test that the wrapper has the same hubble_distance as the wrapped object."""
        assert wrapper.hubble_distance == cosmo.hubble_distance
        assert isinstance(wrapper.hubble_distance, u.Quantity)
        assert wrapper.hubble_distance.unit == u.Unit("Mpc")

    def test_hubble_time(self, wrapper, cosmo):
        """Test that the wrapper has the same hubble_time as the wrapped object."""
        assert wrapper.hubble_time == cosmo.hubble_time
        assert isinstance(wrapper.hubble_time, u.Quantity)
        assert wrapper.hubble_time.unit == u.Unit("Gyr")

    @given(z_arr_st())
    def test_H(self, wrapper, cosmo, z):
        """Test that the wrapper's H is the same as the wrapped object's."""
        H = wrapper.H(z)
        assert np.array_equal(H, cosmo.H(z))
        assert isinstance(H, u.Quantity)
        assert H.unit == u.Unit("km / (Mpc s)")

    @given(z_arr_st())
    def test_h_over_h0(self, wrapper, cosmo, z):
        """Test that the wrapper's efunc is the same as the wrapped object's."""
        e = wrapper.h_over_h0(z)
        assert np.array_equal(e, cosmo.efunc(z))
        assert isinstance(e, np.ndarray)
        assert isinstance(e, u.Quantity)
