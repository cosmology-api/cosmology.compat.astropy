"""Test the Cosmology API compat library."""

from __future__ import annotations

import astropy.units as u
import numpy as np
from hypothesis import given

from .conftest import z_arr_st
from cosmology.compat.astropy._distances import HasDistanceMeasures

################################################################################
# TESTS
################################################################################


class DistanceMeasures_Test:
    def test_wrapper_is_compliant(self, wrapper):
        """Test that AstropyCosmology is a BackgroundCosmologyWrapper."""
        if hasattr(super(), "test_wrapper_is_compliant"):
            super().test_wrapper_is_compliant(wrapper)

        assert isinstance(wrapper, HasDistanceMeasures)

    # =========================================================================

    def test_scale_factor0(self, wrapper):
        """Test that scale_factor(0) returns 1."""
        assert wrapper.scale_factor0 == 1 * u.one
        assert isinstance(wrapper.scale_factor0, np.ndarray)
        assert isinstance(wrapper.scale_factor0, u.Quantity)

    @given(z_arr_st(min_value=None))
    def test_scale_factor(self, wrapper, cosmo, z):
        """Test that the wrapper's scale_factor is the same as the wrapped object's."""
        a = wrapper.scale_factor(z)
        assert np.array_equal(a, cosmo.scale_factor(z))
        assert isinstance(a, np.ndarray)
        assert isinstance(a, u.Quantity)

    def test_T_cmb0(self, wrapper, cosmo):
        """Test that the wrapper has the same Tcmb0 as the wrapped object."""
        assert wrapper.T_cmb0 == cosmo.Tcmb0
        assert isinstance(wrapper.T_cmb0, u.Quantity)
        assert wrapper.T_cmb0.unit == u.Unit("K")

    @given(z_arr_st())
    def test_T_cmb(self, wrapper, cosmo, z):
        """Test that the wrapper's Tcmb is the same as the wrapped object's."""
        T = cosmo.T_cmb(z)
        assert np.array_equal(T, cosmo.Tcmb(z))
        assert isinstance(T, u.Quantity)
        assert T.unit == u.K

    @given(z_arr_st())
    def test_age(self, wrapper, cosmo, z):
        """Test the wrapper's age."""
        age = wrapper.age(z)
        assert np.array_equal(age, cosmo.age(z))
        assert isinstance(age, u.Quantity)
        assert age.unit == u.Unit("Gyr")

    @given(z_arr_st())
    def test_lookback_time(self, wrapper, cosmo, z):
        """Test the wrapper's lookback_time."""
        t = wrapper.lookback_time(z)
        assert np.array_equal(t, cosmo.lookback_time(z))
        assert isinstance(t, u.Quantity)
        assert t.unit == u.Unit("Gyr")

    @given(z_arr_st())
    def test_comoving_distance(self, wrapper, cosmo, z):
        """Test the wrapper's comoving_distance."""
        d = wrapper.comoving_distance(z)
        assert np.array_equal(d, cosmo.comoving_distance(z))
        assert isinstance(d, u.Quantity)
        assert d.unit == u.Unit("Mpc")

    @given(z_arr_st())
    def test_comoving_transverse_distance(self, wrapper, cosmo, z):
        """Test the wrapper's comoving_transverse_distance."""
        d = wrapper.comoving_transverse_distance(z)
        assert np.array_equal(d, cosmo.comoving_transverse_distance(z))
        assert isinstance(d, u.Quantity)
        assert d.unit == u.Unit("Mpc")

    @given(z_arr_st())
    def test_comoving_volume(self, wrapper, cosmo, z):
        """Test the wrapper's comoving_volume."""
        v = wrapper.comoving_volume(z)
        assert np.array_equal(v, cosmo.comoving_volume(z))
        assert isinstance(v, u.Quantity)
        assert v.unit == u.Unit("Mpc3")

    @given(z_arr_st())
    def test_differential_comoving_volume(self, wrapper, cosmo, z):
        """Test the wrapper's differential_comoving_volume."""
        v = wrapper.differential_comoving_volume(z)
        assert np.array_equal(v, cosmo.differential_comoving_volume(z))
        assert isinstance(v, u.Quantity)
        assert v.unit == u.Unit("Mpc3 / sr")

    @given(z_arr_st())
    def test_angular_diameter_distance(self, wrapper, cosmo, z):
        """Test the wrapper's angular_diameter_distance."""
        d = wrapper.angular_diameter_distance(z)
        assert np.array_equal(d, cosmo.angular_diameter_distance(z))
        assert isinstance(d, u.Quantity)
        assert d.unit == u.Unit("Mpc")

    @given(z_arr_st())
    def test_luminosity_distance(self, wrapper, cosmo, z):
        """Test the wrapper's luminosity_distance."""
        d = wrapper.luminosity_distance(z)
        assert np.array_equal(d, cosmo.luminosity_distance(z))
        assert isinstance(d, u.Quantity)
        assert d.unit == u.Unit("Mpc")
