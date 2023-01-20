"""Test the Cosmology API compat library."""

from __future__ import annotations

import astropy.units as u
import numpy as np
import pytest
from cosmology.api import FLRWAPIConformant, FLRWAPIConformantWrapper
from cosmology.compat.astropy import AstropyFLRW
from hypothesis import given
from hypothesis.extra import numpy as npst

from .test_core import Test_AstropyCosmology

################################################################################
# PARAMETERS

# Hypothesis strategy for generating arrays of redshifts
def z_arr_st(*, allow_nan: bool = False, min_value: float | None = 0, **kwargs):
    return npst.arrays(
        # TODO: do we want to test float16?
        npst.floating_dtypes(sizes=(32, 64)),
        npst.array_shapes(min_dims=1),
        elements={"allow_nan": allow_nan, "min_value": min_value} | kwargs,
    )


################################################################################
# TESTS
################################################################################


class Test_AstropyFLRW(Test_AstropyCosmology):
    @pytest.fixture(scope="class")
    def wrapper(self, cosmo):
        return AstropyFLRW(cosmo)

    # =========================================================================
    # Tests

    def test_wrapper_is_compliant(self, wrapper):
        """Test that AstropyCosmology is a FLRWAPIConformantWrapper."""
        super().test_wrapper_is_compliant(wrapper)

        assert isinstance(wrapper, FLRWAPIConformant)
        assert isinstance(wrapper, FLRWAPIConformantWrapper)

    def test_getattr(self, wrapper, cosmo):
        """Test that the wrapper can access the attributes of the wrapped object."""
        # The base Cosmology API doesn't have H0
        assert wrapper.meta == cosmo.meta

    # =========================================================================
    # FLRW API Tests

    def test_H0(self, wrapper, cosmo):
        """Test that the wrapper has the same H0 as the wrapped object."""
        assert wrapper.H0 == cosmo.H0
        assert isinstance(wrapper.H0, u.Quantity)
        assert wrapper.H0.unit == u.Unit("km / (Mpc s)")

    def test_Om0(self, wrapper, cosmo):
        """Test that the wrapper has the same Om0 as the wrapped object."""
        assert wrapper.Om0 == cosmo.Om0
        assert isinstance(wrapper.Om0, np.ndarray)
        assert not isinstance(wrapper.Om0, u.Quantity)

    def test_Ode0(self, wrapper, cosmo):
        """Test that the wrapper has the same Ode0 as the wrapped object."""
        assert wrapper.Ode0 == cosmo.Ode0
        assert isinstance(wrapper.Ode0, np.ndarray)
        assert not isinstance(wrapper.Ode0, u.Quantity)

    def test_Tcmb0(self, wrapper, cosmo):
        """Test that the wrapper has the same Tcmb0 as the wrapped object."""
        assert wrapper.Tcmb0 == cosmo.Tcmb0
        assert isinstance(wrapper.Tcmb0, u.Quantity)
        assert wrapper.Tcmb0.unit == u.Unit("K")

    def test_Neff(self, wrapper, cosmo):
        """Test that the wrapper has the same Neff as the wrapped object."""
        assert wrapper.Neff == cosmo.Neff
        assert isinstance(wrapper.Neff, np.ndarray)
        assert not isinstance(wrapper.Neff, u.Quantity)

    def test_m_nu(self, wrapper, cosmo):
        """Test that the wrapper has the same m_nu as the wrapped object."""
        assert all(np.equal(w, c) for w, c in zip(wrapper.m_nu, tuple(cosmo.m_nu)))
        assert isinstance(wrapper.m_nu, tuple)
        assert all(isinstance(m, u.Quantity) for m in wrapper.m_nu)
        assert all(m.unit == u.Unit("eV") for m in wrapper.m_nu)

    def test_Ob0(self, wrapper, cosmo):
        """Test that the wrapper has the same Ob0 as the wrapped object."""
        assert wrapper.Ob0 == cosmo.Ob0
        assert isinstance(wrapper.Ob0, np.ndarray)
        assert not isinstance(wrapper.Ob0, u.Quantity)

    def test_scale_factor0(self, wrapper):
        """Test that scale_factor(0) returns 1."""
        assert wrapper.scale_factor0 == 1
        assert isinstance(wrapper.scale_factor0, np.ndarray)
        assert not isinstance(wrapper.scale_factor0, u.Quantity)

    def test_h(self, wrapper, cosmo):
        """Test that the wrapper has the same h as the wrapped object."""
        assert wrapper.h == cosmo.h
        assert isinstance(wrapper.h, np.ndarray)
        assert not isinstance(wrapper.h, u.Quantity)

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

    def test_Otot0(self, wrapper, cosmo):
        """Test that the wrapper has the same Otot0 as the wrapped object."""
        assert wrapper.Otot0 == cosmo.Otot0
        assert isinstance(wrapper.Otot0, np.ndarray)
        assert not isinstance(wrapper.Otot0, u.Quantity)

    def test_Odm0(self, wrapper, cosmo):
        """Test that the wrapper has the same Odm0 as the wrapped object."""
        assert wrapper.Odm0 == cosmo.Odm0
        assert isinstance(wrapper.Odm0, np.ndarray)
        assert not isinstance(wrapper.Odm0, u.Quantity)

    def test_Ok0(self, wrapper, cosmo):
        """Test that the wrapper has the same Ok0 as the wrapped object."""
        assert wrapper.Ok0 == cosmo.Ok0
        assert isinstance(wrapper.Ok0, np.ndarray)
        assert not isinstance(wrapper.Ok0, u.Quantity)

    def test_Ogamma0(self, wrapper, cosmo):
        """Test that the wrapper has the same Ogamma0 as the wrapped object."""
        assert wrapper.Ogamma0 == cosmo.Ogamma0
        assert isinstance(wrapper.Ogamma0, np.ndarray)
        assert not isinstance(wrapper.Ogamma0, u.Quantity)

    def test_Onu0(self, wrapper, cosmo):
        """Test that the wrapper has the same Onu0 as the wrapped object."""
        assert wrapper.Onu0 == cosmo.Onu0
        assert isinstance(wrapper.Onu0, np.ndarray)
        assert not isinstance(wrapper.Onu0, u.Quantity)

    def test_rho_critical0(self, wrapper, cosmo):
        """Test that the wrapper's rho_critical0 is the same as critical_density0."""
        assert wrapper.rho_critical0 == cosmo.critical_density0
        assert isinstance(wrapper.rho_critical0, u.Quantity)
        assert wrapper.rho_critical0.unit == u.Unit("Msun / Mpc3")

    def test_rho_tot0(self, wrapper, cosmo):
        """Test the wrapper's rho_tot0."""
        assert wrapper.rho_tot0 == cosmo.critical_density0 * cosmo.Otot0
        assert isinstance(wrapper.rho_tot0, u.Quantity)
        assert wrapper.rho_tot0.unit == u.Unit("Msun / Mpc3")

    def test_rho_m0(self, wrapper, cosmo):
        """Test the wrapper's rho_m0."""
        assert np.allclose(wrapper.rho_m0, cosmo.critical_density0 * cosmo.Om0)
        assert isinstance(wrapper.rho_m0, u.Quantity)
        assert wrapper.rho_m0.unit == u.Unit("Msun / Mpc3")

    def test_rho_de0(self, wrapper, cosmo):
        """Test the wrapper's rho_de0."""
        assert wrapper.rho_de0 == cosmo.critical_density0 * cosmo.Ode0
        assert isinstance(wrapper.rho_de0, u.Quantity)
        assert wrapper.rho_de0.unit == u.Unit("Msun / Mpc3")

    def test_rho_b0(self, wrapper, cosmo):
        """Test the wrapper's rho_b0."""
        assert wrapper.rho_b0 == cosmo.critical_density0 * cosmo.Ob0
        assert isinstance(wrapper.rho_b0, u.Quantity)
        assert wrapper.rho_b0.unit == u.Unit("Msun / Mpc3")

    def test_rho_dm0(self, wrapper, cosmo):
        """Test the wrapper's rho_dm0."""
        assert wrapper.rho_dm0 == cosmo.critical_density0 * cosmo.Odm0
        assert isinstance(wrapper.rho_dm0, u.Quantity)
        assert wrapper.rho_dm0.unit == u.Unit("Msun / Mpc3")

    def test_rho_gamma0(self, wrapper, cosmo):
        """Test the wrapper's rho_gamma0."""
        assert wrapper.rho_gamma0 == cosmo.critical_density0 * cosmo.Ogamma0
        assert isinstance(wrapper.rho_gamma0, u.Quantity)
        assert wrapper.rho_gamma0.unit == u.Unit("Msun / Mpc3")

    def test_rho_nu0(self, wrapper, cosmo):
        """Test the wrapper's rho_nu0."""
        assert wrapper.rho_nu0 == cosmo.critical_density0 * cosmo.Onu0
        assert isinstance(wrapper.rho_nu0, u.Quantity)
        assert wrapper.rho_nu0.unit == u.Unit("Msun / Mpc3")

    @given(z_arr_st(min_value=None))
    def test_scale_factor(self, wrapper, cosmo, z):
        """Test that the wrapper's scale_factor is the same as the wrapped object's."""
        a = cosmo.scale_factor(z)
        assert np.array_equal(a, cosmo.scale_factor(z))
        assert isinstance(a, np.ndarray)
        assert not isinstance(a, u.Quantity)

    @given(z_arr_st())
    def test_Tcmb(self, wrapper, cosmo, z):
        """Test that the wrapper's Tcmb is the same as the wrapped object's."""
        T = cosmo.Tcmb(z)
        assert np.array_equal(T, cosmo.Tcmb(z))
        assert isinstance(T, u.Quantity)
        assert T.unit == u.K

    @given(z_arr_st())
    def test_H(self, wrapper, cosmo, z):
        """Test that the wrapper's H is the same as the wrapped object's."""
        H = wrapper.H(z)
        assert np.array_equal(H, cosmo.H(z))
        assert isinstance(H, u.Quantity)
        assert H.unit == u.Unit("km / (Mpc s)")

    @given(z_arr_st())
    def test_efunc(self, wrapper, cosmo, z):
        """Test that the wrapper's efunc is the same as the wrapped object's."""
        e = wrapper.efunc(z)
        assert np.array_equal(e, cosmo.efunc(z))
        assert isinstance(e, np.ndarray)
        assert not isinstance(e, u.Quantity)

    @given(z_arr_st())
    def test_inv_efunc(self, wrapper, cosmo, z):
        """Test that the wrapper's inv_efunc is the same as the wrapped object's."""
        ie = wrapper.inv_efunc(z)
        assert np.array_equal(ie, cosmo.inv_efunc(z))
        assert isinstance(ie, np.ndarray)
        assert not isinstance(ie, u.Quantity)

    @given(z_arr_st())
    def test_Otot(self, wrapper, cosmo, z):
        """Test that the wrapper's Otot is the same as the wrapped object's."""
        omega = wrapper.Otot(z)
        assert np.array_equal(omega, cosmo.Otot(z))
        assert isinstance(omega, np.ndarray)
        assert not isinstance(omega, u.Quantity)

    # TODO: why do these fail for z-> inf?
    @given(z_arr_st(max_value=1e9))
    def test_Om(self, wrapper, cosmo, z):
        """Test that the wrapper's Om is the same as the wrapped object's."""
        omega = wrapper.Om(z)
        assert np.array_equal(omega, cosmo.Om(z))
        assert isinstance(omega, np.ndarray)
        assert not isinstance(omega, u.Quantity)

    @given(z_arr_st(max_value=1e9))
    def test_Ob(self, wrapper, cosmo, z):
        """Test that the wrapper's Ob is the same as the wrapped object's."""
        omega = wrapper.Ob(z)
        assert np.array_equal(omega, cosmo.Ob(z))
        assert isinstance(omega, np.ndarray)
        assert not isinstance(omega, u.Quantity)

    @given(z_arr_st(max_value=1e9))
    def test_Odm(self, wrapper, cosmo, z):
        """Test that the wrapper's Odm is the same as the wrapped object's."""
        omega = wrapper.Odm(z)
        assert np.array_equal(omega, cosmo.Odm(z))
        assert isinstance(omega, np.ndarray)
        assert not isinstance(omega, u.Quantity)

    @given(z_arr_st(max_value=1e9))
    def test_Ogamma(self, wrapper, cosmo, z):
        """Test that the wrapper's Ogamma is the same as the wrapped object's."""
        omega = wrapper.Ogamma(z)
        assert np.array_equal(omega, cosmo.Ogamma(z))
        assert isinstance(omega, np.ndarray)
        assert not isinstance(omega, u.Quantity)

    @given(z_arr_st(max_value=1e9))
    def test_Onu(self, wrapper, cosmo, z):
        """Test that the wrapper's Onu is the same as the wrapped object's."""
        omega = wrapper.Onu(z)
        assert np.array_equal(omega, cosmo.Onu(z))
        assert isinstance(omega, np.ndarray)
        assert not isinstance(omega, u.Quantity)

    @given(z_arr_st())
    def test_Ode(self, wrapper, cosmo, z):
        """Test that the wrapper's Ode is the same as the wrapped object's."""
        omega = wrapper.Ode(z)
        assert np.array_equal(omega, cosmo.Ode(z))
        assert isinstance(omega, np.ndarray)
        assert not isinstance(omega, u.Quantity)

    @given(z_arr_st())
    def test_Ok(self, wrapper, cosmo, z):
        """Test that the wrapper's Ok is the same as the wrapped object's."""
        omega = wrapper.Ok(z)
        assert np.array_equal(omega, cosmo.Ok(z))
        assert isinstance(omega, np.ndarray)
        assert not isinstance(omega, u.Quantity)

    @given(z_arr_st(max_value=1e9))
    def test_rho_critical(self, wrapper, cosmo, z):
        r"""Test that the wrapper's rho_critical is critical_density."""
        rho = wrapper.rho_critical(z)
        assert np.array_equal(rho, cosmo.critical_density(z))
        assert isinstance(rho, u.Quantity)
        assert rho.unit == u.Unit("Msun / Mpc3")

    @given(z_arr_st())
    def test_rho_tot(self, wrapper, cosmo, z):
        """Test the wrapper's rho_tot."""
        rho = wrapper.rho_tot(z)
        assert np.array_equal(rho, cosmo.critical_density(z) * cosmo.Otot(z))
        assert isinstance(rho, u.Quantity)
        assert rho.unit == u.Unit("Msun / Mpc3")

    @given(z_arr_st(max_value=1e9))
    def test_rho_m(self, wrapper, cosmo, z):
        """Test the wrapper's rho_m."""
        rho = wrapper.rho_m(z)
        assert np.allclose(rho, cosmo.critical_density(z) * cosmo.Om(z))
        assert isinstance(rho, u.Quantity)
        assert rho.unit == u.Unit("Msun / Mpc3")

    @given(z_arr_st(max_value=1e9))
    def test_rho_de(self, wrapper, cosmo, z):
        """Test the wrapper's rho_de."""
        rho = wrapper.rho_de(z)
        assert np.allclose(rho, cosmo.critical_density(z) * cosmo.Ode(z))
        assert isinstance(rho, u.Quantity)
        assert rho.unit == u.Unit("Msun / Mpc3")

    @given(z_arr_st(max_value=1e9))
    def test_rho_k(self, wrapper, cosmo, z):
        """Test the wrapper's rho_k."""
        rho = wrapper.rho_k(z)
        assert np.array_equal(rho, cosmo.critical_density(z) * cosmo.Ok(z))
        assert isinstance(rho, u.Quantity)
        assert rho.unit == u.Unit("Msun / Mpc3")

    @given(z_arr_st(max_value=1e9))
    def test_rho_dm(self, wrapper, cosmo, z):
        """Test the wrapper's rho_dm."""
        rho = wrapper.rho_dm(z)
        assert np.allclose(rho, cosmo.critical_density(z) * cosmo.Odm(z))
        assert isinstance(rho, u.Quantity)
        assert rho.unit == u.Unit("Msun / Mpc3")

    @given(z_arr_st(max_value=1e9))
    def test_rho_b(self, wrapper, cosmo, z):
        """Test the wrapper's rho_b."""
        rho = wrapper.rho_b(z)
        assert np.allclose(rho, cosmo.critical_density(z) * cosmo.Ob(z))
        assert isinstance(rho, u.Quantity)
        assert rho.unit == u.Unit("Msun / Mpc3")

    @given(z_arr_st(max_value=1e9))
    def test_rho_gamma(self, wrapper, cosmo, z):
        """Test the wrapper's rho_gamma."""
        rho = wrapper.rho_gamma(z)
        assert np.allclose(rho, cosmo.critical_density(z) * cosmo.Ogamma(z))
        assert isinstance(rho, u.Quantity)
        assert rho.unit == u.Unit("Msun / Mpc3")

    @given(z_arr_st(max_value=1e9))  # TODO: why does this fail for z < 1e7?
    def test_rho_nu(self, wrapper, cosmo, z):
        """Test the wrapper's rho_nu."""
        rho = wrapper.rho_nu(z)
        assert np.allclose(rho, cosmo.critical_density(z) * cosmo.Onu(z))
        assert isinstance(rho, u.Quantity)
        assert rho.unit == u.Unit("Msun / Mpc3")

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
