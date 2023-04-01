"""Test the Cosmology API compat library."""

import astropy.cosmology as apycosmo
import pytest

from cosmology.api import Cosmology, CosmologyWrapper

from cosmology.compat.astropy import AstropyCosmology

################################################################################
# TESTS
################################################################################


class Test_AstropyCosmology:
    @pytest.fixture(scope="class")
    def cosmo(self):
        return apycosmo.Planck18

    @pytest.fixture(scope="class")
    def wrapper(self, cosmo):
        return AstropyCosmology(cosmo)

    # =========================================================================
    # Tests

    def test_wrapper_is_compliant(self, wrapper):
        """Test that AstropyCosmology is a CosmologyWrapper."""
        if hasattr(super(), "test_wrapper_is_compliant"):
            super().test_wrapper_is_compliant(wrapper)

        assert isinstance(wrapper, Cosmology)
        assert isinstance(wrapper, CosmologyWrapper)

    def test_getattr(self, wrapper, cosmo):
        """Test that the wrapper can access the attributes of the wrapped object."""
        assert wrapper.meta == cosmo.meta
