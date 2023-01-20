"""Test the Cosmology API compat library."""

import pytest
from cosmology.api import FLRWAPIConformant, FLRWAPIConformantWrapper
from cosmology.compat.astropy import AstropyFLRW

from .test_core import Test_AstropyCosmology

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
