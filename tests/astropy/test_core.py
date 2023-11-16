"""Test the Cosmology API compat library."""

import astropy.cosmology as apycosmo
import pytest

from cosmology.api import Cosmology as CosmologyAPI
from cosmology.api import CosmologyWrapper as CosmologyWrapperAPI

from cosmology.compat.astropy._core import CosmologyWrapper

################################################################################
# TESTS
################################################################################


class Test_CosmologyWrapper:
    @pytest.fixture(scope="class")
    def cosmo(self):
        return apycosmo.Planck18

    @pytest.fixture(scope="class")
    def wrapper(self, cosmo):
        return CosmologyWrapper(cosmo)

    # =========================================================================
    # Tests

    def test_wrapper_is_compliant(self, wrapper):
        """Test that CosmologyWrapper is a CosmologyWrapper."""
        if hasattr(super(), "test_wrapper_is_compliant"):
            super().test_wrapper_is_compliant(wrapper)

        assert isinstance(wrapper, CosmologyAPI)
        assert isinstance(wrapper, CosmologyWrapperAPI)

    def test_getattr(self, wrapper, cosmo):
        """Test that the wrapper can access the attributes of the wrapped object."""
        assert wrapper.meta == cosmo.meta
