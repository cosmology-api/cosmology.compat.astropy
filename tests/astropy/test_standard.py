"""Test the Cosmology API compat library."""

from __future__ import annotations

import pytest

from cosmology.api import StandardCosmology
from cosmology.api import StandardCosmologyWrapper as StandardCosmologyWrapperAPI

from .test_components import (
    HasBaryonComponent_Test,
    HasDarkEnergyComponent_Test,
    HasDarkMatterComponent_Test,
    HasGlobalCurvatureComponent_Test,
    HasMatterComponent_Test,
    HasNeutrinoComponent_Test,
    HasPhotonComponent_Test,
    HasTotalComponent_Test,
)
from .test_core import Test_CosmologyWrapper
from .test_distances import HasDistanceMeasures_Test
from .test_extras import HasCriticalDensity_Test, HasHubbleParameter_Test
from cosmology.compat.astropy import StandardCosmologyWrapper

################################################################################
# TESTS
################################################################################


class Test_StandardCosmologyWrapper(
    HasTotalComponent_Test,
    HasGlobalCurvatureComponent_Test,
    HasMatterComponent_Test,
    HasBaryonComponent_Test,
    HasNeutrinoComponent_Test,
    HasDarkEnergyComponent_Test,
    HasDarkMatterComponent_Test,
    HasPhotonComponent_Test,
    HasCriticalDensity_Test,
    HasHubbleParameter_Test,
    HasDistanceMeasures_Test,
    Test_CosmologyWrapper,
):
    @pytest.fixture(scope="class")
    def wrapper(self, cosmo):
        return StandardCosmologyWrapper(cosmo)

    # =========================================================================
    # Tests

    def test_wrapper_is_compliant(self, wrapper):
        """Test that StandardCosmologyWrapper is a StandardCosmologyWrapper."""
        super().test_wrapper_is_compliant(wrapper)

        assert isinstance(wrapper, StandardCosmology)
        assert isinstance(wrapper, StandardCosmologyWrapperAPI)
