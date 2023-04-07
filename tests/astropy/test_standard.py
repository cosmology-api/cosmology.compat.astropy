"""Test the Cosmology API compat library."""

from __future__ import annotations

import pytest

from cosmology.api import StandardCosmology
from cosmology.api import StandardCosmologyWrapper as StandardCosmologyWrapperAPI

from .test_components import (
    BaryonComponent_Test,
    CurvatureComponent_Test,
    DarkEnergyComponent_Test,
    DarkMatterComponent_Test,
    MatterComponent_Test,
    NeutrinoComponent_Test,
    PhotonComponent_Test,
    TotalComponent_Test,
)
from .test_core import Test_CosmologyWrapper
from .test_distances import DistanceMeasures_Test
from .test_extras import CriticalDensity_Test, HubbleParameter_Test
from cosmology.compat.astropy import StandardCosmologyWrapper

################################################################################
# TESTS
################################################################################


class Test_StandardCosmologyWrapper(
    TotalComponent_Test,
    CurvatureComponent_Test,
    MatterComponent_Test,
    BaryonComponent_Test,
    NeutrinoComponent_Test,
    DarkEnergyComponent_Test,
    DarkMatterComponent_Test,
    PhotonComponent_Test,
    CriticalDensity_Test,
    HubbleParameter_Test,
    DistanceMeasures_Test,
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
