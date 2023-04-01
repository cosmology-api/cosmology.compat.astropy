"""Test the Cosmology API compat library."""

from __future__ import annotations

import pytest

from cosmology.api import StandardCosmology, StandardCosmologyWrapper

from .test_components import (
    AstropyBaryonComponent_Test,
    AstropyCurvatureComponent_Test,
    AstropyDarkEnergyComponent_Test,
    AstropyDarkMatterComponent_Test,
    AstropyMatterComponent_Test,
    AstropyNeutrinoComponent_Test,
    AstropyPhotonComponent_Test,
    AstropyTotalComponent_Test,
)
from .test_core import Test_AstropyCosmology
from .test_distances import AstropyDistanceMeasures_Test
from .test_extras import AstropyCriticalDensity_Test, AstropyHubbleParameter_Test
from cosmology.compat.astropy import AstropyStandardCosmology

################################################################################
# TESTS
################################################################################


class Test_AstropyStandardCosmology(
    AstropyTotalComponent_Test,
    AstropyCurvatureComponent_Test,
    AstropyMatterComponent_Test,
    AstropyBaryonComponent_Test,
    AstropyNeutrinoComponent_Test,
    AstropyDarkEnergyComponent_Test,
    AstropyDarkMatterComponent_Test,
    AstropyPhotonComponent_Test,
    AstropyCriticalDensity_Test,
    AstropyHubbleParameter_Test,
    AstropyDistanceMeasures_Test,
    Test_AstropyCosmology,
):
    @pytest.fixture(scope="class")
    def wrapper(self, cosmo):
        return AstropyStandardCosmology(cosmo)

    # =========================================================================
    # Tests

    def test_wrapper_is_compliant(self, wrapper):
        """Test that AstropyCosmology is a BackgroundCosmologyWrapper."""
        super().test_wrapper_is_compliant(wrapper)

        assert isinstance(wrapper, StandardCosmology)
        assert isinstance(wrapper, StandardCosmologyWrapper)
