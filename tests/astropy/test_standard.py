"""Test the Cosmology API compat library."""

from __future__ import annotations

import pytest

from cosmology.api import StandardCosmology, StandardCosmologyWrapper
from cosmology.compat.astropy import AstropyStandardCosmology

from .test_components import (
    AstropyHasBaryonComponent_Test,
    AstropyHasDarkEnergyComponent_Test,
    AstropyHasDarkMatterComponent_Test,
    AstropyHasGlobalCurvatureComponent_Test,
    AstropyHasMatterComponent_Test,
    AstropyHasNeutrinoComponent_Test,
    AstropyHasPhotonComponent_Test,
    AstropyHasTotalComponent_Test,
)
from .test_core import Test_AstropyCosmology
from .test_distances import AstropyHasDistanceMeasures_Test
from .test_extras import AstropyHasCriticalDensity_Test, AstropyHasHubbleParameter_Test

################################################################################
# TESTS
################################################################################


class Test_AstropyStandardCosmology(
    AstropyHasTotalComponent_Test,
    AstropyHasGlobalCurvatureComponent_Test,
    AstropyHasMatterComponent_Test,
    AstropyHasBaryonComponent_Test,
    AstropyHasNeutrinoComponent_Test,
    AstropyHasDarkEnergyComponent_Test,
    AstropyHasDarkMatterComponent_Test,
    AstropyHasPhotonComponent_Test,
    AstropyHasCriticalDensity_Test,
    AstropyHasHubbleParameter_Test,
    AstropyHasDistanceMeasures_Test,
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
