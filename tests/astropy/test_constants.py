# Copyright (c) 2022, Nathaniel Starkman and Nicolas Tessore
"""Test the Cosmology API compat library."""

from cosmology.api import CosmologyConstantsNamespace

from cosmology.compat.astropy import constants


def test_namespace_is_compliant():
    """Test :mod:`cosmology.compat.astropy.constants`."""
    assert isinstance(constants, CosmologyConstantsNamespace)
