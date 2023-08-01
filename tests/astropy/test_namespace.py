"""Test the Cosmology API compat library."""

import cosmology.compat.astropy as namespace
from cosmology.api import CosmologyNamespace


def test_namespace_is_compliant():
    """Test :mod:`cosmology.compat.astropy.constants`."""
    assert isinstance(namespace, CosmologyNamespace)
