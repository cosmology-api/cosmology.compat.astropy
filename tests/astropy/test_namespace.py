"""Test the Cosmology API compat library."""

from cosmology.api import CosmologyNamespace

import cosmology.compat.astropy as namespace


def test_namespace_is_compliant():
    """Test :mod:`cosmology.compat.astropy.constants`."""
    assert isinstance(namespace, CosmologyNamespace)
