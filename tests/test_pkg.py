"""Test the Cosmology API compat library."""


def test_imported():
    """This is a namespace package, so it should be importable."""
    import cosmology.compat.astropy

    assert cosmology.compat.astropy.__name__ == "cosmology.compat.astropy"
