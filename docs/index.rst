#############################################
The Cosmology API Astropy Compatability Layer
#############################################

.. currentmodule:: cosmology.compat.astropy

:mod:`cosmology.compat.astropy` is a `Cosmology API`_ compatibility layer for
:mod:`astropy`. With this module, you can use any
:class:`~astropy.cosmology.FLRW`-based :mod:`astropy` cosmology in any context
that accepts a `Cosmology API`_ cosmology.

The two components of this module are:

.. toctree::
   :hidden:
   :maxdepth: 1

   cosmology.compat.astropy <self>

.. autosummary::
   :nosignatures:
   :toctree: api
   :caption: API

   StandardCosmologyWrapper
   constants


Quick Start
===========

We can write a function that accepts any `Cosmology API`_ cosmology...

.. testsetup:: python

    from cosmology.api._array_api import Array

.. code-block:: python

    # No implementation, just a description of the interface!
    from cosmology.api import StandardCosmology


    def flat_angular_diameter_distance(
        cosmo: StandardCosmology[Array, Array], z: Array
    ) -> Array:
        # Do some cosmology with any object that implements the API
        if cosmo.Omega_k0 != 0:
            raise ValueError("This function only works for flat cosmologies")
        return cosmo.comoving_distance(z) / (1 + z)


and then use it with any `Cosmology API`_ cosmology, specifically
:mod:`cosmology.compat.astropy`!

We make a compatible :mod:`astropy` cosmology using this library...

.. code-block:: python

    from astropy.cosmology import Planck18 as astropy_planck18
    from cosmology.compat.astropy import StandardCosmologyWrapper

    planck18 = StandardCosmologyWrapper(astropy_planck18)


.. code-block:: python

    import astropy.units as u
    import astropy.cosmology.units as cu

    z = u.Quantity([0.1, 0.2, 0.3], cu.redshift)

which now works with the above function.

    >>> flat_angular_diameter_distance(planck18, z)
    <Quantity [393.2415434 , 702.98158286, 948.2358979 ] Mpc>


Contributors
============

.. include:: ../AUTHORS.rst
