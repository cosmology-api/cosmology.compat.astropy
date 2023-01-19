"""Astropy cosmology constants.

Note that :mod:`astropy` constants have astropy units.

From the :mod:`cosmology.api`, the list of required constants is:

- G: Gravitational constant G in pc km2 s-2 Msol-1.
"""

from astropy.constants import G as _G

__all__ = ["G"]


G = _G.to("pc km2 / (Msun s2)")
