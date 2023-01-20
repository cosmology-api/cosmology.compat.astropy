"""Astropy cosmology constants.

Note that :mod:`astropy` constants have astropy units.

From the :mod:`cosmology.api`, the list of required constants is:

- G: Gravitational constant G in pc km2 s-2 Msol-1.
"""

from astropy.constants import G as _G
from astropy.constants import c as _c

__all__ = ["G", "speed_of_light"]


G = _G.to("pc km2 / (Msun s2)")
speed_of_light = _c.to("km / s")
