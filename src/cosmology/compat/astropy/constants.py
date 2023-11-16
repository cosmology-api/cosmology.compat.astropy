"""Astropy cosmology constants.

Note that :mod:`astropy` constants have astropy units.

From the :mod:`cosmology.api`, the list of required constants is:

- G: Gravitational constant G in pc km2 s-2 Msol-1.
- c: Speed of light in km s-1.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from astropy.constants import G as _G
from astropy.constants import c as _c

__all__ = ["G", "c"]

if TYPE_CHECKING:
    from astropy.quantity import Quantity


G: Quantity = _G.to("pc km2 / (Msun s2)")
"""Gravitational constant G in pc km2 s-2 Msol-1."""

c: Quantity = _c.to("km / s")
"""Speed of light in km s-1."""
