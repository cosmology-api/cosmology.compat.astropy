"""The Cosmology API compatability wrapper for Astropy."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import astropy.units as u
import numpy as np
from astropy.cosmology import FLRW  # noqa: TCH002
from astropy.units import Quantity

from cosmology.api import HasDistanceMeasures as CoreHasDistanceMeasures

from cosmology.compat.astropy._core import InputT

__all__: list[str] = []


################################################################################
# PARAMETERS

_MPC3_UNITS = u.Mpc**3
_MPC3_SR_UNITS = _MPC3_UNITS / u.sr


################################################################################


@dataclass(frozen=True)
class AstropyHasDistanceMeasures(
    CoreHasDistanceMeasures[Quantity, InputT],
    Protocol,
):
    cosmo: FLRW

    @property
    def scale_factor0(self) -> Quantity:
        return np.asarray(1.0) << u.one

    @property
    def Tcmb0(self) -> Quantity:
        return self.cosmo.Tcmb0.to(u.K)

    # ==============================================================

    def scale_factor(self, z: InputT, /) -> Quantity:
        return np.asarray(self.cosmo.scale_factor(z)) << u.one

    def Tcmb(self, z: InputT, /) -> Quantity:
        return self.cosmo.Tcmb(z).to(u.K)

    # ----------------------------------------------
    # Time

    def age(self, z: InputT, /) -> Quantity:
        return self.cosmo.age(z).to(u.Gyr)

    def lookback_time(self, z: InputT, /) -> Quantity:
        return self.cosmo.lookback_time(z).to(u.Gyr)

    # ----------------------------------------------
    # Comoving distance

    def comoving_distance(self, z: InputT, /) -> Quantity:
        return self.cosmo.comoving_distance(z).to(u.Mpc)

    def comoving_transverse_distance(
        self,
        z: InputT,
        /,
    ) -> Quantity:
        return self.cosmo.comoving_transverse_distance(z).to(u.Mpc)

    def comoving_volume(self, z: InputT, /) -> Quantity:
        return self.cosmo.comoving_volume(z).to(_MPC3_UNITS)

    def differential_comoving_volume(
        self,
        z: InputT,
        /,
    ) -> Quantity:
        return self.cosmo.differential_comoving_volume(z).to(_MPC3_SR_UNITS)

    # ----------------------------------------------
    # Angular diameter distance

    def angular_diameter_distance(
        self,
        z: InputT,
        /,
    ) -> Quantity:
        return self.cosmo.angular_diameter_distance(z).to(u.Mpc)

    # ----------------------------------------------
    # Luminosity distance

    def luminosity_distance(self, z: InputT, /) -> Quantity:
        return self.cosmo.luminosity_distance(z).to(u.Mpc)
