"""The Cosmology API compatability wrapper for Astropy."""

from __future__ import annotations

from contextlib import suppress
from typing import TYPE_CHECKING, Any, overload

import astropy.units as u
import numpy as np
import scipy.integrate as si
from astropy.units import Quantity

from cosmology.api import ComovingDistanceMeasures as ComovingDistanceMeasuresAPI
from cosmology.api import HasConstantsNamespace
from cosmology.api import LookbackDistanceMeasures as LookbackDistanceMeasuresAPI
from cosmology.api import ProperDistanceMeasures as ProperDistanceMeasuresAPI
from cosmology.api import ScaleFactor as ScaleFactorAPI
from cosmology.api import TemperatureCMB as TemperatureCMBAPI

from cosmology.compat.astropy._core import InputT

__all__: list[str] = []


if TYPE_CHECKING:
    from astropy.cosmology import FLRW
    from numpy.typing import NDArray


_MPC3_UNITS = u.Mpc**3
_MPC3_SR_UNITS = _MPC3_UNITS / u.sr


class TemperatureCMB(TemperatureCMBAPI[Quantity, InputT]):
    cosmo: FLRW

    @property
    def T_cmb0(self) -> Quantity:
        """CMB temperature in K at z=0."""
        return self.cosmo.Tcmb0.to(u.K)

    def T_cmb(self, z: InputT, /) -> Quantity:
        """CMB temperature in K at redshift z.

        Parameters
        ----------
        z : Quantity, positional-only
            Input redshift.

        Returns
        -------
        Quantity
        """
        return self.cosmo.Tcmb(z).to(u.K)


class ScaleFactor(ScaleFactorAPI[Quantity, InputT]):
    cosmo: FLRW

    @property
    def scale_factor0(self) -> Quantity:
        """Scale factor at z=0."""
        return np.asarray(1.0) << u.one

    def scale_factor(self, z: InputT, /) -> Quantity:
        """Redshift-dependenct scale factor.

        The scale factor is defined as :math:`a = a_0 / (1 + z)`.

        Parameters
        ----------
        z : Quantity or float, positional-only
            The redshift(s) at which to evaluate the scale factor.

        Returns
        -------
        Quantity
        """
        return np.asarray(self.cosmo.scale_factor(z)) << u.one


class ComovingDistanceMeasures(ComovingDistanceMeasuresAPI[Quantity, InputT]):
    cosmo: FLRW

    @overload
    def comoving_distance(self, z: InputT, /) -> Quantity:
        ...

    @overload
    def comoving_distance(self, z1: InputT, z2: InputT, /) -> Quantity:
        ...

    def comoving_distance(self, z1: InputT, z2: InputT | None = None, /) -> Quantity:
        r"""Comoving line-of-sight distance :math:`d_c(z1, z2)` in Mpc.

        The comoving distance along the line-of-sight between two objects
        remains constant with time for objects in the Hubble flow.

        Parameters
        ----------
        z : Quantity, positional-only
        z1, z2 : Quantity, positional-only
            Input redshifts. If one argument ``z`` is given, the time
            :math:`t_T(0, z)` is returned. If two arguments ``z1, z2`` are
            given, the time :math:`t_T(z_1, z_2)` is returned.

        Returns
        -------
        Quantity
        """
        z1, z2 = (0.0, z1) if z2 is None else (z1, z2)
        return self.cosmo._comoving_distance_z1z2(z1, z2).to(u.Mpc)  # noqa: SLF001

    @overload
    def transverse_comoving_distance(self, z: InputT, /) -> Quantity:
        ...

    @overload
    def transverse_comoving_distance(self, z1: InputT, z2: InputT, /) -> Quantity:
        ...

    def transverse_comoving_distance(
        self, z1: InputT, z2: InputT | None = None, /
    ) -> Quantity:
        r"""Transverse comoving distance :math:`d_M(z1, z2)` in Mpc.

        This value is the transverse comoving distance at redshift ``z``
        corresponding to an angular separation of 1 radian. This is the same as
        the comoving distance if :math:`\Omega_k` is zero (as in the current
        concordance Lambda-CDM model).

        Parameters
        ----------
        z : Quantity, positional-only
        z1, z2 : Quantity, positional-only
            Input redshifts. If one argument ``z`` is given, the time
            :math:`d_M(0, z)` is returned. If two arguments ``z1, z2`` are
            given, the time :math:`d_M(z_1, z_2)` is returned.

        Returns
        -------
        Quantity
        """
        z1, z2 = (0.0, z1) if z2 is None else (z1, z2)
        return self.cosmo._comoving_transverse_distance_z1z2(z1, z2).to(  # noqa: SLF001
            u.Mpc
        )

    @overload
    def comoving_volume(self, z: InputT, /) -> Quantity:
        ...

    @overload
    def comoving_volume(self, z1: InputT, z2: InputT, /) -> Quantity:
        ...

    def comoving_volume(self, z1: InputT, z2: InputT | None = None, /) -> Quantity:
        r"""Comoving volume in cubic Mpc.

        This is the volume of the universe encompassed by redshifts less than
        ``z``. For the case of :math:`\Omega_k = 0` it is a sphere of radius
        `comoving_distance` but it is less intuitive if :math:`\Omega_k` is not.

        Parameters
        ----------
        z : Quantity, positional-only
        z1, z2 : Quantity, positional-only
            Input redshifts. If one argument ``z`` is given, the volume
            :math:`V_c(0, z)` is returned. If two arguments ``z1, z2`` are
            given, the volume :math:`V_c(z_1, z_2)` is returned.

        Returns
        -------
        Quantity
        """
        z1, z2 = (0.0, z1) if z2 is None else (z1, z2)
        return (self.cosmo.comoving_volume(z2) - self.cosmo.comoving_volume(z1)).to(
            _MPC3_UNITS
        )

    def differential_comoving_volume(self, z: InputT, /) -> Quantity:
        r"""Differential comoving volume in cubic Mpc per steradian.

        If :math:`V_c` is the comoving volume of a redshift slice with solid
        angle :math:`\Omega`, this function returns

        .. math::

            \mathtt{differential\_comoving\_volume(z)}
            = \frac{dV_c}{d\Omega \, dz}
            = \frac{c \, d_M^2(z)}{H(z)} \;.

        Parameters
        ----------
        z : Quantity, positional-only
            Input redshift

        Returns
        -------
        Quantity
            The differential comoving volume :math:`dV_c` in Mpc3 sr-1.
        """
        return self.cosmo.differential_comoving_volume(z).to(_MPC3_SR_UNITS)


class ProperDistanceMeasures(ProperDistanceMeasuresAPI[Quantity, InputT]):
    """The object has a proper distance method."""

    cosmo: FLRW

    @overload
    def proper_distance(self, z: InputT, /) -> Quantity:
        ...

    @overload
    def proper_distance(self, z1: InputT, z2: InputT, /) -> Quantity:
        ...

    def proper_distance(self, z1: InputT, z2: InputT | None = None, /) -> Quantity:
        r"""Proper distance :math:`d` in Mpc.

        The proper distance is the distance between two objects at redshifts
        ``z1`` and ``z2``, including the effects of the expansion of the
        universe.

        Parameters
        ----------
        z : Quantity, positional-only
        z1, z2 : Quantity, positional-only
            Input redshifts. If one argument ``z`` is given, the distance
            :math:`d(0, z)` is returned. If two arguments ``z1, z2`` are
            given, the distance :math:`d(z_1, z_2)` is returned.

        Returns
        -------
        Quantity
            The proper distance :math:`d` in Mpc.
        """
        raise NotImplementedError

    @overload
    def proper_time(self, z: InputT, /) -> Quantity:
        ...

    @overload
    def proper_time(self, z1: InputT, z2: InputT, /) -> Quantity:
        ...

    def proper_time(self, z1: InputT, z2: InputT | None = None, /) -> Quantity:
        r"""Proper time :math:`t` in Gyr.

        The proper time is the proper distance divided by
        :attr:`~cosmology.api.CosmologyConstantsNamespace.c`.

        Parameters
        ----------
        z : Quantity, positional-only
        z1, z2 : Quantity, positional-only
            Input redshifts. If one argument ``z`` is given, the time
            :math:`t(0, z)` is returned. If two arguments ``z1, z2`` are
            given, the time :math:`t(z_1, z_2)` is returned.

        Returns
        -------
        Quantity
            The proper time :math:`t` in Gyr.
        """
        raise NotImplementedError


def _lookback_time_z1z2(cosmo: FLRW, z1: InputT, z2: InputT, /) -> NDArray:
    """Lookback time to redshift ``z``. Value in units of Hubble time."""
    return si.quad(cosmo._lookback_time_integrand_scalar, z1, z2)[0]  # noqa: SLF001


class LookbackDistanceMeasures(
    LookbackDistanceMeasuresAPI[Quantity, InputT], HasConstantsNamespace
):
    cosmo: FLRW

    def __post_init__(self) -> None:
        with suppress(AttributeError):
            super().__post_init__()

        self._cosmo_fn: dict[str, Any]
        self._cosmo_fn.update(
            {
                "lookback_time": np.vectorize(_lookback_time_z1z2, excluded=["cosmo"]),
            }
        )

    @overload
    def lookback_distance(self, z: InputT, /) -> Quantity:
        ...

    @overload
    def lookback_distance(self, z1: InputT, z2: InputT, /) -> Quantity:
        ...

    def lookback_distance(self, z1: InputT, z2: InputT | None = None, /) -> Quantity:
        r"""Lookback distance :math:`d_T` in Mpc.

        The lookback distance is the subjective distance it took light to travel
        from redshift ``z1`` to  ``z2``.

        Parameters
        ----------
        z : Quantity, positional-only
        z1, z2 : Quantity, positional-only
            Input redshifts. If one argument ``z`` is given, the distance
            :math:`d_T(0, z)` is returned. If two arguments ``z1, z2`` are
            given, the distance :math:`d_T(z_1, z_2)` is returned.

        Returns
        -------
        Quantity
            The lookback distance :math:`d_T` in Mpc.
        """
        return (self.lookback_time(z1, z2) * self.constants.c).to(u.Mpc)

    @overload
    def lookback_time(self, z: InputT, /) -> Quantity:
        ...

    @overload
    def lookback_time(self, z1: InputT, z2: InputT, /) -> Quantity:
        ...

    def lookback_time(self, z1: InputT, z2: InputT | None = None, /) -> Quantity:
        """Lookback time in Gyr.

        The lookback time is the time that it took light from being emitted at
        one redshift to being observed at another redshift. Effectively it is
        the difference between the age of the Universe at the two redshifts.

        Parameters
        ----------
        z : Quantity, positional-only
        z1, z2 : Quantity, positional-only
            Input redshifts. If one argument ``z`` is given, the time
            :math:`t_T(0, z)` is returned. If two arguments ``z1, z2`` are
            given, the time :math:`t_T(z_1, z_2)` is returned.

        Returns
        -------
        Quantity
        """
        z1, z2 = (0.0, z1) if z2 is None else (z1, z2)
        return (
            self.cosmo.hubble_time * self._cosmo_fn["lookback_time"](self.cosmo, z1, z2)
        ).to(u.Gyr)


class DistanceMeasures(
    TemperatureCMB,
    ScaleFactor,
    ComovingDistanceMeasures,
    ProperDistanceMeasures,
    LookbackDistanceMeasures,
):
    """Cosmology API protocol for distance measures.

    This is a protocol class that defines the standard API for distance
    measures. It is not intended to be instantiated. Instead, it should be used
    for ``isinstance`` checks or as an ABC for libraries that wish to define a
    compatible cosmology class.
    """

    cosmo: FLRW

    # ----------------------------------------------
    # Time

    def age(self, z: InputT, /) -> Quantity:
        """Age of the universe in Gyr at redshift ``z``.

        Parameters
        ----------
        z : Quantity
            Input redshift.

        Returns
        -------
        Quantity
        """
        return self.cosmo.age(z).to(u.Gyr)

    # ----------------------------------------------
    # Angular diameter distance

    @overload
    def angular_diameter_distance(self, z: InputT, /) -> Quantity:
        ...

    @overload
    def angular_diameter_distance(self, z1: InputT, z2: InputT, /) -> Quantity:
        ...

    def angular_diameter_distance(
        self, z1: InputT, z2: InputT | None = None, /
    ) -> Quantity:
        """Angular diameter distance :math:`d_A(z)` in Mpc.

        This gives the proper (sometimes called 'physical') transverse distance
        corresponding to an angle of 1 radian for an object at redshift ``z``
        ([1]_, [2]_, [3]_).

        Parameters
        ----------
        z : Quantity, positional-only
        z1, z2 : Quantity, positional-only
            Input redshifts. If one argument ``z`` is given, the distance
            :math:`d_A(0, z)` is returned. If two arguments ``z1, z2`` are
            given, the distance :math:`d_A(z_1, z_2)` is returned.

        Returns
        -------
        Quantity

        References
        ----------
        .. [1] Weinberg, 1972, pp 420-424; Weedman, 1986, pp 421-424.
        .. [2] Weedman, D. (1986). Quasar astronomy, pp 65-67.
        .. [3] Peebles, P. (1993). Principles of Physical Cosmology, pp 325-327.
        """
        z1, z2 = (0.0, z1) if z2 is None else (z1, z2)
        return self.cosmo.angular_diameter_distance_z1z2(z1, z2).to(u.Mpc)

    # ----------------------------------------------
    # Luminosity distance

    def luminosity_distance(self, z1: InputT, z2: InputT | None = None, /) -> Quantity:
        """Redshift-dependent luminosity distance :math:`d_L(z1, z2)` in Mpc.

        This is the distance to use when converting between the bolometric flux
        from an object at redshift ``z`` and its bolometric luminosity [1]_.

        Parameters
        ----------
        z : Quantity, positional-only
        z1, z2 : Quantity, positional-only
            Input redshifts. If one argument ``z`` is given, the distance
            :math:`d_L(0, z)` is returned. If two arguments ``z1, z2`` are
            given, the distance :math:`d_L(z_1, z_2)` is returned.

        Returns
        -------
        Quantity

        References
        ----------
        .. [1] Weinberg, 1972, pp 420-424; Weedman, 1986, pp 60-62.
        """
        return (z1 + 1.0) * self.transverse_comoving_distance(z1, z2).to(u.Mpc)
