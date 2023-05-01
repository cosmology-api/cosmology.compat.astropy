"""The Cosmology API compatability wrapper for Astropy."""

from __future__ import annotations

from contextlib import suppress
from typing import TYPE_CHECKING, Any, overload

import astropy.units as u
import numpy as np
import scipy.integrate as si
from astropy.cosmology import FLRW

__all__: list[str] = []


if TYPE_CHECKING:
    from astropy.cosmology import FLRW
    from astropy.units import Quantity
    from numpy.typing import NDArray

    from cosmology.compat.astropy._core import InputT

_MPC3_UNITS = u.Mpc**3
_MPC3_SR_UNITS = _MPC3_UNITS / u.sr


class TemperatureCMB:
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


class ScaleFactor:
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


class ComovingDistanceMeasures:
    cosmo: FLRW

    def comoving_distance(self, z: InputT, zp: InputT | None = None, /) -> Quantity:
        r"""Comoving line-of-sight distance :math:`d_c(z1, z2)` in Mpc.

        The comoving distance along the line-of-sight between two objects
        remains constant with time for objects in the Hubble flow.

        Parameters
        ----------
        z, zp : Quantity, positional-only
            Input redshifts. If ``zp`` is `None` (default), then the distance
            :math:`d_c(0, z)` is returned, otherwise the distance :math:`d_c(z,
            zp)` is returned.

        Returns
        -------
        Quantity
            The comoving distance :math:`d_c(z1, z2)` in Mpc, where ``(z1, z2)``
            is (0, `z`) if `zp` is `None` else (`z`, `zp`).
        """
        z1, z2 = (0.0, z) if zp is None else (z, zp)
        return self.cosmo._comoving_distance_z1z2(z1, z2).to(u.Mpc)  # noqa: SLF001

    def transverse_comoving_distance(
        self, z: InputT, zp: InputT | None = None, /
    ) -> Quantity:
        r"""Transverse comoving distance :math:`d_M(z1, z2)` in Mpc.

        This value is the transverse comoving distance at redshift ``z``
        corresponding to an angular separation of 1 radian. This is the same as
        the comoving distance if :math:`\Omega_k` is zero (as in the current
        concordance Lambda-CDM model).

        Parameters
        ----------
        z, zp : Quantity, positional-only
            Input redshifts. If ``zp`` is `None` (default), then the distance
            :math:`d_M(0, z)` is returned, otherwise the distance :math:`d_M(z,
            zp)` is returned.

        Returns
        -------
        Quantity
            The comoving transverse distance :math:`d_M(z1, z2)` in Mpc, where
            ``(z1, z2)`` is (0, `z`) if `zp` is `None` else (`z`, `zp`).
        """
        z1, z2 = (0.0, z) if zp is None else (z, zp)
        return self.cosmo._comoving_transverse_distance_z1z2(z1, z2).to(  # noqa: SLF001
            u.Mpc
        )

    def comoving_volume(self, z: InputT, zp: InputT | None = None, /) -> Quantity:
        r"""Comoving volume in cubic Mpc.

        This is the volume of the universe encompassed by redshifts less than
        ``z``. For the case of :math:`\Omega_k = 0` it is a sphere of radius
        `comoving_distance` but it is less intuitive if :math:`\Omega_k` is not.

        Parameters
        ----------
        z, zp : Quantity, positional-only
            Input redshifts. If ``zp`` is `None` (default), then the
            volume :math:`V_c(0, z)` is returned, otherwise the
            volume :math:`V_c(z, zp)` is returned.

        Returns
        -------
        Quantity
            The comoving volume :math:`V_c(z1, z2)` in Mpc, where
            ``(z1, z2)`` is (0, `z`) if `zp` is `None` else (`z`, `zp`).
        """
        z1, z2 = (0.0, z) if zp is None else (z, zp)
        return (self.cosmo.comoving_volume(z2) - self.cosmo.comoving_volume(z1)).to(
            _MPC3_UNITS
        )

    def differential_comoving_volume(
        self, z: InputT, zp: InputT | None = None, /
    ) -> Quantity:
        r"""Differential comoving volume in cubic Mpc per steradian.

        If :math:`V_c` is the comoving volume of a redshift slice with solid
        angle :math:`\Omega`, this function returns

        .. math::

            \mathtt{dvc(z)}
            = \frac{1}{d_H^3} \, \frac{dV_c}{d\Omega \, dz}
            = \frac{x_M^2(z)}{E(z)}
            = \frac{\mathtt{xm(z)^2}}{\mathtt{ef(z)}} \;.

        Parameters
        ----------
        z, zp : Quantity, positional-only
            Input redshifts. If ``zp`` is `None` (default), then the
            differential volume :math:`dV_c(0, z)` is returned, otherwise the
            differential volume :math:`dV_c(z, zp)` is returned.

        Returns
        -------
        Quantity
            The differential comoving volume :math:`dV_c(z1, z2)` in Mpc,
            where ``(z1, z2)`` is (0, `z`) if `zp` is `None` else (`z`, `zp`).
        """
        z1, z2 = (0.0, z) if zp is None else (z, zp)
        return (
            self.cosmo.differential_comoving_volume(z2)
            - self.cosmo.differential_comoving_volume(z1)
        ).to(_MPC3_SR_UNITS)


def _lookback_time_z1z2(cosmo: FLRW, z1: InputT, z2: InputT, /) -> NDArray:
    """Lookback time to redshift ``z``. Value in units of Hubble time."""
    return si.quad(cosmo._lookback_time_integrand_scalar, z1, z2)[0]  # noqa: SLF001


class DistanceMeasures(TemperatureCMB, ScaleFactor, ComovingDistanceMeasures):
    """Cosmology API protocol for distance measures.

    This is a protocol class that defines the standard API for distance
    measures. It is not intended to be instantiated. Instead, it should be used
    for ``isinstance`` checks or as an ABC for libraries that wish to define a
    compatible cosmology class.
    """

    cosmo: FLRW
    _cosmo_fn: dict[str, Any]

    def __post_init__(self) -> None:
        with suppress(AttributeError):
            super().__post_init__()  # type: ignore[misc]

        self._cosmo_fn: dict[str, Any]
        self._cosmo_fn.update(
            {
                "lookback_time": np.vectorize(_lookback_time_z1z2, excluded=["cosmo"]),
            }
        )

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

    # ----------------------------------------------
    # Angular diameter distance

    def angular_diameter_distance(
        self, z: InputT, zp: InputT | None = None, /
    ) -> Quantity:
        """Angular diameter distance :math:`d_A(z)` in Mpc.

        This gives the proper (sometimes called 'physical') transverse distance
        corresponding to an angle of 1 radian for an object at redshift ``z``
        ([1]_, [2]_, [3]_).

        Parameters
        ----------
        z, zp : Quantity, positional-only
            Input redshifts. If ``zp`` is `None` (default), then the distance
            :math:`d_A(0, z)` is returned, otherwise the distance :math:`d_A(z,
            zp)` is returned.

        Returns
        -------
        Quantity
            The angular diameter distance :math:`d_A(z1, z2)` in Mpc, where
            ``(z1, z2)`` is (0, `z`) if `zp` is `None` else (`z`, `zp`).

        References
        ----------
        .. [1] Weinberg, 1972, pp 420-424; Weedman, 1986, pp 421-424.
        .. [2] Weedman, D. (1986). Quasar astronomy, pp 65-67.
        .. [3] Peebles, P. (1993). Principles of Physical Cosmology, pp 325-327.
        """
        z1, z2 = (0.0, z) if zp is None else (z, zp)
        return self.cosmo.angular_diameter_distance_z1z2(z1, z2).to(u.Mpc)

    # ----------------------------------------------
    # Luminosity distance

    def luminosity_distance(self, z: InputT, zp: InputT | None = None, /) -> Quantity:
        """Redshift-dependent luminosity distance :math:`d_L(z1, z2)` in Mpc.

        This is the distance to use when converting between the bolometric flux
        from an object at redshift ``z`` and its bolometric luminosity [1]_.

        Parameters
        ----------
        z, zp : Quantity, positional-only
            Input redshifts. If ``zp`` is `None` (default), then the
            distance :math:`d_L(0, z)` is returned, otherwise the
            distance :math:`d_L(z, zp)` is returned.

        Returns
        -------
        Quantity
            The luminosity distance :math:`d_L(z1, z2)` in Mpc, where
            ``(z1, z2)`` is (0, `z`) if `zp` is `None` else (`z`, `zp`).

        References
        ----------
        .. [1] Weinberg, 1972, pp 420-424; Weedman, 1986, pp 60-62.
        """
        return (z + 1.0) * self.transverse_comoving_distance(z, zp).to(u.Mpc)
