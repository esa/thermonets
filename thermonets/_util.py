import numpy as np
import heyoka as hy

# WGS84 values
a_earth = 6378137.0
b_earth = 6356752.314245


def cart2geo(
    x, y, z, e2=1 - b_earth**2 / a_earth**2, R_eq=a_earth, iters=4, symbolic=False
):
    """Converts from Cartesian to Geodetic coordinates
       We use the iterations from: "Physical Geodesy" by Heiskanen and Moritz, 1967

    Args:
        x (`float` or `hy.expression`): x coordinate (units must be consistent with R_e, defaults to m).
        y (`float` or `hy.expression`): y coordinate (units must be consistent with R_eq, defaults to m).
        z (`float` or `hy.expression`): z coordinate (units must be consistent with R_eq, defaults to m).
        e2 (`float`, optional): 1-b^2/a^2 the ellipsoid oblateness (squared).
        R_eq (`float`, optional): the equatorial radius (units must be consistent with x,y,z, defaults to WGS84 Earth equatorial radius in m).
        iters (int, optional): Nmber of fixed iterations. Defaults to 4.
        symbolic (bool, optional): Performs the computations using heyoka as a backend and thus computes a symbolic expression. Defaults to False.

    Returns:
        h, phi, long: altitude in same units as x,y,z and R_eq, latitude in [-pi/2, pi/2] and longitude in [-pi,pi].
    """
    if symbolic:
        backend = hy
        atan2 = hy.atan2
        atan = hy.atan
    else:
        backend = np
        atan2 = np.arctan2
        atan = np.arctan
    long = atan2(y, x)
    p = backend.sqrt(x**2 + y**2)
    phi = atan(z / p / (1 - e2))
    # we iterate to improve the solution
    for _ in range(iters):
        N = R_eq / backend.sqrt(1 - e2 * backend.sin(phi) ** 2)
        h = p / backend.cos(phi) - N
        phi = atan(z / p / (1 - e2 * N / (N + h)))
    return h, phi, long


def geo2cart(h, lat, lon, e2=1 - b_earth**2 / a_earth**2, R_eq=a_earth, symbolic=False):
    """COnverts from Geodetic to Cartesian

    Args:
        h (`float` or `hy.expression`): Altitude.
        lat (`float` or `hy.expression`): Geodetic latitude.
        lon (`float` or `hy.expression`): Geodetic Longitude.
        e2 (`float`, optional): 1-b^2/a^2 the ellipsoid oblateness (squared).
        R_eq (`float`, optional): the equatorial radius (units must be consistent with x,y,z, defaults to WGS84 Earth equatorial radius in m).
        symbolic (bool, optional): Performs the computations using heyoka as a backend and thus computes a symbolic expression. Defaults to False.

    Returns:
        x, y, z: cartesian coordinates (same units as h and R_eq).

    """
    if symbolic:
        backend = hy
    else:
        backend = np
    N = R_eq / backend.sqrt(1 - e2 * backend.sin(lat) ** 2)
    x = (N + h) * backend.cos(lat) * backend.cos(lon)
    y = (N + h) * backend.cos(lat) * backend.sin(lon)
    z = ((1 - e2) * N + h) * backend.sin(lat)
    return x, y, z

