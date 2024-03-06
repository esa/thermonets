import numpy as np
import heyoka as hy
import torch
import spaceweather
import pyatmos
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
    """Converts from Geodetic to Cartesian

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

def earth_rotation_matrix(mjd_date,symbolic=False):
    """
    This function returns the Earth rotation angle at a given date: https://en.wikipedia.org/wiki/Sidereal_time#ERA.

    Args:
        - mjd_date (`float`): modified julian date
    
    Returns:
        - `list`: Earth rotation matrix
    """
    if symbolic:
        backend = hy
    else:
        backend = np
    era= 2*np.pi*(0.7790572732640 + 1.00273781191135448*(mjd_date+2400000.5-2451545.0))
    R=[[backend.cos(era),backend.sin(era),0],[-backend.sin(era),backend.cos(era),0],[0,0,1]]
    return R

def mean_absolute_percentage_error(y_true, y_pred):
    """
    Compute the mean absolute percentage error (MAPE) between true and predicted values.
    
    Args:
        y_true (`torch.tensor`): True values.
        y_pred (`torch.tensor`): Predicted values.
        
    Returns:
        `torch.tensor`: Mean absolute percentage error.
    """
    return torch.mean(torch.abs((y_true - y_pred) / y_true)) * 100

def normalize_min_max(data, min_val = None, max_val = None):
    """_summary_

    Args:
        data (_type_): _description_
        min_val (_type_): _description_
        max_val (_type_): _description_

    Returns:
        _type_: _description_
    """
    if min_val is None:
        min_val = data.min()
    
    if max_val is None:
        max_val = data.max()
    
    normalized_data = (2 * (data - min_val) / (max_val - min_val)) - 1
    return normalized_data

def unnormalize_min_max(data,min_val,max_val):
    """_summary_

    Args:
        data (_type_): _description_
        min_val (_type_): _description_
        max_val (_type_): _description_

    Returns:
        _type_: _description_
    """
    unnormalized_data = 1/2 * (data + 1) * (max_val - min_val) + min_val
    return unnormalized_data

def mjd(date):
    """
    Converts a datetime object to a modified julian date. The julian date conversion is the one valid from 1900 to 2100, and reported in Chapter 3.5 of 
    Vallado's book Fundamentals of Astrodynamics and Applications.

    Args:
        - date (`datetime.datetime` or `list`): date(s) as datetime object(s)

    Returns:
        - mjd (`float` or `list`): modified julian date
    """
    if isinstance(date,list):
        year,month,day,hour,minute,second,microsecond=np.zeros((len(date),)),np.zeros((len(date),)), np.zeros((len(date),)), np.zeros((len(date),)), np.zeros((len(date),)), np.zeros((len(date),)), np.zeros((len(date),))
        for idx,d in enumerate(date):
            year[idx], month[idx], day[idx], hour[idx], minute[idx], second[idx], microsecond[idx]= d.year, d.month, d.day, d.hour, d.minute, d.second, d.microsecond  
    else:
        year, month, day, hour, minute, second, microsecond= date.year, date.month, date.day, date.hour, date.minute, date.second, date.microsecond

    #first we compute the julian date:
    jd=(367.0 * year -
            7.0 * (year + ((month + 9.0) // 12.0)) * 0.25 // 1.0 +
            275.0 * month // 9.0 +
            day + 1721013.5)
    fr=(microsecond/1e6 + second + minute * 60.0 + hour * 3600.0) / 86400.0
    #and then the modified julian date:
    mjd = jd+fr-2400000.5
    return mjd

def get_nrlmsise00_spaceweather_indices(date):
    """
    Takes a date, or list of dates, and returns the corresponding ap, f107, f107A (either as single values or arrays).

    Args:
        - date (`datetime.datetime` or `list` of `datetime.datetime`): date or list of dates at which the space weather is queried

    Returns:
        - ap (`int` or `np.array`): Ap value(s) corresponding to that date(s)
        - F10.7 (`float` or `np.array`): F10.7 value(s) corresponding to that date(s)
        - F10.7A (`float` or `np.array`): F10.7 81-day average value(s) corresponding to that date(s)
    """
    sw_data=spaceweather.sw_daily()
    ap_data = sw_data[["Apavg"]]
    f107_data = sw_data[["f107_obs"]]
    f107a_data = sw_data[["f107_81ctr_obs"]]
    if isinstance(date,list):
        #we prepare the input dates:
        dates=[f'{int(d.year)}-{int(d.month)}-{int(d.day)}' for d in date]
        #we extract the space weather indices
        ap=ap_data.loc[dates].values.flatten()
        f107=f107_data.loc[dates].values.flatten()
        f107A=f107a_data.loc[dates].values.flatten()
    else:
        ap=ap_data.loc[f'{int(date.year)}-{int(date.month)}-{int(date.day)}'].values[0]
        f107=f107_data.loc[f'{int(date.year)}-{int(date.month)}-{int(date.day)}'].values[0]
        f107A=f107a_data.loc[f'{int(date.year)}-{int(date.month)}-{int(date.day)}'].values[0]
    return ap,f107,f107A

def get_nrlmsise00_attributes(date):
    ap,f107,f107A = get_nrlmsise00_spaceweather_indices(date)
    doy = date.timetuple().tm_yday
    sid = date.hour*3600+date.minute*60+date.second+date.microsecond/1e6
    return ap,f107,f107A, doy,sid

def get_jb08_spaceweather_indices(date,swdata=None):
    """
    Takes a date, or list of dates, and returns the corresponding ap, f107, f107A (either as single values or arrays).

    Args:
        - date (`datetime.datetime` or `list` of `datetime.datetime`): date or list of dates at which the space weather is queried

    Returns:
        - F10.7 (`float` or `np.array`): F10.7 value(s) corresponding to that date(s)
        - F10.7A (`float` or `np.array`): F10.7 81-day average value(s) corresponding to that date(s)
        - S10.7 (`float` or `np.array`): S10.7 value(s) corresponding to that date(s)
        - S10.7A (`float` or `np.array`): S10.7 81-day average value(s) corresponding to that date(s)
        - M10.7 (`float` or `np.array`): F10.7 value(s) corresponding to that date(s)
        - M10.7A (`float` or `np.array`): F10.7 81-day average value(s) corresponding to that date(s)
        - Y10.7 (`float` or `np.array`): F10.7 value(s) corresponding to that date(s)
        - Y10.7A (`float` or `np.array`): F10.7 81-day average value(s) corresponding to that date(s)
        - dDst/dT (`float` or `np.array`): Dst index change due to temperature change
    """
    from pyatmos.jb2008.spaceweather import get_sw
    if swdata is None:
        swfile = pyatmos.download_sw_jb2008()
        swdata = pyatmos.read_sw_jb2008(swfile)
    t_mjd = mjd(date)
    if isinstance(t_mjd,float):
        return get_sw(swdata,t_mjd)
    else:
        f107,f107a,s107,s107a,m107,m107a,y107,y107a,dDstdT=np.zeros((len(t_mjd),)), np.zeros((len(t_mjd),)), np.zeros((len(t_mjd),)), np.zeros((len(t_mjd),)), np.zeros((len(t_mjd),)), np.zeros((len(t_mjd),)), np.zeros((len(t_mjd),)), np.zeros((len(t_mjd),)) ,np.zeros((len(t_mjd),))
        for i, v in enumerate(t_mjd):
            f107[i],f107a[i],s107[i],s107a[i],m107[i],m107a[i],y107[i],y107a[i],dDstdT[i]=get_sw(swdata,v)
    return f107,f107a,s107,s107a,m107,m107a,y107,y107a,dDstdT

def get_jb08_attributes(date):
    f107,f107a,s107,s107a,m107,m107a,y107,y107a,dDstdT = get_jb08_spaceweather_indices(date)
    doy = date.timetuple().tm_yday
    sid = date.hour*3600+date.minute*60+date.second+date.microsecond/1e6
    return f107,f107a,s107,s107a,m107,m107a,y107,y107a,dDstdT,doy,sid