#!python3
"""
    GPHS/PHYS 441 plate rotation functions
    Rupert Sutherland - Geodynamics module
"""
import numpy as np
import matplotlib.pyplot as plt

rEarth = 6371.0  # km

def length(vector):
    """
    Find the length of a vector of floats.
    
    Parameters
    ----------
    vector : n-dimensional array-like object

    Returns
    -------
    vectorLength : float with positive or zero value
    """
    v = np.array(vector, dtype=float)
    return np.sqrt(v.dot(v))

def positionVector(lonlat):
    """
    Converts geocentric [longitude,latitude] decimal degrees point location 
    to Earth-centred Earth-fixed (ECEF) position vector [x,y,z] (unit length).
    x = towards (lon=0,  lat=0 )
    y = towards (lon=90E, lat=0 )
    z = towards (lon=0,  lat=90N)

    Parameters
    ----------
    lonlat : array-like object [longitude,latitude] 

    Returns
    -------
    positionVector : np.array([x,y,z]) with unit length
    """    
    lon = np.radians(float(lonlat[0]))
    lat = np.radians(float(lonlat[1]))
    x = np.cos(lon)*np.cos(lat)
    y = np.sin(lon)*np.cos(lat)
    z = np.sin(lat)
    return np.array([x,y,z])

def lonlat(vector):
    """
    Converts a 3-vector to [longitude, latitude] decimal degrees.
    If not unit length, it is the projected position.
    Longitude is defined 0-360, to avoid problems at -180/180 near NZ

    Parameters
    ----------
    vector : [x,y,z] (array-like object)

    Returns
    -------
    np.array([longitude,latitude])
    """
    # Ensure a 3-vector of floating point numbers else raise exception
    v = np.array(vector, dtype=float)
    if np.shape(v) != (3,) :
        raise ValueError('lonlat(): expects 3-vector')
        
    # Convert to position vector
    L = length(v)
    x = v[0]/L
    y = v[1]/L
    z = v[2]/L
    
    lat = np.degrees(np.arcsin(z))
    
    # longitude: arctan only defined for -90 to 90, and error if x=0
    # longitude: require -180 to +180, so use sign of x and y
    if x == 0 :
        lon = np.sign(y) * 90.
    elif x > 0 :
        lon = np.degrees(np.arctan(y/x))
    elif y == 0 :
        lon = 180. + np.degrees(np.arctan(y/x))
    else:
        lon = np.sign(y) * 180. + np.degrees(np.arctan(y/x))
    
    # comment out the line below to keep longitude as -180/180
    lon = np.mod(lon,360)
    
    return np.array([lon,lat])

def M(h):
    """
    Function that returns the 3x3 matrix H from 3-vector h, such that
        Hu = h x u
    where u is 3-vector. In python, then this should be TRUE:
    np.dot(H,u) == np.cross(h,u)

    Parameters
    ----------
    h : array-like 3-vector of floats [hx,hy,hz]

    Returns
    -------
    H : 3x3 np.array 

    """
    h = np.array(h, dtype=float)
    if np.shape(h) != (3,) :
        raise ValueError('M(h): expects 3-vector h')
        
    return np.array([[ 0   ,-h[2], h[1]],
                     [ h[2], 0   ,-h[0]],
                     [-h[1], h[0], 0   ]])

def rotationMatrix(h):
    """
    Converts the 3-vector h into a 3x3 rotation matrix.

    Parameters
    ----------
    h : 3-vector (array-like)

    Returns
    -------
    R : rotation matrix np.array with shape = (3, 3)

    """
    h = np.array(h, dtype=float)
    if np.shape(h) != (3,) :
        raise ValueError('rotationMatrix(h): expects 3-vector h')        
    angle = length(h)
    I = np.identity(3)
    if angle == 0 :
        return I
    else:
        polePositionVector = h/angle
        P = M(polePositionVector)       
        return I + np.sin(angle)*P + (1. - np.cos(angle))*np.dot(P,P)

def hVector(R):
    """
    Find the 3-vector of rotation parameters used to define a rotation matrix.
    
    Parameters
    ----------
    R  : 3x3 numpy array rotation matrix

    Returns
    -------
    h : numpy array 3-vector rotation parameters (length is angle in radians)

    """
    angle = np.arccos((np.trace(R)-1)/2)
    vec = np.array([R[2,1]-R[1,2],R[0,2]-R[2,0],R[1,0]-R[0,1]])
    pole = vec/length(vec)
    
    return angle * pole

def readGMTxy(filename):
    """
    Reads a multi-segment gmt xy file.

    Parameters
    ----------
    filename : path to file
        
    Returns
    -------
    xyList : list [[[x,...],[y,...]],[[x,...],[y,...]],...]

    """
    feature = list()
    x = list()
    y = list()
    with open(filename,'r') as f :
        data = f.readlines()
    for line in data:
        w = line.strip().split()
        #print(data)
        if w[0] == '>' :
            if x :
                feature.append([x,y])
            x = list()
            y = list()
        else:
            x.append(float(w[0]))
            y.append(float(w[1]))
    return feature

def mapSetup(MAP_BOUNDS=[170,180,-42,-34],TITLE="New Zealand"):
    """
    Instantiates a new map of New Zealand for plotting data.
    Cylindrical equidistant, scaled for Wellington latitude (41S).
    Reads coastline.xy and adds to map.

    Parameters
    ----------
    MAP_BOUNDS : array-like [lonMin,lonMax,latMin,latMax]
    TITLE      : string

    Returns
    -------
    fig,ax   : matplotlib figure and axes objects

    """
    fig,ax = plt.subplots(1, 1, figsize=(8,8))
    latitude = np.radians(41.3)
    scale = np.cos(latitude)
    ax.set_aspect(1/scale)
    ax.set_xlim(MAP_BOUNDS[0],MAP_BOUNDS[1])
    ax.set_ylim(MAP_BOUNDS[2],MAP_BOUNDS[3])
    ax.set_title(TITLE)
    coast = readGMTxy('nzcoast.xy')
    for line in coast:
        x,y = line
        ax.plot(x,y,linewidth=1,color='darkgrey')        
    return fig,ax

if __name__ == '__main__' :
    """
    unit tests - only executes if run as stand-alone code, not when imported
    """    
    fig,ax = mapSetup()
    plt.show()
