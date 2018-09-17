import numpy as np

def nmad(data):
    """Normalized NMAD=1.48 * `~.astropy.stats.median_absolute_deviation`
    
    """
    import astropy.stats
    return 1.48*astropy.stats.median_absolute_deviation(data)

def shapely_polygon_to_region(shape, prefix=['image\n']):
    """
    Convert a `~shapely` region to a DS9 region polygon
    
    TBD
    
    Parameters
    ----------
    shape : `~shapely.geometry` object
        Can handle `Polygon` and `MultiPolygon` shapes, or any shape with 
        a valid `boundary.xy` attribute.
    
    prefix : list of strings
        Strings to prepend to the polygon listself.
    
    Returns
    -------
    polystr : list of strings
        Region strings
        
    """
    if hasattr(shape, '__len__'):
        multi = shape
    else:
        multi = [shape]
    
    polystr = [p for p in prefix]
    for p in multi:
        try:
            if hasattr(p.boundary, '__len__'):
                for pb in p.boundary:
                    coords = np.array(pb.xy).T
                    p_i = coords_to_polygon_region(coords)
                    polystr.append(p_i)
            else:
                coords = np.array(p.boundary.xy).T
                p_i = coords_to_polygon_region(coords)
                polystr.append(p_i)
        except:
            pass
    
    return polystr

def boundary_coords(shape):
    coo = []
    
def coords_to_polygon_region(coords):
    """
    Coordinate list to `ds9` polygon string
    
    Parameters
    ----------
    coords : (N,2) `~numpy.ndarray`
        Polygon vertices.
    
    Returns
    -------
    polystr : type
    """
    polystr = 'polygon('+','.join(['{0}'.format(xi) for xi in list(coords.flatten())])+')\n'
    return polystr
    
LINE_PARAM_IMAGING = {'NK': 3, 'lo':5, 'hi': 10, 'line_length': 140, 'line_thresh': 2, 'line_gap': 15, 'med_size':5}

LINE_PARAM_GRISM = {'NK': 16, 'hi': 7, 'lo': 4, 'line_length': 180, 'line_thresh': 2, 'line_gap': 15, 'med_size':5}
