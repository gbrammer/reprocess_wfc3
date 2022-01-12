import os
import sys
import numpy as np
import warnings


def which_calwf3(executable='calwf3.e'):
    """
    Return result of `which calwf3.e`
    """
    import subprocess
    
    env_bin = os.path.join(sys.exec_prefix, 'bin')
    if (env_bin not in os.getenv('PATH')) & os.path.exists(env_bin):
        _path = ':'.join([env_bin, os.getenv('PATH')])
    else:
        _path = os.getenv('PATH')

    proc = subprocess.Popen(
        ['which', executable],
        stderr=subprocess.STDOUT,
        stdout=subprocess.PIPE,
        env={'PATH':_path}
    )
    
    result = ''.join([line.decode('utf8').strip() 
                      for line in proc.stdout])
    
    return result


def run_calwf3(file, clean=True, log_func=None):
    """
    Call calwf3.e executable with PATH fixes
    
    Parameters
    ----------
    file : str
        Filename, e.g., `ixxxxxxq_raw.fits`

    clean : bool
        Clean existing `ima` and `flt` files if `file` is has 'raw' extension
        
    log_func : func
        Function to handle `stdout` from the `calwf3.e` executable
    
    Returns
    -------
    return_code : int
        Value returned by the `calwf3.e` process
        
    """
    import subprocess
    
    env_bin = os.path.join(sys.exec_prefix, 'bin')
    if (env_bin not in os.getenv('PATH')) & os.path.exists(env_bin):
        _path = ':'.join([env_bin, os.getenv('PATH')])
    else:
        _path = os.getenv('PATH')
    
    if clean & ('_raw.fits' in file):
        for ext in ['_ima.fits','_flt.fits']:
            _file = file.replace('_raw.fits',ext)
            if os.path.exists(_file):
                print(f'$ rm {_file}')  
                os.remove(_file)
    
    exec_file = which_calwf3()
    print(f'$ {exec_file} {file}')  
    
    proc = subprocess.Popen(
        ['calwf3.e', file],
        stderr=subprocess.STDOUT,
        stdout=subprocess.PIPE,
        env={'PATH':_path, 'iref':os.getenv('iref')}
    )
    
    if log_func is not None:
        for line in proc.stdout:
            log_func(line.decode('utf8'))

    return_code = proc.wait()
    return return_code


def set_warnings(numpy_level='ignore', astropy_level='ignore'):
    """
    Set global numpy and astropy warnings
    
    Parameters
    ----------
    numpy_level : {'ignore', 'warn', 'raise', 'call', 'print', 'log'}
        Numpy error level (see `~numpy.seterr`).
        
    astropy_level : {'error', 'ignore', 'always', 'default', 'module', 'once'}
        Astropy error level (see `~warnings.simplefilter`).
    
    """
    from astropy.utils.exceptions import AstropyWarning
    
    np.seterr(all=numpy_level)
    warnings.simplefilter(astropy_level, category=AstropyWarning)


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
