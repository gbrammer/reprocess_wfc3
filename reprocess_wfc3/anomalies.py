import numpy as np
from numpy import ma

import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.figure import Figure
from matplotlib.backends.backend_agg import FigureCanvasAgg

from skimage.feature import canny
from skimage.transform import probabilistic_hough_line
from skimage import morphology as morph

import scipy.ndimage as nd

from shapely.geometry import LineString

from . import utils

def compute_earthshine(cube, dq, time, sl_left=slice(40,240), sl_right=slice(760,960)):
    """
    Check for scattered earthshine light as the difference between median 
    values of the left and right column-averages of an image.
    
    `cube`, `dq` and `time` come from `split_multiaccum`, run on an IMA file.
    """
            
    diff = ma.masked_array(np.diff(cube, axis=0), mask=(dq[1:,:,:] > 0))
    
    column_average = ma.median(diff, axis=1)
    left = ma.median(column_average[:, sl_left], axis=1)/np.diff(time)
    right = ma.median(column_average[:, sl_right], axis=1)/np.diff(time)
    
    return left-right

def test():
    import glob
    import os
    import astropy.io.fits as pyfits
    import wfc3tools
    
    from . import utils, reprocess_wfc3
    from .reprocess_wfc3 import split_multiaccum
    from . import anomalies
    
    files=glob.glob('*ima.fits'); i=-1
    files.sort()
    
    i+=1; file=files[i]
        
    for file in files:

        if not os.path.exists(file.replace('_raw','_ima')):
            try:
                os.remove(file.replace('_raw','_flt'))
                os.remove(file.replace('_raw','_ima'))
            except:
                pass

            ima = pyfits.open(file, mode='update')
            ima[0].header['CRCORR'] = 'PERFORM'
            ima.flush()
            
            wfc3tools.calwf3(file, log_func=reprocess_wfc3.silent_log)

        ima = pyfits.open(file.replace('_raw','_ima'))
        cube, dq, time, NS = split_multiaccum(ima, scale_flat=False)

        is_grism = ima[0].header['FILTER'] in ['G102','G141']
        if is_grism:
            params = [LINE_PARAM_GRISM_LONG, LINE_PARAM_GRISM_SHORT]
        else:
            params = [LINE_PARAM_IMAGING_LONG, LINE_PARAM_IMAGING_SHORT]

        out = trails_in_cube(cube, dq, time,
                             line_params=params[0],
                             subtract_column=is_grism)

        image, edges, lines = out

        if len(lines) == 0:
            out = trails_in_cube(cube, dq, time,
                                 line_params=params[1],
                                 subtract_column=is_grism)

            image, edges, lines = out

        root = ima.filename().split('_')[0]
        print(root, len(lines))

        if len(lines) > 0:
            fig = sat_trail_figure(image, edges, lines, label=root)
            #fig.savefig('{0}_trails.png'.format(root))
            canvas = FigureCanvasAgg(fig)
            canvas.print_figure(root+'_trails.png', dpi=200)
            
            reg = anomalies.segments_to_mask(lines, params[0]['NK'],
                                             image.shape[1],
                                             buf=params[0]['NK']*(1+is_grism))

            fpr = open('{0}_trails.reg'.format(root),'w')
            fpr.writelines(reg)
            fpr.close()
        
# Imaging
LINE_PARAM_IMAGING_LONG = {'sn_thresh': 4, 'line_length': 700, 'line_thresh': 2, 'med_size': [12,1], 'use_canny': True, 'lo': 5, 'hi': 10, 'NK': 3, 'line_gap': 7}

LINE_PARAM_IMAGING_SHORT = {'sn_thresh': 4, 'line_length': 100, 'line_thresh': 2, 'med_size': [12,1], 'use_canny': True, 'lo': 9, 'hi': 14, 'NK': 3, 'line_gap': 1}


# # Grism
# LINE_PARAM_GRISM_SHORT = {'line_length': 400, 'line_thresh': 2, 'NK': 30, 'med_size': 5, 'use_canny': False, 'hi': 7, 'lo': 3, 'line_gap': 1, 'sn_thresh':4}
# 
# LINE_PARAM_GRISM_LONG = {'line_length': 600, 'line_thresh': 2, 'NK': 30, 'med_size': 5, 'use_canny': False, 'hi': 7, 'lo': 3, 'line_gap': 1, 'sn_thresh':2}

LINE_PARAM_GRISM_LONG = {'sn_thresh': 2, 'line_length': 600, 'line_thresh': 2, 'use_canny': True, 'med_size': [12,1], 'lo': 3, 'hi': 7, 'NK': 30, 'line_gap': 3}

LINE_PARAM_GRISM_SHORT = {'sn_thresh': 2, 'line_length': 250, 'line_thresh': 2, 'use_canny': True, 'med_size': [12,1], 'lo': 10, 'hi': 15, 'NK': 30, 'line_gap': 1}

def auto_flag_trails(cube, dq, time, is_grism=False, root='satellite', earthshine_mask=False, pop_reads=[]):
    """
    Automatically flag satellite trails
    """
    
    #is_grism = ima[0].header['FILTER'] in ['G102','G141']
    print('reprocess_wfc3.anomalies: {0}, is_grism={1}'.format(root, is_grism))
    if is_grism:
        params = [LINE_PARAM_GRISM_LONG, LINE_PARAM_GRISM_SHORT]
    else:
        params = [LINE_PARAM_IMAGING_LONG, LINE_PARAM_IMAGING_SHORT]
        
    print('reprocess_wfc3.anomalies: {0}, Long trail params\n   {1}'.format(root, params[0]))
    out = trails_in_cube(cube, dq, time,
                         line_params=params[0],
                         subtract_column=is_grism,
                         earthshine_mask=earthshine_mask, 
                         pop_reads=pop_reads)

    image, edges, lines = out
    
    is_short=False
    if len(lines) == 0:
        is_short=True
        print('reprocess_wfc3.anomalies: {0}, Short trail params\n   {1}'.format(root, params[1]))
        #print('Try trail params: {0}'.format(params[1]))
        out = trails_in_cube(cube, dq, time,
                             line_params=params[1],
                             subtract_column=is_grism,
                             earthshine_mask=earthshine_mask,
                             pop_reads=pop_reads)
    
        image, edges, lines = out
    
    #root = ima.filename().split('_')[0]
    print('reprocess_wfc3: {0} has {1} satellite trails'.format(root, len(lines)))
    
    fig = sat_trail_figure(image, edges, lines, label=root)
    canvas = FigureCanvasAgg(fig)
    canvas.print_figure(root+'_trails.png', dpi=200)

    # fig.savefig('{0}_trails.png'.format(root))
    # plt.close(fig)
    
    if len(lines) > 0:
        reg = segments_to_mask(lines, params[0]['NK'], image.shape[1],
                               buf=params[0]['NK']*4)
                                         
        fpr = open('{0}.01.mask.reg'.format(root),'a')
        fpr.writelines(reg)
        fpr.close()
        
def trails_in_cube(cube, dq, time, line_params=LINE_PARAM_IMAGING_LONG, subtract_column=True, earthshine_mask=False, pop_reads=[]):
    """
    Find satellite trails in MultiAccum sequence
    
    Parameters
    ----------
    `cube`, `dq` and `time` come from `split_multiaccum`, run on an IMA file.
    
    line_params : dict
        Line-finding parameters, see `~reprocess_wfc3.anomalies.LINE_PARAM_IMAGING`.
    
    subtract_column : bool
        Subtract column average from the smoothed image.
    
    Returns
    -------
    image : `~numpy.ndarray`
        The image from which the linear features were detected.
    
    edges : `~numpy.ndarray`
        The `skimage.feature.canny` edge image.
        
    lines : (N,2,2)
        `N` linear segments found on the edge image with
        `skimage.transform.probabilistic_hough_line`.
    
    """    
    from .reprocess_wfc3 import split_multiaccum
    
    utils.set_warnings()
    
    ## Line parameters
    lp = LINE_PARAM_IMAGING_LONG
    for k in line_params:
        lp[k] = line_params[k]
    
    NK = lp['NK']
    lo = lp['lo']
    hi = lp['hi']
    line_thresh = lp['line_thresh']
    line_length = lp['line_length']
    line_gap = lp['line_gap']
    med_size = lp['med_size']
    use_canny = lp['use_canny']
    sn_thresh = lp['sn_thresh']
    
    # Parse FITS file if necessary
    if hasattr(cube, 'filename'):
        cube, dq, time, NS = split_multiaccum(cube, scale_flat=False)
    
    # DQ masking in the cube
    cdq = (dq - (dq & 8192)) == 0
    cube[~cdq] = 0

    # Difference images
    dt = np.diff(time)[1:]
    arr = (np.diff(cube, axis=0)[1:].T/dt).T

    dq0 = (dq[-1,:,:] - (dq[-1,:,:] & 8192)) == 0

    # Max diff image minus median
    #diff = arr.max(axis=0) - np.median(arr, axis=0)
    
    arr_so = np.sort(arr, axis=0)
    
    # This will except out if too few reads
    diff = arr_so[-1,:,:] - arr_so[-3,:,:]

    # Global median
    med = np.median(diff[5:-5,5:-5])
    
    # Median filter
    medfilt = nd.median_filter(diff*dq0-med, size=med_size)

    # wagon wheel
    medfilt[:300,900:] = 0
    medfilt[380:600,810:950] = 0

    medfilt[990:,] = 0
    medfilt[:16,:] = 0
    medfilt[:,:16] = 0
    medfilt[:,-16:] = 0
    
    medfilt = (medfilt*dq0)[5:-5,5:-5]
                
    # Smoothed image
    kern = np.ones((NK, NK))/NK**2
    mk = nd.convolve(medfilt, kern)[NK//2::NK,NK//2::NK][1:-1,1:-1]
    
    mask = mk != 0
    
    # Apply earthshine mask
    if earthshine_mask:
        yp, xp = np.indices((1014,1014))
        x0, y0 = [578, 398], [0,1014]
        cline = np.polyfit(x0, y0, 1)
        yi = np.polyval(cline, xp)
        emask = yp > yi
        #medfilt *= emask
        mask &= emask[NK//2::NK,NK//2::NK][1:-1,1:-1]
        
    else:
        # Mask center to only get trails that extend from one edge to another
        if line_length < 240:
            #print('xxx mask center', line_length, mask.sum())
            sl = slice(2*line_length//NK,-2*line_length//NK)
            mask[sl,sl] = False
            #print('yyy mask center', line_length, mask.sum())
        
    # Column average
    if subtract_column:
        col = np.median(mk, axis=0)
        mk -= col
    
    image = mk
    #image *= mask
    
    # Image ~ standard deviation
    nmad = utils.nmad(image[mask])

    if use_canny:
        edges = canny(image, sigma=1, low_threshold=lo*nmad, high_threshold=hi*nmad, mask=mask)
    else:
        edges = image-np.median(image)*0 > sn_thresh*nmad
        small_edge=np.maximum(50//NK, 1)
        morph.remove_small_objects(edges, min_size=small_edge*5, 
                                   connectivity=small_edge*5,
                                   in_place=True)  
    
    # edges[0,:] = False
    # edges[-1,:] = False
    # edges[:,0] = False
    # edges[:,-1] = False
    
    lines = probabilistic_hough_line(edges, threshold=line_thresh,
                                     line_length=line_length//NK,
                                     line_gap=line_gap)
    
    return image, edges, np.array(lines)
    
def segments_to_mask(lines, NK, NX, buf=None):
    """
    """
    
    xl = np.array([-int(0.1*NX), int(1.1*NX)])
    if buf is None:
        buf = NK
    
    line_shape = None
    for l in lines:
        c = np.polyfit(l[:,0], l[:,1], 1)
        xx = (xl+1.5)*NK
        yy = (np.polyval(c, xl)+1.5)*NK
        
        line_i = LineString(coordinates=np.array([xx, yy]).T)
        line_buf = line_i.buffer(buf, resolution=2)
        
        if line_shape is None:
            line_shape = line_buf
        else:
            line_shape = line_shape.union(line_buf)
    
    return utils.shapely_polygon_to_region(line_shape)
         
def sat_trail_figure(image, edges, lines, label='Exposure'):
    """
    Make a figure showing the detected features
    TBD
    
    Parameters
    ----------
    image : `~numpy.ndarray`
        The image from which the linear features were detected.
    
    edges : `~numpy.ndarray`
        The `skimage.feature.canny` edge image.
        
    lines : (N,2,2)
        `N` linear segments found on the edge image with
        `skimage.transform.probabilistic_hough_line`.
        
    label : string
    
    
    Returns
    -------
    fig : `~matplotlib.figure.Figure`
        Figure object.
        
    """
    utils.set_warnings()
    
    nmad = utils.nmad(image[image != 0])
    
    # Generating figure 2
    fig, axes = plt.subplots(1, 2, figsize=(4, 2), sharex=True, sharey=True)
    ax = axes.ravel()

    # fig = Figure(figsize=(4,2))
    # ax = [fig.add_subplot(121+i) for i in range(2)]
        
    ax[0].imshow(image, cmap=cm.gray, origin='lower', 
                 vmin=-1*nmad, vmax=5*nmad)
                 
    ax[0].text(0.1, 0.95, label, ha='left', va='top',
               transform=ax[0].transAxes, color='w', size=6)

    ax[1].imshow(edges, cmap=cm.gray, origin='lower')
    ax[1].text(0.1, 0.95, 'Canny edges', ha='left', va='top', 
               transform=ax[1].transAxes, color='w', size=6)

    # ax[2].imshow(edges * 0, origin='lower')
    for line in lines:
        p0, p1 = line
        ax[1].plot((p0[0], p1[0]), (p0[1], p1[1]), alpha=0.6)
        ax[0].plot((p0[0], p1[0]), (p0[1], p1[1]), alpha=0.6)
        
    # ax[2].set_xlim((0, image.shape[1]))
    # ax[2].set_ylim((0, image.shape[0]))
    # ax[2].text(0.1, 0.95, 'Hough lines', ha='left', va='top', 
    #            transform=ax[2].transAxes, color='w', size=6)

    for a in ax:
        a.set_xticklabels([])
        a.set_yticklabels([])
        #a.set_axis_off()

    fig.tight_layout(pad=0.1)
    return fig
    