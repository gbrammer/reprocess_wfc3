"""
Scripts to reprocess WFC3 exposures with time-variable backgrounds 
or satellite trails.  
"""

import os
import glob
import shutil
import traceback

import numpy as np
import numpy.ma

import time

try:
    import astropy.io.fits as pyfits
except:
    import pyfits

#### WFC3 tools (pip, conda)
import wfc3tools

from . import anomalies, utils
utils.set_warnings() # silence numpy warnings

#### Logging
import logging
logger = logging.getLogger('reprocess_wfc3')
logger.setLevel(logging.DEBUG)

ch = logging.StreamHandler()
ch.setLevel(logging.DEBUG)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(message)s', datefmt='%m/%d/%Y %H:%M:%S')
ch.setFormatter(formatter)
if len(logger.handlers) == 0:
    logger.addHandler(ch)

def silent_log(input_string):
    """
    Quiet calwf3 logs
    """
    pass


def get_flat(hdulist):
    """
    Get the flat-field file specified in the header
    """
    flat_file = hdulist[0].header['PFLTFILE'].replace('iref$', os.getenv('iref')+'/')
    flat_im = pyfits.open(flat_file)
    flat = flat_im[1].data
    return flat_im, flat


def fetch_calibs(ima_file, ftpdir='https://hst-crds.stsci.edu/unchecked_get/references/hst/', verbose=True, remove_corrupt=True):
    """
    Fetch necessary calibration files needed for running calwf3 from STScI FTP
    
    Old FTP dir: ftp://ftp.stsci.edu/cdbs/iref/"""
    import os
    
    if not os.getenv('iref'):
        print('No $iref set!  Put it in ~/.bashrc or ~/.cshrc.')
        return False
        
    im = pyfits.open(ima_file)
    for ctype in ['BPIXTAB', 'CCDTAB', 'OSCNTAB', 'CRREJTAB', 'DARKFILE', 'NLINFILE', 'DFLTFILE', 'PFLTFILE', 'IMPHTTAB', 'IDCTAB']:
        
        if ctype not in im[0].header:
            continue
            
        if im[0].header[ctype] == 'N/A':
            continue
        
        cimg = im[0].header[ctype].split('iref$')[1]
        
        iref_file = os.path.join(os.getenv('iref'), cimg)
        if not os.path.exists(iref_file):
            if verbose:
                print('Calib: %s=%s' %(ctype, im[0].header[ctype]))

            os.system('curl -o %s %s/%s' %(iref_file, ftpdir, cimg))
            
            # Check that file is OK, since curl makes a file anyway
            if 'fits' in iref_file:
                try:
                    pyfits.open(iref_file)
                except:
                    msg = ('Downloaded file {0} appears to be corrupt.\n'
                           'Check that {1}/{2} exists and is a valid file')

                    print(msg.format(iref_file, ftpdir, cimg))
                    if remove_corrupt:
                        os.remove(iref_file)

                    return False
                    
    return True
    
def split_multiaccum(ima, scale_flat=True, get_err=False):
    """
    Pull out the MultiAccum reads of a RAW or IMA file into a single 3D 
    matrix.
    
    Returns cube[NSAMP,1024,1014], time, NSAMP
    """    
    skip_ima = ('ima' in ima.filename()) & (ima[0].header['FLATCORR'] == 'COMPLETE')
    if scale_flat & ~skip_ima:
        #FLAT_F140W = pyfits.open(os.path.join(os.getenv('iref'), 'uc721143i_pfl.fits'))[1].data
        flat_im, flat = get_flat(ima)
    else:
        #FLAT_F140W = 1
        flat_im, flat = None, 1.
    
    is_dark = 'drk' in ima.filename()
    if is_dark:
        flat = 1
                
    NSAMP = ima[0].header['NSAMP']
    sh = ima['SCI',1].shape
    
    cube = np.zeros((NSAMP, sh[0], sh[1]))
    if 'ima' in ima.filename():
        dq = np.zeros((NSAMP, sh[0], sh[1]), dtype=np.int32)
    else:
        dq = 0
    
    if get_err:
        cube_err = cube*0
        
    times = np.zeros(NSAMP)
    for i in range(NSAMP):
        if (ima[0].header['UNITCORR'] == 'COMPLETE') & (~is_dark):
            cube[NSAMP-1-i, :, :] = ima['SCI',i+1].data*ima['TIME',i+1].header['PIXVALUE']/flat
        else:
            #print 'Dark'
            cube[NSAMP-1-i, :, :] = ima['SCI',i+1].data/flat
        
        if get_err:
            if ima[0].header['UNITCORR'] == 'COMPLETE':
                cube_err[NSAMP-1-i, :, :] = ima['ERR',i+1].data*ima['TIME',i+1].header['PIXVALUE']/flat
            else:
                cube_err[NSAMP-1-i, :, :] = ima['ERR',i+1].data/flat
            
        if 'ima' in ima.filename():
            dq[NSAMP-1-i, :, :] = ima['DQ',i+1].data
        
        times[NSAMP-1-i] = ima['TIME',i+1].header['PIXVALUE']
    
    if get_err:
        return cube, cube_err, dq, times, NSAMP
    else:
        return cube, dq, times, NSAMP
             
def make_IMA_FLT(raw='ibhj31grq_raw.fits', pop_reads=[], remove_ima=True, fix_saturated=True, flatten_ramp=True, stats_region=[[300,714], [300,714]], earthshine_threshold=0.1, log_func=silent_log, auto_trails=True):
    """
    Run calwf3, if necessary, to generate ima & flt files.  Then put the last
    read of the ima in the FLT SCI extension and let Multidrizzle flag 
    the CRs.
    
    Optionally pop out reads affected by satellite trails or earthshine.  The 
    parameter `pop_reads` is a `list` containing the reads to remove, where
    a value of 1 corresponds to the first real read after the 2.9s flush.
    
    """
        
    #### Remove existing products or calwf3 will die
    for ext in ['flt','ima']:
        if os.path.exists(raw.replace('raw', ext)):
            os.remove(raw.replace('raw', ext))
    
    #### Turn off CR rejection    
    raw_im = pyfits.open(raw, mode='update')
    if raw_im[0].header['DETECTOR'] == 'UVIS':
        return True
        
    status = fetch_calibs(raw) #, ftpdir='ftp://ftp.stsci.edu/cdbs/iref/')
    if not status:
        msg = 'Problem with `fetch_calibs(\'{0}\')`, aborting...'
        print(msg.format(raw))
        return False
        
    if not pop_reads:
        raw_im[0].header['CRCORR'] = 'OMIT'
        raw_im.flush()
    
    #### Run calwf3
    print('reprocess_wfc3: wfc3tools.calwf3(\'{0}\')'.format(raw))
    try:
        #wfc3tools.calwf3(raw, log_func=log_func)
        utils.run_calwf3(raw, log_func=log_func)
    except:
        trace = traceback.format_exc(limit=2)
        log = '\n########################################## \n'
        log += '# ! Exception ({0})\n'.format(time.ctime())
        log += '#\n# !'+'\n# !'.join(trace.split('\n'))
        log += '\n######################################### \n\n'
        print(log)
        
    flt = pyfits.open(raw.replace('raw', 'flt'), mode='update')
    ima = pyfits.open(raw.replace('raw', 'ima'))
    
    #### Pull out the data cube, order in the more natural sense
    #### of first reads first
    cube, dq, times, NSAMP = split_multiaccum(ima, scale_flat=False)
    
    earthshine_mask = False
    
    samp_seq = raw_im[0].header['SAMP_SEQ']
    
    if (earthshine_threshold > 0) & (not samp_seq.startswith('STEP')):
        earth_diff = anomalies.compute_earthshine(cube, dq, times)
        flag_earth = earth_diff > earthshine_threshold

        flagged_reads = list(np.where(flag_earth)[0])
        
        # Just zeroth read
        if flagged_reads == [0]:
            flag_earth[0] = False
            
        # if True:
        #     flagged_reads = list(np.where(flag_earth)[0])
        #     #print('Flagged reads: ', flagged_reads)
            
        ### If all but two or few flagged, make a mask
        NR = len(flag_earth)
        if (~flag_earth).sum() <= 2:
            print('reprocess_wfc3: {0} - {1}/{2} reads affected by scattered earthshine: {3} **make mask**'.format(raw, flag_earth.sum(), NR, flagged_reads))
            
            mask_reg = """# Region file format: DS9 version 4.1
global color=green dashlist=8 3 width=1 font="helvetica 10 normal roman" select=1 highlite=1 dash=0 fixed=0 edit=1 move=1 delete=1 include=1 source=1
image
polygon(-46.89536,1045.4771,391.04896,1040.5005,580.16128,-12.05888,-51.692518,-9.3910781)"""
            fp = open(raw.replace('_raw.fits','.01.mask.reg'), 'w')
            fp.write(mask_reg)
            fp.close()
            
            earthshine_mask = True
            
        elif (flag_earth.sum() > 0) & ((~flag_earth).sum() > 2):

            print('reprocess_wfc3: {0} - {1}/{2} reads affected by scattered earthshine: {3}'.format(raw, flag_earth.sum(), NR, flagged_reads))
            
            pop_reads = pop_reads + flagged_reads
            
            earthshine_mask = True
            
        else:
            pass
    
    if auto_trails:
        if earthshine_mask:
            # Recompute cube
            print('reprocess_wfc3: {0} Recompute cube'.format(raw))
            cube, dq, times, NSAMP = split_multiaccum(ima, scale_flat=False)
            
        is_grism = ima[0].header['FILTER'] in ['G102','G141']
        root = os.path.basename(ima.filename()).split('_')[0]        
        try:
            anomalies.auto_flag_trails(cube*1, dq*1, times*1,
                                   is_grism=is_grism,
                                   root=root, earthshine_mask=earthshine_mask)
        except:
            # Probably too few reads
            pass
        
        # if earthshine_mask:
        #     # Revert cube
        #     ima = pyfits.open(raw.replace('raw', 'ima'))
        #     cube, dq, times, NSAMP = split_multiaccum(ima, scale_flat=False)
            
        
    #### Readnoise in 4 amps
    readnoise_2D = np.zeros((1024,1024))
    readnoise_2D[512: ,0:512] += ima[0].header['READNSEA']
    readnoise_2D[0:512,0:512] += ima[0].header['READNSEB']
    readnoise_2D[0:512, 512:] += ima[0].header['READNSEC']
    readnoise_2D[512: , 512:] += ima[0].header['READNSED']
    readnoise_2D = readnoise_2D**2

    #### Gain in 4 amps
    gain_2D = np.zeros((1024,1024))
    gain_2D[512: ,0:512] += ima[0].header['ATODGNA']
    gain_2D[0:512,0:512] += ima[0].header['ATODGNB']
    gain_2D[0:512, 512:] += ima[0].header['ATODGNC']
    gain_2D[512: , 512:] += ima[0].header['ATODGND']
    
    ### Pop out reads affected by satellite trails or earthshine
    masks = glob.glob(raw.replace('.fits', '*mask.reg'))
    if (len(pop_reads) > 0): # | (len(masks) > 0):
        print('reprocess_wfc3: Pop reads {0} from {1}'.format(pop_reads, ima.filename()))
        
        #### Need to put dark back in for Poisson
        dark_file = ima[0].header['DARKFILE'].replace('iref$', os.getenv('iref')+'/')
        dark = pyfits.open(dark_file)
        dark_cube, dark_dq, dark_time, dark_NSAMP = split_multiaccum(dark, scale_flat=False)
        
        #### Need flat for Poisson
        flat_im, flat = get_flat(ima)
        
        #### Subtract diffs of flagged reads
        diff = np.diff(cube, axis=0)
        dark_diff = np.diff(dark_cube, axis=0)

        dt = np.diff(times)
        final_exptime = np.ones((1024, 1024))*times[-1]
        final_sci = cube[-1,:,:]*1
        final_dark = dark_cube[NSAMP-1,:,:]*1        
        for read in pop_reads:
            final_sci -= diff[read,:,:]
            final_dark -= dark_diff[read,:,:]
            final_exptime -= dt[read]

            # ds9.view(final_sci/final_exptime)
                
        if False:
            ### Experimenting with automated flagging
            sh = (1024,1024)
            drate = (diff.reshape((14,-1)).T/dt).T
            med = np.median(drate, axis=0)
            fmed = np.median(med)
            nmad = 1.48*np.median(np.abs(drate-med), axis=0)
            
            drate_ma = np.ma.masked_array(drate, mask=~np.isfinite(drate))
            wht_ma = drate_ma*0
            
            excess = med*0.
            for read in range(1,NSAMP-1):
                med_i = np.percentile(drate[read,:]-med, 20)
                excess_electrons = (drate[read,:]-med-med_i)*dt[read]
                rms = np.sqrt((fmed + med_i)*dt[read])
                
                hot = (excess_electrons / rms) > 10
                #sm = nd.median_filter(excess_electrons.reshape(sh), 10).flatten()
                #hot |= (sm / rms) > 3
                
                med_i = np.percentile((drate[read,:]-med)[~hot], 50)
                print(med_i)
                
                drate_ma.mask[read, hot] |= True
                drate_ma.data[read,:] -= med_i
                wht_ma.mask[read, hot] |= True
                wht_ma.data[read,:] = dt[read]
            
            wht_ma.mask[0,:] = True
            
            avg = (drate_ma*wht_ma).sum(axis=0)/wht_ma.sum(axis=0)
            pyfits.writeto('%s_avg.fits' %(raw.split('_raw')[0]), data=avg.data.reshape(sh)[5:-5,5:-5], overwrite=True)
                
        #### Removed masked regions of individual reads
        # if len(masks) > 0:
        #     import pyregion
        #     for mask in masks:
        #         mask_read = int(mask.split('.')[-3])
        #         if mask_read in pop_reads:
        #             continue
        #         
        #         print('Mask pixels in read %d (%s)' %(mask_read, mask))
        #         
        #         refhdu = ima['SCI', 1]
        #         r = pyregion.open(mask).as_imagecoord(header=refhdu.header)
        #         mask_array = r.get_mask(hdu=refhdu)
        #         final_exptime -= mask_array*dt[mask_read]
        #         final_sci -= diff[mask_read,:,:]*mask_array
        #         final_dark -= dark_diff[mask_read,:,:]*mask_array
                
        #### Variance terms
        ## read noise
        final_var = readnoise_2D*1
        ## poisson term
        final_var += (final_sci*flat + final_dark*gain_2D)*(gain_2D/2.368)
        ## flat errors
        final_var += (final_sci*flat*flat_im['ERR'].data)**2
        final_err = np.sqrt(final_var)/flat/(gain_2D/2.368)/1.003448/final_exptime
        
        final_sci /= final_exptime
                
        flt[0].header['EXPTIME'] = np.max(final_exptime)
        
    else:
        if flatten_ramp:
            #### Subtract out the median of each read to make background flat
            fix_saturated = False
            
            print('reprocess_wfc3: Flatten ramp {0}'.format(raw))
            ima = pyfits.open(raw.replace('raw', 'ima'), mode='update')
            
            #### Grism exposures aren't flat-corrected
            filter = ima[0].header['FILTER']
            if 'G1' in filter:
                flats = {'G102': 'uc72113oi_pfl.fits', 
                         'G141': 'uc721143i_pfl.fits'}
                
                flat = pyfits.open('%s/%s' %(os.getenv('iref'), flats[filter]))[1].data
            else:
                flat = 1.
            
            #### Remove the variable ramp            
            slx = slice(stats_region[0][0], stats_region[0][1])
            sly = slice(stats_region[1][0], stats_region[1][1])
            total_countrate = np.median((ima['SCI',1].data/flat)[sly, slx])
            for i in range(ima[0].header['NSAMP']-2):
                ima['SCI',i+1].data /= flat
                med = np.median(ima['SCI',i+1].data[sly, slx])
                print('reprocess_wfc3:   read {0:>2d}, bg= {1:.2f}'.format(i+1, med))
                ima['SCI',i+1].data += total_countrate - med
            
            if 'G1' in filter:
                for i in range(ima[0].header['NSAMP']-2):
                    ima['SCI',i+1].data *= flat
            
            ima[0].header['CRCORR'] = 'PERFORM'
            ima[0].header['DRIZCORR'] = 'OMIT'
            
            ### May need to generate a simple dummy ASN file for a single exposure
            ### Not clear why calwf3 needs an ASN if DRIZCORR=OMIT, but it does
            need_asn = False
            if ima[0].header['ASN_ID'] == 'NONE':
                need_asn=True
            else:
                if not os.path.exists(ima[0].header['ASN_TAB']):
                    need_asn=True
            
            if need_asn:
                import stsci.tools
                
                exp = ima.filename().split('_ima')[0]
                params = stsci.tools.asnutil.ASNMember()
                asn = stsci.tools.asnutil.ASNTable(output=exp)
                asn['members'] = {exp:params}
                asn['order'] = [exp]
                asn.write()
                
                ima[0].header['ASN_ID'] = exp.upper()
                ima[0].header['ASN_TAB'] = '%s_asn.fits' %(exp)
                
            ima.flush()
                                
            #### Initial cleanup
            files=glob.glob(raw.replace('raw', 'ima_*'))
            for file in files:
                print('reprocess_wfc3: cleanup - rm {0}'.format(file))
                os.remove(file)
        
            #### Run calwf3 on cleaned IMA
            #wfc3tools.calwf3(raw.replace('raw', 'ima'), log_func=log_func)
            utils.run_calwf3(raw.replace('raw', 'ima'), log_func=log_func)
            
            #### Put results into an FLT-like file
            try:
                ima = pyfits.open(raw.replace('raw', 'ima_ima'))
                flt_new = pyfits.open(raw.replace('raw', 'ima_flt'))
            except:
                ima = pyfits.open(raw.replace('raw', 'ima'))
                flt_new = pyfits.open(raw.replace('raw', 'flt'))
                
            flt['DQ'].data = flt_new['DQ'].data*1
            flt['TIME'] = flt_new['TIME']
            flt['SAMP'] = flt_new['SAMP']
            
            final_sci = ima['SCI', 1].data*1
            final_sci[5:-5,5:-5] = flt_new['SCI'].data*1
            #final_err = ima['ERR', 1].data*1
            
            ### Need original ERR, something gets messed up
            final_err = ima['ERR', 1].data*1
            final_err[5:-5,5:-5] = flt['ERR'].data*1
            
            ### Clean up
            files=glob.glob(raw.replace('raw', 'ima_*'))
            for file in files:
                print('reprocess_wfc3: cleanup - rm {0}'.format(file))
                os.remove(file)
                
        else:
            final_sci = ima['SCI', 1].data*1
            final_err = ima['ERR', 1].data*1
    
    final_dq = ima['DQ', 1].data*1
    
    #### For saturated pixels, look for last read that was unsaturated
    #### Background will be different under saturated pixels but maybe won't
    #### matter so much for such bright objects.
    if (fix_saturated):
        #### Saturated pixels
        zi, yi, xi = np.indices(dq.shape)
        saturated = (dq & 256) > 0
        # 1024x1024 index array of reads where pixels not saturated
        zi_flag = zi*1
        zi_flag[saturated] = 0
        ### 2D array of the last un-saturated read
        last_ok_read = np.max(zi_flag, axis=0)
        sat_zero = last_ok_read == 0        
        # pyfits.writeto(raw.replace('_raw','_lastread'), data=last_ok_read[5:-5,5:-5], header=flt[1].header, overwrite=True)
        ### keep pixels from first read even if saturated
        last_ok_read[sat_zero] = 1
        
        zi_idx = zi < 0
        for i in range(1, NSAMP-1):
            zi_idx[i,:,:] = zi[i,:,:] == last_ok_read

        time_array = times[zi]
        time_array[0,:,:] = 1.e-3 # avoid divide-by-zero
        # pixels that saturated before the last read
        fix = (last_ok_read < (ima[0].header['NSAMP'] - 1)) & (last_ok_read > 0)
        #err = np.sqrt(ima[0].header['READNSEA']**2 + cube)/time_array
        err = np.sqrt(readnoise_2D + cube)/time_array

        final_sci[fix] = np.sum((cube/time_array)*zi_idx, axis=0)[fix]
        final_err[fix] = np.sum(err*zi_idx, axis=0)[fix]

        fixed_sat = (zi_idx.sum(axis=0) > 0) & ((final_dq & 256) > 0)
        final_dq[fixed_sat] -= 256
        final_dq[sat_zero] |= 256
        
        print('reprocess_wfc3: Fix {0} saturated pixels'.format(fixed_sat.sum()))
        flt['DQ'].data |= final_dq[5:-5,5:-5] & 256
        
    else:
        #### Saturated pixels
        flt['DQ'].data |= ima['DQ',1].data[5:-5,5:-5] & 256
        
    flt['SCI'].data = final_sci[5:-5,5:-5]
    flt['ERR'].data = final_err[5:-5,5:-5]
    
    #### Some earthshine flares DQ masked as 32: "unstable pixels"
    mask = (flt['DQ'].data & 32) > 0
    if mask.sum() > 1.e4:
        print('reprocess_wfc3: Take out excessive DQ=32 flags (N={0:.2e})'.format(mask.sum()))
        #flt['DQ'].data[mask] -= 32
        mask = flt['DQ'].data & 32
        ### Leave flagged 32 pixels around the edges
        flt['DQ'].data[5:-5,5:-5] -= mask[5:-5,5:-5]
        
    ### Update the FLT header
    flt[0].header['IMA2FLT'] = (1, 'FLT extracted from IMA file')
    flt[0].header['IMASAT'] = (fix_saturated*1, 'Manually fixed saturation')
    flt[0].header['NPOP'] = (len(pop_reads), 'Number of reads popped from the sequence')
    if len(pop_reads) > 0:
        pop_str = ' '.join('{0}'.format(r) for r in pop_reads)
    else:
        pop_str = None
    
    flt[0].header['POPREADS'] = (pop_str, 'Reads popped out of MultiACCUM sequence')
    # for iread, read in enumerate(pop_reads):
    #     flt[0].header['POPRD%02d' %(iread+1)] = (read, 'Read kicked out of the MULTIACCUM sequence')
        
    flt.flush()
    
    ### Remove the IMA file
    if remove_ima:
        os.remove(raw.replace('raw', 'ima'))

def reprocess_parallel(files, cpu_count=0, skip=True):
    """
    """
    
    import multiprocessing as mp
    import time
    
    nskip = 0
    if skip:
        for i in range(len(files))[::-1]:
            if os.path.exists(files[i].replace('_raw.fits', '_flt.fits')):
                p = files.pop(i)
                nskip += 1
                
    if len(files) == 0:
        return True
                        
    t0_pool = time.time()
    if cpu_count <= 0:
        cpu_count = np.minimum(mp.cpu_count(), len(files))
    
    msg = 'reprocess_wfc3: Running `make_IMA_FLT` for {0} files on {1} CPUs ({2} skipped)'
    print(msg.format(len(files), cpu_count, nskip))
      
    pool = mp.Pool(processes=cpu_count)
    
    results = [pool.apply_async(make_IMA_FLT, (file, [], True, True, True, [[300,700],[300,700]], 0.1, silent_log, True)) for file in files]
    pool.close()
    pool.join()
    
    t1_pool = time.time()
    
def show_ramps_parallel(files, cpu_count=0, skip=True):
    """
    """
    
    import multiprocessing as mp
    import time
    
    nskip = 0
    if skip:
        for i in range(len(files))[::-1]:
            if os.path.exists(files[i].replace('_raw.fits', '_ramp.png')):
                p = files.pop(i)
                nskip += 1
                
    if len(files) == 0:
        return True
        
    t0_pool = time.time()
    if cpu_count <= 0:
        cpu_count = np.minimum(mp.cpu_count(), len(files))
    
    msg = 'reprocess_wfc3: Running `show_MultiAccum_reads` for {0} files on {1} CPUs ({2} skipped)'
    print(msg.format(len(files), cpu_count, nskip))
    
    pool = mp.Pool(processes=cpu_count)
    
    results = [pool.apply_async(show_MultiAccum_reads, (file, False, False, [[300,700],[300,700]])) for file in files]
    pool.close()
    pool.join()

    t1_pool = time.time()
    
def show_MultiAccum_reads(raw='ibp329isq_raw.fits', flatten_ramp=False, verbose=True, stats_region=[[0,1014], [0,1014]]):
    """
    Make a figure (.ramp.png) showing the individual reads of an 
    IMA or RAW file.
    """    
    import scipy.ndimage as nd

    import matplotlib.pyplot as plt
    
    from matplotlib.figure import Figure
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    
    if verbose:
        logger.setLevel(logging.DEBUG)
    else:
        logger.setLevel(logging.WARN)

    status = fetch_calibs(raw) #, ftpdir='ftp://ftp.stsci.edu/cdbs/iref/')
    if not status:
        return False
        
    img = pyfits.open(raw)
    
    if 'raw' in raw:
        gains = [2.3399999, 2.3699999, 2.3099999, 2.3800001]
        gain = np.zeros((1024,1024))
        gain[512: ,0:512] += gains[0]
        gain[0:512,0:512] += gains[1]
        gain[0:512, 512:] += gains[2]
        gain[512: , 512:] += gains[3]
    else:
        gain=1
    
    logger.info('Make MULTIACCUM cube')
        
    #### Split the multiaccum file into individual reads    
    cube, dq, times, NSAMP = split_multiaccum(img, scale_flat=False)
    
    if 'raw' in raw:
        dark_file = img[0].header['DARKFILE'].replace('iref$', os.getenv('iref')+'/')
        dark = pyfits.open(dark_file)
        dark_cube, dark_dq, dark_time, dark_NSAMP = split_multiaccum(dark, scale_flat=False)

        diff = np.diff(cube-dark_cube[:NSAMP,:,:], axis=0)*gain
        dt = np.diff(times)
    
        #### Need flat for Poisson
        flat_im, flat = get_flat(img)
        diff /= flat
    else:
        diff = np.diff(cube, axis=0)
        dt = np.diff(times)
    
    ####  Average ramp
    slx = slice(stats_region[0][0], stats_region[0][1])
    sly = slice(stats_region[1][0], stats_region[1][1])
    ramp_cps = np.median(diff[:, sly, slx], axis=1)
    avg_ramp = np.median(ramp_cps, axis=1)
    
    #### Initialize the figure
    logger.info('Make plot')
    
    plt.ioff()
    #fig = plt.figure(figsize=[10,10])
    fig = Figure(figsize=[10,10])

    ## Smoothing
    smooth = 1
    kernel = np.ones((smooth,smooth))/smooth**2
    
    ## Plot the individual reads
    for j in range(1,NSAMP-1):
        ax = fig.add_subplot(4,4,j)
        smooth_read = nd.convolve(diff[j,:,:],kernel)
        ax.imshow(smooth_read[5:-5:smooth, 5:-5:smooth]/dt[j], 
                  vmin=0, vmax=4, origin='lower', cmap=plt.get_cmap('cubehelix'))
        
        ax.set_xticklabels([]); ax.set_yticklabels([])
        ax.text(20,5,'%d' %(j), ha='left', va='bottom', backgroundcolor='white')
    
    ## Show the ramp
    fig.tight_layout(h_pad=0.3, w_pad=0.3, pad=0.5)
    
    ax = fig.add_axes((0.6, 0.05, 0.37, 0.18))
    #ax = fig.add_subplot(428)
    ax.plot(times[2:], (ramp_cps[1:,16:-16:4].T/np.diff(times)[1:]).T, 
            alpha=0.1, color='black')
    ax.plot(times[2:], avg_ramp[1:]/np.diff(times)[1:], alpha=0.8, 
            color='red', linewidth=2)
    ax.set_xlabel('time'); ax.set_ylabel('background [e/s]')

    #fig.tight_layout(h_pad=0.3, w_pad=0.3, pad=0.5)
    root=raw.split('_')[0]
    #plt.savefig(root+'_ramp.png')
    
    canvas = FigureCanvasAgg(fig)
    canvas.print_figure(root+'_ramp.png', dpi=200)
    
    #### Same ramp data file    
    np.savetxt('%s_ramp.dat' %(root), np.array([times[1:], avg_ramp/np.diff(times)]).T, fmt='%.3f')
    
    if flatten_ramp:
        #### Flatten the ramp by setting background countrate to the average.  
        #### Output saved to "*x_flt.fits" rather than the usual *q_flt.fits.
        
        flux = avg_ramp/np.diff(times)
        avg = avg_ramp.sum()/times[-1]
        min = flux[1:].min()
        subval = np.cumsum((flux-avg)*np.diff(times))
        
        imraw = pyfits.open(raw.replace('ima','raw'))
        for i in range(1, NSAMP):
            logger.info('Remove excess %.2f e/s from read #%d (t=%.1f)' %(flux[-i]-min, NSAMP-i+1, times[-i]))
            
            imraw['SCI',i].data = imraw['SCI',i].data - np.cast[int](subval[-i]/2.36*flat)
                
        files=glob.glob(raw.split('q_')[0]+'x_*')
        for file in files:
            os.remove(file)
            
        imraw[0].header['CRCORR'] = 'PERFORM'
        imraw.writeto(raw.replace('q_raw', 'x_raw'), overwrite=True)
        
        ## Run calwf3
        #wfc3tools.calwf3(raw.replace('q_raw', 'x_raw'))
        utils.run_calwf3(raw.replace('q_raw', 'x_raw'), clean=True)
        
    return fig
    
def in_shadow(file='ibhj07ynq_raw.fits'):
    """
    Compute which reads in a RAW file were obtained within the Earth SHADOW.  
    
    Requires the associated JIF files that contain this information, for example
    "ibhj07040_jif.fits" for the default data file.  These can be obtained by requesting
    the "observation log" files from MAST.
    
    """
    import astropy.time
    import astropy.io.fits as pyfits
    import numpy as np
    
    #### Open the raw file
    raw = pyfits.open(file)
    NSAMP = raw[0].header['NSAMP']

    #### Find JIF file.  Can either be association or single files
    if raw[0].header['ASN_ID'] == 'NONE':
        exp = raw[0].header['ROOTNAME']
        jif = pyfits.open(exp[:-1]+'j_jif.fits')[1]
    else:
        exp = raw[0].header['ROOTNAME']
        asn = raw[0].header['ASN_TAB']
        jif = pyfits.open(asn.replace('asn', 'jif'))
        for i in range(len(jif)-1):
            if jif[i+1].header['EXPNAME'][:-1] == exp[:-1]:
                jif = jif[i+1]
                break
    
    #### Shadow timing (last entry and exit)
    shadow_in = astropy.time.Time(jif.header['SHADOENT'].replace('.',':'), 
                        format='yday', in_subfmt='date_hms', scale='utc')
                        
    shadow_out = astropy.time.Time(jif.header['SHADOEXT'].replace('.',':'), 
                        format='yday', in_subfmt='date_hms', scale='utc')
    
    #### Array of read timings
    t0 = []
    for i in range(NSAMP):
        h = raw['sci',i+1].header
        ti = astropy.time.Time(h['ROUTTIME'], format='mjd', scale='utc')
        t0.append(ti)
        
    t0 = astropy.time.Time(t0)
    
    #### Test if reads were taken during shadow
    test_in_shadow = ((t0-shadow_in).sec < (t0-shadow_out).sec) | ((t0-shadow_out).sec < 0)

    return t0, test_in_shadow
    
                
    
        
        