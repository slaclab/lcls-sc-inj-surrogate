import scipy
from scipy.optimize import curve_fit
import scipy.ndimage as ndimage
from lcls_tools import image_processing as imp
import numpy as np
import matplotlib.pyplot as plt



def gauss(x, a, x0, sigma, offset):
    '''
    Calculate and returns a basic Gaussian.
    
    Positional arguments: 
    x values -- x
    Amplitude of Gaussian -- a
    Mean of dist -- x0
    Std. Dev. of Gaussian -- sigma
    Constant -- offset
    '''
    return a*np.exp(-(x-x0)**2/(2*sigma**2)) + offset

def fitProjection(projection, guess_vec, noise):
    '''
    Fits Gaussian to 1-d array. Returns fits, covariance values and errors.
    
    Positional arguments:
    1-d array -- projection
    Initial guesses for Gaussian fit (see gauss), 4 elements of the form -- gauss_vec:[a, x0, sigma, offest]
    '''
    noise = np.ones_like(projection)*noise
    #print(noise[0])
    #print(np.shape(noise))
    opt_gauss, cov_gauss = scipy.optimize.curve_fit(gauss, range(len(projection)), projection, sigma = noise, maxfev = 1000000, p0=guess_vec, bounds = [[0,0,0,0],[100000, 5000, 300, 100000]] )
    err_gauss = np.sqrt(np.diag(cov_gauss))
    return opt_gauss, cov_gauss, err_gauss

def makeMaskedImage(image, threshold = 1.05):
    '''
    Applied background removal via masking. 
    
    Positional argument: 
    Image with background -- image

    Optional argument:
    Multiplier (scalar larger than 1) for mean image value, used for masking -- threshold
    '''
    im = image
    mask = (im > threshold*im.mean()).astype(np.float)
    im = im*mask
    open_img = ndimage.binary_opening(mask)
    close_img = ndimage.binary_closing(open_img)
    close_img = close_img*im
    return close_img

def makeGaussFitData(projection, gauss_fit):
    '''
    Returns Gaussian for associated fit values. 
    
    Positional arguments: 
    X values (only needs length of projection, but projection is supplied) -- projection
    4-vec with Gassian fit values of the form -- gauss_fit:[a, x0, sigma, offest]
    '''
    return gauss(range(len(projection)), gauss_fit[0], gauss_fit[1], gauss_fit[2], gauss_fit[3])

def makeXProjection(image):
    '''
    Applies LCLS-Tools projection operation. 
    
    Positional argument:
    Image to project -- image
    '''
    return imp.image_processing.x_projection(image)
    
def makeYProjection(image):
    '''
    Applies LCLS-Tools projection operation. 
    
    Positional argument:
    Image to project -- image
    '''
    return imp.image_processing.y_projection(image)
    
def round_nearest(x,num=50000):
    return int(np.floor(float(x)/num)*num)


def GetImageNoise(image):
    '''
    A very rough method for estimating pixel noise for an image
    Positional arguments: 
    image -- n by m image before masking or trimming
    
    Returns:
    Sigma noise
    '''
    img = image[0:100,0:100]
    xproj = makeXProjection(img)
    xsig = 5*np.std(xproj)
    yproj = makeYProjection(img)
    ysig = 5*np.std(yproj)
    
    return xsig, ysig


def VCCAnalysis(image, plotting = True, amp = 5000, scale_factor = 5, sigma = 50, offset = 200, npix = 150, resolution = 1, verbose = False):
    '''
    Applied following analysis for VCC images: 
    - finds center of mass (CoM) of image
    - cuts down 2D matrix to range around CoM (npix)
    - makes and fits projections for calculating Gaussian sigma values
    - returns all analysis and properly sized images for re-binning
    
    Positional argument: 
    Single image (2D array) for processing -- image
    
    Optional arguments: 
    Plotting -- T/F
    Pixel value guess for Gaussian fits -- [amp, sigma, offset]
    Number of pixels for trimming images (image will be 2*npix by 2*npix) -- npix
    Dimensionfull scalar with units length/pixel -- resolution
    '''
    ## get noise from unaltered image
    xnoise, ynoise = GetImageNoise(image)
    
    ## finds CoM and applies some correction if needed
    xcm, ycm = imp.image_processing.center_of_mass(image)
    if np.isnan(xcm):
        xcm = npix
        ycm = npix
    if xcm < npix:
        npix = xcm
    if ycm < npix:
        npix = ycm
        
    ## trims image down around CoM
    img = image[int(xcm-npix):int(xcm+npix),int(ycm-npix):int(ycm+npix)]
    nn, mm = np.shape(img)

    ## pre-processing for ensuring proper re-binning
    newn = round_nearest(nn, num = 50)
    newm = round_nearest(mm, num = 50)
    
    fit_initial = [amp, npix, sigma, offset]
    
    xproj = makeXProjection(img)
    if np.remainder(len(xproj),2) == 1:
        xproj = xproj[:-1]
    xgauss_fit, xcov, xerr =  fitProjection(xproj, fit_initial, xnoise)
    xdata = makeGaussFitData(xproj, xgauss_fit)
    
    yproj = makeYProjection(img)
    if np.remainder(len(yproj),2) == 1:
        yproj = yproj[:-1]

    ygauss_fit, ycov, yerr =  fitProjection(yproj, fit_initial, ynoise)
    ydata = makeGaussFitData(yproj, ygauss_fit)
    
    ## in order to cut the image properly, the fit is used to calculate 
    ## the sigma for the distribution, and the number of sigma to keep is
    ## determined by the scale_factor. 
    scale_factor = scale_factor

    xnpix = int(round_nearest(scale_factor*np.ceil(xgauss_fit[2]), num = 50))
    ynpix = int(round_nearest(scale_factor*np.ceil(ygauss_fit[2]), num = 50))
    
    #### Ensures that the entire image is included and that the image has 1:1 aspect ratio
    if xnpix>ynpix:
        ynpix = xnpix
    else:
        xnpix = ynpix
        
    newimg = image[np.abs(int(xcm-xnpix/2)):int(xcm+xnpix/2),np.abs(int(ycm-ynpix/2)):int(ycm+ynpix/2)]
    
    if verbose:
        print("xgauss fit", scale_factor*np.ceil(xgauss_fit[2]))
        print("yguass fit", scale_factor*np.ceil(ygauss_fit[2]))
        print("centers of mass", xcm, ycm)
        print("new sizing information", xnpix, ynpix)
        print("X min pixel: ", str(int(xcm-xnpix/2)))
        print("X max pixel: ", str(int(xcm+xnpix/2)))
        print("Y min pixel: ", str(int(ycm-ynpix/2)))
        print("Y max pixel: ", str(int(ycm+ynpix/2)))
        print("shape of the new cut image", np.shape(newimg))
        
    masked = makeMaskedImage(newimg)
        

    mask_nn, mask_mm = np.shape(masked)
    mask_nn = round_nearest(mask_nn, num = 50)
    mask_mm = round_nearest(mask_mm, num = 50)
    masked = masked[:mask_nn, :mask_mm]
    
    if plotting:
        plt.figure(figsize = (5,5))
        plt.title('Original Image')
        plt.imshow(img, extent = [-npix*resolution , npix*resolution, -npix*resolution, npix*resolution], origin = "lower")
        plt.xlabel('microns')
        plt.show()
            
        n,m = np.shape(masked)
        axis = (np.arange(n)-int(n/2))*resolution
        np.min(axis)
        plt.figure(figsize = (5,5))
        plt.title("Masked Image")
        plt.imshow(masked,extent=[np.min(axis),np.max(axis),np.min(axis),np.max(axis)], origin = "lower")
        plt.show()
        
        plt.figure(figsize = (6,4))
        plt.title("sigma_x = {:.3f}, sigma_y = {:.3f}, x_err = {:.3f}, y_err = {:.3f}".format(xgauss_fit[2]*resolution*1E-3, ygauss_fit[2]*resolution*1E-3, xerr[2]*resolution*1E-3, yerr[2]*resolution*1E-3))
        plt.plot((np.arange(int(-len(xproj)/2),int(len(xproj)/2)))*resolution, xproj, 'b.', label = 'x projection')
        plt.plot((np.arange(int(-len(xdata)/2),int(len(xdata)/2)))*resolution, xdata, 'b-', label = 'x fit')
        plt.plot((np.arange(int(-len(yproj)/2),int(len(yproj)/2)))*resolution, yproj, 'r.', label = 'y projection')
        plt.plot((np.arange(int(-len(ydata)/2),int(len(ydata)/2)))*resolution, ydata, 'r-', label = 'y fit')
        plt.xlabel('microns')
        plt.legend()
        plt.show()
        
    return img, masked, xproj, xdata, xgauss_fit, xcov, xerr, yproj, ydata, ygauss_fit, ycov, yerr

def YAGAnalysis(image, plotting = True, scale_factor = 5, amp = 50000, sigma = 100, offset = 200, npix = 150, resolution = 1, verbose = False):
    '''
    Applied following analysis for YAG images: 
    - finds center of mass (CoM) of image
    - cuts down 2D matrix to range around CoM (npix)
    - makes and fits projections for calculating Gaussian sigma values
    - returns all analysis and properly sized images for re-binning
    
    Positional argument: 
    Single image (2D array) for processing -- image
    
    Optional arguments: 
    Plotting -- T/F
    Pixel value guess for Gaussian fits -- [amp, sigma, offset]
    Number of pixels for trimming images -- npix
    Dimensionfull scalar with units length/pixel -- resolution
    '''
    ## get noise from unaltered image
    xnoise, ynoise = GetImageNoise(image)
    
    ## finds CoM and applies some correction if needed
    xcm, ycm = imp.image_processing.center_of_mass(image)
    if np.isnan(xcm):
        xcm = npix
        ycm = npix
    if xcm < npix:
        npix = xcm
    if ycm < npix:
        npix = ycm
        
    ## trims image down around CoM
    img = image[int(xcm-npix):int(xcm+npix),int(ycm-npix):int(ycm+npix)]
    nn, mm = np.shape(img)

    ## pre-processing for ensuring proper re-binning
    newn = round_nearest(nn, num = 50)
    newm = round_nearest(mm, num = 50)
    
    fit_initial = [amp, npix, sigma, offset]
    
    xproj = makeXProjection(img)
    if np.remainder(len(xproj),2) == 1:
        xproj = xproj[:-1]
    xgauss_fit, xcov, xerr =  fitProjection(xproj, fit_initial, xnoise)
    xdata = makeGaussFitData(xproj, xgauss_fit)
    
    yproj = makeYProjection(img)
    if np.remainder(len(yproj),2) == 1:
        yproj = yproj[:-1]

    ygauss_fit, ycov, yerr =  fitProjection(yproj, fit_initial, ynoise)
    ydata = makeGaussFitData(yproj, ygauss_fit)
    
    ## in order to cut the image properly, the fit is used to calculate 
    ## the sigma for the distribution, and the number of sigma to keep is
    ## determined by the scale_factor. 
    scale_factor = scale_factor

    xnpix = int(round_nearest(scale_factor*np.ceil(xgauss_fit[2]), num = 50))
    ynpix = int(round_nearest(scale_factor*np.ceil(ygauss_fit[2]), num = 50))
    
    #### Ensures that the entire image is included and that the image has 1:1 aspect ratio
    if xnpix>ynpix:
        ynpix = xnpix
    else:
        xnpix = ynpix
        
    newimg = image[np.abs(int(xcm-xnpix/2)):int(xcm+xnpix/2),np.abs(int(ycm-ynpix/2)):int(ycm+ynpix/2)]
    
    if verbose:
        print("xgauss fit", scale_factor*np.ceil(xgauss_fit[2]))
        print("yguass fit", scale_factor*np.ceil(ygauss_fit[2]))
        print("centers of mass", xcm, ycm)
        print("new sizing information", xnpix, ynpix)
        print("X min pixel: ", str(int(xcm-xnpix/2)))
        print("X max pixel: ", str(int(xcm+xnpix/2)))
        print("Y min pixel: ", str(int(ycm-ynpix/2)))
        print("Y max pixel: ", str(int(ycm+ynpix/2)))
        print("shape of the new cut image", np.shape(newimg))
        
    masked = makeMaskedImage(newimg)
        

    mask_nn, mask_mm = np.shape(masked)
    mask_nn = round_nearest(mask_nn, num = 50)
    mask_mm = round_nearest(mask_mm, num = 50)
    masked = masked[:mask_nn, :mask_mm]

    if plotting:
        ns, ms = np.shape(newimg)
        plt.figure(figsize = (5,5))
        plt.imshow(newimg, extent = [-int(ns/2)*resolution , int(ns/2)*resolution, -int(ms/2)*resolution, int(ms/2)*resolution], origin = "lower")
        plt.xlabel('microns')
        plt.show()
            
        n,m = np.shape(masked)
        axis = (np.arange(n)-int(n/2))*resolution
        np.min(axis)
        plt.figure(figsize = (5,5))
        plt.imshow(masked,extent=[np.min(axis),np.max(axis),np.min(axis),np.max(axis)], origin = "lower")
        plt.show()

        plt.figure(figsize = (6,4))

        plt.title("sigma_x = {:.3f}, sigma_y = {:.3f}, x_err = {:.3f}, y_err = {:.3f}".format(xgauss_fit[2]*resolution*1E-3, 
                                                                                              ygauss_fit[2]*resolution*1E-3, xerr[2]*resolution*1E-3,
                                                                                              yerr[2]*resolution*1E-3))
        plt.plot((range(int(-len(xproj)/2),int(len(xproj)/2)))*resolution, xproj, 'b.', label = 'x projection')
        plt.plot((range(int(-len(xdata)/2),int(len(xdata)/2)))*resolution, xdata, 'b-', label = 'x fit')
        plt.plot((range(int(-len(yproj)/2),int(len(yproj)/2)))*resolution, yproj, 'r.', label = 'y projection')
        plt.plot((range(int(-len(ydata)/2),int(len(ydata)/2)))*resolution, ydata, 'r-', label = 'y fit')
        plt.xlabel('microns')
        plt.legend()
        plt.show()
        
    return img, masked, xproj, xdata, xgauss_fit, xcov, xerr, yproj, ydata, ygauss_fit, ycov, yerr
def doAnalysisAndReturnDict(images, plotting = True, VCC = True, npix = 150, resolution = 1, verbose = False, scale_factor = 5):
    '''
    This is the main function which preforms the analysis on images and returns a dictionary of information. 
    
    Positional argument: 
    Array (3D, stacked 2D images) of images to be processed -- images
    
    Optional arguments: 
    Plotting -- T/F
    VCC -- T/F (set to F for YAG processing)
    VCC or YAG resolution scalar -- resolution
    Estimated # of pixels for trimming images -- npix
    '''
    cutimages = []
    maskedimages = []
    xprojs = []
    xdatas = []
    xfits = []
    xcovs = []
    xerrs = []
    yprojs = []
    ydatas = []
    yfits = []
    ycovs = []
    yerrs = []
    rebinned = []
    extents = []
    ind = 0
    

    for i in range(np.shape(images)[2]):
        img = images[:,:,i]
        if plotting:
            print(ind)

        
        if VCC:
            img, masked, xproj, xdata, xgauss_fit, xcov, xerr, yproj, ydata, ygauss_fit, ycov, yerr = VCCAnalysis(img, plotting = plotting, resolution = resolution, npix = npix, verbose = verbose)
        else:
            img, masked, xproj, xdata, xgauss_fit, xcov, xerr, yproj, ydata, ygauss_fit, ycov, yerr = YAGAnalysis(img, scale_factor = scale_factor, plotting = plotting, resolution = resolution, npix = npix, verbose = verbose)
        maskedimages.append(masked)
        cutimages.append(img)
        xprojs.append(xproj)
        xdatas.append(xdata)
        xfits.append(xgauss_fit)
        xcovs.append(xcov)
        xerrs.append(xerr)
        yprojs.append(yproj)
        ydatas.append(ydata)
        yfits.append(ygauss_fit)
        ycovs.append(ycov)
        yerrs.append(yerr)
        binned = bin_ndarray(masked,(50,50))
        rebinned.append(binned)
        
        n,m = np.shape(masked)
        axis = (np.arange(n)-int(n/2))*resolution
        extent=[np.min(axis),np.max(axis),np.min(axis),np.max(axis)]
        
        extents.append(extent)
        
        if plotting:
            plt.title('rebinned image')
            plt.imshow(binned,extent=[np.min(axis),np.max(axis),np.min(axis),np.max(axis)],origin = "lower")
            plt.show()
        ind = ind + 1
        print("________________________________________________________")
        
    myDict = {
    'trimmed_images': cutimages,
    'binned': rebinned,
    'masked_images':  maskedimages,
    'xprojections': xprojs,
    'xdata': xdatas,
    'xfits': xfits,
    'xcovs': xcovs,
    'xerrs': xerrs,
    'ydata': ydatas,
    'yfits': yfits,
    'ycovs': ycovs,
    'yerrs': yerrs,
    'extents': extents    
    }
    return myDict



def bin_ndarray(ndarray, new_shape, operation='sum'):
    """
    All credit for this function goes to original creator: J.F. Sebastian
    Copied from: https://stackoverflow.com/questions/8090229/resize-with-averaging-or-rebin-a-numpy-2d-array
    Bins an ndarray in all axes based on the target shape, by summing or
        averaging.

    Number of output dimensions must match number of input dimensions and 
        new axes must divide old ones.

    Example
    -------
    >>> m = np.arange(0,100,1).reshape((10,10))
    >>> n = bin_ndarray(m, new_shape=(5,5), operation='sum')
    >>> print(n)

    [[ 22  30  38  46  54]
     [102 110 118 126 134]
     [182 190 198 206 214]
     [262 270 278 286 294]
     [342 350 358 366 374]]

    """
    operation = operation.lower()
    if not operation in ['sum', 'mean']:
        raise ValueError("Operation not supported.")
    if ndarray.ndim != len(new_shape):
        raise ValueError("Shape mismatch: {} -> {}".format(ndarray.shape,
                                                           new_shape))
    compression_pairs = [(d, c//d) for d,c in zip(new_shape,
                                                  ndarray.shape)]
    flattened = [l for p in compression_pairs for l in p]
    ndarray = ndarray.reshape(flattened)
    for i in range(len(new_shape)):
        op = getattr(ndarray, operation)
        ndarray = op(-1*(i+1))
    return ndarray


def getEachPV(pv, whichDataset, timestamps, f):
    '''
    Helper function for getting PV data from dataset h5s
    '''
    x = []
    for k in timestamps:
        temp = dict(f[whichDataset][k]['pvdata'].attrs)
        x.append(temp[pv][0])
    return x

def getPVsVCC(keys, timestamp, f, whichDataset = 'lcls_sc_image_and_pv_data_vcc'):
    '''
    Helper function for getting PV data from dataset h5s
    '''
    pvData = dict(f[whichDataset][timestamp]['pvdata'].attrs)
    vals = []
    for k in keys:
        vals.append(pvData[k][0])
    return vals

def getPVsYAG(keys, timestamp, f, whichDataset = 'lcls_sc_image_and_pv_data_yag'):
    '''
    Helper function for getting PV data from dataset h5s
    '''
    pvData = dict(f[whichDataset][timestamp]['pvdata'].attrs)
    vals = []
    for k in keys:
        vals.append(pvData[k][0])
    return vals

def makeDataPerVCC(keys, dataset, timestamp, f):
    '''
    Helper function for getting PV data from dataset h5s
    '''
    dataset[timestamp] = getPVsVCC(keys, timestamp, f)
    return dataset

def makeDataPerYAG(keys, dataset, timestamp, f):
    '''
    Helper function for getting PV data from dataset h5s
    '''
    dataset[timestamp] = getPVsYAG(keys, timestamp, f)
    return dataset
    
def package(vcc, vccpvs, vcc_res, yag, yagpvs, yag_res, pvnames):
    '''
    Packaging utility for 
    '''
    inputs = np.array(vccpvs)
    inputs[:,-1] = np.array(yagpvs[:,-1])
    inputs = dict(zip(pvnames,inputs.T))
    vcc_binned = np.array(vcc["binned"])
    yag_binned = np.array(yag['binned'])
    vcc_extents = np.array(vcc["extents"])
    yag_extents = np.array(yag['extents'])
    vcc_images = []
    yag_images = []
    yag_fits = []
    yag_errs = []
    for i in range(np.shape(vccpvs)[0]):
        vcc_images.append(np.append(vcc_extents[i,:],vcc_binned[i,:,:].flatten()))
        yag_images.append(np.append(yag_extents[i,:],yag_binned[i,:,:].flatten()))
        yag_fits.append([yag['xfits'][i][2]*vcc_res, yag['yfits'][i][2]*yag_res])
        yag_errs.append([yag['xerrs'][i][2]*vcc_res, yag['yerrs'][i][2]*yag_res])
    return inputs, vcc_images, yag_images, np.array(yag_fits), np.array(yag_errs)