import numpy as np
import scipy.interpolate as sinterp


def high_pass_filter(img, filtersize=10):
    """
    A FFT implmentation of high pass filter.

    Args:
        img: a 2D image
        filtersize: size in Fourier space of the size of the space. In image space, size=img_size/filtersize

    Returns:
        filtered: the filtered image

    Originally written for pyKLIP package (Wang et al. 2015)
    Additional comments for ASTR341 by William Balmer
    """
    # mask NaNs if there are any
    nan_index = np.where(np.isnan(img))
    if np.size(nan_index) > 0:
        good_index = np.where(~np.isnan(img))
        y, x = np.indices(img.shape)
        good_coords = np.array([x[good_index], y[good_index]]).T # shape of Npix, ndimage
        nan_fixer = sinterp.NearestNDInterpolator(good_coords, img[good_index])
        fixed_dat = nan_fixer(x[nan_index], y[nan_index])
        img[nan_index] = fixed_dat

    transform = np.fft.fft2(img)  # take img into fourier space

    # coordinate system in FFT image
    u, v = np.meshgrid(np.fft.fftfreq(transform.shape[1]), np.fft.fftfreq(transform.shape[0]))
    # scale u,v so it has units of pixels in FFT space
    rho = np.sqrt((u*transform.shape[1])**2 + (v*transform.shape[0])**2)
    # scale rho up so that it has units of pixels in FFT space
    # rho *= transform.shape[0]
    # create the filter
    filt = 1. - np.exp(-(rho**2/filtersize**2))

    filtered = np.real(np.fft.ifft2(transform*filt))  # returns filter*img into img space

    # restore NaNs
    filtered[nan_index] = np.nan
    img[nan_index] = np.nan

    return filtered
