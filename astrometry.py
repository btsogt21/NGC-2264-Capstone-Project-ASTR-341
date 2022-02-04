#!/usr/bin/env python
# coding: utf-8

# In[83]:


import numpy as np
import matplotlib.pyplot as plt
from astropy.wcs import WCS
from astropy.io import fits, ascii
from astropy.table import Table, Column
from astropy.visualization import ZScaleInterval
import pandas as pd
import glob 
import scipy.signal
import scipy.ndimage.interpolation as interp
import shutil
import pdb
import astropy.stats as stat
from astropy.stats import mad_std
from astropy.stats import sigma_clip
from photutils.utils import calc_total_error

from photutils import aperture_photometry, CircularAperture, CircularAnnulus, DAOStarFinder


# In[66]:


#Load in the image
hdu = fits.open('C:/Users/batma/Documents/reduction_combination_pipeline/20210302/ReducedB/new-image.fits')
w = WCS(hdu[0].header)

image = hdu[0].data

#Load in the target list
targ = ascii.read('C:/Users/batma/Documents/reduction_combination_pipeline/astrometry_list.txt', data_start=3, delimiter=';')
targ['_RA'].name = 'RA'
targ['_DE'].name = 'Dec'

#Put RA/Dec into an array
sky = np.array([list(targ['RA']), list(targ['Dec'])]).T


# In[67]:


targ


# In[68]:


#Transform the RA and Dec into pixel coordinates.
pos = np.array(w.wcs_world2pix(sky[:,0], sky[:,1], 0)).T
aperture = photutils.CircularAperture(pos, r=10)
pos_x = []
pos_y = []
for i in np.arange(len(pos)):
    pos_x.append(pos[i][0])
    pos_y.append(pos[i][1])


# In[69]:


#Plot the apertures
get_ipython().run_line_magic('matplotlib', 'inline')

#Scale your image to something reasonble.
mn, mx = ZScaleInterval().get_limits(np.log10(image))

plt.figure(figsize = [10,7])

#Use the wcs solution to use sky coordinates when plotting. 
ax = plt.subplot(projection = w)
plt.subplots_adjust(right = 0.98, top = 0.98)

#Plot the apertures.
ax.imshow(np.log10(image), origin = 'lower', vmin = mn, vmax = mx)
aperture.plot(color='w', lw=1.5, alpha=0.8)

ax.set_xlabel('RA [J2000]')
ax.set_ylabel('Dec [J2000]')


# In[70]:


plt.scatter(mn, mx)


# In[71]:


yuh = np.arange(10)
yuh1 = np.arange(10)


# In[72]:


plt.plot(yuh, yuh1)


# In[44]:


def measurePhotometry(fitsfile, star_xpos, star_ypos, aperture_radius, sky_inner, sky_outer, error_array):
    """
    This function takes a fitsfile, positions of stars, an aperture radius, annuli specifications, and an error image
    and outputs a photometry table for the image.
    """
    # Read in the data from the fits file:
    image = fits.getdata(fitsfile)
    
    #Creates the aperture (around the stars) and the annulus (a shell around the aperture, for bkg calcs)
    #Makes a list of apertures and annuli for each star
    #starapertures = CircularAperture((star_xpos, star_ypos),r = aperture_radius)
    pos = [(star_xpos[i],star_ypos[i]) for i in np.arange(len(star_xpos))]
    starapertures = CircularAperture(pos,r = aperture_radius)
    skyannuli = CircularAnnulus(pos, r_in = sky_inner, r_out = sky_outer)
    phot_apers = [starapertures, skyannuli]
    
    # What is new about the way we're calling aperture_photometry?
    # Last time, we didn't have an error array to pass to the function. This time, our calculations will be more accurate because they consider 
    # median background AND noise 
    phot_table = aperture_photometry(image, phot_apers, error=error_array)
        
    # Calculate mean background in annulus and subtract from aperture flux
    bkg_mean = phot_table['aperture_sum_1'] / skyannuli.area
    bkg_starap_sum = bkg_mean * starapertures.area
    final_sum = phot_table['aperture_sum_0']-bkg_starap_sum
    phot_table['bg_subtracted_star_counts'] = final_sum
    
    # Calculating the mean error in the background. First, the error in the sum of values within the aperture
    #is divided by the area of the annulus to compute the mean error from the background. Sum error is computed by 
    #multiplying this by the area of the apertures - this is the total flux error within the aperture
    bkg_mean_err = phot_table['aperture_sum_err_1'] / skyannuli.area
    bkg_sum_err = bkg_mean_err * starapertures.area
    
    # Propagating the error to find the total error: taking the square root of the sum of the squares of the Poisson noise and the background error calculated above
    phot_table['bg_sub_star_cts_err'] = np.sqrt((phot_table['aperture_sum_err_0']**2)+(bkg_sum_err**2)) 
    
    return phot_table


# In[63]:


def bg_error_estimate(fitsfile):
    """
    This function takes a .fits file as an input, gets the data from the file, masks data greater than 3
    standard deviations from the median, replaces the masked data with NaNs, and calculates the median of this 
    error. Then, it writes two files, a background error file and a total error file, the latter considering the
    gain of the observing device (and thus Poisson noise). It outputs the raw data of the error_image.
    """
    fitsdata = fits.getdata(fitsfile) #gets the data from a .fits file
    hdr = fits.getheader(fitsfile) #saves the header as hdr
    
    # What is happening in the next step? Read the docstring for sigma_clip.
    # Answer: takes the median of the data and iterates over the data, rejecting values that are more than
    #3 standard deviations (sigma=3) away from the median. Returns an array with the same shape, with the 
    #rejected data masked 

    filtered_data = sigma_clip(fitsdata, sigma=3.,copy=False)
    
    # Summarize the following steps:
    # Takes the masked array and fills all the maxed points with a specified value, in this case a NaN
    # the background error = the square root of the above result
    # takes every nan point in the above line's output and sets it to the median of the bakground error
    bkg_values_nan = filtered_data.filled(fill_value=np.nan)
    bkg_error = np.sqrt(bkg_values_nan)
    bkg_error[np.isnan(bkg_error)] = np.nanmedian(bkg_error)
    
    print("Writing the background-only error image: ", fitsfile.split('.')[0]+"_bgerror.fit")
    fits.writeto(fitsfile.split('.')[0]+"_bgerror.fit", bkg_error, hdr, overwrite=True)
    
    effective_gain = 1.4 # electrons per ADU
    
    error_image = calc_total_error(fitsdata, bkg_error, effective_gain)  
    
    print("Writing the total error image: ", fitsfile.split('.')[0]+"_error.fit")
    fits.writeto(fitsfile.split('.')[0]+"_error.fit", error_image, hdr, overwrite=True)
    
    return error_image


# In[91]:


error_array = bg_error_estimate('C:/Users/batma/Documents/reduction_combination_pipeline/20210302/ReducedB/new-image.fits')


# In[92]:


measurePhotometry('C:/Users/batma/Documents/reduction_combination_pipeline/20210302/ReducedB/new-image.fits', pos_x, pos_y, 10, 15, 20, error_array)


# In[ ]:




