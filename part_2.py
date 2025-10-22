
import numpy as np
import matplotlib.pyplot as plt
import skimage as sk

#load images
source = sk.io.imread('avian_blood.jpg')
template = sk.io.imread('avian_blood_template.jpg')

#convert to greyscale
source_grey = sk.color.rgb2gray(source)
template_grey = sk.color.rgb2gray(template)

#get rid of noise with gaussian filter
sigma_source = np.std(source_grey) * 0.5
sigma_template = np.std(template_grey) * 0.5
source_smooth = sk.filters.gaussian(source_grey, sigma=sigma_source)
template_smooth = sk.filters.gaussian(template_grey, sigma=sigma_template)

#normalise images

#automated template matching with rotation (to do)
#get angles
#find best score
#find best match
#find best angle

#for angles match with template

#update scores

#find match location

#find threshold

#plotting

