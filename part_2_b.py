#PART 2B
import numpy as np
import matplotlib.pyplot as plt
from skimage import io, color, feature, transform, exposure
from skimage.feature import match_template
from skimage.draw import rectangle_perimeter

#load images
source = io.imread('avian_blood.jpg')
template = io.imread('avian_blood_template.jpg')

#convert to greyscale
source_grey = color.rgb2gray(source)
template_grey = color.rgb2gray(template)

#normalise images to reduce noise
source_grey = exposure.equalize_hist(source_grey)
template_grey = exposure.equalize_hist(template_grey)
