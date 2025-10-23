#PART 3A
import numpy as np
import matplotlib.pyplot as plt
from skimage import io, color, filters, morphology, exposure, transform, measure, segmentation

from part_1_filters import img_resized

# load and resize
img = io.imread('HeLa_cells.jpg')
img_resized = transform.resize(img, (512, 512), anti_aliasing=True)

# convert to greyscale
img_grey = color.rgb2gray(img_resized)

#DENOISE
#enhance local contrast
img_eq = exposure.equalize_adapthist(img_resized, clip_limit=0.03)

#gaussian
sigma = max(1, (min(img_eq.shape) / 512) * 1.5)
img_smooth = filters.gaussian(img_eq, sigma=sigma)

#thresholding
threshold_value = filters.threshold_otsu(img_smooth)
binary = img_smooth > threshold_value

#remove small noise and fill small holes

#separate touching cells

#remove boundary touching objects

#count labelled regions



