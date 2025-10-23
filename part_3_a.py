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
image_eq = exposure.equalize_adapthist(img_resized, clip_limit=0.03)

#gaussian

# enhance contrast#

#thresholding

#remove small noise and fill small holes

#separate touching cells

#remove boundary touching objects

#count labelled regions



