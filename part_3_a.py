#PART 3A
import numpy as np
import matplotlib.pyplot as plt
from skimage import io, color, filters, morphology, exposure, transform, measure, segmentation
from scipy import ndimage as ndi

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
binary_cleaned = morphology.remove_small_objects(binary, min_size=50)
binary_cleaned = morphology.remove_small_holes(binary_cleaned, area_threshold=100)

#separate touching cells
distance = ndi.distance_transform_edt(binary_cleaned)

#local max and min
footprint_size = int(np.mean(img_eq.shape) / 50)
local_max = morphology.local_maxima(distance, footprint=morphology.disk(footprint_size))

markers, _ = ndi.label(local_max)
labels = morphology.watershed(-distance, markers, mask=binary_cleaned)

#remove boundary touching objects

#count labelled regions



