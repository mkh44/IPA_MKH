#PART 2B
import numpy as np
import matplotlib.pyplot as plt
from skimage import io, color, feature, transform, exposure
from skimage.feature import match_template, peak_local_max
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

#edge detection
source_edges = feature.canny(source_grey, sigma=2)
template_edges = feature.canny(template_grey, sigma=2)

#speed
scale_factor = 0.3
source_edges_small = transform.rescale(source_edges, scale_factor, anti_aliasing=False)
template_edges_small = transform.rescale(template_edges, scale_factor, anti_aliasing=False)

# template matching (no rotation)
result = match_template(source_edges_small, template_edges_small)

# adaptive threshold selection
all_scores = result.ravel()
adaptive_threshold = np.mean(all_scores) + 2 * np.std(all_scores)





