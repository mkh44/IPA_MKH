#PART 2B
import numpy as np
import matplotlib.pyplot as plt
from skimage import io, color, feature, transform, exposure, filters
from skimage.feature import match_template, peak_local_max
from skimage.draw import rectangle_perimeter
from skimage.transform import rotate
from skipy import stats
from skipy.stats import median_abs_deviation

#load images
source = io.imread('avian_blood.jpg')
template = io.imread('avian_blood_template.jpg')

#convert to greyscale
source_grey = color.rgb2gray(source)
template_grey = color.rgb2gray(template)

#PARAMETERS FROM IMAGE
#adaptive clip lim form img size
source_clip_limit = 0.005 * source_grey.size / 256
template_clip_limit = 0.005 * template_grey.size / 256

#estemated noise level (median abs more robust than std
source_noise = median_abs_deviation(source_grey.flatten())
template_noise = median_abs_deviation(template_grey.flatten())

#sigma for gaussian filter. set as proportional to noise level
gaussian_sigma_source = max(1, int(source_noise * 2))
gaussian_sigma_template = max(1, int(template_noise * 2))

min_distance_factor = 0.8
adaptive_min_distance = int(min(template_grey.shape) * min_distance_factor)
adaptive_min_distance = max(1, adaptive_min_distance)

#noise reduction (adaptive)
source_eq = exposure.equalize_adapthist(source_grey, clip_limit=source_clip_limit,)
template_eq = exposure.equalize_adapthist(template_grey, clip_limit=template_clip_limit,)

#gaussian noise reduction
source_smooth = filters.gaussian(source_eq, sigma=1)
template_smooth = filters.gaussian(template_eq, sigma=1)

#edge detection
source_edges = feature.canny(source_smooth, sigma=2)
template_edges = feature.canny(template_smooth, sigma=2)

#edges + intensities
source_combined = 0.5 * source_eq + 0.5 * source_edges
template_combined = 0.5 * template_eq + 0.5 * template_edges


#speed
# scale_factor = 0.9
# source_edges_small = transform.rescale(source_edges, scale_factor, anti_aliasing=False)
# template_edges_small = transform.rescale(template_edges, scale_factor, anti_aliasing=False)

# template matching (no rotation)
result = match_template(source_edges, template_edges)

# adaptive threshold selection
all_scores = np.ravel(result)
adaptive_threshold = np.mean(all_scores) + 2 * np.std(all_scores)

# peak detection
peaks = peak_local_max(result, min_distance=20, threshold_abs=adaptive_threshold)
detections = [(int(y), int(x), int(template_edges.shape[0]), int(template_edges.shape[1])) for (y, x) in peaks]
# scaling
# scaled_results = [(int(y/scale_factor), int(x/scale_factor), int(template_edges.shape[0]), int(template_edges.shape[1]))
#                   for (y, x) in peaks]


source_detected = source.copy()

for (y, x, h, w) in detections:
    rr, cc = rectangle_perimeter((y, x), end=(y+h, x+w), shape=source_detected.shape)
    source_detected[rr, cc] = (255, 0, 0)


fig, ax = plt.subplots(1, figsize=(8, 6))
plt.imshow(source_detected)
plt.title(f"All Detections (threshold={adaptive_threshold:.2f})")
plt.axis('off')
plt.tight_layout()
plt.show()




