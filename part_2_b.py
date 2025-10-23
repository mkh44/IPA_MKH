#PART 2B
import numpy as np
import matplotlib.pyplot as plt
from skimage import io, color, feature, transform, exposure, filters
from skimage.feature import match_template, peak_local_max
from skimage.draw import rectangle_perimeter
from skimage.transform import rotate
from scipy import stats
from scipy.stats import median_abs_deviation

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

canny_sigma_source = max(1, int(source_noise * 3))
canny_sigma_template = max(1, int(template_noise * 3))

min_distance_factor = 0.8
adaptive_min_distance = int(min(template_grey.shape) * min_distance_factor)
adaptive_min_distance = max(1, adaptive_min_distance)

#NOISE
#noise reduction (adaptive)
source_eq = exposure.equalize_adapthist(source_grey, clip_limit=source_clip_limit,)
template_eq = exposure.equalize_adapthist(template_grey, clip_limit=template_clip_limit,)

#gaussian noise reduction
source_smooth = filters.gaussian(source_eq, sigma=gaussian_sigma_source)
template_smooth = filters.gaussian(template_eq, sigma=gaussian_sigma_template)

#edge detection
source_edges = feature.canny(source_smooth, sigma=canny_sigma_source)
template_edges = feature.canny(template_smooth, sigma=canny_sigma_template)

#edges + intensities
source_combined = 0.5 * source_eq + 0.5 * source_edges
template_combined = 0.5 * template_eq + 0.5 * template_edges


#speed
# scale_factor = 0.9
# source_edges_small = transform.rescale(source_edges, scale_factor, anti_aliasing=False)
# template_edges_small = transform.rescale(template_edges, scale_factor, anti_aliasing=False)

#template matching rotation invariance
angles = np.arange(0, 360, 10)
best_result = None
best_angle = 0
all_detections = []

for angle in angles:
    rotated_template = rotate(template_edges, angle, resize=True)

    # template matching
    result = match_template(source_edges, rotated_template)

    adaptive_threshold = np.percentile(result, 95) # Use 95th percentile as threshold

    peaks = peak_local_max(result, min_distance=adaptive_min_distance, threshold_abs=adaptive_threshold)

    for (y, x) in peaks:
        all_detections.append((int(y), int(x), int(rotated_template.shape[0]), int(rotated_template.shape[1]), angle))

source_detected = source.copy()

for (y, x, h, w, angle) in all_detections:
    rr, cc = rectangle_perimeter((y, x), end=(y+h, x+w), shape=source_detected.shape)
    source_detected[rr, cc] = (255, 0, 0)




fig, ax = plt.subplots(1, figsize=(8, 6))
plt.imshow(source_detected)
plt.title(f"All Detections (threshold={adaptive_threshold:.2f})")
plt.axis('off')
plt.tight_layout()
plt.show()




