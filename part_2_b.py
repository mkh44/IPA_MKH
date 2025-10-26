#PART 2B
import numpy as np
import matplotlib.pyplot as plt
from skimage import io, color, feature, transform, exposure, filters, restoration
from skimage.feature import match_template, peak_local_max
from skimage.draw import rectangle_perimeter
from skimage.transform import rotate
from scipy import stats
from scipy.stats import median_abs_deviation
from skimage.filters import threshold_otsu


#load images
source = io.imread('avian_blood.jpg')
template = io.imread('avian_blood_template.jpg')

#convert to greyscale
source_grey = color.rgb2gray(source)
template_grey = color.rgb2gray(template)

source_grey = exposure.equalize_hist(source_grey)
template_grey = exposure.equalize_hist(template_grey)

#making sure image is interpreted as 2D
source_grey = np.atleast_2d(source_grey)
template_grey = np.atleast_2d(template_grey)

#PARAMETERS FROM IMAGE
#adaptive clip lim form img size
source_clip_limit = 0.005 * source_grey.size / 255
template_clip_limit = 0.005 * template_grey.size / 255

#NOISE
#noise reduction (adaptive)
source_eq = exposure.equalize_adapthist(source_grey, clip_limit=source_clip_limit,)
template_eq = exposure.equalize_adapthist(template_grey, clip_limit=template_clip_limit,)

est_sigma_source = restoration.estimate_sigma(source_grey, channel_axis=None)
sigma_source = np.clip(2.0 * est_sigma_source * 255, 0.5, 3.0)

est_sigma_template = restoration.estimate_sigma(template_grey, channel_axis=None)
sigma_template = np.clip(2.0 * est_sigma_template * 255, 0.5, 3.0)

min_distance_factor = 0.8
adaptive_min_distance = int(0.6 * np.mean(template_grey.shape))
adaptive_min_distance = max(1, adaptive_min_distance)


#edge detection
source_edges = feature.canny(source_eq, sigma=sigma_source)
template_edges = feature.canny(template_eq, sigma=sigma_template)

#edges + intensities
source_combined = 0.5 * source_eq + 0.5 * source_edges
template_combined = 0.5 * template_eq + 0.5 * template_edges

#template matching rotation invariance
angles = np.arange(0, 360, 15)
best_result = []
best_angle = 0


for angle in angles:
    rotated_template = transform.rotate(template_edges, float(angle), resize=True)
    result = match_template(source_edges, rotated_template)
    threshold = np.mean(result) + 3 * np.std(result)
    y, x = np.unravel_index(np.argmax(result), result.shape)
    score = result[y, x]
    best_result.append((y, x, rotated_template.shape[0], rotated_template.shape[1], score))


source_detected = source.copy()

for (y, x, h, w, score) in best_result:
    h, w = rotated_template.shape
    y1, x1 = int(max(0, y)), int(max(0, x))
    y2, x2 = int(min(source_detected.shape[0] - 1, y + h)), int(min(source_detected.shape[1] - 1, x + w))
    if y1 >= y2 or x1 >= x2:
        continue
    if y1 <= 0 or x1 <= 0 or y2 >= source_detected.shape[0] - 1 or x2 >= source_detected.shape[1] - 1:
        continue
    rr, cc = rectangle_perimeter((y1, x1), end=(y2, x2), shape=source_detected.shape, clip=True)
    source_detected[rr, cc] = (255, 0, 0)



fig, ax = plt.subplots(1, figsize=(8, 6))
plt.imshow(source_detected)
plt.title(f"All Detections (threshold={threshold:.2f})")
plt.axis('off')
plt.tight_layout()
plt.show()




