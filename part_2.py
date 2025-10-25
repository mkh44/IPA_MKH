#PART 2A
import numpy as np
import matplotlib.pyplot as plt
from skimage import io, color, feature, transform, exposure, restoration
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

est_sigma_source = restoration.estimate_sigma(source_grey, channel_axis=None)
sigma_source = np.clip(3.0 * est_sigma_source * 255, 0.5, 3.0)

est_sigma_template = restoration.estimate_sigma(template_grey, channel_axis=None)
sigma_template = np.clip(3.0 * est_sigma_template * 255, 0.5, 3.0)

#edge detection
source_edges = feature.canny(source_grey, sigma=sigma_source)
template_edges = feature.canny(template_grey, sigma=sigma_template)

#rotation invariance
angles = np.arange(0, 360, 15) #checking every 15 degrees
best_score = []
threshold = 0.3

for angle in angles:
    rotated_template = transform.rotate(template_edges, angle, resize=True)
    result = match_template(source_edges, rotated_template)
    score = np.max(result)

    #peaks
    match_indices = np.where(result >= threshold)
    for (y, x) in zip(*match_indices):
        score = result[y, x]
        best_score.append((y, x, rotated_template.shape[0], rotated_template.shape[1], score))

source_detected = source.copy()

for (y, x, h, w, score) in best_score:
    rr, cc = rectangle_perimeter((y, x), end=(y+h, x+w), shape=source_detected.shape)
    source_detected[rr, cc] = (255, 0, 0)#red box thing

#plotting
plt.figure(figsize=(8, 6))
plt.imshow(source_detected)
plt.title('Detections (red boxes)')
plt.axis('off')
plt.tight_layout()
plt.show()