#PART 2A
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
source_norm = exposure.equalize_hist(source_grey)
template_norm = exposure.equalize_hist(template_grey)

#edge detection
source_edges = feature.canny(source_norm, sigma=2)
template_edges = feature.canny(template_norm, sigma=2)

#rotation invariance
angles = np.arange(0, 360, 15) #checking every 15 degrees
best_score = -np.inf
best_match = None
best_angle = None

for angle in angles:
    rotated = sk.transform.rotate(source_smooth, angle, resize=True)
    result = sk.feature.match_template(rotated, template_smooth, pad_input=True)
    score = np.max(result)

    if score > best_score:
        best_score = score
        best_angle = angle
        best_match = result

#find match location
ij = np.unravel_index(np.argmax(best_match), best_match.shape)
x, y, = ij[::-1]

#find threshold from mean + k*std
threshold = np.mean(best_match) + 2 * np.std(best_match)
match_mask = best_match > threshold

#plotting
fig, ax = plt.subplots(1, 2, figsize=(12, 6))
ax[0].imshow(sk.transform.rotate(source, best_angle, resize=False))
ax[0].set_title(f'Best match at {best_angle:.1f} degrees')
rect = plt.Rectangle((x, y), template.shape[1], template.shape[0], edgecolor='red', facecolor='none', linewidth=2)
ax[0].add_patch(rect)

ax[1].imshow(best_match, cmap='viridis')
ax[1].set_title('Template Match Correlation Map')

for a in ax:
    a.axis('off')
plt.tight_layout()
plt.show()