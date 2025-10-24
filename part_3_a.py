#PART 3A
import numpy as np
import matplotlib.pyplot as plt
from skimage import io, color, filters, feature, morphology, exposure, transform, measure, segmentation
from scipy import ndimage as ndi

# load and resize
img = io.imread('HeLa_cells.jpg')
img_resized = transform.resize(img, (512, 512), anti_aliasing=True)

# convert to greyscale
img_grey = color.rgb2gray(img_resized)

#DENOISE
#enhance local contrast
img_eq = exposure.equalize_adapthist(img_grey, clip_limit=0.03)

#gaussian
sigma = max(1, (min(img_eq.shape) / 512) * 1.5)
img_smooth = filters.gaussian(img_eq, sigma=sigma)

#edges
edges = feature.canny(img_smooth)
plt.imshow(edges)

#thresholding
threshold_value = filters.threshold_otsu(img_smooth)
binary = img_smooth > threshold_value

#remove small noise and fill small holes
binary_cleaned = ndi.binary_fill_holes(binary)

selem = morphology.disk(max(1, int(min(img_eq.shape) / 200)))
binary_cleaned = morphology.opening(binary_cleaned, selem)

#separate touching cells
distance = ndi.distance_transform_edt(binary_cleaned)
smooth_sigma = max(1, int(min(img_eq.shape) / 300))
distance_smooth = filters.gaussian(distance, sigma=smooth_sigma)
h_value = 0.4 * np.std(distance_smooth)
distance_hmax = morphology.h_maxima(distance_smooth, h=h_value)
markers, _ = ndi.label(distance_hmax)
labels = segmentation.watershed(-distance_smooth, markers, mask=binary_cleaned)

#remove boundary touching objects
labels_no_border = segmentation.clear_border(labels)

#count labelled regions
regions = measure.regionprops(labels_no_border)
cell_count = len(regions)

print(f"Total number of HeLa cells (excluding boundary cells): {cell_count}")

#plotting
plt.figure(figsize=(10, 10))
plt.imshow(color.label2rgb(labels_no_border, bg_label=0))
plt.title(f'Final Segmentation (Count = {cell_count})')
plt.axis('off')
plt.tight_layout()
plt.show()


