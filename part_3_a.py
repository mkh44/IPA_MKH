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
img_eq = exposure.equalize_adapthist(img_grey, clip_limit=0.03)

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
window_size = max(3, int(min(img_eq.shape) / 50))
if window_size % 2 == 0:
    window_size += 1

local_max = (distance == ndi.maximum_filter(distance, size=window_size))
local_max[binary_cleaned == 0] = False

markers, _ = ndi.label(local_max)
labels = segmentation.watershed(-distance, markers, mask=binary_cleaned)

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


