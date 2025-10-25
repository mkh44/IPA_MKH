#PART 3C
import numpy as np
import matplotlib.pyplot as plt
from skimage import io, color, filters, morphology, exposure, transform, measure, segmentation
from skimage.restoration import estimate_sigma
from scipy import ndimage as ndi
from scipy.optimize import linear_sum_assignment
from skimage.measure import label as sk_label

# defining segmentation pipeline
def segment_cells(image_color, resize_shape=(512,512))
    if img.ndim == 3:
        img_grey = color.rgb2gray(img)
    else:
        img_grey = img.copy()

    if img_grey.shape != resize_shape:
        img_gray = transform.resize(img_grey, resize_shape, anti_aliasing=True)
        img_color = transform.resize(img, resize_shape, anti_aliasing=True)
    else:
        img_color = img.copy()

    img_eq = exposure.equalize_adapthist(img_grey, clip_limit=0.03)

    #noise
    noise_sigma = estimate_sigma(img_eq, channel_axis=None)
    sigma = np.clip(2.0 * noise_sigma * 255, 0.5, 3.0)

    img_smooth = filters.gaussian(img_eq, sigma=sigma)

    #otsu thresholding
    threshold = filters.threshold_otsu(img_smooth)
    binary = img_smooth > threshold

    #cleaning
    min_size = max(20, int(0.0005 * img_smooth.size))
    binary = morphology.remove_small_objects(binary, min_size=min_size)
    binary = morphology.remove_small_holes(binary, area_threshold=int(0.0005 * img_smooth.size))

    selem = morphology.disk(max(1, int(min(img_eq.shape) / 200)))
    binary = morphology.opening(binary, selem)

    distance = ndi.distance_transform_edt(binary)
    distance_smooth = filters.gaussian(distance, sigma=np.clip(min(img_eq.shape)/400, 1, 2))

    h_value = 0.6 * np.std(distance_smooth)
    markers = morphology.h_maxima(distance_smooth, h=h_value)
    markers_labeled, _ = ndi.label(markers)

    labels = segmentation.watershed(-distance_smooth, markers_labeled, mask=binary)
    labels_no_border = segmentation.clear_border(labels)


    labels_no_border = sk_label(labels_no_border > 0)

    props_final = measure.regionprops(labels_no_border)

    return labels_no_border, props_final, img_color

