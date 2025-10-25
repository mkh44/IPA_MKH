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