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

def iou_matrix(labels_a, labels_b):
    regions_a = measure.regionprops(labels_a)
    regions_b = measure.regionprops(labels_b)
    n_a = len(regions_a)
    n_b = len(regions_b)
    if n_a == 0 or n_b == 0:
        return np.zeros((n_a, n_b)), regions_a, regions_b

    la = labels_a.ravel()
    lb = labels_b.ravel()
    max_a = labels_a.max()
    max_b = labels_b.max()

    pairs = la.astype(np.int64) * (max_b + 1) + lb.astype(np.int64)
    vals, counts = np.unique(pairs, return_counts=True)

    # decode
    ious = np.zeros((n_a, n_b), dtype=float)
    area_a = np.array([p.area for p in regions_a])
    area_b = np.array([p.area for p in regions_b])

    label_to_idx_a = {p.label: idx for idx, p in enumerate(regions_a)}
    label_to_idx_b = {p.label: idx for idx, p in enumerate(regions_b)}
    for val, cnt in zip(vals, counts):
        la_val = val // (max_b + 1)
        lb_val = val % (max_b + 1)
        if la_val == 0 or lb_val == 0:
            continue
        ia = label_to_idx_a.get(la_val, None)
        ib = label_to_idx_b.get(lb_val, None)
        if ia is None or ib is None:
            continue
        inter = cnt
        union = area_a[ia] + area_b[ib] - inter
        if union > 0:
            ious[ia, ib] = inter / union
    return ious, regions_a, regions_b


def run_rotation_experiment(image_path, angles=np.arange(0, 360, 10), resize_shape=(512,512), plot_results=True):
    img_orig = io.imread(image_path)
    labels_ref, regions_ref, img_color_ref = segment_cells(img_orig, resize_shape=resize_shape)
    count_ref = len(regions_ref)
    print(f"Reference count (0°): {count_ref}")

    results = []
    for angle in angles:
        # rotate image
        rotated = transform.rotate(img_orig, angle=angle, resize=False, mode='reflect')
        labels_rot, props_rot, img_color_rot = segment_cells(rotated, resize_shape=resize_shape)
        count_rot = len(props_rot)