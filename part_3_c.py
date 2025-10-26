#PART 3C
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from skimage import io, color, filters, morphology, exposure, transform, measure, segmentation
from skimage.restoration import estimate_sigma
#import pywavelets as pywt
from scipy import ndimage as ndi
from scipy.optimize import linear_sum_assignment
from skimage.measure import label as sk_label
import warnings

warnings.filterwarnings('ignore')

# defining segmentation pipeline
def segment_cells(image_color, resize_shape=(512,512)):
    img = image_color.astype(float)
    if img.max() > 1:
        img = img / 255
    if img.ndim == 3:
        img_grey = color.rgb2gray(img)
    else:
        img_grey = img.copy()

    if img_grey.shape != resize_shape:
        img_grey = transform.resize(img_grey, resize_shape, anti_aliasing=True)
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
        labels_rot, regions_rot, img_color_rot = segment_cells(rotated, resize_shape=resize_shape)
        count_rot = len(regions_rot)
        i, pa, pb = iou_matrix(labels_ref, labels_rot)

        #compare
        if i.size == 0:
            mean_iou = 0.0
            matched = 0
            centroid_disp_median = np.nan
            area_ratio_mean = np.nan
            unmatched_ref = len(pa)
            unmatched_rot = len(pb)
        else:
            cost = 1.0 - i
            n, m = cost.shape
            if n > m:
                cost_pad = np.hstack([cost, np.ones((n, n - m))])
            elif m > n:
                cost_pad = np.vstack([cost, np.ones((m - n, m))])
            else:
                cost_pad = cost
            row_ind, col_ind = linear_sum_assignment(cost_pad)
            matches = []
            ious_matched = []
            centroid_disps = []
            area_ratios = []
            for r, c in zip(row_ind, col_ind):
                if r < n and c < m:
                    iou_score = i[r, c]
                    if iou_score > 0.1:
                        matches.append((r, c, iou_score))
                        ious_matched.append(iou_score)
                        ca = pa[r].centroid
                        cb = pb[c].centroid
                        disp = np.sqrt((ca[0] - cb[0]) ** 2 + (ca[1] - cb[1]) ** 2)
                        centroid_disps.append(disp)
                        area_ratios.append(pb[c].area / pa[r].area)
            if len(ious_matched) == 0:
                mean_iou = 0.0
                matched = 0
                centroid_disp_median = np.nan
                area_ratio_mean = np.nan
            else:
                mean_iou = np.mean(ious_matched)
                matched = len(ious_matched)
                centroid_disp_median = np.median(centroid_disps)
                area_ratio_mean = np.mean(area_ratios)
            unmatched_ref = max(0, len(pa) - matched)
            unmatched_rot = max(0, len(pb) - matched)

        results.append({
            "angle": angle,
            "count_rot": count_rot,
            "count_ref": count_ref,
            "count_diff": count_rot - count_ref,
            "mean_iou": mean_iou,
            "matched": matched,
            "unmatched_ref": unmatched_ref,
            "unmatched_rot": unmatched_rot,
            "centroid_disp_median": centroid_disp_median,
            "area_ratio_mean": area_ratio_mean
        })
        print(f"Angle {angle:3d}°: count={count_rot:3d}, Δ={count_rot - count_ref:3d}, meanIoU={mean_iou:.3f}, matched={matched}")

    df = pd.DataFrame(results)
    df = df.rename(columns={
            "angle": "Rotation Angle (°)",
            "count_rot": "Cell Count",
            "count_diff": "Δ Count",
            "mean_iou": "Mean IoU",
            "centroid_disp_median": "Median Centroid Displacement (px)"
    })
    print(df)

    print(f"Angle {angle:3d}°: count={count_rot:3d}, Δ={count_rot - count_ref:3d}, meanIoU={mean_iou:.3f}, matched={matched}")

    #plotting

    if plot_results:
        fig, axs = plt.subplots(2, 2, figsize=(12, 8))
        axs = axs.ravel()
        axs[0].plot(df.angle, df.count_rot, '-o', label='rotated count')
        axs[0].axhline(df.count_ref.iloc[0], color='k', linestyle='--', label='reference')
        axs[0].set_title('Cell count vs rotation angle')
        axs[0].set_xlabel('angle (deg)')
        axs[0].set_ylabel('count')
        axs[0].legend()

        axs[1].plot(df.angle, df.mean_iou, '-o')
        axs[1].set_title('Mean IoU (matched regions)')
        axs[1].set_xlabel('angle (deg)')
        axs[1].set_ylabel('mean IoU')

        axs[2].plot(df.angle, df.centroid_disp_median, '-o')
        axs[2].set_title('Median centroid displacement (pixels)')
        axs[2].set_xlabel('angle (deg)')
        axs[2].set_ylabel('pixels')

        axs[3].plot(df.angle, df.unmatched_rot, '-o', label='unmatched_rot')
        axs[3].plot(df.angle, df.unmatched_ref, '-o', label='unmatched_ref')
        axs[3].set_title('Unmatched regions')
        axs[3].legend()

        plt.tight_layout()
        plt.show()

    return df, labels_ref


if __name__ == "__main__":
    image_path = "HeLa_cells.jpg"
    angles = np.arange(0, 360, 10)  # test every 10 degrees
    df, labels_ref = run_rotation_experiment(image_path, angles=angles, resize_shape=(512,512), plot_results=True)

