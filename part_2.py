
import numpy as np
import matplotlib.pyplot as plt
import skimage as sk

#load images
source = sk.io.imread('avian_blood.jpg')
template = sk.io.imread('avian_blood_template.jpg')

#convert to greyscale
source_grey = sk.color.rgb2gray(source)
template_grey = sk.color.rgb2gray(template)

#get rid of noise with gaussian filter
sigma_source = np.std(source_grey) * 0.5
sigma_template = np.std(template_grey) * 0.5
source_smooth = sk.filters.gaussian(source_grey, sigma=sigma_source)
template_smooth = sk.filters.gaussian(template_grey, sigma=sigma_template)

#normalise images
source_smooth = sk.exposure.rescale_intensity(source_smooth, out_range=(0, 1))
template_smooth = sk.exposure.rescale_intensity(template_smooth, out_range=(0, 1))


#automated template matching with rotation (to do)

angles = np.arange(0, 360, 10)
best_score = -np.inf
best_match = None
best_angle = None

for angle in angles:
    rotated = sk.transform.rotate(source_smooth, angle, resize=False)
    result = sk.feature.match_template(rotated, template_smooth, pad_input=True)
    score = np.max(result)

    if score > best_score:
        best_score = score
        best_angle = angle
        best_match = result

#find match location
ij = np.unravel_index(np.argmax(best_match), best_match.shape)
x, y, = ij[::-1]

#find threshold


#plotting

