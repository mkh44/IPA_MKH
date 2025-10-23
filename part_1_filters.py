#PART 1A: RESIZING IMAGE
import numpy as np
import matplotlib.pyplot as plt
import skimage as sk
import PIL as pil

#correcting image orrientation
img_pil = pil.Image.open('my_face.jpg')
img_pil = pil.ImageOps.exif_transpose(img_pil)
img = np.array(img_pil)
#resizing image
img_resized = sk.transform.resize(img, (512, 512), anti_aliasing=True)
#desplaying image
plt.figure(figsize=(5, 5))
plt.imshow(img_resized)
plt.title('Resized RGB Image')
plt.axis('off')


#PART 1A: GREYSCALE
#converting to greyscale
grey_img = sk.color.rgb2gray(img_resized)

plt.figure(figsize=(5, 5))
plt.imshow(grey_img, cmap='gray', VMIN=0, VMAX=1)
plt.title('Resized Greyscale Image')
plt.axis('off')
plt.show()

#PART 1A: MAX, MIN AND MEAN
#min, max and mean values
min_val = np.min(grey_img)
max_val = np.max(grey_img)
mean_val = np.mean(grey_img)

print(f'min_val: {min_val:.4f}')
print(f'max_val: {max_val:.4f}')
print(f'mean_val: {mean_val:.4f}')

#PART 1B: ADDING GAUSSIAN
mean = 0.0
variance = 0.04
sigma = np.sqrt(variance)
gaussian_noise = np.random.normal(mean, sigma, grey_img.shape)
noisy_img = grey_img + gaussian_noise
noisy_img = np.clip(noisy_img, 0, 255).astype(np.unit8)
plt.figure(figsize=(5, 5))
plt.imshow(noisy_img, cmap='gray')
plt.title('Greyscale Image with Gaussian Noise')
plt.axis('off')
plt.show()

#PART 1C: EDGE DETECTORS
#apply sobel edge detector
edges_sobel = sk.filters.sobel(noisy_img)

#appky prewitt edge detector
edges_prewitt = sk.filters.prewitt(noisy_img)

# apply canny
edges_canny = sk.feature.canny(noisy_img, sigma=1)

# displaying results
plt.figure(figsize=(5, 5))
plt.imshow(edges_sobel, cmap='gray')
plt.title('Edges Sobel')
plt.axis('off')

plt.figure(figsize=(5, 5))
plt.imshow(edges_prewitt, cmap='gray')
plt.title('Edges Prewitt')
plt.axis('off')

plt.figure(figsize=(5, 5))
plt.imshow(edges_canny, cmap='gray')
plt.title('Edges Canny')
plt.axis('off')

plt.show()
