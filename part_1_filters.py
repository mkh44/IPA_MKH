
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

#converting to greyscale
grey_img = sk.color.rgb2gray(img_resized)

#desplaying images
plt.figure(figsize=(5, 5))
plt.imshow(img)
plt.title('Original Image')
plt.axis('off')

plt.figure(figsize=(5, 5))
plt.imshow(img_resized)
plt.title('Resized RGB Image')
plt.axis('off')

plt.figure(figsize=(5, 5))
plt.imshow(grey_img)
plt.title('Resized Greyscale Image')
plt.axis('off')
plt.show()
