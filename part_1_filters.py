
import numpy as np
import matplotlib.pyplot as plt
import skimage as sk

# loading image
img = sk.io.imread('my_face.jpg')

#resizing image
img_resized = sk.transform.resize(img, (512, 512), anti_aliasing=True)

#converting to greyscale
gray_img = sk.color.rgb2gray(img_resized)

#desplaying images
plt.plot(img)
plt.plot(img_resized)
plt.plot(gray_img)
plt.show()
