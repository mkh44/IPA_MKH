
import numpy as np
import matplotlib.pyplot as plt
import skimage as sk

img = io.imread('my_face.jpg')

img_resized = transform.resize(img, (512, 512), anti_aliasing=True)
