
import numpy as np
import matplotlib.pyplot as plt
import skimage as sk

img = sk.io.imread('my_face.jpg')

img_resized = sk.transform.resize(img, (512, 512), anti_aliasing=True)


