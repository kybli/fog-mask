import numpy as np
from PIL import Image
import skimage, skimage.io as io, skimage.morphology

# value of 0 = masked
# value of 1 = unmasked

def dark_channel(img, size):
    dark_channel_img = np.amin(img, axis=2)
    footprint = np.ones((size, size))
    dark_channel_img = skimage.morphology.erosion(dark_channel_img, footprint)
    return dark_channel_img

def atmospheric_light(img, dark_channel_img):
    return

if __name__ == '__main__':
    file_path = "img/input/foggy_cityscape.png"

    im = io.imread(file_path)
    dc = dark_channel(im, 50)

    io.imsave('img/dark-channel/dark_channel.png', dc)
    
    