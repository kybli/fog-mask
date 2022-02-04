import numpy as np
import math
import skimage, skimage.io as io, skimage.morphology
import cv2

# value of 0 = masked
# value of 1 = unmasked

def dark_channel(img, size):
    dark_channel_img = np.amin(img, axis=2)
    footprint = np.ones((size, size))
    dark_channel_img = skimage.morphology.erosion(dark_channel_img, footprint)
    return dark_channel_img

def atmospheric_light(img, dark_channel_img):
    h, w = img.shape[:2]
    img_size = h * w
    top_point_one_percent = int(max(math.floor(img_size / 1000), 1))

    flat_dark_channel_img = dark_channel_img.reshape(img_size)
    flat_img = img.reshape(img_size, 3)
    flat_img_intensities = np.mean(img, 2).reshape(img_size)

    brightest_dc_indexes = np.argpartition(flat_dark_channel_img, -top_point_one_percent)[-top_point_one_percent:]
    
    top_atm_pixel_count = int(max(top_point_one_percent * 0.5, 1))
    atm_light_pixel_indexes = np.argpartition(flat_img_intensities[brightest_dc_indexes], -top_atm_pixel_count)[-top_atm_pixel_count:]

    atm_light = np.mean(np.take(np.take(flat_img, brightest_dc_indexes, 0), atm_light_pixel_indexes, 0), 0)

    return atm_light

def transmission_estimate(img, atm_light, size):
    w = 0.95
    return (255 * (1 - (w * dark_channel((img / atm_light), size)))).astype(np.uint8)

if __name__ == '__main__':
    file_path = "~/proj-x/fog-mask/img/input/foggy_cityscape.png"

    img = io.imread(file_path)
    dark_channel_img = dark_channel(img, 15)
    
    atm_light = atmospheric_light(img, dark_channel_img)
    transmission_estimate_img = transmission_estimate(img, atm_light, 15)

    io.imsave('img/dark-channel/dark_channel.png', dark_channel_img)
    io.imsave('img/dark-channel/transmission_estimate.png', transmission_estimate_img)
    