import numpy as np
import math
import skimage, skimage.io as io, skimage.morphology as morph, skimage.color as color, skimage.filters.rank as rank
import cv2

# value of 0 = masked
# value of 1 = unmasked

def dark_channel(img, size):
    dark_channel_img = np.amin(img, axis=2)
    footprint = np.ones((size, size))
    dark_channel_img = morph.erosion(dark_channel_img, footprint)
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

def refine_transmission(img, transmission_estimate_img):
    r = 60      # radius of kernel
    e = 0.0001  # regularization

    gray_scale_img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY);
    gray_scale_img = np.float64(gray_scale_img)/255;

    transmission_estimate_img = transmission_estimate_img / 255

    mean_I = cv2.boxFilter(gray_scale_img, cv2.CV_64F, (r,r))
    mean_p = cv2.boxFilter(transmission_estimate_img, cv2.CV_64F, (r,r))
    corr_I = cv2.boxFilter(gray_scale_img * gray_scale_img,cv2.CV_64F, (r,r))
    corr_Ip = cv2.boxFilter(gray_scale_img * transmission_estimate_img, cv2.CV_64F, (r,r))

    var_I   = corr_I - (mean_I * mean_I)
    cov_Ip = corr_Ip - mean_I * mean_p

    a = cov_Ip / (var_I + e)
    b = mean_p - (a * mean_I)

    mean_a = cv2.boxFilter(a, cv2.CV_64F, (r,r))
    mean_b = cv2.boxFilter(b, cv2.CV_64F, (r,r))

    q = (mean_a * gray_scale_img) + mean_b
    return (q * 255).astype(np.uint8)


    gray_scale_img = (255 * color.rgb2gray(img)).astype(np.uint8)
    temp_gsi = color.rgb2gray(img)
    footprint = morph.disk(r)

    mean_I = rank.mean(gray_scale_img, footprint=footprint) / 255
    mean_p = rank.mean(gray_scale_img, footprint=footprint) / 255
    corr_I = rank.mean((gray_scale_img * (gray_scale_img / 255)).astype(np.uint8), footprint=footprint) / 255
    corr_Ip = rank.mean((gray_scale_img * (transmission_estimate_img / 255)).astype(np.uint8), footprint=footprint) / 255

    var_I = corr_I - (mean_I * mean_I)
    cov_Ip = corr_Ip - (mean_I * mean_p)

    a = (255 * (cov_Ip / (var_I + e))).astype(np.uint8)
    a_temp = cov_Ip / (var_I + e)
    b = (255 * ((mean_p - (a_temp * mean_I)))).astype(np.uint8)

    mean_a = rank.mean(a, footprint=footprint) / 255
    mean_b = rank.mean(b, footprint=footprint) / 255

    # q = (mean_a * gray_scale_img / 1000000) + mean_b / 1000
    q = (mean_a * temp_gsi) + mean_b

    return (q * 255).astype(np.uint8)


if __name__ == '__main__':
    file_name = "extremely_foggy_highway"
    file_path = "~/proj-x/fog-mask/img/input/{}.png".format(file_name)

    img = io.imread(file_path)
    dark_channel_img = dark_channel(img, 15)
    
    atm_light = atmospheric_light(img, dark_channel_img)
    transmission_estimate_img = transmission_estimate(img, atm_light, 15)

    refined_transmission_img = refine_transmission(img, transmission_estimate_img)

    io.imsave('img/dark-channel/dark_channel_{}.png'.format(file_name), dark_channel_img)
    io.imsave('img/dark-channel/transmission_estimate_{}.png'.format(file_name), transmission_estimate_img)
    io.imsave('img/dark-channel/refined_transmission_{}.png'.format(file_name), refined_transmission_img)
    