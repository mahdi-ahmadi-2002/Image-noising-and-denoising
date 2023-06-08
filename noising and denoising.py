import math

import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt


def mse(img1, img2):
    squared_diff = (img1 - img2) ** 2
    diff_sum = np.sum(squared_diff)
    pixel_num = img1.shape[0] * img1.shape[1]
    return diff_sum / pixel_num


def psnr(image_mse, l):
    return 10 * np.log10((l * l) / image_mse)


def best_sigma_mse(image, first_image):
    arr = [[], []]
    min_sigma = [0, np.inf]
    for i in range(1, 3000):
        t = 2 * math.floor(3 * (i / 1000)) + 1
        arr[0].append(mse(cv.GaussianBlur(image, (t, t), i / 1000), first_image))
        arr[1].append(i / 1000)
        if arr[0][i - 1] < min_sigma[1]:
            min_sigma[0] = arr[1][i - 1]
            min_sigma[1] = arr[0][i - 1]
    return arr, min_sigma


def best_sigma_psnr(image, first_image):
    arr = [[], []]
    min_sigma = [0, -np.inf]
    for i in range(1, 3000):
        t = 2 * math.floor(3 * (i / 1000)) + 1
        arr[0].append(psnr(mse(cv.GaussianBlur(image, (t, t), i / 1000), first_image), 255))
        arr[1].append(i / 1000)
        if arr[0][i - 1] > min_sigma[1]:
            min_sigma[0] = arr[1][i - 1]
            min_sigma[1] = arr[0][i - 1]
    return arr, min_sigma


def display_info(first_image, image, image_name, noise_sigma):
    arr, min_sigma = best_sigma_mse(image, first_image)
    image = cv.GaussianBlur(image, (5, 5), min_sigma[0])
    cv.imshow('edited ' + image_name + ' mse for noise sigma :' + noise_sigma + '', image)
    plt.plot(arr[1], arr[0])
    plt.title('MSE vs different sigmas on ' + image_name + ' for noise sigma : ' + noise_sigma)
    plt.xlabel("Sigma")
    plt.ylabel("MSE")
    plt.show()
    image_mse = mse(first_image, image)
    print('image : ', image_name, ', noise sigma : ', noise_sigma, ', MSE : ', image_mse, ', PSNR : ',
          psnr(image_mse, 255), ', sigma : ', min_sigma[0])


def sigma_vs_sigma_plot(image, iters):
    temp = [[], []]
    for i in range(iters):
        imgnew = image.copy()
        s = np.random.normal(0, i, size=(image.shape[0], image.shape[1]))
        imgnew = (imgnew + s).astype('uint8')
        arr, min_sigma = best_sigma_mse(imgnew, image)
        temp[0].append(min_sigma[0])
        temp[1].append(i)
        print(i)

    plt.plot(temp[1], temp[0])
    plt.title('relation of sigmas')
    plt.xlabel("noise Sigma")
    plt.ylabel("filter sigma")
    plt.show()


np.random.seed(42)

gorilla = cv.imread('Resources/Photos/baboon.bmp')
gorilla = cv.cvtColor(gorilla, cv.COLOR_BGR2GRAY)
first_goorilla = gorilla.copy()
cv.imshow('first gorilla', gorilla)
mu = 0
sigma = 25
s = np.random.normal(mu, sigma, size=(gorilla.shape[0], gorilla.shape[1]))
gorilla = (gorilla + s).astype('uint8')
cv.imshow('noised gorilla for noise sigma : 25', gorilla)

goorilla1 = gorilla.copy()
goorilla_arr, goorilla_min_sigma = best_sigma_psnr(goorilla1, first_goorilla)
goorilla1 = cv.GaussianBlur(goorilla1, (5, 5), goorilla_min_sigma[0])
cv.imshow('edited gorilla psnr for noise : 25', goorilla1)
plt.plot(goorilla_arr[1], goorilla_arr[0])
plt.title("PSNR vs different sigmas")
plt.xlabel("Sigma")
plt.ylabel("PSNR")
plt.show()
goorilla_mse = mse(first_goorilla, goorilla1)
print('MSE : ', goorilla_mse, ', PSNR : ', psnr(goorilla_mse, 255), 'sigma : ', goorilla_min_sigma[0])

display_info(first_goorilla, gorilla, 'gorilla mse', '25')

gorilla = cv.imread('Resources/Photos/baboon.bmp')
gorilla = cv.cvtColor(gorilla, cv.COLOR_BGR2GRAY)
first_goorilla = gorilla.copy()
cv.imshow('first gorilla', gorilla)
mu = 0
sigma = 50
s = np.random.normal(mu, sigma, size=(gorilla.shape[0], gorilla.shape[1]))
gorilla = (gorilla + s).astype('uint8')
cv.imshow('noised gorilla for sigma : 50', gorilla)

display_info(first_goorilla, gorilla, 'gorilla', '50')

gorilla = cv.imread('Resources/Photos/baboon.bmp')
gorilla = cv.cvtColor(gorilla, cv.COLOR_BGR2GRAY)
first_goorilla = gorilla.copy()
cv.imshow('first gorilla', gorilla)
mu = 0
sigma = 10
s = np.random.normal(mu, sigma, size=(gorilla.shape[0], gorilla.shape[1]))
gorilla = (gorilla + s).astype('uint8')
cv.imshow('noised gorilla for sigma : 10', gorilla)

display_info(first_goorilla, gorilla, 'gorilla', '10')
sigma_vs_sigma_plot(first_goorilla, 100)



lenna = cv.imread('Resources/Photos/lena.tif')
lenna = cv.cvtColor(lenna, cv.COLOR_BGR2GRAY)
first_lenna = lenna.copy()
cv.imshow('first lenna', lenna)
mu = 0
sigma = 10
s = np.random.normal(mu, sigma, size=(lenna.shape[0], lenna.shape[1]))
lenna = (lenna + s).astype('uint8')
cv.imshow('noised lenna for sigma : 10', lenna)

display_info(first_lenna, lenna, 'lenna', '10')

lenna = cv.imread('Resources/Photos/lena.tif')
lenna = cv.cvtColor(lenna, cv.COLOR_BGR2GRAY)
first_lenna = lenna.copy()
cv.imshow('first lenna', lenna)
mu = 0
sigma = 25
s = np.random.normal(mu, sigma, size=(lenna.shape[0], lenna.shape[1]))
lenna = (lenna + s).astype('uint8')
cv.imshow('noised lenna for sigma : 25', lenna)

display_info(first_lenna, lenna, 'lenna', '25')

lenna = cv.imread('Resources/Photos/lena.tif')
lenna = cv.cvtColor(lenna, cv.COLOR_BGR2GRAY)
first_lenna = lenna.copy()
cv.imshow('first lenna', lenna)
mu = 0
sigma = 50
s = np.random.normal(mu, sigma, size=(lenna.shape[0], lenna.shape[1]))
lenna = (lenna + s).astype('uint8')
cv.imshow('noised lenna for sigma : 50', lenna)

display_info(first_lenna, lenna, 'lenna', '50')
sigma_vs_sigma_plot(first_lenna, 100)





cameraman = cv.imread('Resources/Photos/caman.tif')
cameraman = cv.cvtColor(cameraman, cv.COLOR_BGR2GRAY)
first_cameraman = cameraman.copy()
cv.imshow('first cameraman', cameraman)
mu = 0
sigma = 10
s = np.random.normal(mu, sigma, size=(cameraman.shape[0], cameraman.shape[1]))
cameraman = (cameraman + s).astype('uint8')
cv.imshow('noised cameraman for sigma : 10', cameraman)

display_info(first_cameraman, cameraman, 'cameraman', '10')

cameraman = cv.imread('Resources/Photos/caman.tif')
cameraman = cv.cvtColor(cameraman, cv.COLOR_BGR2GRAY)
first_cameraman = cameraman.copy()
cv.imshow('first cameraman', cameraman)
mu = 0
sigma = 25
s = np.random.normal(mu, sigma, size=(cameraman.shape[0], cameraman.shape[1]))
cameraman = (cameraman + s).astype('uint8')
cv.imshow('noised cameraman for sigma : 25', cameraman)

display_info(first_cameraman, cameraman, 'cameraman', '25')

cameraman = cv.imread('Resources/Photos/caman.tif')
cameraman = cv.cvtColor(cameraman, cv.COLOR_BGR2GRAY)
first_cameraman = cameraman.copy()
cv.imshow('first cameraman', cameraman)
mu = 0
sigma = 50
s = np.random.normal(mu, sigma, size=(cameraman.shape[0], cameraman.shape[1]))
cameraman = (cameraman + s).astype('uint8')
cv.imshow('noised cameraman for sigma : 50', cameraman)

display_info(first_cameraman, cameraman, 'cameraman', '50')
sigma_vs_sigma_plot(first_cameraman, 100)

cv.waitKey(0)
