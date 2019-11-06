# ASSIGNEMENT 2 PART B SUBMITTED BY NITISH RAJ 2018MCS2140
import numpy as npy
import scipy as scp
import cv2
import matplotlib.pyplot as mtplt
from scipy import r_ as rfc
from scipy import fftpack as fft_scp

original_image = cv2.imread('TestSubject.jpg')
original_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2GRAY)

bitsize = 16
size_image = original_image.shape
dct = npy.zeros(size_image)
for i in rfc[:size_image[0]:bitsize]:
    for j in rfc[:size_image[1]:bitsize]:
        dct[i:(i + bitsize), j:(j + bitsize)] = fft_scp.dct(
            fft_scp.dct(original_image[i:(i + bitsize), j:(j + bitsize)], axis=0, norm='ortho'), axis=1, norm='ortho')

threshold = 0.031
dct_thresh = dct * (abs(dct) > (threshold * npy.max(dct)))
percent_nonzeros = npy.sum(dct_thresh != 0.0) / (size_image[0] * size_image[1] * 1.0)

dct_image = npy.zeros(size_image)
for i in rfc[:size_image[0]:bitsize]:
    for j in rfc[:size_image[1]:bitsize]:
        dct_image[i:(i + bitsize), j:(j + bitsize)] = fft_scp.idct(
            fft_scp.idct(dct_thresh[i:(i + bitsize), j:(j + bitsize)], axis=0, norm='ortho'), axis=1, norm='ortho')

dft = npy.zeros(size_image, dtype='complex')
dft_image = npy.zeros(size_image, dtype='complex')

for i in rfc[:size_image[0]:bitsize]:
    for j in rfc[:size_image[1]:bitsize]:
        dft[i:(i + bitsize), j:(j + bitsize)] = npy.fft.fft2(original_image[i:(i + bitsize), j:(j + bitsize)])

threshold = 0.031
dft_thresh = dft * (abs(dft) > (threshold * npy.max(abs(dft))))

for i in rfc[:size_image[0]:bitsize]:
    for j in rfc[:size_image[1]:bitsize]:
        dft_image[i:(i + bitsize), j:(j + bitsize)] = npy.fft.ifft2(dft_thresh[i:(i + bitsize), j:(j + bitsize)])

percent_nonzeros_dft = npy.sum(dft_thresh != 0.0) / (size_image[0] * size_image[1] * 1.0)
mse = npy.square(original_image - dct_image).mean(axis=None)
mse2 = npy.square(original_image - dft_image.astype('uint8')).mean(axis=None)
print(mse, mse2)
mtplt.figure("Assignment 2 PART B 2018MCS2140")
mtplt.imshow(
    npy.vstack((abs(original_image.astype('uint8')), abs(dct_image.astype('uint8')), abs(dft_image.astype('uint8')))),
    cmap='gray')
mtplt.show()