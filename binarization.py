import cv2
import numpy as np
from matplotlib import pyplot as plt

"""
Function takes in a grayscale image returns a binary image
Thresholding method performed is Otsu's
Param:
	img: Grayscale image passed in
Return:
	img_copy: image thats been thresholding and binarized
"""
def binary(img):
    # img = cv2.imread('can_pix.png',0)
    bin_number = 256
    # find normalized_histogram, and its probability density
    hist, bins = np.histogram(img.flatten(), 256, [0,256])

    total_hist = np.sum(hist, dtype=np.float32)
    pdf = (hist / total_hist).astype(np.float32)
    # print img.shape
    w_0 = w_1 = u_0 = u_1 = 0
    max_var = 0
    thresh = -1
    for t in range(1,256):
	   p1, p2 = pdf[t], pdf[256-t]
	   w_0 = np.sum(pdf[0:t-1])
	   if w_0 == 0.0:
		  w_0 = 1e-7
	   w_1 = np.sum(pdf[t:256-1])
	   if w_1 == 0.0:
		  w_1 = 1e-7
	   # print np.arange(t).shape
	   # print pdf[0:t].shape
	   u_0 = np.dot(np.arange(t), np.transpose(pdf[0:t]))/w_0
	   # print np.arange(t,256).shape
	   # print pdf[t:256].shape
	   u_1 = np.dot(np.arange(t,256), np.transpose(pdf[t:256]))/w_1
	   # print u_0, u_1
	   btwn_class_var = w_0 * w_1 * (u_0-u_1) * (u_0-u_1)
	   if btwn_class_var > max_var:
		  max_var = btwn_class_var
		  thresh = t
    # print max_var, thresh
    img_copy = np.copy(img)
    # cv2.imshow('copy', img_copy)
    for i in range(img_copy.shape[0]):
	   for j in range(img_copy.shape[1]):
		  # print img_copy[i,j], t
		  if img_copy[i,j] < thresh:
			 img_copy[i,j] = 0
		  else:
			 img_copy[i,j] = 1

    # cv2.imshow('threshed', img_copy)
    # #cv2.imshow('original', img)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    # plt.imshow(img_copy, 'gray')
    # plt.show()
    return img_copy

# img = cv2.imread('own_contacts.jpg', 0)
# plt.imshow(binary(img), 'gray')
# plt.show()
# img = cv2.imread('can_pix.png', 0)
# plt.imshow(binary(img), 'gray')
# plt.xticks([]), plt.yticks([])
# plt.show()