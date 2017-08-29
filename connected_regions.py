import cv2
import sys
import numpy as np
from binarization import binary
from matplotlib import pyplot as plt
from scipy import misc


# Redefine recursion limit and allocate larger stack size
iMaxStackSize = 50000
sys.setrecursionlimit(iMaxStackSize)

marker = 0
"""
Recursive function for labeling all connected components
Param:
	bin_img (array) : Binary image to be passed in and analyzed
	x (int): x coordinate of current pixel
	y (int): y coordinate of current pixel
"""
def label(bin_img, x, y):
	global mark
	# global marker
	# print marker
	mark[x,y] = marker
	Neighbors = bin_img[x-1:x+1, y-1:y+1]
	for i in range(x-1,x+2):
		for j in range(y-1,y+2):
			if bin_img[i,j] == 1 and mark[i,j] == 0:
				label(bin_img,i,j)
"""
Main function to read in a file and then call labels
"""
def connect_call(img):
	# Pass in the binary_img into a global variable for access
	binary_img = binary(img)
	
	# plt.imshow(binary_img, 'gray')
	# plt.show()

	global marker

	# instead of initializing empty, treat '0' value in 
	# mark matrix as None
	global mark
	mark = np.zeros(binary_img.shape)

	rows = binary_img.shape[0]
	cols = binary_img.shape[1]
	for x in range(1,rows-1):
		for y in range(1,cols-1):
			if binary_img[x,y] == 1 and mark[x,y] == 0:
				marker = marker + 1
				label(binary_img,x,y)
	print marker

	return mark

def main():
	# img = cv2.imread('can_pix.png', 0)
	# connect_call(img)
	# contacts = cv2.imread('own_contacts.jpg', 0)
	# connect_call(contacts)
	# tape = cv2.imread('own_tape.jpg', 0)
	# connect_call(tape)
	# pen = cv2.imread('own_pen.jpg', 0)
	# connect_call(pen)
	# card = cv2.imread('card.jpg', 0)
	# card = misc.imresize(card, 15, interp='bilinear')
	# connect_call(card)
	# charge = cv2.imread('own_charger.JPG', 0)
	# connect_call(charge)
	heart = cv2.imread('heart.jpg', 0)
	connect_call(heart)
	plt.imshow(mark)
	plt.show()

if __name__ == "__main__":
	main()
