import cv2
from matplotlib import pyplot as plt
import numpy as np
from binarization import binary
from connected_regions import connect_call, label
from scipy import misc
import math

# should input for this be binary image or marked_matrix???
def moments(marked_mat,j,k,d):
	total_moment = 0
	rows = marked_mat.shape[0]
	cols = marked_mat.shape[1]
	for x in range(rows):
		for y in range(cols):
			if marked_mat[x,y] == d:
				# write code to calculate moment
				total_moment = total_moment + np.power(x,k)*np.power(y,j)
	return total_moment

def central_moments(marked_mat,j,k,d):
	# Output should be central moment u_jk
	counter = 0
	central_moment = 0
	rows = marked_mat.shape[0]
	cols = marked_mat.shape[1]

	x_bar = moments(marked_mat,1,0,d)/float(moments(marked_mat,0,0,d))
	y_bar = moments(marked_mat,0,1,d)/float(moments(marked_mat,0,0,d))

	for x in range(rows):
		for y in range(cols):
			if marked_mat[x,y] == d:
				# write code to calculate moment
				central_moment = central_moment + np.power((x-x_bar),k)*np.power((y-y_bar),j)
	return central_moment

def normalized_moments(marked_mat,j,k,d):
	# Output should be normalized moment m_jk
	mu_20 = central_moments(marked_mat,2,0,d)
	mu_02 = central_moments(marked_mat,0,2,d)
	M_00 = moments(marked_mat,0,0,d)

	o_x = np.sqrt(mu_20/float(M_00))
	o_y = np.sqrt(mu_02/float(M_00))

	x_bar = moments(marked_mat,1,0,d)/float(moments(0,0,d))
	y_bar = moments(marked_mat,0,1,d)/float(moments(0,0,d))

	rows = marked_mat.shape[0]
	cols = marked_mat.shape[1]

	for y in range(rows):
		for x in range(cols):
			if marked_mat[x,y] == d:
				norm_moment = norm_moment + np.power(((x-x_bar)/o_x),k)*np.power(((y-y_bar)/o_y),j)

	return norm_moment
def centroid(marked_mat,d):
	 moment0 = moments(marked_mat,0,0,d)
	 xbar = moments(marked_mat,1,0,d)/moment0
	 ybar = moments(marked_mat,0,1,d)/moment0

	 cent = [xbar,ybar]
	 print "centroid is: ", cent
	 return cent

def main():
	# img = cv2.imread('card.jpg', 0)
	# img = misc.imresize(img, 15, interp='bilinear')
	# binary_matrix = binary(img)
	# mat = connect_call(img)

	img = cv2.imread('heart.jpg', 0)
	# img = misc.imresize(img, 15, interp='bilinear')
	binary_matrix = binary(img)
	mat = connect_call(img)

	o_20 = central_moments(mat,2,0,1)
	o_11 = central_moments(mat,1,1,1)
	o_02 = central_moments(mat,0,2,1)

	mom_mat =  np.array([[o_20,o_11],[o_11,o_02]])
	w, v = np.linalg.eig(mom_mat)
	print "Eigen Values: " , w
	e1 = v[:,0]
	e2 = v[:,1]

	c = centroid(mat,1)
	y = c[1]
	x = c[0]

	e1_x, e1_y = e1[1], e1[0]
	e2_x, e2_y = e2[1], e2[0]

	# print "Central M. 10 is: ", central_moments(mat,1,0,1)
	# print "Central M. 01 is: ", central_moments(mat,0,1,1)
	# print "Central M. 20 is: ", central_moments(mat,2,0,1)
	# print "Central M. 02 is: ", central_moments(mat,0,2,1)
	# print "Central M. 11 is: ", central_moments(mat,1,1,1)

	# print "Moment 00 is: ", moments(mat,0,0,1)
	# print "Moment 10 is: ", moments(mat,1,0,1)
	# print "Moment 01 is: ", moments(mat,0,1,1)

	# print "Eigen 1 is: ", e1
	# print "Eigen 2 is: ", e2
	# print "Centroid is: ", c

	y_axis = np.array([1,0])
	cos_theta = np.dot(e2, y_axis)/float(np.sqrt(np.dot(e2,e2)) * 
		np.sqrt(np.dot(y_axis,y_axis)))
	theta = np.arccos(cos_theta)
	rot_m = np.matrix([[np.cos(theta), -np.sin(theta)],
		[np.sin(theta), np.cos(theta)]])

	rows, cols = mat.shape

	new_img = np.zeros(mat.shape)

	for i in range(mat.shape[0]):
		for j in range(mat.shape[1]):
			new_img = rot_m*np.array([[i-x],[j-y]])
			new_x = int(new_img[0]+x)
			new_y = int(new_img[1]+y)
			# print new_x, new_y
			if (new_x > 0) and (new_x < rows) and (new_y > 0) and (new_y < cols):
				value = binary_matrix[i,j]
				new_img[new_x, new_y] = mat[i,j]


	fig,ax1 = plt.subplots(1)
	ax1.imshow(new_img, 'gray')
	# plt.axes().arrow(x,y,e1_x,e1_y,head_width=0.05,head_length=0.1,color ='b')
	# ax1.plot(x, y, e1_x, e1_y, color='r')
	# ax1.plot(x, y, e2_x, e2_y, color='b')
	# ax1.plot(x, y, color='g')
	circle1 = plt.Circle((x,y), 1, color='g')
	ax1.add_artist(circle1)
	ax1.set_xlim([0,240])
	ax1.set_ylim([0,320])
	plt.quiver(x, y, e1_x, e1_y, color='r')
	plt.quiver(x, y, e2_x, e2_y, color='b')
	plt.show()

	# print marked_matrix.shape


if __name__ == "__main__":
	main()