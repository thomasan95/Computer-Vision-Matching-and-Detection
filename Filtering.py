import cv2
import numpy as np
from matplotlib import pyplot as plt
from matplotlib import patches as patches
from scipy import ndimage
from scipy import misc
import collections

Rect_P = collections.namedtuple('Rect_P', ['x_p', 'y_p', 'height_p', 'width_p'])

"""
Initializing ground boxes and Rectangle coordinates
"""
x_g_1, y_g_1 = 175, 145
x_g_2, y_g_2 = 69, 205
x_g_3, y_g_3 = 329, 251
width_g_1, height_g_1 = 522-175, 245-145
width_g_2, height_g_2 = 488-69, 357-205
width_g_3, height_g_3 = 480-329, 345-251

car1_ground = patches.Rectangle((175,145),522-175,245-145, linewidth=2, edgecolor='g', 
	facecolor='none')
car2_ground = patches.Rectangle((69,205),488-69,357-205, linewidth=2, edgecolor='g', 
	facecolor='none')
car3_ground = patches.Rectangle((329,251),480-329,345-251, linewidth=2, edgecolor='g', 
	facecolor='none')



"""
Defining function to draw intersection area:
Returns the rectangle for intersection and if theres
no intersection, returns nothing
"""
def intersectionRect(x_p, y_p, width_p, height_p, x_g, y_g, width_g, height_g):
	x_p_max = x_p + width_p
	x_g_max = x_g + width_g
	y_p_max = y_p + height_p
	y_g_max = y_g + height_g
	dx = min(x_p_max, x_g_max) - max(x_p, x_g)
	dy = min(y_p_max, y_g_max) - max(y_p, y_g)
	if (dx >= 0) and (dy >= 0): 
		x = max(x_p, x_g)
		y = max(y_p, y_g)
		return patches.Rectangle((x, y), dx, dy, 
	 		linewidth=2, edgecolor='purple', facecolor='none')
	return patches.Rectangle((0, 0), 0, 0, 
	 		linewidth=2, edgecolor='purple', facecolor='none')

"""
Function to detect template inside image
"""
def detection(template, synthetic):

	# car 3
	# rect_h = template.shape[0]*0.1
	# rect_w = template.shape[1]*0.1
	# # car 1
	# rect_h = template.shape[0]*0.1
	# rect_w = template.shape[1]*0.15
	# # car 2
	# rect_h = template.shape[0]*0.1
	# rect_w = template.shape[1]*0.1

	rect_h = template.shape[0]
	rect_w = template.shape[1]

	# template = misc.imresize(template, 15)
	template_t = template - np.mean(template)
	synthetic_t = synthetic - np.mean(synthetic)

	# template_t = np.rot90(template_t,3)
	conv = ndimage.convolve(synthetic_t, template_t, mode='constant', cval=0.0)
	# plt.imshow(conv, 'jet')
	# plt.show()

	fig,ax = plt.subplots(1)
	ax.imshow(synthetic, 'gray')

	# changed to 1 for car, 3 for mickeys
	for z in range(0,3):

	 	curr_y,curr_x = np.unravel_index(conv.argmax(), conv.shape)
	 	conv[curr_y, curr_x] = 0

	 	rect = patches.Rectangle( (curr_x-rect_w/2, curr_y-(rect_h/2)), rect_w, rect_h, 
	 		linewidth=2, edgecolor='b', facecolor='none')

	 	# inter_rect = intersectionRect(curr_x-rect_w/2, (curr_y-rect_h/2), rect_w, 
	 	# 	rect_h, x_g_1, y_g_1, width_g_1, height_g_1)
	 	# inter_rect = intersectionRect(curr_x-rect_w/2, curr_y-rect_h/2, rect_w, 
	 	# 	rect_h, x_g_2, y_g_2, width_g_2, height_g_2)
	 	# inter_rect = intersectionRect(curr_x-rect_w/2, curr_y-rect_h/2, rect_w, 
	 	# 	rect_h, x_g_3, y_g_3, width_g_3, height_g_3)

	 	if z == 0:
	 		p = Rect_P(x_p=curr_x-rect_w/2 ,y_p=curr_y-rect_h/2, width_p=rect_w, 
	 			height_p=rect_h)
	 	ax.add_patch(rect)

	# ax.add_patch(car1_ground)
	# ax.add_patch(car2_ground)
	# ax.add_patch(car3_ground)
	# ax.add_patch(inter_rect)
		rect1 = patches.Rectangle( ((curr_x-rect_w/2 + 5 ), (curr_y-rect_h/2 + 5)), rect_w-5, rect_h-5, 
	 			linewidth=2, edgecolor='red', facecolor='none')
		ax.add_patch(rect1)
	plt.show()
	return p

"""
Function to determine the quality of intersection area
"""
def detection_quality(x_p, y_p, width_p, height_p, x_g, y_g, width_g, height_g):

	x_p_max = x_p + width_p
	x_g_max = x_g + width_g
	y_p_max = y_p + height_p
	y_g_max = y_g + height_g
	intersectionArea = 0
	rect_p = np.array([[x_p, x_p + width_p], [y_p, y_p + height_p]])
	rect_g = np.array([[x_g, x_g + width_g], [y_g, y_g + height_g]])
	dx = min(x_p_max, x_g_max) - max(x_p, y_g)
	dy = min(y_p_max, y_g_max) - max(y_p, y_g)
	if (dx >= 0) and (dy >= 0):
		intersectionArea = dx*dy
	unionArea = (width_g*height_g) + (width_p*height_p) - intersectionArea
	detectionArea = intersectionArea/float(unionArea)

	return detectionArea

"""
Main
"""
def main():
	template = cv2.imread('filter.jpg', 0)
	synthetic = cv2.imread('toy.png', 0)
	x_p, y_p, width_p, height_p = detection(template, synthetic)

	# template = cv2.imread('cartemplate.jpg', 0)
	# synthetic = cv2.imread('car1.jpg', 0)
	# synthetic = cv2.imread('car2.jpg', 0)
	# synthetic = cv2.imread('car3.jpg', 0)
	# detection(template, synthetic)

	# x_p, y_p, width_p, height_p = detection(template, synthetic)
	# quality_car_1 = detection_quality(x_p, y_p, width_p, height_p, 
	# 	x_g_1, y_g_1, width_g_1, height_g_1)
	# print quality_car_1

	# x_p, y_p, width_p, height_p = detection(template, synthetic)
	# quality_car_2 = detection_quality(x_p, y_p, width_p, height_p, 
	# 	x_g_2, y_g_2, width_g_2, height_g_2)
	# print quality_car_2

	x_p, y_p, width_p, height_p = detection(template, synthetic)
	quality_car_3 = detection_quality(x_p, y_p, width_p, height_p, 
		x_g_3, y_g_3, width_g_3, height_g_3)
	print quality_car_3

	# print x_p, y_p, width_p, height_p
	# x_g1, y_g1, width_g1, height_g1 = x_p*0.8, y_p*0.8, 127*0.8, 127*0.8
	# quality_1 = detection_quality(x_p, y_p, width_p, height_p, x_g1, y_g1, width_g1, height_g1)
	# x_g2, y_g2, width_g2, height_g2 = x_p*0.9, y_p*0.9, 127*0.9, 127*0.9
	# quality_2 = detection_quality(x_p, y_p, width_p, height_p, x_g2, y_g2, width_g2, height_g2)
	# x_g3, y_g3, width_g3, height_g3 = x_p*0.8, y_p*0.95, 127*0.95, 127*0.95
	# quality_3 = detection_quality(x_p, y_p, width_p, height_p, x_g3, y_g3, width_g3, height_g3)
	# print quality_1, quality_2, quality_3


if __name__ == "__main__":
	main()