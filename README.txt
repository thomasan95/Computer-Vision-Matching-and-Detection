binarization.py
	binary()
	This file acts as the helper function for the rest of the codes.
	The file takes in an image file as a parameter and performs Otsu’s
	Thesholding and returns a Binary Image:

connected_regions.py
	label()
	This function recursively labels all neighbors. If it detects a 1 neighbor, it
	then labels that at the current marker and then recursively calls the neighbor
	
	connect_call()
	It takes in the grayscale image as a parameter, then this function iterates
	through the binary image and goes through the image looking for 1’s in the binary.
	Once a 1 is found, it then calls label for the recursive labeling

moment_calc.py
	moments()
	Takes in the marked matrix from connect_call(), j, k, d for the specific mark
	and moment.
	It calculates and returns the moment

	central_moments(marked_mat,j,k,d)
	Function takes in the marked matrix, j,k, and d and returns the central moment
	from the given parameters
	
	normalized_moments(marked_mat,j,k,d)
	Function computed the normalized moments for parameters j,k,d

	centroid(marked_mat, d)
	calculates the centroid from the given mark 

	main()
	Executes all the images necessary

Filtering.py
	intersectionRect(x_p, y_p, width_p, height_p, x_g, y_g, width_g, height_g)
	This function takes in the x_p and y_p coordinates and x_g y_g along with the 
	respective widths and heights for each rectangle. It then computes and returns the 
	rectangle of intersection.

	detection(template, synthetic)
	This function performs the convolution and finds the regions of highest intensity
	and creates a blue box around that region. It returns the x_p, y_p, width_p, and heigh_p
	to be used later for intersection calculations

	detection_quality(x_p, y_p, width_p, height_p, x_g, y_g, width_g, height_g)
	This function is a simple mathematical function to calculate regions of intersection
	between two rectangles given

	main()
	Main execution script. Used to execute cars(1-3) and toy
	
	