import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import random
import math
import os
import argparse
import numpy as np
from scipy.spatial import ConvexHull
import scipy.stats as st
from matplotlib.patches import Circle

MAX_X = 200
MAX_Y = 200
MIN_RADIUS = 1
MAX_RADIUS = 95
MIN_NUM = 1
MAX_NUM = 4
NUM_OF_STIMULI = 10
FEW_MANY_THRESHOLD_IN_PERCENTAGES = 0.4
EPSILON = 5

current_area = 0
PI = 3.14
TOTAL_AREA = MAX_X * MAX_Y


class Circle:
	def __init__(self, center_x, center_y, radius):
		self.center_x = center_x
		self.center_y = center_y
		self.radius = radius

	def get_area(self):
		return PI * self.radius ** 2


def create_circle(area, avg_area, iter):
	if area == 0:
		return Circle(0, 0, 0)
	try:
		avg_radius = math.sqrt(avg_area/PI)
		rad = avg_radius * random.uniform(0.1, 1)
		randX = random.uniform(rad, MAX_X - rad)
		randY = random.uniform(rad, MAX_Y - rad)
		circle = Circle(randX, randY, rad)
	except Exception as e:
		#print('cannot create circle: %s' % e)
		return None
	return circle


def is_colliding(c1, c2):
	if math.hypot(c1.center_x - c2.center_x, c1.center_y - c2.center_y) <= c1.radius + c2.radius:
		return True
	return False


def is_valid_circle(c):
	if ((c.center_y + c.radius >= MAX_Y-EPSILON)
		or (c.center_x + c.radius >= MAX_X-EPSILON)
		or (c.center_y - c.radius <= 0)
		or (c.center_x - c.radius <= 0)):
		return False
	return True


def collide(circle, circles):
	for c in circles:
		if dist2d((c.center_x, c.center_y), (circle.center_x, circle.center_y)) < c.radius + circle.radius:
			return True
	return False


def dist2d(p, q):
	distance = np.sqrt((p[0] - q[0]) ** 2 + (p[1] - q[1]) ** 2)
	return distance


def draw_circles(index, numeric_val, request_classification, dest_dir):
	if request_classification is 'many':
		requested_area = random.randint(TOTAL_AREA * FEW_MANY_THRESHOLD_IN_PERCENTAGES + 1, TOTAL_AREA)
		num_circles = numeric_val
	else:
		requested_area = random.randint(0, TOTAL_AREA * FEW_MANY_THRESHOLD_IN_PERCENTAGES)
		num_circles = numeric_val

	curr_area = requested_area
	curr_num_circles = num_circles
	# print('requested_area: %s pixels, num_circles: %s,
	# in percentages: %s' % (requested_area, num_circles, requested_area/TOTAL_AREA))
	circles = []
	if num_circles == 0:
		avg_area_of_single_circle = 0
	else:
		avg_area_of_single_circle = requested_area/num_circles
	# dynamic programing
	while curr_area > 0 and curr_num_circles > 0:
		# print('current area: %s, current num of circles: %s' %(curr_area, curr_num_circles))
		circle = create_circle(curr_area, avg_area_of_single_circle, 0)
		if circle is None:
			# start over
			return False
		if len(circles) == 0:
			circles.append(circle)
			curr_area -= circle.get_area()
			curr_num_circles -= 1
		else:
			new_circles = []
			iter = 0
			while collide(circle, circles):
				circle = create_circle(curr_area, avg_area_of_single_circle, ++iter)

			new_circles.append(circle)
			curr_area -= circle.get_area()
			curr_num_circles -= 1
			circles += new_circles

	actual_area = 0
	for cir in circles:
		actual_area += cir.get_area()

	area_in_percentages = actual_area/TOTAL_AREA
	number_of_circles = len(circles)
	num_of_verteces, convex_hull_perimeter, convex_hull_area = calculate_convex_hull(circles)
	max_density = calculate_density(circles)
	# print('current area: %s, current num of circles: %s, in percentages: %s'
	#  % (actual_area, number_of_circles, area_in_percentages))

	fig, ax = plt.subplots()

	for c in circles:
		circle_obj = plt.Circle((c.center_x, c.center_y), c.radius, color='k')
		# print('circle: x: %s, y: %s, radius: %s, area: %s' % (c.center_x, c.center_y, c.radius, c.get_area()))
		ax.add_artist(circle_obj)

	plt.xlim([0, MAX_X])
	plt.ylim([0, MAX_Y])

	plt.axis('off')

	if area_in_percentages > FEW_MANY_THRESHOLD_IN_PERCENTAGES:
		classification = "many"
	else:
		classification = "few"

	# validation of the requested classifications:
	if classification != request_classification:
		print('starting over because classification is wrong: image {} of numeric val {}, request class: {}, observed class: {}'.format(index, numeric_val, request_classification, classification))
		return False # start over

	# validate that the requested numer of circles is indeed found in the image
	if numeric_val != number_of_circles:
		print('starting over because num of circles is wrong: image {} requested circles:  {}, observed circles: {}'.format(index, numeric_val, number_of_circles))
		return False  # start over

	area_in_percentages_str = "%.2f" % area_in_percentages
	convex_hull_in_percentages_str = "%.2f" % (convex_hull_area/TOTAL_AREA)
	convex_hull_perimeter_str = "%.2f" % convex_hull_perimeter
	density_str ="%.6f" % max_density
	file_name = "index_" + str(index) + "_" + classification +\
				"_circles_" + str(number_of_circles) +\
				"_ch_" + str(num_of_verteces) + \
				"_chp_" + str(convex_hull_perimeter_str) + \
 				"_cha_" + convex_hull_in_percentages_str + \
				"_density_" + density_str + \
				"_area_" + area_in_percentages_str + ".png"

	fig.savefig(dest_dir + os.sep + file_name)
	plt.close()
	print('saving image: {} of numeric val: {}, class: {}'.format(index, numeric_val, request_classification))
	return True


def is_bigger_than_all(c, circles):
	if len(circles) <= 1:
		return False
	is_bigger = 0
	for cir in circles:
		# not comparing a circle to itself
		if c.get_area() != cir.get_area():
			if c.get_area() / cir.get_area() >= 2.5:
				is_bigger += 1
	if is_bigger == (len(circles)-1):
		return True
	return False


def calculate_density(circles):
	if len(circles) < 2:
		return 0
	x = []
	y = []
	for c in circles:
		x.append(c.center_x)
		y.append(c.center_y)

	# Define the borders
	deltaX = (max(x) - min(x)) / MIN_RADIUS
	deltaY = (max(y) - min(y)) / MIN_RADIUS
	xmin = min(x) - deltaX
	xmax = max(x) + deltaX
	ymin = min(y) - deltaY
	ymax = max(y) + deltaY

	# Create meshgrid
	xx, yy = np.mgrid[xmin:xmax:100j, ymin:ymax:100j]

	positions = np.vstack([xx.ravel(), yy.ravel()])
	values = np.vstack([x, y])
	kernel = st.gaussian_kde(values)
	f = np.reshape(kernel(positions).T, xx.shape)
	max_f = np.max(f)
	print(max_f)
	return max_f


def calculate_convex_hull(circles):
	if len(circles) < 3:
		if len(circles) == 0:
			return 0, 0, 0
		if len(circles) == 1:
			return 1, 0, 0
		if len(circles) == 2:
			convex_hull_perimeter = dist2d((circles[0].center_x, circles[0].center_y), (circles[1].center_x, circles[1].center_y))
			return 2, convex_hull_perimeter, 0
	points = []
	for circle in circles:
		points.append([circle.center_x, circle.center_y])
	points = np.array(points)
	#plt.plot(points[:, 0], points[:, 1], 'o')
	hull = ConvexHull(points)

	convex_hull_perimeter = hull.area
	convex_hull_area = hull.volume
	num_of_verteces = len(hull.simplices)

	return num_of_verteces, convex_hull_perimeter, convex_hull_area
	#len(hull.simplices)))
	#for simplex in hull.simplices:
	#	plt.plot(points[simplex, 0], points[simplex, 1], 'k-')
	#plt.show()


def delete_dir_content(folder):
	for the_file in os.listdir(folder):
		file_path = os.path.join(folder, the_file)
		try:
			os.unlink(file_path)
		except Exception as e:
			print(e)


def create_stimuli(dest_dir, should_delete_prev_content):
	if not os.path.isdir(dest_dir):
		os.mkdir(dest_dir)
	if should_delete_prev_content:
		delete_dir_content(dest_dir)
	general_many_index = 0
	general_few_index = 0
	for numeric_val in range(MIN_NUM, MAX_NUM + 1):
		many_index = 0
		few_index = 0
		# if numerical val is zero only few classification is relevant
		if numeric_val == 0:
			while few_index < NUM_OF_STIMULI:
				is_success = False
				while not is_success:
					is_success, classification, actual_area, area_in_percentages, index, circles = nocollide_random_unif_circles(numeric_val, general_many_index, general_few_index, dest_dir)
					if is_success and few_index <= NUM_OF_STIMULI:
						draw_and_save_circles_to_images(circles, classification, actual_area, area_in_percentages, index, dest_dir)
						few_index += 1
						general_few_index += 1
		else:
			while many_index < NUM_OF_STIMULI or few_index < NUM_OF_STIMULI:
				is_success = False
				while not is_success:
					is_success, classification, actual_area, area_in_percentages, index, circles = nocollide_random_unif_circles(
						numeric_val, general_many_index, general_few_index, dest_dir)
					if is_success and classification == 'many':
						if many_index <= NUM_OF_STIMULI:
							draw_and_save_circles_to_images(circles, classification, actual_area, area_in_percentages,
															index, dest_dir)
							many_index += 1
							general_many_index += 1
					else:
						if is_success and few_index <= NUM_OF_STIMULI:
							draw_and_save_circles_to_images(circles, classification, actual_area, area_in_percentages,
															index, dest_dir)
							few_index += 1
							general_few_index += 1
		print('few index {}'.format(few_index))
		print('many index {}'.format(many_index))
		print('done creating {} stimuli for number: {}'.format(NUM_OF_STIMULI, numeric_val))


def add_circle(ax, center=(0, 0), r=1, face_color=(0, 0, 1, 1), edge_color=(0, 0, 1, 1), line_width=None):
	# circle_obj = plt.Circle(center, radius=r, color='k')
	# # print('circle: x: %s, y: %s, radius: %s, area: %s' % (c.center_x, c.center_y, c.radius, c.get_area()))
	# ax.add_artist(circle_obj)

	C = matplotlib.patches.Circle(center, radius=r, facecolor=face_color, edgecolor=edge_color, linewidth=line_width)
	ax.add_patch(C)


def dist2d(p, q):
	distance = np.sqrt((p[0] - q[0]) ** 2 + (p[1] - q[1]) ** 2)
	return distance


def collide(center, rad, centers, rads):
	for c, r in zip(centers, rads):
		if dist2d(c, center) < r + rad:
			return True
	return False


def nocollide_random_unif_circles(num_of_circles, many_index, few_index, dest_dir):
	ncircle = num_of_circles
	min_r = MIN_RADIUS
	max_r = MAX_RADIUS

	circles = []
	rads = [random.uniform(min_r, max_r)]
	centers = [(random.uniform(rads[0], MAX_X-rads[0]-EPSILON), random.uniform(rads[0], MAX_Y-rads[0]-EPSILON))]
	if centers[0][0]+rads[0] > MAX_X or centers[0][1]+rads[0] > MAX_Y:
		print("center x: {}, center y: {}, rad: {} ".format(centers[0][0], centers[0][1], rads[0]))
	# here we create the first circle
	if num_of_circles > 0:
		circles.append(Circle(centers[0][0], centers[0][1], rads[0]))
	for n in range(ncircle-1):
		rad = random.uniform(min_r, max_r)
		center = (random.uniform(rad, MAX_X-rad-EPSILON), random.uniform(rad, MAX_Y-rad-EPSILON))
		if center[0] + rads[0] > MAX_X or center[1] + rads[0] > MAX_Y:
			print("center x: {}, center y: {}, rad: {} ".format(center[0], center[1], rads[0]))
		while collide(center, rad, centers, rads):
			center = (random.uniform(rad, MAX_X-rad-EPSILON), random.uniform(rad, MAX_Y-rad-EPSILON))
			rad = random.uniform(min_r, max_r)
			if center[0] + rads[0] > MAX_X or center[1] + rads[0] > MAX_Y:
				print("center x: {}, center y: {}, rad: {} ".format(center[0], center[1], rads[0]))
		centers.append(center)
		rads.append(rad)
		circles.append(Circle(center[0], center[1], rad))

	if ncircle != len(circles):
		print('Error in number of requested circles. requested: {}, observed: {}'.format(ncircle, len(circles)))
		return False, None

	actual_area = 0
	for cir in circles:
		actual_area += cir.get_area()

	area_in_percentages = actual_area / TOTAL_AREA
	if area_in_percentages >= FEW_MANY_THRESHOLD_IN_PERCENTAGES:
		classification = 'many'
		index = many_index
	else:
		classification = 'few'
		index = few_index
	return True, classification, actual_area, area_in_percentages, index, circles


def draw_and_save_circles_to_images(circles,  classification, actual_area, area_in_percentages, index, dest_dir):
	color = 'black'
	sizex = [0, MAX_X]
	sizey = [0, MAX_Y]
	fig, ax = plt.subplots(1, 1)
	for cir in circles:
		add_circle(ax, center=(cir.center_x, cir.center_y), r=cir.radius, face_color=color, edge_color=(0, 0, 1, 1))

	ax.axis('equal')
	ax.set_xlim(sizex);
	ax.set_ylim(sizey)
	plt.axis('off')
	fig.tight_layout()

	# analyze image:
	num_of_verteces, convex_hull_perimeter, convex_hull_area = calculate_convex_hull(circles)
	if convex_hull_area == 0:
		density = 0
	else:
		density = actual_area / convex_hull_area

	if TOTAL_AREA > 0:
		convex_hull_in_percentages = convex_hull_area / TOTAL_AREA
	else:
		convex_hull_in_percentages = 0
	area_in_percentages_str = "%.2f" % area_in_percentages
	convex_hull_in_percentages_str = "%.2f" % convex_hull_in_percentages
	convex_hull_perimeter_str = "%.2f" % convex_hull_perimeter
	density_str = "%.2f" % density
	file_name = "index_" + str(index) + "_" + classification + \
				"_circles_" + str(len(circles)) + \
				"_ch_" + str(num_of_verteces) + \
				"_chp_" + str(convex_hull_perimeter_str) + \
				"_cha_" + convex_hull_in_percentages_str + \
				"_density_" + density_str + \
				"_area_" + area_in_percentages_str + ".png"

	fig.savefig(dest_dir + os.sep + file_name)
	plt.close()
	print('saving image: {} of numeric val: {}, class: {}'.format(index, len(circles), classification))


def main(args):
	create_stimuli(args.dest_dir, False)
	print('Done!')


if __name__ == '__main__':
	parser = argparse.ArgumentParser(description='evolve arguments')
	parser.add_argument('--dest_dir', dest='dest_dir', type=str, required=True, help='The dest dir')
	args = parser.parse_args()
	main(args)


