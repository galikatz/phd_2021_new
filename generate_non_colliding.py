import random
import matplotlib
#matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
import numpy


def nocollide_random_unif_circles():

	def add_circle(ax, center=(0,0), r=1, face_color=(0,0,1,1), edge_color=(0,0,1,1), line_width = None):
		C = Circle(center, radius = r, facecolor = face_color, edgecolor = edge_color, linewidth = line_width)
		ax.add_patch(C)

	def dist2d(p, q):
		distance = numpy.sqrt( (p[0]-q[0])**2 + (p[1]-q[1])**2 )
		return distance

	def collide(center, rad, centers, rads):
		for c, r in zip(centers, rads):
			if dist2d(c, center) < r + rad:
				return True
		return False
	ncircle = 10
	min_r = 1
	max_r = 50
	color = 'black'

	sizex = [0, 200]
	sizey = [0, 200]

	fig, ax = plt.subplots(1, 1)

	rads = [random.uniform(min_r, max_r)]
	centers = [(random.uniform(rads[0], sizex[1]-rads[0]), random.uniform(rads[0], sizey[1]-rads[0]))]

	add_circle(ax, center=centers[0], r=rads[0], face_color=color, edge_color=(0, 0, 0, 0))
	for n in range(ncircle):
		rad = random.uniform(min_r, max_r)
		center = (random.uniform(rad, sizex[1]-rad), random.uniform(rad, sizey[1]-rad))

		while collide(center, rad, centers, rads):
			center = (random.uniform(rad, sizex[1]-rad), random.uniform(rad, sizey[1]-rad))
			rad = random.uniform(min_r, max_r)
		centers.append(center)
		rads.append(rad)
		add_circle(ax, center=centers[-1], r=rads[-1], face_color=color, edge_color=(0, 0, 0, 0))
		print(n)

	ax.axis('equal')
	ax.set_xlim(sizex);
	ax.set_ylim(sizey)
	ax.tick_params(colors=(0, 0, 0, 0))
	ax.set_title('min radius = {}, max radius = {}, n = {} circles'.format(min_r, max_r, ncircle))

	fig.tight_layout()
	fig.show()
	pass

if __name__ == '__main__':


	nocollide_random_unif_circles()