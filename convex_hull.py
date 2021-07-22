from scipy.spatial import ConvexHull, convex_hull_plot_2d
import numpy as np
import matplotlib
#matplotlib.use('TkAgg')
import matplotlib.pyplot as plt


def convex_hull():
	points = np.random.rand(30, 2)   # 30 random points in 2-D
	hull = ConvexHull(points)
	plt.plot(points[:,0], points[:,1], 'o')
	print('hull len: {}'.format(len(hull.simplices)))
	for simplex in hull.simplices:
		plt.plot(points[simplex, 0], points[simplex, 1], 'k-')
	plt.show()



if __name__ == '__main__':
	convex_hull()