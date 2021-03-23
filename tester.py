import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import random

# fig, ax = plt.subplots()
#
# #plt.plot(xList, yList, marker = '.', color = 'k', linestyle = 'None')
#
# for i in range(10):
#     circle1 = plt.Circle((i, i), i, color = 'r')
#     ax.add_artist(circle1)
# plt.show()


def create_circle():
	circle = plt.Circle((0, 0), radius=5)
	return circle

def show_shape(patch):
	ax = plt.gca()
	ax.add_patch(patch)
	plt.axis('scaled')
	plt.show()


if __name__ == '__main__':
	c = create_circle()
	show_shape(c)
