import numpy as np
import matplotlib
#matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import scipy.stats as st
from sklearn.datasets.samples_generator import make_blobs

n_components = 20
X, truth = make_blobs(n_samples=n_components, centers=1,
                      cluster_std = [3],
                      random_state=100)
fig1 = plt.figure(figsize=(6,6))
plt.scatter(X[:, 0], X[:, 1], s=50, c = truth)
plt.title("Example of a mixture of {n_components} distributions")
plt.xlabel("x")
plt.ylabel("y")
fig1.savefig("dots.png")
plt.close()

# Extract x and y
x = X[:, 0]
y = X[:, 1]
# Define the borders
deltaX = (max(x) - min(x))/10
deltaY = (max(y) - min(y))/10
xmin = min(x) - deltaX
xmax = max(x) + deltaX
ymin = min(y) - deltaY
ymax = max(y) + deltaY
#print(xmin, xmax, ymin, ymax)
# Create meshgrid
xx, yy = np.mgrid[xmin:xmax:100j, ymin:ymax:100j]

positions = np.vstack([xx.ravel(), yy.ravel()])
values = np.vstack([x, y])
kernel = st.gaussian_kde(values)
f = np.reshape(kernel(positions).T, xx.shape)
max_f = np.max(f)
print(max_f)
fig2 = plt.figure(figsize=(6, 6))
ax = fig2.gca()
ax.set_xlim(xmin, xmax)
ax.set_ylim(ymin, ymax)
cfset = ax.contourf(xx, yy, f, cmap='coolwarm')
ax.imshow(np.rot90(f), cmap='coolwarm', extent=[xmin, xmax, ymin, ymax])
cset = ax.contour(xx, yy, f, colors='k')
ax.clabel(cset, inline=1, fontsize=10)
ax.set_xlabel('X')
ax.set_ylabel('Y')
plt.title('2D Gaussian Kernel density estimation')
fig2.savefig("gkd.png")
plt.close()

pass