"""
You have a grid-like representation of an image with n rows and m columns, where each cell contains a pixel intensity value. Your goal is to adjust the pixel intensities to improve object visibility.

Each pixel at position (i, j) has an intensity value pixelIntensity[j], where 0 ≤ i < n and 0 ≤ j < m. The requirement is that no pixel in the previous rows of the same column should have a brightness greater than or equal to the current pixel.

You can increase a pixel's intensity at a cost of one unit per unit of increase. Determine the minimum total cost to achieve the desired intensity pattern across all pixels.
"""

import numpy as np
import matplotlib.pyplot as plt

x = np.random.randint(0, 255, (40, 40))
plt.figure(1)
plt.imshow(x)

for j in range(x.shape[1]):
	max_column = x[0][j]
	for i in range(x.shape[0]):
		if x[i][j] <= max_column:
			x[i][j] = max_column + 1
		else:
			max_column = x[i][j]

plt.figure(2)
plt.imshow(x)
plt.show()