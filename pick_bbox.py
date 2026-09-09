import sys
import matplotlib.pyplot as plt
from matplotlib.image import imread

path = sys.argv[1]
img = imread(path)
fig, ax = plt.subplots()
ax.imshow(img)
ax.set_title("Click top-left corner, then bottom-right corner")
pts = plt.ginput(2, timeout=0)
plt.close(fig)

(x1, y1), (x2, y2) = pts
x_min, x_max = sorted([x1, x2])
y_min, y_max = sorted([y1, y2])
print(f'"bbox": [{x_min:.0f}, {y_min:.0f}, {x_max:.0f}, {y_max:.0f}]')