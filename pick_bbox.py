"""
pick_bbox.py — manually get a bounding box from an image by clicking two corners.

Stand-in for a real object detector, which isn't built yet in this POC. Used
to generate the test detections in example_detections.json.

Usage:
    python3 pick_bbox.py path/to/frame.png

Click the top-left corner of the object, then the bottom-right corner.
Prints a bbox in the format used by example_detections.json.
"""

import sys

import matplotlib.pyplot as plt
from matplotlib.image import imread

if len(sys.argv) != 2:
    print("Usage: python3 pick_bbox.py path/to/frame.png")
    sys.exit(1)

path = sys.argv[1]
img = imread(path)

fig, ax = plt.subplots()
ax.imshow(img)
ax.set_title("Click top-left corner, then bottom-right corner")
pts = plt.ginput(2, timeout=0)
plt.close(fig)

if len(pts) != 2:
    print("Need exactly 2 clicks — got", len(pts))
    sys.exit(1)

(x1, y1), (x2, y2) = pts
x_min, x_max = sorted([x1, x2])
y_min, y_max = sorted([y1, y2])
print(f'"bbox": [{x_min:.0f}, {y_min:.0f}, {x_max:.0f}, {y_max:.0f}]')
