# import the necessary packages
from imutils import contours
from skimage import measure
import numpy as np
import argparse
import imutils
import cv2
from matplotlib import pyplot as plt

def grab_rgb(image, c):

	# Initialize empty list
	lst_intensities = []
	# Create a mask image that contains the contour filled in
	cimg = np.zeros_like(image)
	cv2.drawContours(cimg, c, i, color=255, thickness=-1)

	# plt.imshow(cimg)
	# plt.show()

	# Access the image pixels and create a 1D numpy array then add to list
	pts = np.where(cimg == 255)
	lst_intensities.append(image[pts[0], pts[1]])
	pixel_string = '{0},{1},{2}'.format(r, g, b)

	return pixel_string
 
# parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True,
	help="path to the image file")
args = vars(ap.parse_args())

# load the image, convert it to grayscale, and blur it
image = cv2.imread(args["image"])
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
blurred = cv2.GaussianBlur(gray, (5, 5), 0)

# threshold the image to reveal light regions in the blurred image
thresh = cv2.threshold(blurred, 30, 255, cv2.THRESH_BINARY)[1]

# perform a connected component analysis on the thresholded
# image, then initialize a mask to store only the "small blobs"
labels = measure.label(thresh, neighbors=8, background=0)
mask = np.zeros(thresh.shape, dtype="uint8")

# loop over the unique components
for label in np.unique(labels):
	# if this is the background label, ignore it
	if label == 0:
		continue

	# otherwise, construct the label mask and count the
	# number of pixels 
	labelMask = np.zeros(thresh.shape, dtype="uint8")
	labelMask[labels == label] = 255
	numPixels = cv2.countNonZero(labelMask)

	# if the number of pixels in the component is sufficiently
	# small, then add it to our mask of "small blobs"
	if numPixels < 30 and numPixels > 3:
		mask = cv2.add(mask, labelMask)

# find the contours in the mask, then sort them from left to
# right
cnts = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL,
	cv2.CHAIN_APPROX_SIMPLE)
cnts = cnts[0] if imutils.is_cv2() else cnts[1]
cnts = contours.sort_contours(cnts)[0]

b,g,r = cv2.split(image)
 
# loop over the contours
for (i, c) in enumerate(cnts):
	# draw the bright spot on the image
	(x, y, w, h) = cv2.boundingRect(c)
	# if c contains red count
	rgb = grab_rgb(image, cnts)
	print(rgb)

	#((cX, cY), radius) = cv2.minEnclosingCircle(c)
	# place the counts on the image
	cv2.putText(image, "{}".format(i + 1), (x, y - 5),
		cv2.FONT_HERSHEY_PLAIN, 0.6, (255, 255, 255), 1)
 
# show the output image
plt.imshow(image)
plt.show()

