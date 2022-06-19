import numpy as np
import cv2


def calc_metric(image, x, y, w, h):
	img = cv2.imread(image)
	img = img[y:y+h, x:x+w]

	img = np.reshape(img, (-1,3))
	data = np.float32(img)

	criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
	flags = cv2.KMEANS_RANDOM_CENTERS
	compactness,labels,centers = cv2.kmeans(data,1,None,criteria,10,flags)

	bgr_color = centers[0].astype(np.int32)
	print('Dominant color is: bgr({})'.format(bgr_color))

	b = int(bgr_color[0])
	g = int(bgr_color[1])
	r = int(bgr_color[2])

	prev_img = cv2.imread(image)
	cv2.rectangle(prev_img, (x, y), (x+w, y+h), (b,g,r), -1)


	cv2.imshow('',prev_img)
	cv2.waitKey(0)

	return centers[0].astype(np.int32)

# calc_metric('./00QCBzqD1EY.jpg', 245, 70, 60, 60)