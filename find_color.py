import numpy as np
import cv2
import glob
import pandas as pd


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

	return centers[0].astype(np.int32)


def find_color(input_dir, output_file="output_color.csv"):
	black = (-25,-25,-25)
	blue_cyan = (255,255,0)
	green = (0,255,0)
	red = (0,0,255)
	white = (255,255,255)
	yellow = (0, 255, 255)

	colors = [black, blue_cyan, green, red, white, yellow]

	files = glob.glob(input_dir+'/*')
	file_names = []
	colors_list = []

	for file in files:
		print(file)
		file_n = file.replace("\\", "/").split('/')[-1]
		file_names.append(file_n)

		img = cv2.imread(file)

		# get center part
		y = img.shape[0]//4
		x = img.shape[1]//4
		w = img.shape[1]//2
		h = img.shape[0]//2

		color = calc_metric(file, x, y, w, h)

		diff_c = []

		for c in colors:
			# dist = sqrt(((color[2]-c[2])*0.3)**2 + ((color[1]-c[1])*0.59)**2 + ((color[0]-c[0])*0.11)**2)
			b_ = abs(color[0] - c[0])
			g_ = abs(color[1] - c[1])
			r_ = abs(color[2] - c[2])
			diff_c.append(b_+g_+r_)

		color_index = np.argmin(np.array(diff_c))
		if color_index == 0:
			colors_list.append('black')
			print('black')
		if color_index == 1:
			colors_list.append('blue_cyan')
			print('blue_cyan')
		if color_index == 2:
			colors_list.append('green')
			print('green')
		if color_index == 3:
			colors_list.append('red')
			print('red')
		if color_index == 4:
			colors_list.append('white_silver')
			print('white_silver')
		if color_index == 5:
			colors_list.append('yellow')
			print('yellow')

	output = pd.DataFrame({'file': file_names, 'pred': colors_list})
	output.to_csv(output_file, index=False, header=False)



find_color('./output')