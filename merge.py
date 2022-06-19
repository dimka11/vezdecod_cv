import cv2
import numpy as np
import pandas as pd
import glob

def merge_channels(input_dir, output_dir):
    files_list = glob.glob(input_dir+'/*')
    files_list = sorted(files_list)
    files_list_set = set()
    for file in files_list:
        files_list_set.add(file.replace('_b.jpg', '').replace('_g.jpg', '').replace('_r.jpg', ''))
        
    for file in files_list_set:
        file_name = file.replace("\\", "/").split('/')[-1]+'.jpg'
        b = cv2.imread(file+'_b.jpg', cv2.IMREAD_UNCHANGED)
        g = cv2.imread(file+'_g.jpg', cv2.IMREAD_UNCHANGED)
        r = cv2.imread(file+'_r.jpg',cv2.IMREAD_UNCHANGED)

        #     print(b.shape, g.shape, r.shape)

        bgr = cv2.merge([b, g, r])
        cv2.imwrite(output_dir+'/'+file_name, bgr)
