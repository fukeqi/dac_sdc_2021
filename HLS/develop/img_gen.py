import numpy as np
import math
import os
from PIL import Image
import cv2
np.set_printoptions(threshold=np.inf)

image_buff  = []
image = np.zeros((4,160,320,4),np.uint8)
img_size = 320

root = "/home/dd/fkq_vivado_prj/Skynet_sxl_q8_c/test_sample"
imgdir1 = os.path.join(root,'136.jpg')
imgdir2 = os.path.join(root,'137.jpg')
imgdir3 = os.path.join(root,'138.jpg')
imgdir4 = os.path.join(root,'139.jpg')

raw_img = cv2.imread(imgdir1)
raw_width, raw_height = raw_img.shape[1], raw_img.shape[0]
img = cv2.cvtColor(raw_img, cv2.COLOR_BGR2RGBA)
img = cv2.resize(img, (img_size, img_size // 2), interpolation=cv2.INTER_LINEAR)
image[0] = np.array(img)

raw_img = cv2.imread(imgdir2)
raw_width, raw_height = raw_img.shape[1], raw_img.shape[0]
img = cv2.cvtColor(raw_img, cv2.COLOR_BGR2RGBA)
img = cv2.resize(img, (img_size, img_size // 2), interpolation=cv2.INTER_LINEAR)
image[1] = np.array(img)

raw_img = cv2.imread(imgdir3)
raw_width, raw_height = raw_img.shape[1], raw_img.shape[0]
img = cv2.cvtColor(raw_img, cv2.COLOR_BGR2RGBA)
img = cv2.resize(img, (img_size, img_size // 2), interpolation=cv2.INTER_LINEAR)
image[2] = np.array(img)

raw_img = cv2.imread(imgdir4)
raw_width, raw_height = raw_img.shape[1], raw_img.shape[0]
img = cv2.cvtColor(raw_img, cv2.COLOR_BGR2RGBA)
img = cv2.resize(img, (img_size, img_size // 2), interpolation=cv2.INTER_LINEAR)
image[3] = np.array(img)

print_log = open('print_img.txt','w')
print("================================================",file=print_log)
print(image,file=print_log)
print(image.shape,file=print_log)
image=image.astype(np.uint8)
image.tofile("raw_image.bin")
