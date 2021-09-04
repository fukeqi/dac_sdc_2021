import os
import torch
import math
import numpy as np
from mymodel import *
from quant_dorefa import *
import cv2
from PIL import Image
from torchvision import transforms

np.set_printoptions(suppress=True)
np.set_printoptions(threshold=np.inf)

print_log = open('print_imgtest.txt','w')
print_fm_log = open('print_imgtest_fm.txt','w')
print_fm2_log = open('print_imgtest_fm2.txt','w')


model = UltraNet_3fz()
model.load_state_dict(torch.load('test_best.pt')['model'])
model = model.cuda()
model.eval()


IMAGE_RAW_ROW = 360
IMAGE_RAW_COL = 640
IMAGE_ROW = 160
IMAGE_COL = 320
GRID_ROw = 10
GRID_COL = 20
X_SCALE = IMAGE_RAW_COL / IMAGE_COL
Y_SCALE = IMAGE_RAW_ROW / IMAGE_ROW

trans = transforms.Compose([
    transforms.ToTensor(),
    #transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.25, 0.25, 0.25]),
])

def sigmoid(x):
    s = 1 / (1 + np.exp(-x))
    return s

actq = activation_quantize_fn(4)

def get_boxes(pred_boxes, pred_conf):
    n = pred_boxes.size(0)
    # pred_boxes = pred_boxes.view(n, -1, 4)
    # pred_conf = pred_conf.view(n, -1, 1)
    FloatTensor = torch.cuda.FloatTensor if pred_boxes.is_cuda else torch.FloatTensor
    p_boxes = FloatTensor(n, 4)
    # print(pred_boxes.shape, pred_conf.shape)

    for i in range(n):
        _, index = pred_conf[i].max(0)
        p_boxes[i] = pred_boxes[i][index]
    print(index)
    print(index%20)
    print(index//20)
    return p_boxes

def bbox_iou(box1):
    """
    Returns the IoU of two bounding boxes
    """
    box1[:,0] *= 2
    box1[:,2] *= 2
    box1[:,1] *= (360/160)
    box1[:,3] *= (360/160)
    # Transform from center and width to exact coordinates
    b1_x1, b1_x2 = box1[:, 0] - box1[:, 2] / 2, box1[:, 0] + box1[:, 2] / 2
    b1_y1, b1_y2 = box1[:, 1] - box1[:, 3] / 2, box1[:, 1] + box1[:, 3] / 2
    return b1_x1,b1_y1,b1_x2,b1_y2

def solve(raw_img,flag):
    #img = raw_img.convert('RGB').resize((320, 160))
    img = cv2.cvtColor(raw_img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (320,160), interpolation=cv2.INTER_LINEAR)
    img = trans(img)
    img = img.view(1, 3, 160, 320)
    img = img.cuda()

    if(flag==1):
        x = model.layers[0](img)
        x = model.layers[1](x)
        x = model.layers[2](x)
        x = model.layers[3](x)
        x = model.layers[4](x)
        print("===========================================================================================",file=print_log)
        y = x.cpu().detach().numpy()
        out = np.array([])
        for c in range(y.shape[1]):
            for i in range(y.shape[2]):
                for j in range(y.shape[3]):
                    out = np.append(out,(y[0][c][i][j]*15).astype(np.int8))   
        print(out.shape,file=print_log) 
        out = out.astype(np.int8)
        out.tofile("pool0.bin")

        print("===========================================================================================",file=print_log)
        x = model.layers[5](x)
        x = model.layers[6](x)
        x = model.layers[7](x)
        x = model.layers[8](x)
        y = x.cpu().detach().numpy()
        out = np.array([])
        for c in range(y.shape[1]):
            for i in range(y.shape[2]):
                for j in range(y.shape[3]):
                    out = np.append(out,(y[0][c][i][j]*15).astype(np.int8))   
        print(out.shape,file=print_log) 
        out = out.astype(np.int8)
        out.tofile("pool1.bin")

        print("===========================================================================================",file=print_log)
        x = model.layers[9](x)
        x = model.layers[10](x)
        x = model.layers[11](x)
        x = model.layers[12](x)
        y = x.cpu().detach().numpy()
        out = np.array([])
        for c in range(y.shape[1]):
            for i in range(y.shape[2]):
                for j in range(y.shape[3]):
                    out = np.append(out,(y[0][c][i][j]*15).astype(np.int8))   
        print(out.shape,file=print_log) 
        out = out.astype(np.int8)
        out.tofile("pool2.bin")

    output,p = model(img)
    batch_n = 1
    output = output.cpu().detach().numpy()
    res_np = np.array(output[:batch_n]).reshape(batch_n, -1, 6, 6)
    conf = res_np[...,4].max(axis=2)
    max_index = conf.argmax(1)
    print(max_index)
    grid_x = max_index % 20
    grid_y = max_index // 20
    print(grid_x)
    print(grid_y)
    boxs = np.zeros((batch_n, 6, 4))
    for i in range(batch_n):
        boxs[i, :, :] = res_np[i, max_index[i], :, :4] 
    
    xy = boxs[..., :2].mean(axis=1)
    wh = boxs[..., 2:4].mean(axis=1)
    
    #xy[:, 0] += grid_x
    #xy[:, 1] += grid_y

    #xy *= 16
    #wh *= 20

    xy[:, 0] *= X_SCALE
    xy[:, 1] *= Y_SCALE
    wh[:, 0] *= X_SCALE
    wh[:, 1] *= Y_SCALE
    xmin = xy[:, 0] - wh[:, 0] / 2
    xmax = xy[:, 0] + wh[:, 0] / 2
    ymin = xy[:, 1] - wh[:, 1] / 2
    ymax = xy[:, 1] + wh[:, 1] / 2
    
    print(xmin," ",ymin," ",xmax," ",ymax)
   
   

def main():
    global raw_width, raw_height
    
    raw_img = cv2.imread('/samples/0.jpg')
    raw_width, raw_height = raw_img.shape[1], raw_img.shape[0]
    solve(raw_img,1)

    raw_img = cv2.imread('/samples/6.jpg')
    raw_width, raw_height = raw_img.shape[1], raw_img.shape[0]
    solve(raw_img,0)

    raw_img = cv2.imread('/samples/49.jpg')
    raw_width, raw_height = raw_img.shape[1], raw_img.shape[0]
    solve(raw_img,0)

    raw_img = cv2.imread('/samples/88.jpg')
    raw_width, raw_height = raw_img.shape[1], raw_img.shape[0]
    solve(raw_img,0)

if __name__ == '__main__':
    main()