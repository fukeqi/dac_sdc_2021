import torch
import torch.nn as nn
import math
import numpy as np
from PIL import Image
from quant import *

np.set_printoptions(suppress=True)
np.set_printoptions(threshold=np.inf)
model = torch.load('test_best.pt')

torch.save(model['model'],'model_state_dict.pt')
model_param = torch.load('model_state_dict.pt')

def intToBin4(i):
    return (bin(((1 << 4) - 1) & i)[2:]).zfill(4)

def weight_reg(weight_q,f):
    N = round(weight_q.shape[1]/16)
    M = round(weight_q.shape[0]/32)
    for m in range(M):
        for n in range(N):
            for co in range(32):
                for h in range(3):
                    for w in range(3): 
                        s = '0x'
                        for ci in range(16):
                            tmp = int(intToBin4(weight_q[co+m*32][ci+n*16][h][w]),2)
                            s = s + hex(tmp)[2:]
                        if h==0 and w==0:
                            print('{',s,end=', ',file=f)
                        elif h==2 and w==2:
                            print(s,end='},',file=f)
                        else:
                            print(s,end=', ',file=f)
                print('',file=f)

def pweight_reg(weight_q,f):
    for co in range(12):
        for ci in range(64):
            if ci==0:
                print('{',weight_q[co][ci][0][0],end=', ', file=f)  
            elif ci==63:
                print(weight_q[co][ci][0][0],end='},', file=f)  
            else:
                print(weight_q[co][ci][0][0],end=', ', file=f)  
        print('',file=f)       
                       
                
   

def weight_quantize_int(weight):
    weight = weight.cpu().detach().numpy()
    weight_q = weight * 7
    weight_q = np.round(weight_q).astype(np.int8)
    return weight_q

def print_w(w,f):
    f.write("====================================================================")
    L = w.shape[0]
    f.write(str(L))
    f.write('\n')
    for i in range(round(L/32)):
        for j in range(32):
            f.write(str((w[i*32+j]+15)//2))
            if j==31:
                f.write(',\n')
            else:
                f.write(', ')

quantize_fn = weight_quantize_fn(4)

print_log = open('print_weight.txt','w')
print_w_log = open('print_w.txt','w')
f = open('weight.txt','w')
layers = []
for v in model_param:
    if len(model_param[v].size())==4:
        layers.append(v)
        print(v," ", model_param[v].size(), file=print_w_log)
        

print("======================================================================",file=print_log)
print("======================================================================",file=print_w_log)
layer = layers[0]  #3,32
print(layer, file=print_log)
print(layer, file=print_w_log)
weight=quantize_fn(model_param[layer])
weight_q=weight_quantize_int(weight)
print(weight_q,file=print_log)
print(weight_q.shape,file=print_log)
c = np.zeros((32,16,3,3))
for i in range(32):
    a = weight_q[i]
    b = np.zeros((13,3,3))
    c[i] = np.append(a,b,axis=0)
ww = c.astype(np.int8)
weight_reg(ww, print_w_log)
'''
for co in range(32):
    for kh in range(3):
        for kw in range(3):
            for ci in range(3):
                if ci==0 and kh==0 and kw==0:
                    print('{',weight_q[co][ci][kh][kw],end=', ', file=print_w_log)
                elif ci==2 and kh==2 and kw==2:
                    print(weight_q[co][ci][kh][kw],end='},', file=print_w_log)
                else:
                    print(weight_q[co][ci][kh][kw],end=', ', file=print_w_log)
    print('',file=print_w_log)
'''
print("======================================================================",file=print_log)
print("======================================================================",file=print_w_log)
layer = layers[1] #32 32
print(layer, file=print_log)
print(layer, file=print_w_log)
weight=quantize_fn(model_param[layer])
weight_q=weight_quantize_int(weight)
print(weight_q,file=print_log)
print(weight_q.shape,file=print_log)
weight_reg(weight_q, print_w_log)
print("======================================================================",file=print_log)
print("======================================================================",file=print_w_log)
layer = layers[2] #32, 64
print(layer, file=print_log)
print(layer, file=print_w_log)
weight=quantize_fn(model_param[layer])
weight_q=weight_quantize_int(weight)
print(weight_q,file=print_log)
print(weight_q.shape,file=print_log)
weight_reg(weight_q, print_w_log)
print("======================================================================",file=print_log)
print("======================================================================",file=print_w_log)
layer = layers[3] #64, 64
print(layer, file=print_log)
print(layer, file=print_w_log)
weight=quantize_fn(model_param[layer])
weight_q=weight_quantize_int(weight)
print(weight_q,file=print_log)
print(weight_q.shape,file=print_log)
weight_reg(weight_q, print_w_log)
print("======================================================================",file=print_log)
print("======================================================================",file=print_w_log)
layer = layers[4] #64, 64
print(layer, file=print_log)
print(layer, file=print_w_log)
weight=quantize_fn(model_param[layer])
weight_q=weight_quantize_int(weight)
print(weight_q,file=print_log)
print(weight_q.shape,file=print_log)
weight_reg(weight_q, print_w_log)
print("======================================================================",file=print_log)
print("======================================================================",file=print_w_log)
layer = layers[5] #64, 64
print(layer, file=print_log)
print(layer, file=print_w_log)
weight=quantize_fn(model_param[layer])
weight_q=weight_quantize_int(weight)
print(weight_q,file=print_log)
print(weight_q.shape,file=print_log)
weight_reg(weight_q, print_w_log)
print("======================================================================",file=print_log)
print("======================================================================",file=print_w_log)
layer = layers[6] #64, 64
print(layer, file=print_log)
print(layer, file=print_w_log)
weight=quantize_fn(model_param[layer])
weight_q=weight_quantize_int(weight)
print(weight_q,file=print_log)
print(weight_q.shape,file=print_log)
weight_reg(weight_q, print_w_log)


#head
print("======================================================================",file=print_log)
print("======================================================================",file=print_w_log)
layer = layers[7] #64, 64
print(layer, file=print_log)
print(layer, file=print_w_log)
weight=quantize_fn(model_param[layer])
weight_q=weight_quantize_int(weight)
print(weight_q,file=print_log)
print(weight_q.shape,file=print_log)
weight_reg(weight_q, print_w_log)
print("======================================================================",file=print_log)
print("======================================================================",file=print_w_log)
layer = layers[8] #12 64
print(layer, file=print_log)
print(layer, file=print_w_log)
weight=quantize_fn(model_param[layer])
weight_q=weight_quantize_int(weight)
print(weight_q,file=print_log)
print(weight_q.shape,file=print_log)
pweight_reg(weight_q, print_w_log)


#head
print("======================================================================",file=print_log)
print("======================================================================",file=print_w_log)
layer = layers[9] #64, 64
print(layer, file=print_log)
print(layer, file=print_w_log)
weight=quantize_fn(model_param[layer])
weight_q=weight_quantize_int(weight)
print(weight_q,file=print_log)
print(weight_q.shape,file=print_log)
weight_reg(weight_q, print_w_log)
print("======================================================================",file=print_log)
print("======================================================================",file=print_w_log)
layer = layers[10] #12 64
print(layer, file=print_log)
print(layer, file=print_w_log)
weight=quantize_fn(model_param[layer])
weight_q=weight_quantize_int(weight)
print(weight_q,file=print_log)
print(weight_q.shape,file=print_log)
pweight_reg(weight_q, print_w_log)


#head
print("======================================================================",file=print_log)
print("======================================================================",file=print_w_log)
layer = layers[11] #64, 64
print(layer, file=print_log)
print(layer, file=print_w_log)
weight=quantize_fn(model_param[layer])
weight_q=weight_quantize_int(weight)
print(weight_q,file=print_log)
print(weight_q.shape,file=print_log)
weight_reg(weight_q, print_w_log)
print("======================================================================",file=print_log)
print("======================================================================",file=print_w_log)
layer = layers[12] #12 64
print(layer, file=print_log)
print(layer, file=print_w_log)
weight=quantize_fn(model_param[layer])
weight_q=weight_quantize_int(weight)
print(weight_q,file=print_log)
print(weight_q.shape,file=print_log)
pweight_reg(weight_q, print_w_log)






'''
for v in model_param:
    print(v,file=f)
    if len(model_param[v].size())==4:
        print(quantize_fn(model_param[v]),file=t)
        print(weight_quantize_int(quantize_fn(model_param[v])), file=f)
    else:
         print(model_param[v], file=f)
'''

#
#print(model_param,file=f)




