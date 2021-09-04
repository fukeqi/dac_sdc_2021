import torch
import torch.nn as nn
import math
import numpy as np
from PIL import Image
from quant import *

def bn_act_w_bias_float(gamma, beta, mean, var, eps=1e-5):
    w = gamma / (np.sqrt(var) + eps)
    b = beta - (mean / (np.sqrt(var) + eps) * gamma)
    return w, b

def bn_act_quantize_int(gamma, beta, mean, var, eps=1e-5, w_bit=4, in_bit=4, out_bit=4, l_shift=8):
    w, b = bn_act_w_bias_float(gamma, beta, mean, var, eps)
    #print('inc:',w)
    #print('bias:',b)

    n = 2**(w_bit - 1 + in_bit + l_shift) / ((2 ** (w_bit-1) - 1) * (2 ** in_bit - 1)) #2**15/7x15
    inc_q = (2 ** out_bit - 1) * n * w 
    bias_q = (2 ** (w_bit-1) - 1) * (2 ** in_bit - 1) * (2 ** out_bit - 1) * n * b
    inc_q = np.round(inc_q).astype(np.int32)
    bias_q = np.round(bias_q).astype(np.int32)
    #print('inc_q: ', inc_q)
    #print('bias_q: ', bias_q)
    return inc_q, bias_q

def bias_int(bias,w_bit=4, in_bit=4):
    w = bias * (2 ** (w_bit-1) - 1) * (2 ** in_bit - 1) * (2**8)
    w = np.round(w).astype(np.int32)
    return w




# 确定 inc 位宽 
def get_inc_bit_width(inc):
    abs_max = np.abs(inc).max()
    bit_width = len(str(bin(abs_max))) - 2
    return bit_width + 1

# 确定bias的位宽
# bias 有整数和负数
# 当前算法得出的还不是最优
def get_bias_bit_width(bias):
    abs_max = np.abs(bias).max()
    bit_width = len(str(bin(abs_max))) - 2
    return bit_width + 1

def bias_to_txt(bias,f): #bias numpy列向量
    L = bias.size
    bias_t = bias.reshape(1,L)
    bias_t = bias_t.tolist()
    for i in range(int(L/32)):
        f.write('{')
        for v in range(4):
            for h in range(8):
                f.write(str(bias_t[0][i*32+v*8+h]))
                if(h==7):
                    if(v==3):
                        f.write('},\n')
                    else:
                        f.write(',\n')
                else:
                    f.write(', ')

np.set_printoptions(suppress=True)
np.set_printoptions(threshold=np.inf)
model = torch.load('test_best.pt')

torch.save(model['model'],'model_state_dict.pt')
model_param = torch.load('model_state_dict.pt')


print_log = open('print_information.txt','w')
layers = []
for v in model_param:
    if len(model_param[v].size())==1:
        layers.append(v)
        print(v," ", model_param[v].size(), file=print_log)
f = open('bias.txt','w')
print("layers len",len(layers))

print("======================================================================",file=print_log)
weight = model_param[layers[0]].cpu().detach().numpy()
bias = model_param[layers[1]].cpu().detach().numpy()
mean = model_param[layers[2]].cpu().detach().numpy()
var = model_param[layers[3]].cpu().detach().numpy()
inc_q, bias_q=bn_act_quantize_int(weight,bias,mean,var)
print('layer0_inc:',inc_q,file=print_log)
print('layer0_bias:',bias_q,file=print_log)
print(get_inc_bit_width(inc_q),file=print_log)
print(get_inc_bit_width(bias_q),file=print_log)
f.write('layer0_inc\n')
bias_to_txt(inc_q,f)
f.write('layer0_bias\n')
bias_to_txt(bias_q,f)

print("======================================================================",file=print_log)
weight = model_param[layers[4]].cpu().detach().numpy()
bias = model_param[layers[5]].cpu().detach().numpy()
mean = model_param[layers[6]].cpu().detach().numpy()
var = model_param[layers[7]].cpu().detach().numpy()
inc_q, bias_q=bn_act_quantize_int(weight,bias,mean,var)
print('layer1_inc:',inc_q,file=print_log)
print('layer1_bias:',bias_q,file=print_log)
print(get_inc_bit_width(inc_q),file=print_log)
print(get_inc_bit_width(bias_q),file=print_log)
f.write('layer1_inc\n')
bias_to_txt(inc_q,f)
f.write('layer1_bias\n')
bias_to_txt(bias_q,f)

print("======================================================================",file=print_log)
weight = model_param[layers[8]].cpu().detach().numpy()
bias = model_param[layers[9]].cpu().detach().numpy()
mean = model_param[layers[10]].cpu().detach().numpy()
var = model_param[layers[11]].cpu().detach().numpy()
inc_q, bias_q=bn_act_quantize_int(weight,bias,mean,var)
print('layer2_inc:',inc_q,file=print_log)
print('layer2_bias:',bias_q,file=print_log)
print(get_inc_bit_width(inc_q),file=print_log)
print(get_inc_bit_width(bias_q),file=print_log)
f.write('layer2_inc\n')
bias_to_txt(inc_q,f)
f.write('layer2_bias\n')
bias_to_txt(bias_q,f)

print("======================================================================",file=print_log)
weight = model_param[layers[12]].cpu().detach().numpy()
bias = model_param[layers[13]].cpu().detach().numpy()
mean = model_param[layers[14]].cpu().detach().numpy()
var = model_param[layers[15]].cpu().detach().numpy()
inc_q, bias_q=bn_act_quantize_int(weight,bias,mean,var)
print('layer3_inc:',inc_q,file=print_log)
print('layer3_bias:',bias_q,file=print_log)
print(get_inc_bit_width(inc_q),file=print_log)
print(get_inc_bit_width(bias_q),file=print_log)
f.write('layer3_inc\n')
bias_to_txt(inc_q,f)
f.write('layer3_bias\n')
bias_to_txt(bias_q,f)

print("======================================================================",file=print_log)
weight = model_param[layers[16]].cpu().detach().numpy()
bias = model_param[layers[17]].cpu().detach().numpy()
mean = model_param[layers[18]].cpu().detach().numpy()
var = model_param[layers[19]].cpu().detach().numpy()
inc_q, bias_q=bn_act_quantize_int(weight,bias,mean,var)
print('layer4_inc:',inc_q,file=print_log)
print('layer4_bias:',bias_q,file=print_log)
print(get_inc_bit_width(inc_q),file=print_log)
print(get_inc_bit_width(bias_q),file=print_log)
f.write('layer4_inc\n')
bias_to_txt(inc_q,f)
f.write('layer4_bias\n')
bias_to_txt(bias_q,f)

print("======================================================================",file=print_log)
weight = model_param[layers[20]].cpu().detach().numpy()
bias = model_param[layers[21]].cpu().detach().numpy()
mean = model_param[layers[22]].cpu().detach().numpy()
var = model_param[layers[23]].cpu().detach().numpy()
inc_q, bias_q=bn_act_quantize_int(weight,bias,mean,var)
print('layer5_inc:',inc_q,file=print_log)
print('layer5_bias:',bias_q,file=print_log)
print(get_inc_bit_width(inc_q),file=print_log)
print(get_inc_bit_width(bias_q),file=print_log)
f.write('layer5_inc\n')
bias_to_txt(inc_q,f)
f.write('layer5_bias\n')
bias_to_txt(bias_q,f)

print("======================================================================",file=print_log)
weight = model_param[layers[24]].cpu().detach().numpy()
bias = model_param[layers[25]].cpu().detach().numpy()
mean = model_param[layers[26]].cpu().detach().numpy()
var = model_param[layers[27]].cpu().detach().numpy()
inc_q, bias_q=bn_act_quantize_int(weight,bias,mean,var)
print('layer6_inc:',inc_q,file=print_log)
print('layer6_bias:',bias_q,file=print_log)
print(get_inc_bit_width(inc_q),file=print_log)
print(get_inc_bit_width(bias_q),file=print_log)
f.write('layer6_inc\n')
bias_to_txt(inc_q,f)
f.write('layer6_bias\n')
bias_to_txt(bias_q,f)


#head
print("======================================================================",file=print_log)
weight = model_param[layers[28]].cpu().detach().numpy()
bias = model_param[layers[29]].cpu().detach().numpy()
mean = model_param[layers[30]].cpu().detach().numpy()
var = model_param[layers[31]].cpu().detach().numpy()
inc_q, bias_q=bn_act_quantize_int(weight,bias,mean,var)
print('layer7_inc:',inc_q,file=print_log)
print('layer7_bias:',bias_q,file=print_log)
print(get_inc_bit_width(inc_q),file=print_log)
print(get_inc_bit_width(bias_q),file=print_log)
f.write('layer7_inc\n')
bias_to_txt(inc_q,f)
f.write('layer7_bias\n')
bias_to_txt(bias_q,f)


#head
print("======================================================================",file=print_log)
weight = model_param[layers[33]].cpu().detach().numpy()
bias = model_param[layers[34]].cpu().detach().numpy()
mean = model_param[layers[35]].cpu().detach().numpy()
var = model_param[layers[36]].cpu().detach().numpy()
inc_q, bias_q=bn_act_quantize_int(weight,bias,mean,var)
print('layer9_inc:',inc_q,file=print_log)
print('layer9_bias:',bias_q,file=print_log)
print(get_inc_bit_width(inc_q),file=print_log)
print(get_inc_bit_width(bias_q),file=print_log)
f.write('layer9_inc\n')
bias_to_txt(inc_q,f)
f.write('layer9_bias\n')
bias_to_txt(bias_q,f)

#head
print("======================================================================",file=print_log)
weight = model_param[layers[38]].cpu().detach().numpy()
bias = model_param[layers[39]].cpu().detach().numpy()
mean = model_param[layers[40]].cpu().detach().numpy()
var = model_param[layers[41]].cpu().detach().numpy()
inc_q, bias_q=bn_act_quantize_int(weight,bias,mean,var)
print('layer11_inc:',inc_q,file=print_log)
print('layer11_bias:',bias_q,file=print_log)
print(get_inc_bit_width(inc_q),file=print_log)
print(get_inc_bit_width(bias_q),file=print_log)
f.write('layer11_inc\n')
bias_to_txt(inc_q,f)
f.write('layer11_bias\n')
bias_to_txt(bias_q,f)

print("======================================================================",file=print_log)
bias0 = model_param[layers[32]].cpu().detach().numpy()
bias1 = model_param[layers[37]].cpu().detach().numpy()
bias2 = model_param[layers[42]].cpu().detach().numpy()
print(bias0)
print(get_inc_bit_width(bias_int(bias0)),file=print_log)
print(bias1)
print(get_inc_bit_width(bias_int(bias1)),file=print_log)
print(bias2)
print(get_inc_bit_width(bias_int(bias2)),file=print_log)
for i in range(12):
    if i==0:
        print('{',bias_int(bias0)[i],end=',',file=f)
    elif i==11:
        print(bias_int(bias0)[i],end='},',file=f)
    else:
        print(bias_int(bias0)[i],end=',',file=f)
print('',file=f)
for i in range(12):
    if i==0:
        print('{',bias_int(bias1)[i],end=',',file=f)
    elif i==11:
        print(bias_int(bias1)[i],end='},',file=f)
    else:
        print(bias_int(bias1)[i],end=',',file=f)
print('',file=f)
for i in range(12):
    if i==0:
        print('{',bias_int(bias2)[i],end=',',file=f)
    elif i==11:
        print(bias_int(bias2)[i],end='},',file=f)
    else:
        print(bias_int(bias2)[i],end=',',file=f)
print('',file=f)