#ifndef HLSNET_H
#define HLSNET_H
#include <stdlib.h>
#include <stdio.h>
#include <iostream>
#include <fstream>
#include <vector>
#include <memory.h>
#include <time.h>
#include <cstring>
#include <math.h>
#include "ap_int.h"
#include <ap_fixed.h>
#include <hls_video.h>

using namespace std;





#ifdef __SDSCC__
#include "sds_lib.h"
#else
#define sds_alloc malloc
#define sds_free free
#endif

#define __AP_INT__
#ifdef __AP_INT__
typedef ap_uint<4> ADT;
typedef ap_int<18> RDT;  //14
typedef ap_int<24> BDT;
typedef ap_int<4>  WDT;
typedef ap_int<16> MDT;

typedef ap_uint<1> BADT;
typedef ap_uint<1> BWDT;
typedef ap_int<6>  BRDT;

typedef ap_int<256> WDT32;  //8bit*32
typedef ap_int<64> WDT16;  //4bit*16
typedef ap_int<128>  ADT32;  //4bit*32 -->ddr
typedef ap_int<32>  ADT4;   //8bit*4
typedef ap_int<24>  ADT3;   //8bit*3

typedef ap_int<256> BDT16;  //16bit*16

#endif

#define layer_count 17
struct layer
{
	char name[10];
	int iw, ih, ic, ow, oh, oc;
	int k, s, p;
};


#define pool0_o  	0
#define pool1_o  	52649 	//(160X2+3)*(80X2+3)*1
#define pool2_o  	66178   // 52649+(80x2+3)*(40x2+3)*1
#define fm_all  	73316   // 66178+(40x2+3)*(20x2+3)*2


/******************utils.cpp************************/
//void load_fm(ADT* fm, layer l);
void load_weight(WDT32 *weight, int length);
void load_img(ADT4* img, int length);

//void check(ADT* result, ADT* golden, int len, layer l);
//void check_fm(ADT* fm, layer l);
//void check_bbox(BDT* bbox, layer l);
//void show_fm(ADT* fm, layer l);


void HlsNet(ADT4* img, ADT32* fm, ADT4* bbox);


#endif
