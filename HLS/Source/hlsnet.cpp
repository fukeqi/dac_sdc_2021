#include "hlsnet.h"
#include "param.h"
#include "weights.h"

static layer config[layer_count] = {
{ "conv0",   320,160,3,  320,160,32, 3,1,1},
{ "pool0",   320,160,32, 160,80,32,  2,2,0},
{ "conv1",   160,80,32,  160,80,32,  3,1,1},
{ "pool1",   160,80,32,  80,40,32,   2,2,0},
{ "conv2",   80,40,32,   80,40,64,   3,1,1},
{ "pool2",   80,40,64,   40,20,64,   2,2,0},
{ "conv3",   40,20,64,   40,20,64,   3,1,1},
{ "pool3",   40,20,64,   20,10,64,   2,2,0},
{ "conv4",   20,10,64,   20,10,64,   3,1,1},
{ "conv5",   20,10,64,   20,10,64,   3,1,1},
{ "conv6",   20,10,64,   20,10,64,   3,1,1},
{ "conv7",   20,10,64,   20,10,64,   3,1,1},
{ "conv8",   20,10,64,   20,10,36,   1,1,0},
};
/***************************
 * CONV3X3 MODULE  in:32 out:32
 **************************/
template<unsigned A_BIT>
void CONV3X3(ap_uint<A_BIT> IFM[16][23][43], RDT OFM[32][23][43], WDT WBUF3x3[32][16][3][3])
{
#pragma HLS ARRAY_PARTITION variable=OFM dim=1 complete
#pragma HLS ARRAY_PARTITION variable=IFM dim=1 complete
#pragma HLS ARRAY_PARTITION variable=WBUF3x3 dim=1 complete
#pragma HLS ARRAY_PARTITION variable=WBUF3x3 dim=2 complete

	for(int kh=0; kh<3; kh++)
	{
		for(int kw=0; kw<3; kw++)
		{
			for(int h=1; h<22; h++)
			{
				for(int w=1; w<42; w++)
				{
				#pragma HLS PIPELINE II=1
					for(int co=0; co<32; co++)
					{
						for(int ci=0; ci<16; ci++)
						{
							OFM[co][h][w] += IFM[ci][h+kh-1][w+kw-1]*WBUF3x3[co][ci][kh][kw];
						}
					}
				}
			}
		}
	}

}

/********************************
 * CONV1x1 MODULE
 ********************************/
ap_int<12> MAC16(
	WDT w0,  ADT b0,
	WDT w1,  ADT b1,
	WDT w2,  ADT b2,
	WDT w3,  ADT b3,
	WDT w4,  ADT b4,
	WDT w5,  ADT b5,
	WDT w6,  ADT b6,
	WDT w7,  ADT b7,
	WDT w8,  ADT b8,
	WDT w9,  ADT b9,
	WDT w10, ADT b10,
	WDT w11, ADT b11,
	WDT w12, ADT b12,
	WDT w13, ADT b13,
	WDT w14, ADT b14,
	WDT w15, ADT b15)
{
#pragma HLS INLINE off
	ap_int<8> mul0, mul1, mul2,  mul3,  mul4,  mul5,  mul6,  mul7;
	ap_int<8> mul8, mul9, mul10, mul11, mul12, mul13, mul14, mul15;
	ap_int<9> add0, add1, add2, add3, add4,  add5, add6, add7;
	ap_int<10> add8, add9, add10, add11;
    ap_int<11> add12, add13;
    ap_int<12> res;

    mul0 = w0 * b0;
    mul1 = w1 * b1;
    mul2 = w2 * b2;
    mul3 = w3 * b3;
    mul4 = w4 * b4;
    mul5 = w5 * b5;
    mul6 = w6 * b6;
    mul7 = w7 * b7;
    mul8 = w8 * b8;
    mul9 = w9 * b9;
    mul10 = w10 * b10;
    mul11 = w11 * b11;
    mul12 = w12 * b12;
    mul13 = w13 * b13;
    mul14 = w14 * b14;
    mul15 = w15 * b15;

    add0 = mul0 + mul1;
    add1 = mul2 + mul3;
    add2 = mul4 + mul5;
    add3 = mul6 + mul7;
    add4 = mul8 + mul9;
    add5 = mul10 + mul11;
    add6 = mul12 + mul13;
    add7 = mul14 + mul15;

    add8 = add0 + add1;
    add9 = add2 + add3;
    add10 = add4 + add5;
    add11 = add6 + add7;

    add12 = add8 + add9;
    add13 = add10 + add11;

    res = add12 + add13;

    return res;
}


void LOAD_W1x1(const WDT WBUF1x1[12][64], WDT W1x1[12][16], int CI)
{
#pragma HLS INLINE off
#pragma HLS PIPELINE
	for(int ci=0; ci<16; ci++){
		for(int co=0; co<12; co++){
			W1x1[co][ci] = WBUF1x1[co][ci+CI];
		}
	}
}

void CONV1X1(ADT IFM[64][23][43], RDT OFM[12][23][43], const WDT WBUF1x1[12][64])
{
#pragma HLS ARRAY_PARTITION variable=OFM dim=1 complete
#pragma HLS ARRAY_PARTITION variable=IFM dim=1 complete
#pragma HLS ARRAY_PARTITION variable=WBUF1x1 dim=1 complete
	WDT W1x1[12][16];
#pragma HLS ARRAY_PARTITION variable=W1x1 dim=1 complete
#pragma HLS ARRAY_PARTITION variable=W1x1 dim=2 complete

	for(int ci=0; ci<64; ci+=16)
	{
		LOAD_W1x1(WBUF1x1, W1x1, ci);
		for(int h=1; h<22; h++)
		{
			for(int w=1; w<42; w++)
			{
			#pragma HLS PIPELINE II=1
				for(int co=0; co<12; co++)
				{
					for(int c=0; c<16; c++)
					{
						OFM[co][h][w] += W1x1[co][c]*IFM[c+ci][h][w];
					}

				}
			}
		}
	}

}
/*********************************
 * Max Pooling
 ********************************/
ADT MAX(ADT a, ADT b, ADT c, ADT d)
{
#pragma HLS INLINE
	ADT t1 = a > b ? a : b;
	ADT t2 = c > d ? c : d;
	return t1 > t2 ? t1 : t2;
}

void POOL(ADT32* fm, ADT IFM[32][23][43], int Hx, int Wx, int Cx, int ow, int oh)
{
#pragma HLS ARRAY_PARTITION variable=IFM dim=1 complete
	int tile = ow/20;
	int h_o = Hx*10 + Hx/tile;
	int w_o = Wx*20 + Wx/tile;
	for (int h=1; h<=10; h++)
	{
		for (int w=1; w<=20; w++)
		{
#pragma HLS PIPELINE II=4
			int fm_index = Cx*(oh*2+3)*(ow*2+3) + (h+h_o)*(ow*2+3) + (w+w_o);
			ADT32 DATA;
			for (int c=0; c<32; c++)
			{
				DATA.range(4*c+3,4*c) = MAX(IFM[c][2*h-1][2*w-1],IFM[c][2*h-1][2*w],IFM[c][2*h][2*w-1],IFM[c][2*h][2*w]);
			}
			fm[fm_index] = DATA;
		}
	}
}
void POOL2FM(ADT FM[64][23][43], ADT IFM[32][23][43], int Hx, int Wx, int Cx)
{
#pragma HLS ARRAY_PARTITION variable=IFM dim=1 complete
#pragma HLS ARRAY_PARTITION variable=FM dim=1 complete

	int h_o = Hx*10 + Hx;
	int w_o = Wx*20 + Wx;
	for (int h=1; h<=10; h++)
	{
		for (int w=1; w<=20; w++)
		{
#pragma HLS PIPELINE
			for (int c=0; c<32; c++)
			{
				FM[c+Cx*32][h+h_o][w+w_o] = MAX(IFM[c][2*h-1][2*w-1],IFM[c][2*h-1][2*w],IFM[c][2*h][2*w-1],IFM[c][2*h][2*w]);
			}
		}
	}
}

/*********************************
 * ReLu
 ********************************/
void Load_BBUF(const BDT bias[][32], BDT BBUF[32], int Mx)
{
#pragma HLS ARRAY_PARTITION variable=bias complete dim=2
#pragma HLS ARRAY_PARTITION variable=BBUF complete dim=1
#pragma HLS PIPELINE
	for(int c=0; c<32; c++){
		BBUF[c] = bias[Mx][c];
	}
}
void Load_MBUF(const MDT bias[][32], MDT MBUF[32], int Mx)
{
#pragma HLS ARRAY_PARTITION variable=bias complete dim=2
#pragma HLS ARRAY_PARTITION variable=MBUF complete dim=1
#pragma HLS PIPELINE
	for(int c=0; c<32; c++){
		MBUF[c] = bias[Mx][c];
	}
}

void ACTIVATION(RDT IFM[32][23][43], ADT OFM[32][23][43], const BDT BBUF[32], const MDT MBUF[32])
{
#pragma HLS INLINE off
#pragma HLS ARRAY_PARTITION variable=BBUF dim=1 complete
#pragma HLS ARRAY_PARTITION variable=MBUF dim=1 complete

	const unsigned D = 1 << 15;

	for (int h = 1; h < 22; h++)
	{
		for (int w = 1; w < 42; w++)
		{
#pragma HLS PIPELINE
            for (int c = 0; c < 32; c++)
            {
                int qy = IFM[c][h][w]*MBUF[c]+BBUF[c];
                if (qy > 0) {
					qy = (qy+(D>>1)) >> 15;
					if (qy > 15){
						OFM[c][h][w] = ADT(15);
					} else {
						OFM[c][h][w] = ADT(qy);
					}
				} else {
					OFM[c][h][w] = ADT(0);
				}
                IFM[c][h][w] = 0;
            }
        }
	}
}


/******************************************************
 * Load image
 *****************************************************/
/*
void Load_IMG(ADT4* img, ap_uint<8> IFM[3][23][43], int Hx, int Wx, int b)
{
    int h_o = Hx*20-1;
    int w_o = Wx*40-1;
    for (int h=0; h<22; h++)
    {
        for (int w=0; w<42; w++)
        {
#pragma HLS PIPELINE II=1
            ADT4 DATA = img[b*320*160 + (h+h_o)*320 + (w+w_o)];  //ADT4 DATA = img[b*320*160 + (h+h_o)*320 + (w+w_o)];
            for (int c=0; c<3; c++)
            {
                if (h+h_o<0||w+w_o<0||h+h_o>159||w+w_o>319)
                    IFM[c][h][w] = 0;
                else
                    IFM[c][h][w] = DATA.range((c<<3)+7,(c<<3));  //quantization 4bit
            }
        }
    }
}
*/
ADT img_norm_buf[256]={
0 , 0 , 0 , 0 , 0 , 0 , 0 , 0 , 0 , 1 , 1 , 1 , 1 , 1 , 1 , 1 ,
1 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 2 , 2 , 2 , 2 , 2 , 2 ,
2 , 2 , 2 , 2 , 2 , 2 , 2 , 2 , 2 , 2 , 2 , 3 , 3 , 3 , 3 , 3 ,
3 , 3 , 3 , 3 , 3 , 3 , 3 , 3 , 3 , 3 , 3 , 3 , 4 , 4 , 4 , 4 ,
4 , 4 , 4 , 4 , 4 , 4 , 4 , 4 , 4 , 4 , 4 , 4 , 4 , 5 , 5 , 5 ,
5 , 5 , 5 , 5 , 5 , 5 , 5 , 5 , 5 , 5 , 5 , 5 , 5 , 5 , 6 , 6 ,
6 , 6 , 6 , 6 , 6 , 6 , 6 , 6 , 6 , 6 , 6 , 6 , 6 , 6 , 6 , 7 ,
7 , 7 , 7 , 7 , 7 , 7 , 7 , 7 , 7 , 7 , 7 , 7 , 7 , 7 , 7 , 7 ,
8 , 8 , 8 , 8 , 8 , 8 , 8 , 8 , 8 , 8 , 8 , 8 , 8 , 8 , 8 , 8 ,
8 , 9 , 9 , 9 , 9 , 9 , 9 , 9 , 9 , 9 , 9 , 9 , 9 , 9 , 9 , 9 ,
9 , 9 , 10 , 10 , 10 , 10 , 10 , 10 , 10 , 10 , 10 , 10 , 10 , 10 , 10 , 10 ,
10 , 10 , 10 , 11 , 11 , 11 , 11 , 11 , 11 , 11 , 11 , 11 , 11 , 11 , 11 , 11 ,
11 , 11 , 11 , 11 , 12 , 12 , 12 , 12 , 12 , 12 , 12 , 12 , 12 , 12 , 12 , 12 ,
12 , 12 , 12 , 12 , 12 , 13 , 13 , 13 , 13 , 13 , 13 , 13 , 13 , 13 , 13 , 13 ,
13 , 13 , 13 , 13 , 13 , 13 , 14 , 14 , 14 , 14 , 14 , 14 , 14 , 14 , 14 , 14 ,
14 , 14 , 14 , 14 , 14 , 14 , 14 , 15 , 15 , 15 , 15 , 15 , 15 , 15 , 15 , 15
};
void Load_IMG(ADT4* img, ADT IFM[16][23][43], int Hx, int Wx, int b)
{
    int h_o = Hx*20-1;
    int w_o = Wx*40-1;
    for (int h=0; h<22; h++)
    {
        for (int w=0; w<42; w++)
        {
#pragma HLS PIPELINE II=1
        	ADT4 DATA = img[b*320*160 + (h+h_o)*320 + (w+w_o)];
            for (int c=0; c<3; c++)
            {
                if (h+h_o<0||w+w_o<0||h+h_o>159||w+w_o>319)
                    IFM[c][h][w] = 0;
                else
                    IFM[c][h][w] = img_norm_buf[DATA.range((c<<3)+7,(c<<3)+0)];  //quantization 4bit
            }
        }
    }
}

/******************************************************
 * Load fm: fm_size>40x20
 *****************************************************/
void Load_FM(ADT32* ifm, ADT IFM1[16][23][43], ADT IFM2[16][23][43], int Hx, int Wx, int Cx, int ow, int oh)
{
//#pragma HLS ARRAY_PARTITION variable=IFM dim=1 complete
    int tile = ow/40;
    int h_o, w_o;
    if(tile)
    {
        h_o = Hx*20 + Hx/tile;
        w_o = Wx*40 + Wx/tile;
    }
    else
    {
        h_o = 0;
        w_o = 0;
    }

    for (int h=0; h<22; h++)
    {
        for (int w=0; w<42; w++)
        {
#pragma HLS PIPELINE II=1
            int ifm_index = Cx*(oh*2+3)*(ow*2+3) + (h+h_o)*(ow*2+3) + (w+w_o);
            ADT32 DATA;
            DATA = ifm[ifm_index];
            for (int c=0; c<16; c++)
            {

				IFM1[c][h][w] = DATA.range(c*4+3,c*4);

				IFM2[c][h][w] = DATA.range((c+16)*4+3,(c+16)*4);

            }
        }
    }
}
/******************************************************
 * Load fm: fm_size<80x40
 *****************************************************/
void Load_FM1(ADT FM[64][23][43], ADT IFM[16][23][43], int Cx)
{
#pragma HLS ARRAY_PARTITION variable=IFM dim=1 complete
#pragma HLS ARRAY_PARTITION variable=FM dim=1 complete
    for (int h=0; h<23; h++)
    {
        for (int w=0; w<43; w++)
        {
#pragma HLS PIPELINE II=1
            for (int c=0; c<16; c++)
            {
                IFM[c][h][w] = FM[c+Cx*16][h][w];
            }
        }
    }
}

/******************************************************
 * export fm :fm_size<80x40
 *****************************************************/
void Export_FM1(ADT FM[64][23][43], ADT OFM[32][23][43], int Cx)
{
#pragma HLS ARRAY_PARTITION variable=FM dim=1 complete
#pragma HLS ARRAY_PARTITION variable=OFM dim=1 complete
    for (int h=0; h<23; h++)
    {
#pragma HLS PIPELINE II=1
        for (int c=0; c<32; c++)
        {
            OFM[c][h][21] = 0;
        }
    }

    for (int w=0; w<43; w++)
    {
#pragma HLS PIPELINE II=1
        for (int c=0; c<32; c++)
        {
            OFM[c][11][w] = 0;
        }
    }

    for (int h=1; h<22; h++)
    {
        for (int w=1; w<42; w++)
        {
#pragma HLS PIPELINE II=1

            for (int c=0; c<32; c++)
            {
                FM[c+Cx*32][h][w] = OFM[c][h][w];
            }

        }
    }
}

/******************************************************
 * Load Weight
 *****************************************************/
void Load_WBUF3x3(const WDT16 (*weights)[9], WDT WBUF3x3[32][16][3][3], int Mx, int Nx, int IC)
{
	int tmp = (IC>>4)<<5;
    for(int m=0; m<3; m++)
    {
        for(int n=0; n<3; n++)
        {
#pragma HLS PIPELINE II=1
        	for (int co=0; co<32; co++)
        	{
        		WDT16 DATA;
        		DATA = weights[co+Nx*32+Mx*tmp][m*3+n];
				for(int ci=0; ci<16; ci++)
				{
					WBUF3x3[co][ci][m][n] = DATA.range((15-ci)*4+3,(15-ci)*4);
				}
        	}
        }
    }
}


/********************************************
 * Compute_bbox
 *******************************************/

void Compute_BBOX(RDT FMO[3][12][23][43], const int last_bias[3][12], ap_int<32> BBOX[4][26])
{
#pragma HLS ARRAY_PARTITION variable=FMO complete dim=1
#pragma HLS ARRAY_PARTITION variable=FMO complete dim=2
#pragma HLS ARRAY_PARTITION variable=last_bias complete dim=0
#pragma HLS ARRAY_PARTITION variable=BBOX complete dim=2
    int H,W,i;
    int hmax, wmax, max_conf;

    for(int b=0; b<4; b++)
    {
    	int conf[10][20]={0};
        switch(b)
        {
            case 0: H=1; W=1; break;
            case 1: H=1; W=22; break;
            case 2: H=12; W=1; break;
            case 3: H=12; W=22; break;
        }
        hmax = H;
        wmax = W;

        for(int h=0; h<10; h++){
			for(int w=0; w<20; w++){
				for(i=0; i<6; i++){
#pragma HLS PIPELINE II=6
					conf[h][w] += FMO[2][i*2][h+H][w+W];
				}
			}
        }
        max_conf = conf[0][0];
        for(int h=0; h<10; h++){
			for(int w=0; w<20; w++){
#pragma HLS PIPELINE II=1
				if(conf[h][w]>max_conf){
					max_conf = conf[h][w];
					hmax = h+H;
					wmax = w+W;
				}
				conf[h][w] = 0;
			}
		}
        for(i=0; i<12; i++)  //xy
        {
#pragma HLS UNROLL
        	BBOX[b][i]=(ap_int<32>(FMO[0][i][hmax][wmax])<<8)+last_bias[0][i];
        }
        for(i=12; i<24; i++) //wh
		{
#pragma HLS UNROLL
			BBOX[b][i]=(ap_int<32>(FMO[1][i-12][hmax][wmax])<<8)+last_bias[1][i-12];
			if(b==0)
				std::cout<<BBOX[b][i]<<std::endl;
		}
        BBOX[b][24] = wmax-W;
        BBOX[b][25] = hmax-H;
        //std::cout<<"wmax,hmax"<<BBOX[b][24]<<" "<<BBOX[b][25]<<std::endl;

    }
}

void Export_BBOX(ADT4* bbox, ap_int<32> BBOX[4][26], RDT FMO[3][12][23][43])
{
#pragma HLS ARRAY_PARTITION variable=FMO complete dim=1
#pragma HLS ARRAY_PARTITION variable=FMO complete dim=2
	for(int i=0; i<4; i++)
	{
		for(int j=0; j<26; j++)
		{
			int index = i*26 + j;
			bbox[index] = BBOX[i][j];
		}
	}
	for(int h=0; h<23; h++)
	{
		for(int w=0; w<43; w++)
		{
#pragma HLS PIPELINE
			for(int co=0;co<3;co++)
			{
				for(int ci=0; ci<12; ci++)
				{
					FMO[co][ci][h][w]=0;
				}
			}
		}
	}
}
/******************************************************
 *  HlsNet
 *****************************************************/
ADT FM1[16][23][43];
ADT FM2[16][23][43];
ADT FM3[32][23][43];
RDT FM4[32][23][43];
RDT FMO[3][12][23][43];

ADT FMX[64][23][43];
ADT FMY[64][23][43];


WDT WBUF3x3[32][16][3][3];
BDT BBUF[32];
MDT MBUF[32];

ap_int<32> BBOX[4][26];

void cout_FMA(ADT FM[16][23][43]){
	std::cout<<"FMA"<<std::endl;

	for(int i=0;i<23;i++){
		for(int j=0;j<43;j++){
			std::cout<<FM[1][i][j]<<" ";
		}
		std::cout<<std::endl;
	}


	std::cout<<std::endl;
}

void cout_FMR(RDT FM[32][23][43]){
	std::cout<<"FMR"<<std::endl;

	for(int i=0;i<23;i++){
		for(int j=0;j<43;j++){
			std::cout<<FM[0][i][j]<<" ";
		}
		std::cout<<std::endl;
	}


	std::cout<<std::endl;
}

void cout_FMACT(ADT FM[32][23][43]){
	std::cout<<"FMACT"<<std::endl;

	for(int i=0;i<23;i++){
		for(int j=0;j<43;j++){
			std::cout<<FM[1][i][j]<<" ";
		}
		std::cout<<std::endl;
	}


	std::cout<<std::endl;
}

void HlsNet(ADT4* img, ADT32* fm, ADT4* bbox)
{

#pragma HLS ALLOCATION instances=Export_FM1 limit=1 function
#pragma HLS ALLOCATION instances=Load_FM1 limit=1 function
#pragma HLS ALLOCATION instances=Load_FM limit=1 function
#pragma HLS ALLOCATION instances=Load_IMG limit=1 function
#pragma HLS ALLOCATION instances=ACTIVATION limit=1 function
#pragma HLS ALLOCATION instances=POOL limit=1 function
#pragma HLS ALLOCATION instances=POOL2FM limit=1 function
#pragma HLS ALLOCATION instances=CONV1X1 limit=1 function
#pragma HLS ALLOCATION instances=Compute_BBOX limit=1 function

#pragma HLS INTERFACE m_axi depth=204800 port=img    offset=slave bundle=fm //320x160x4 204800
#pragma HLS INTERFACE m_axi depth=73316 port=fm     offset=slave bundle=fm
#pragma HLS INTERFACE m_axi depth=104 port=bbox 	offset=slave bundle = bbox
#pragma HLS INTERFACE s_axilite register port=return

	/***********************CONV0(3,32,3,3)********************************/
	//input: 320x2,160x2
	std::cout << "CONV0(3,32,3,3)" << std::endl;
	{
		Load_WBUF3x3(conv0_w, WBUF3x3, 0, 0, 16);
		Load_BBUF(conv0_b, BBUF, 0);
		Load_MBUF(conv0_m, MBUF, 0);
		for(int b=0; b<4; b++)
		{
			int H, W;
			switch(b)
			{
				case 0: H=0; W=0; break;
				case 1: H=0; W=8; break;
				case 2: H=8; W=0; break;
				case 3: H=8; W=8; break;
			}
			for(int Hx=0; Hx<8; Hx++)
			{
				for(int Wx=0; Wx<8; Wx++)
				{
					Load_IMG(img, FM1, Hx, Wx, b);
					CONV3X3<4>(FM1, FM4, WBUF3x3);
					ACTIVATION(FM4, FM3, BBUF, MBUF);
					POOL(fm + pool0_o, FM3, Hx+H, Wx+W, 0, config[1].ow, config[1].oh);


				}
			}
		}
	}
	/***********************CONV1(32,32,3,3)********************************/
	//input: 160x2,80x2
	std::cout << "CONV1(32,32,3,3)" << std::endl;
	{
		for(int Hx=0; Hx<8; Hx++)
		{
			for(int Wx=0; Wx<8; Wx++)
			{
				Load_FM(fm + pool0_o, FM1, FM2, Hx, Wx, 0, config[1].ow, config[1].oh);
				Load_WBUF3x3(conv1_w, WBUF3x3, 0, 0, config[2].ic);
				CONV3X3<4>(FM1, FM4, WBUF3x3);
				Load_WBUF3x3(conv1_w, WBUF3x3, 0, 1, config[2].ic);
				CONV3X3<4>(FM2, FM4, WBUF3x3);

				Load_BBUF(conv1_b, BBUF, 0);
				Load_MBUF(conv1_m, MBUF, 0);
				ACTIVATION(FM4, FM3, BBUF, MBUF);
				POOL(fm + pool1_o, FM3, Hx, Wx, 0, config[3].ow, config[3].oh);

			}
		}
	}
	/***********************CONV2(32,64,3,3)********************************/
	//input: 80x2,40x2
	std::cout << "CONV2(32,64,3,3)" << std::endl;
	{
		for(int Hx=0; Hx<4; Hx++)
		{
			for(int Wx=0; Wx<4; Wx++)
			{
				Load_FM(fm + pool1_o, FM1, FM2, Hx, Wx, 0, config[3].ow, config[3].oh);
				for(int Mx=0; Mx<2; Mx++)
				{
					Load_WBUF3x3(conv2_w, WBUF3x3, Mx, 0, config[4].ic);
					CONV3X3<4>(FM1, FM4, WBUF3x3);
					Load_WBUF3x3(conv2_w, WBUF3x3, Mx, 1, config[4].ic);
					CONV3X3<4>(FM2, FM4, WBUF3x3);


					Load_BBUF(conv2_b, BBUF, Mx);
					Load_MBUF(conv2_m, MBUF, Mx);
					ACTIVATION(FM4, FM3, BBUF, MBUF);
					POOL(fm + pool2_o, FM3, Hx, Wx, Mx, config[5].ow, config[5].oh);

				}
			}
		}
	}
	/***********************CONV3(64,64,3,3)********************************/
	//input: 40x2,20x2
	std::cout << "CONV3(64,64,3,3)" << std::endl;
	{
		for(int Hx=0; Hx<2; Hx++)
		{
			for(int Wx=0; Wx<2; Wx++)
			{
				for(int Mx=0; Mx<2; Mx++)
				{
					for(int Nx=0; Nx<2; Nx++)
					{
						Load_FM(fm + pool2_o, FM1, FM2, Hx, Wx, Nx, config[5].ow, config[5].oh);
						Load_WBUF3x3(conv3_w, WBUF3x3, Mx, 2*Nx, config[6].ic);
						CONV3X3<4>(FM1, FM4, WBUF3x3);
						Load_WBUF3x3(conv3_w, WBUF3x3, Mx, 2*Nx+1, config[6].ic);
						CONV3X3<4>(FM2, FM4, WBUF3x3);

					}
					Load_BBUF(conv3_b, BBUF, Mx);
					Load_MBUF(conv3_m, MBUF, Mx);
					ACTIVATION(FM4, FM3, BBUF, MBUF);
					POOL2FM(FMX, FM3, Hx, Wx, Mx);
				}
			}
		}
	}
	/***********************CONV4~6(64,64,3,3)********************************/
	//input: 20x2,10x2
	std::cout << "CONV4(64,64,3,3)" << std::endl;
	{
		for(int Mx=0; Mx<2; Mx++)
		{
			Load_FM1(FMX, FM1, 0);
			for(int Nx=0; Nx<4; Nx++)
			{
				Load_WBUF3x3(conv4_w, WBUF3x3, Mx, Nx, config[8].ic);
				if(Nx%2==0)
				{
					Load_FM1(FMX, FM2, Nx+1);
					CONV3X3<4>(FM1, FM4, WBUF3x3);
				}
				else
				{
					Load_FM1(FMX, FM1, Nx+1);
					CONV3X3<4>(FM2, FM4, WBUF3x3);
				}
			}
			Load_BBUF(conv4_b, BBUF, Mx);
			Load_MBUF(conv4_m, MBUF, Mx);
			ACTIVATION(FM4, FM3, BBUF, MBUF);
			Export_FM1(FMY, FM3, Mx);
		}
	}
	std::cout << "CONV5(64,64,3,3)" << std::endl;
	{
		for(int Mx=0; Mx<2; Mx++)
		{
			Load_FM1(FMY, FM1, 0);
			for(int Nx=0; Nx<4; Nx++)
			{
				Load_WBUF3x3(conv5_w, WBUF3x3, Mx, Nx, config[9].ic);
				if(Nx%2==0)
				{
					Load_FM1(FMY, FM2, Nx+1);
					CONV3X3<4>(FM1, FM4, WBUF3x3);
				}
				else
				{
					Load_FM1(FMY, FM1, Nx+1);
					CONV3X3<4>(FM2, FM4, WBUF3x3);
				}
			}
			Load_BBUF(conv5_b, BBUF, Mx);
			Load_MBUF(conv5_m, MBUF, Mx);
			ACTIVATION(FM4, FM3, BBUF, MBUF);
			Export_FM1(FMX, FM3, Mx);
		}
	}
	std::cout << "CONV6(64,64,3,3)" << std::endl;
	{
		for(int Mx=0; Mx<2; Mx++)
		{
			Load_FM1(FMX, FM1, 0);
			for(int Nx=0; Nx<4; Nx++)
			{
				Load_WBUF3x3(conv6_w, WBUF3x3, Mx, Nx, config[10].ic);
				if(Nx%2==0)
				{
					Load_FM1(FMX, FM2, Nx+1);
					CONV3X3<4>(FM1, FM4, WBUF3x3);
				}
				else
				{
					Load_FM1(FMX, FM1, Nx+1);
					CONV3X3<4>(FM2, FM4, WBUF3x3);
				}
			}
			Load_BBUF(conv6_b, BBUF, Mx);
			Load_MBUF(conv6_m, MBUF, Mx);
			ACTIVATION(FM4, FM3, BBUF, MBUF);
			Export_FM1(FMY, FM3, Mx);
		}
	}
	/***********************head1********************************/
	std::cout << "head1" << std::endl;
	{
		for(int Mx=0; Mx<2; Mx++)
		{
			Load_FM1(FMY, FM1, 0);
			for(int Nx=0; Nx<4; Nx++)
			{
				Load_WBUF3x3(head1_w1, WBUF3x3, Mx, Nx, config[11].ic);
				if(Nx%2==0)
				{
					Load_FM1(FMY, FM2, Nx+1);
					CONV3X3<4>(FM1, FM4, WBUF3x3);
				}
				else
				{
					Load_FM1(FMY, FM1, Nx+1);
					CONV3X3<4>(FM2, FM4, WBUF3x3);
				}
			}
			Load_BBUF(head1_b, BBUF, Mx);
			Load_MBUF(head1_m, MBUF, Mx);
			ACTIVATION(FM4, FM3, BBUF, MBUF);
			Export_FM1(FMX, FM3, Mx);
		}
	}
	{
		CONV1X1(FMX, FMO[0], head1_w2);
	}


	/***********************head2********************************/
	std::cout << "head2" << std::endl;
	{
		for(int Mx=0; Mx<2; Mx++)
		{
			Load_FM1(FMY, FM1, 0);
			for(int Nx=0; Nx<4; Nx++)
			{
				Load_WBUF3x3(head2_w1, WBUF3x3, Mx, Nx, config[11].ic);
				if(Nx%2==0)
				{
					Load_FM1(FMY, FM2, Nx+1);
					CONV3X3<4>(FM1, FM4, WBUF3x3);
				}
				else
				{
					Load_FM1(FMY, FM1, Nx+1);
					CONV3X3<4>(FM2, FM4, WBUF3x3);
				}
			}
			Load_BBUF(head2_b, BBUF, Mx);
			Load_MBUF(head2_m, MBUF, Mx);
			ACTIVATION(FM4, FM3, BBUF, MBUF);
			Export_FM1(FMX, FM3, Mx);
		}
	}
	{
		CONV1X1(FMX, FMO[1], head2_w2);
	}

	/***********************head3********************************/
	std::cout << "head3" << std::endl;
	{
		for(int Mx=0; Mx<2; Mx++)
		{
			Load_FM1(FMY, FM1, 0);
			for(int Nx=0; Nx<4; Nx++)
			{
				Load_WBUF3x3(head3_w1, WBUF3x3, Mx, Nx, config[11].ic);
				if(Nx%2==0)
				{
					Load_FM1(FMY, FM2, Nx+1);
					CONV3X3<4>(FM1, FM4, WBUF3x3);
				}
				else
				{
					Load_FM1(FMY, FM1, Nx+1);
					CONV3X3<4>(FM2, FM4, WBUF3x3);
				}
			}
			Load_BBUF(head3_b, BBUF, Mx);
			Load_MBUF(head3_m, MBUF, Mx);
			ACTIVATION(FM4, FM3, BBUF, MBUF);
			Export_FM1(FMX, FM3, Mx);
		}
	}
	{
		CONV1X1(FMX, FMO[2], head3_w2);
	}

	Compute_BBOX(FMO, last_bias, BBOX);
	Export_BBOX(bbox, BBOX, FMO);

}






















