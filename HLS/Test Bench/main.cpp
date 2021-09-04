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
float sigmoid(float x)
{
	return 1/(1+exp(-x));
}

void Compute_BBOX(ADT4* bbox_origin)
{
	int i;
	float xs,ys,ws,hs;
    unsigned int bbox[4][4];

	for(int b=0; b<4; b++)
	{
		xs=0;ys=0;ws=0;hs=0;
		for(i=0; i<12; i+=2)
		{
			xs += sigmoid((float)bbox_origin[b*26+i]/105/256);
			ys += sigmoid((float)bbox_origin[b*26+i+1]/105/256);
		}
		for(i=12; i<24; i+=2)
		{
			ws += exp((float)bbox_origin[b*26+i]/105/256);
			hs += exp((float)bbox_origin[b*26+i+1]/105/256);
		}

		xs = xs/6 + (float)bbox_origin[b*26+24];
		ys = ys/6 + (float)bbox_origin[b*26+25];
		ws = ws/6;
		hs = hs/6;


		xs = xs*16*640/320;
		ys = ys*16*360/160;
		ws = ws*20*640/320;
		hs = hs*20*360/160;
		bbox[b][0] = (unsigned int)((xs - ws/2.0)); //xmin
		bbox[b][1] = (unsigned int)((ys - hs/2.0)); //ymin
		bbox[b][2] = (unsigned int)((xs + ws/2.0)); //xmax
		bbox[b][3] = (unsigned int)((ys + hs/2.0)); //ymax
	}

	for(int b=0; b<4; b++)
	{
		printf("img %d xmin: %d, ymin: %d, xmax: %d, ymax: %d\n", b, bbox[b][0], bbox[b][1], bbox[b][2], bbox[b][3]);
	}
}

void fm_DT32_2_DT(ADT32* in, ADT* out, layer l)
{
	for (int Mx = 0; Mx < l.oc / 32; Mx++)
	{
		for (int i = 0; i < (2*l.oh+3)*(2*l.ow+3); i++)
		{
			for (int tm = 0; tm < 32; tm++)
			{
				out[(tm+Mx*32)*(2*l.oh+3)*(2*l.ow+3) + i] = in[Mx*(2*l.oh+3)*(2*l.ow+3)+i].range(4*tm+3,4*tm);
			}
		}
	}
}

void distitch(ADT* ifm, ADT* ofm[4], layer l)
{
    int offset_h[4];
    offset_h[0] = offset_h[1] = 1;
    offset_h[2] = offset_h[3] = l.oh + 2;
    int offset_w[4];
    offset_w[0] = offset_w[2] = 1;
    offset_w[1] = offset_w[3] = l.ow + 2;

    for(int p=0; p<4; p++)
    {
        for(int c=0; c<l.oc; c++)
        {
            for(int hh=0; hh<l.oh; hh++)
            {
                for(int ww=0; ww<l.ow; ww++)
                {
                    int ifm_index = c*(l.oh*2+3)*(l.ow*2+3) + (hh+offset_h[p])*(l.ow*2+3) + (ww+offset_w[p]);
                    int ofm_index = c*l.oh*l.ow + hh*l.ow + ww;
                    ofm[p][ofm_index] = ifm[ifm_index];
                }
            }
        }
    }
}

void check_fm(ADT* fm, layer l)
{
    int len = l.oc*l.ow*l.oh;
    ap_uint<8> *tmp = (ap_uint<8> *)malloc(sizeof(ap_uint<8>)*len);

    char nstr[50];
    sprintf(nstr, "%s.bin", l.name);
    FILE *fp = fopen(nstr, "rb");
    fread(tmp, 1, len*sizeof(ap_uint<8>), fp);
    fclose(fp);

    int err = 0;
    int zero;
    for(int c=0; c<l.oc; c++)
    {
        for(int hh=0; hh<l.oh; hh++)
        {
            for(int ww=0; ww<l.ow; ww++)
            {
                int index = c*l.oh*l.ow + hh*l.ow + ww;
                if (fm[index]!=tmp[index])
                {
                    err++;
                }
            }
        }
    }

    if (err > 0)
        std::cout<<err<<std::endl;
    else
    	std::cout<<"correct"<<std::endl;

    free(tmp);
}

int main()
{

	//*************************************init *********************************
	printf("init HlsNet \n");
	ap_int<32> BBOX[4][26];
	ADT4* img;
	ADT4* bbox;
	ADT* data[4];
	ADT* ofm_blob;
	ADT32* ofm_blob32;
	ADT* ofm[4];
	for(int p=0; p<4; p++)
	{
		ofm[p] = (ADT*)sds_alloc(64*80*160*sizeof(ADT));
	}

	img = (ADT4*)sds_alloc(4*160*320*sizeof(ADT4));
	bbox = (ADT4*)sds_alloc(4*26*sizeof(ADT4));
	//biasm = (BDT16*)sds_alloc(4*sizeof(BDT16));
	ofm_blob32 = (ADT32*)sds_alloc(32*fm_all*sizeof(ADT));
	ofm_blob = (ADT*)sds_alloc(64*163*323*sizeof(ADT));
	//*************************************load data *********************************

	printf("load parameter\n");
	load_img(img,320*160*4);
	//****************skynet*********************
	printf("SkyNet start\n");
	HlsNet(img, ofm_blob32, bbox);
	//****************check**********************

	std::cout<<"check:"<<std::endl;
	std::cout<<"pool0:"<<std::endl;
	fm_DT32_2_DT(& ofm_blob32[pool0_o], ofm_blob, config[1]);
	distitch(ofm_blob, ofm, config[1]);
	check_fm(ofm[0], config[1]);

	std::cout<<"pool1:"<<std::endl;
	fm_DT32_2_DT(& ofm_blob32[pool1_o], ofm_blob, config[3]);
	distitch(ofm_blob, ofm, config[3]);
	check_fm(ofm[0], config[3]);

	std::cout<<"pool2:"<<std::endl;
	fm_DT32_2_DT(& ofm_blob32[pool2_o], ofm_blob, config[5]);
	distitch(ofm_blob, ofm, config[5]);
	check_fm(ofm[0], config[5]);

	Compute_BBOX(bbox);
	
	return 0;


}
