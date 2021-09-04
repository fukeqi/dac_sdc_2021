#include "hlsnet.h"


void load_weight(WDT32* weight , int length)
{
    char nstr[50];
    sprintf(nstr, "weights.bin");
    FILE *fp = fopen(nstr, "rb");
    fread(weight, 1, length*sizeof(WDT), fp);
    fclose(fp);
}

void load_img(ADT4* img, int length)
{
    char nstr[50];
    sprintf(nstr, "raw_image.bin");
    FILE *fp = fopen(nstr, "rb");
    fread(img, 4, length*sizeof(ADT4), fp);
    fclose(fp);
}




