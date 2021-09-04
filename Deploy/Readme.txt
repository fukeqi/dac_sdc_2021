Before running the .impnb, 
please move the file load_image_c below /home/xilinx/jupyter_notebooks/dac_sdc_2021/Nobabyknows/ and use command:

cd /home/xilinx/jupyter_notebooks/dac_sdc_2021/Nobabyknows/load_image_c
g++ -shared -O2 load_resize_image.cpp -o load_image.so -fPIC `pkg-config opencv --cflags --libs` -lpthread
