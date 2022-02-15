"""Main module."""

import PIL
from PIL import Image as PILImage
# import rasterio as rio
# import numpy
# import pycuda.driver as drv

# from pycuda.compiler import SourceModule
# import pycuda
# from pycuda import gpuarray
# from pycuda import compiler
# import pycuda.driver as cuda
# import pycuda.autoinit             # PyCuda autoinit
# import pycuda.driver as cuda       # PyCuda In, Out helpers
# import matplotlib.pyplot as plot   # Library to plot
# import matplotlib.cm as colormap   # Library to plot
# import numpy                       # Fast math library
# import time
# import numpy as np                     # numeric python lib
# import matplotlib.image as mpimg       # reading images to numpy arrays
# import matplotlib.pyplot as plt        # to plot any graph
# import matplotlib.patches as mpatches  # to draw a circle at the mean contour
# import scipy.ndimage as ndi            # to determine shape centrality
# # matplotlib setup
# from pylab import rcParams
# rcParams['figure.figsize'] = (8, 8)      # setting default size of plots


# print("%d device(s) found." % cuda.Device.count())           
# for ordinal in range(cuda.Device.count()):
#     dev = cuda.Device(ordinal)
#     print ("Device #%d: %s" % (ordinal, dev.name()))
# print (cuda)


def hello_world():
    print("Hello World!")



from collections import Counter
from string import punctuation


def load_text(input_file):
    """Load text from a text file and return as a string."""
    with open(input_file, "r") as file:
        text = file.read()
    return text
    
def clean_text(text):
    """Lowercase and remove punctuation from a string."""
    text = text.lower()
    for p in punctuation:
        text = text.replace(p, "")
    return text
    
def count_words(input_file):
    """Count unique words in a string."""
    text = load_text(input_file)
    text = clean_text(text)
    words = text.split()
    return Counter(words)





 #Kernel text
kernel = """

#include <math.h>

#define _X  ( threadIdx.x + blockIdx.x * blockDim.x )
#define _Y  ( threadIdx.y + blockIdx.y * blockDim.y )
#define _WIDTH  ( blockDim.x * gridDim.x )
#define _HEIGHT ( blockDim.y * gridDim.y  )
#define _XM(x)  ( (x + _WIDTH) % _WIDTH )
#define _YM(y)  ( (y + _HEIGHT) % _HEIGHT )
#define _INDEX(x,y)  ( _XM(x)  + _YM(y) * _WIDTH )
#define PI 3.1415926


//https://github.com/AlainPaillou/PyCuda_Denoise_Filters/blob/master/PyCuda_KNN_Denoise_Mono.py
__global__ void svfcalculator(float * lattice_out, float * lattice, float scale) //int w, int h
{
    #define NLM_BLOCK_RADIUS    3
    
    int rangeDist = 200;
    
    int imageW = 2000;
    int imageH = 2000;
    
    const long int   ix = blockDim.x * blockIdx.x + threadIdx.x;
    const long int   iy = blockDim.y * blockIdx.y + threadIdx.y;
    const float  x = (float)ix  + 1.0f;
    const float  y = (float)iy  + 1.0f;
    const float limxmin = -1;      //NLM_BLOCK_RADIUS + 2;
    const float limxmax = imageW; // - NLM_BLOCK_RADIUS - 2;
    const float limymin = -1;      //NLM_BLOCK_RADIUS + 2;
    const float limymax = imageH; // - NLM_BLOCK_RADIUS - 2;
    
    long int index4;    
    
    
    if(ix>limxmin && ix<limxmax && iy>limymin && iy<limymax){
        // sky view factor
        float SVF_res = 0;
        
        //Result accumulator
        float clr00 = 0.0;
        float clrIJ = 0.0;
        
        //Center of the KNN window
        index4 = x + (y * imageW);
        
        // the current pixel
        clr00 = lattice[index4];
        
        for(int thetaN =0; thetaN<360; thetaN++) 
        {
            float theta = PI*float(thetaN)/180;
            float betaMax = 0;
            
            for( float radius = 5; radius < rangeDist; radius = radius + 5)
            {   
                // this is important or you will have memory error
                if (x + int(radius*cos(theta)) > limxmax | x + int(radius*cos(theta)) < limxmin | y - int(radius*sin(theta)) > limymax | y - int(radius*sin(theta)) < 0) 
                {
                    break;
                }
            
                long int index2 = x + int(radius*cos(theta)) + (y - int(radius*sin(theta))) * imageW;
                clrIJ = lattice[index2];
                
                // building height information
                float buildH = clrIJ - clr00;
                
                float beta = atan(scale*buildH/radius); //because the pixel resolution is 2ft, height is in ft
                if (betaMax < beta)
                {
                    betaMax = beta;
                }
            }
            SVF_res += pow(cos(betaMax), 2);
        }
        
        lattice_out[index4] = SVF_res/360.0;
    }
    
}
"""


# #Compile and get kernel function
# mod = SourceModule(kernel)
# print (mod)



# def svfCalculator_RayTracingOnGPU(dsm, scale):
#     '''This code is used to calculate the sky view factor using the ray-tracing
#     algorithm based on the GPU acceleration
#     last modified Jan 27, 2021
#     by Xiaojiang Li, Temple University
    
#     Parameters:
#         dsm: the numpy array of the digital surface model
#         scale: is the scale of the image, read from the gdal,
#                 1px of 2 feet, scale is 0.5; 1px of 3 feet, scale is 0.3333
#     '''
    
#     px = numpy.array(dsm).astype(numpy.float32)
    
#     #print ('Size:' + str(dsm.shape))
#     #print ('Pixels:' + str (dsm.shape[0]*dsm.shape[1]))
#     #print('The px.nbtyle is:', px.nbytes, px.shape)
    
    
#     # allocate memory on the device and transfer data to GPU 
#     d_px = cuda.mem_alloc(px.nbytes)
#     cuda.memcpy_htod(d_px, px)
    
#     height,width = px.shape
#     nb_pixels = height * width
    
#     # Set blocks et Grid sizes
#     nb_ThreadsX = 8
#     nb_ThreadsY = 8
#     nb_blocksX = (width // nb_ThreadsX) + 1
#     nb_blocksY = (height // nb_ThreadsY) + 1

#     #print("Test GPU ",nb_blocksX*nb_blocksY," Blocks ",nb_ThreadsX*nb_ThreadsY," Threads/Block")
#     tps1 = time.time()
    
    
#     # create empty array
#     lattice_gpu = gpuarray.to_gpu(px)
#     newLattice_gpu = gpuarray.empty_like(lattice_gpu)
    
#     # the GPU function
#     KNN_Mono_GPU = mod.get_function("svfcalculator")
#     KNN_Mono_GPU(newLattice_gpu, d_px, np.float32(scale), \
#                block=(nb_ThreadsX,nb_ThreadsY,1), \
#                grid=(nb_blocksX,nb_blocksY))  
    
#     bwPx = numpy.empty_like(px)    
#     bwPx = newLattice_gpu.get()
    
#     bwPx = numpy.float32(bwPx)
#     #pil_im = PILImage.fromarray(bwPx)
    
#     return bwPx



