# Readme
Our lab uses Windows and Visual Studio. To interface C++ and Python we use a tool called Simplified Wrapper Interface Generator (SWIG). It would be difficult to get you to be able to compile it as you would need VS, SWIG, a third party library that we use called FFTW, and environment variables set up. If you think it is important that you be able to compile it let me know and I can help you out with it. SWIG works by the user creating an interface file that has the function and class definitions you want exposed in Python. SWIG parses your interface file and generates C++ code that you can then compile to create a dynamic library (pyd file). 

Here I have a couple of modules that I have developed each one in their own sub-directory. They are described below.

## Image Interpolation

Many image processing algorithms will cause pixels to be shifted by fractional amounts, but we are rarely interested in the pixel intensities at fractional indices, normally we only care about integer indices. In these cases we must resample the image using image interpolation. GPUs are great for this, as they have built in interpolation hardware to help with transforming textures. The downside is that you are limited by whatever interpolation methods are implemented in hardware, which is usually linear interpolation. In the medical field, data integrity is really important, so here we implemented a couple higher order interpolation methods using NVidia CUDA. 

Specifically we developed a bi-cubic interpolation algorithm described here

http://verona.fi-p.unam.mx/boris/practicas/CubConvInterp.pdf

and a bi-cubic spline interpolation algorithm described here:

http://bigwww.epfl.ch/publications/unser9902.pdf

The accuracy of these methods were compared using the python script in that directory.

There is a header file that shows the C++ interface to the module and the implementation is in matching .cu file. All of the functions that start with the keyward __global__ indicate kernels that run on the GPU. There is another directory there for a class called CUDA2DRealMatrix, it is a container class to manage matrices stored on the GPU (as the name implies)

## Normalized cross correlator.

Here we developed a C++ class that perform a very time consuming operation called image correlation. Basically if you have two images and you think they are translated with respect to one another you can use this class to determine the best translation in a least squared error sense. The class is defined in the file root/NormalizedCrossCorrelator/NormalizedCrossCorrelator.h and the implementation is in the matching .cpp file. The .i file represents the interface I want to expose to Python. 

There is a python implementation of image correlation in root/NormalizedCrossCorrelator/Python Test Scripts/CrossCorrelate.py. This implementation is very slow but very easy to read and verify accuracy. NormalizedCrossCorrelatorAccuracyTest.pyw is in the same folder which uses the C++ implementation and python implementation and compares the results. This final script is not a great example of my coding style it's only there to show you how you would use the results of the SWIG generated Python module and verify the calculation is working correctly.

This version uses a FFT library called FFTW but we have another version that uses the CUDA FFT library (CuFFT),
