# Readme
Our lab uses Windows and Visual Studio. To interface C++ and Python we use a tool called Simplified Wrapper Interface Generator (SWIG). It would be difficult to get you to be able to compile it as you would need VS, SWIG, a third party library that we use called FFTW, and environment variables set up. If you think it is important that you be able to compile it let me know and I can help you out with it.

SWIG works by the user creating an interface file that has the function and class definitions you want exposed in Python. SWIG parses your interface file and generates C++ code that you can then compile to create a dynamic library (pyd file). 

Here we developed a C++ class that perform a very time consuming operation called image correlation. Basically if you have two images and you think they are translated with respect to one another you can use this class to determine the best translation in a least squared error sense. The class is defined in the file root/NormalizedCrossCorrelator/NormalizedCrossCorrelator.h and the implementation is in the matching .cpp file. The .i file represents the interface I want to expose to Python. It looks a lot like the class definition except there is a section labeled %extend NormalizedCrossCorrelator {...}. In this section contains the methods that you can use that work with NumPy arrays instead of C pointers. 

There is a python implementation of image correlation in root/NormalizedCrossCorrelator/Python Test Scripts/CrossCorrelate.py. This implementation is very slow but very easy to read and verify accuracy. NormalizedCrossCorrelatorAccuracyTest.pyw is in the same folder which uses the C++ implementation and python implementation and compares the results. This final script is not a great example of my coding style it's only there to show you how you would use the results of the SWIG generated Python module and verify the calculation is working correctly.