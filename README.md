# SampleCode
Our lab uses Windows and Visual Studio. To interface C++ and Python we use a tool called Simplified Wrapper Interface Generator (SWIG). It would be difficult to get you to be able to compile it as you would need VS, SWIG, a third party library that we use called FFTW, and environment variables set up. If you think it is important that you be able to compile it let me know and I can help you out with it.

SWIG works by the user creating an interface file that has the function and class definitions you want exposed in Python. SWIG parses your interface file and generates C++ code that you can then compile to create a dynamic library (pyd files). 

Here we developed a C++ class that perform a very time consuming operation called image correlation. Basically if you have two images and you think they are translated with respect to one another you can use this class to determine the best translation in a least squared error sense. The class is 

I created a github project with one of the Python modules that we developed that performs image correlation. I don't think you would be able to easily compile the project