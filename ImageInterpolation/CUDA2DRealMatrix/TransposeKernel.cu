#include "TransposeKernel.h"

__global__ void TransposeAndCastComplexSingleToDouble(cuDoubleComplex * odata, cuComplex *idata, unsigned long width, unsigned long height) {

	unsigned long int xIndex     = blockIdx.x * blockDim.x + threadIdx.x;
	unsigned long int yIndex     = blockIdx.y * blockDim.y + threadIdx.y;

	if (xIndex < width && yIndex < height) {

		cuComplex inTemp = idata[yIndex * width + xIndex];
		cuDoubleComplex outVal;
		outVal.x                        = (double) inTemp.x;
		outVal.y                        = (double) inTemp.y;
		odata[xIndex * height + yIndex] = outVal;
	}
}

__global__ void TransposeAndCastComplexDoubleToSingle(cuComplex * odata, cuDoubleComplex *idata, unsigned long width, unsigned long height) {

	unsigned long int xIndex     = blockIdx.x * blockDim.x + threadIdx.x;
	unsigned long int yIndex     = blockIdx.y * blockDim.y + threadIdx.y;

	if (xIndex < width && yIndex < height) {

		cuDoubleComplex inTemp = idata[yIndex * width + xIndex];
		cuComplex outVal;
		outVal.x                        = (float) inTemp.x;
		outVal.y                        = (float) inTemp.y;
		odata[xIndex * height + yIndex] = outVal;
	}
}