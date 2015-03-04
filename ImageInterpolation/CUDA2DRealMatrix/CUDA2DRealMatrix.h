#pragma once

/******************************************************************************
 * Copyright 2009                                                             *
 * The University of Rochester                                                *
 * All Rights Reserved                                                        *
 *                                                                            *
 * Zach Harvey   (zgh7555@rit.edu)                                            *
 * Alfredo Dubra (adubra@cvs.rochester.edu)                                   *
 *                                                                            *
 * Created: July 8, 2009                                                      *
 * Last modified: July 17, 2009                                               *
 ******************************************************************************/

// Forward definition of CUDA2DRealMatrix
template <class DataType>
class CUDA2DRealMatrix;

// System includes
#include <cuda_runtime.h>
#include <sstream>
#include <iostream>
#include <windows.h>
#include <fstream>

//Project includes
#include <ROIUtility.cuh>
#include <CUDACastingKernels.h>
#include <TransposeKernel.h>
#include <SearchingKernels.h>
#include <CUDAInterpolation.h>

//Threads per block
//TODO: do profiling to determine wheather this is most efficient
#define THREADS_PER_BLOCK	256

// This has to exist somewhere (cmath?)
#define PI 3.14159265358979323846

/// GPU kernel used to perform affine transformations.
/// The approach used is described in the following book 
/// R. Gonzalez and R. Woods, "Digital Image Processing". Upper Saddle River, NJ: Pearson Education Inc.,2008 pp. 88-89
/// Note, use PerformTransformation on the CUDA2DRealMatrix instead of using this
///
/// \param image_mask
///        The image mask for the result matrix. Some transformations will result in pixels that do not 
///        fall into the orignal image, in this case, these pixels are invalid. a 0 means no pixel data 
///        and a 1 indicates pixel data
///
/// \param n_rows_input
///        The number of rows in the  matrix to be transformed. 
///
/// \param n_cols_input
///        The number of columns in the matrix to be transformed.
///
/// \param n_rows_output
///        The number of rows in the output matrix
/// 
/// \param n_cols_output
///        The number of columns in the output matrix
///
/// \param interpolation_row_coordinates
///        a pointer to the array to store the row coordinates that we need to interpolate on
///
/// \param interpolation_col_coordinates
///        a pointer to the array to store the column coordinates that we need to interpolate on
///
/// \param affine_matrix
///        the matrix that defines the transformation
///
template<class T>
__global__  void AffineTransformation(T * image_mask, unsigned long n_rows_input, unsigned long n_cols_input,
								 unsigned long n_rows_output, unsigned long n_cols_output,
								 float * interpolation_row_coordinates, 
								 float * interpolation_col_coordinates, float * affine_matrix)  {
                                 
	// Get the indexing variables
	int col_index = blockIdx.x*blockDim.x + threadIdx.x;
	int row_index = blockIdx.y*blockDim.y + threadIdx.y;

	if (col_index < n_cols_output &&
		row_index < n_rows_output) {

		// a value of (0, 0) for the col and row index correspond to the upper left hand corner
		// of the matrix. We will shift all of the values over to create a new coordinate system
		// in order to ensure that the new image is centered on the matrix	
		int y = floorf(n_rows_output/2) - row_index;
		int x = col_index - floorf(n_cols_output/2);

		// calculate the inverse affine matrix in order to determine how the output matrix maps
		// to the input matrix. We will find a series of points that we must interpolate on
		float inverse_affine[4];

		// 1 / determinant(affine_matrix)
        float scale = 1.0/(affine_matrix[0]*affine_matrix[3] - affine_matrix[1]*affine_matrix[2]);
		// The inverse
		inverse_affine[0] = scale*affine_matrix[3];
		inverse_affine[1] = -scale*affine_matrix[1];
		inverse_affine[2] = -scale*affine_matrix[2];
		inverse_affine[3] = scale*affine_matrix[0];

		// find the location of this pixel on the input matrix
        float x2 = inverse_affine[0]*x + inverse_affine[1]*y;
        float y2 = inverse_affine[2]*x + inverse_affine[3]*y;

		// store the result
		interpolation_row_coordinates[row_index*n_cols_output + col_index] = floorf(n_rows_input/2) - y2;
		interpolation_col_coordinates[row_index*n_cols_output + col_index] = x2 + floorf(n_cols_input/2);

		// if we are out of bounds, then we put a zero in the image mask
		// otherwise we put a 1 in the image mask
		if (x < -floorf(n_cols_input/2) || x > floorf(n_cols_input/2) ||
			y < -floorf(n_rows_input/2) || y > floorf(n_rows_input/2)) {
			image_mask[row_index*n_cols_output + col_index] = 0;
		} else {
			image_mask[row_index*n_cols_output + col_index] = 1;
		}
	}
}

// this kernel computes, per-block, the sum
// of a block-sized portion of the input
// using a block-wide reduction
template<class T>
__global__ void block_sum(T *input,
                          float *per_block_results,
                          const size_t n)
{
  extern __shared__ float sdata[];
  unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;

  // load input into __shared__ memory
  float x = 0;
  if(i < n)
  {
    x = input[i];
  }
  sdata[threadIdx.x] = x;
  __syncthreads();

  // contiguous range pattern
  for(int offset = blockDim.x / 2;
      offset > 0;
      offset >>= 1)
  {
    if(threadIdx.x < offset)
    {
      // add a partial sum upstream to our own
      sdata[threadIdx.x] += sdata[threadIdx.x + offset];
    }

    // wait until all threads in the block have
    // updated their partial sums
    __syncthreads();
  }

  // thread 0 writes the final result
  if(threadIdx.x == 0)
  {
    per_block_results[blockIdx.x] = sdata[0];
  }
}

/***
 * CUDA2DRealMatrix
 *
 * The instances of this class represent 2-dimensional real matrices with a
 * memory buffer that resides on the (Nvidia) GPU. The data in the buffer is
 * stored in row major format.
 ***/

template <class DataType>
class CUDA2DRealMatrix {

	public:

		// Use example:
		// ...
		// CUDA2DRealMatrix<double> my_double_matrix(numRows, numCols);
		// CUDA2DRealMatrix<single> my_single_matrix(numRows, numCols);
		CUDA2DRealMatrix(unsigned long n_rows, unsigned long n_columns);

		//Copy Constructor
		CUDA2DRealMatrix(CUDA2DRealMatrix<DataType> &other);

		//Destructor
		~CUDA2DRealMatrix();

		// Returns the dimensions in the parameters passed. This method will
		// return false if either of the dimensions are zero or the buffer has
		// not been allocated. The call syntax is
		// unsigned long	numRows, numCols;
		// bool success = GetSize(numRows, numCols);
		bool GetSize(unsigned long &n_rows, unsigned long &n_columns);

		// This method changes the matrix dimensions (if different from current)
		// and returns a boolean indicating success. It the matrix size is left
		// unchanged, then the matrix data is kept, otherwise it is lost, and the
		// matrix will be filled with undetermined values.
		bool SetSize(unsigned long n_rows, unsigned long n_columns);

		// Returns a GPU pointer to the first element in the buffer. This method
		// will return false if the buffer is NULL. The call syntax is
		// double * my_double_buffer;
		// bool success = my_matrix.GetPointerToData((void**) &my_double_buffer);
		bool GetPointerToData(DataType **data);

		// Returns the value of the element in the zero-indexed row and colum
		// specified. The method returns false if the buffer is NULL or if the
		// row/column values passed are invalid.
		bool GetElement(unsigned long row, unsigned long column, DataType &value);

		// This method copies data from a buffer in the host computer to the matrix
		// buffer, which is stored in the GPU (device). All the data is copied at once
		// as casted accordingly when needed. The returned boolean indicates success
		// and the is_row_major parameter indicates whether the source data (in the
		// host)is arranged in row (column) major format. Matlab matrices are stored
		// in column major format.
		template <class DataTypeIn>
		bool CopyDataFromHost(DataTypeIn *source_data, bool is_row_major);

		// This method retrieves the data from the matrix buffer, which is in the GPU
		// (device), to a buffer in the host computer. All the data is copied at once
		// and casted accordingly if needed. The returned boolean indicates success
		// and the is_row_major parameter indicates whether the destination data (in
		// the host computer) is arranged in row (column) major format. Matlab
		// matrices are stored in column major format.
		template <class DataTypeOut>
		bool CopyDataToHost(DataTypeOut *destination_data, bool is_row_major);

		// Copies a ROI from this matrix to a host buffer supplied. The ROI is the 
		// closed intervale between (top_row, left_column) to (bottom_row, right_column)
		// This method will return false if the matrix buffer was not correctly created
		// of if the destination_data buffer is NULL
		// Note: this method uses zero based indexing

		template <class DataTypeOut>
		bool CopyROIToHost(DataTypeOut * destination_data, bool is_row_major, 
						   unsigned long top_row, unsigned long left_column, 
						   unsigned long bottom_row, unsigned long right_column);

		// This method copies data from a buffer in the GPU (device) to the matrix
		// buffer, also stored in the GPU. All the data is copied at once and casted
		// accordingly if needed. The returned boolean indicates success and the
		// is_row_major parameter indicates whether the source data (in the host) is
		// arranged in row (column) major format. Matlab matrices are stored in
		// column major format.
		template <class DataTypeIn>
		bool CopyDataFromDevice(DataTypeIn *source_data, bool is_row_major);

		//CopyDataToDevice??
		template <class DataTypeOut>
		bool CopyDataToDevice(DataTypeOut *destination_data, bool is_row_major);

		// Copies a ROI from this matrix to a device buffer supplied. The ROI is the 
		// closed intervale between (top_row, left_column) to (bottom_row, right_column)
		// This method will return false if the matrix buffer was not correctly created
		// of if the destination_data buffer is NULL
		// Note: this method uses zero based indexing

		template <class DataTypeOut>
		bool CopyROIToDevice(DataTypeOut * destination_data, bool is_row_major, 
						   unsigned long top_row, unsigned long left_column, 
						   unsigned long bottom_row, unsigned long right_column);

        template <class DataTypeIn>
		bool AssignFromDevice(CUDA2DRealMatrix<DataTypeIn>  &input_data,
                              CUDA2DRealMatrix<int>         &row_locations, 
                              CUDA2DRealMatrix<int>         &col_locations);

        template <class DataTypeOut>
        bool Get(CUDA2DRealMatrix<DataTypeOut> &output_data,
                 CUDA2DRealMatrix<int>         &row_locations, 
                 CUDA2DRealMatrix<int>         &col_locations);


		// Retrieve a tab separated string representation of the matrix elements that
		// can be used to print to screen, save to a text file, etc. The user should not
		// call free() on the returned string
		std::string ToString();

		// Overloaded ToString() used to print out only a region of interest enclosed
		// by the rows and columns passed as parameters
		std::string ToString(unsigned long min_row,    unsigned long max_row, 
						unsigned long min_column, unsigned long max_column);

		// Return the matrix maximum value, the corresponding row and column, and a
		// boolean indicating whether the matrix was created successfully.
		bool GetMaximum(DataType &max_value, unsigned long &max_row, unsigned long &max_column);

		// Overloaded GetMax method that returns the max with in a ceartain boundry. 
		// The boundry is specified by the rectangle with the top left corner at top_row and left_column
		// and the bottom right corner at bottom_row and right_column. This method will return
		// false if the rectangle does not fully lie within the matrix or if memory copies from
		// device to host fail
		bool GetMaximum(DataType &max_val, unsigned long &max_row,  unsigned long &max_column,
						unsigned long top_row, unsigned long left_column, unsigned long bottom_row,
						unsigned long right_column);

		// Return the matrix minimum value, the corresponding row and column, and a
		// boolean indicating whether the matrix was created successfully.
		bool GetMinimum(DataType &min_value, unsigned long &min_row, unsigned long &min_column);

		// Overloaded GetMi, method that returns the min with in a ceartain boundry. 
		// The boundry is specified by the rectangle with the top left corner at top_row and left_column
		// and the bottom right corner at bottom_row and right_column. This method will return
		// false if the rectangle does not fully lie within the matrix or if memory copies from
		// device to host fail
		bool GetMinimum(DataType &min_value,     unsigned long &min_row,  unsigned long &min_column,
						unsigned long top_row, unsigned long left_column, unsigned long bottom_row,
						unsigned long right_column);


		//Comparison operator
		bool operator==(CUDA2DRealMatrix<DataType> &rhs);

        template <class DataTypeIn>
        bool Eq(CUDA2DRealMatrix<DataTypeIn> &other);

		// This method returns a copy of this instance.
		template <class DataTypeIn>
		bool Copy(CUDA2DRealMatrix<DataTypeIn>& other);

		// This method allows the user to ensure that all memory allocations were successful.
		// This method will return false if there was an error in allocating resources on the 
		// GPU
		bool IsValid();

		// Returns a string representing the type that this matrix stores
		// uint8  -> 8 bit unsigned integers
		// uint32 -> 32 bit unsigned integers
		// int32  -> 32 bit signed integers
		// int8   -> 8 bit signed integer
		// single -> 32 bit signed floating point numbers
		// double -> 64 bit signed floating point numbers
		// DO NOT free the char array that is returned, this class will handle that
		char * GetType();

		template <class T>
		bool add(CUDA2DRealMatrix<T> &other, unsigned long ROI_width, unsigned long ROI_height, 
				 unsigned long ROI_start_row_this,  unsigned long ROI_start_col_this,
				 unsigned long ROI_start_row_other, unsigned long ROI_start_col_other);

        template <class T>
		bool sub(CUDA2DRealMatrix<T> &other, unsigned long ROI_width, unsigned long ROI_height, 
				 unsigned long ROI_start_row_this,  unsigned long ROI_start_col_this,
				 unsigned long ROI_start_row_other, unsigned long ROI_start_col_other);

        template <class T>
		bool div(CUDA2DRealMatrix<T> &other, unsigned long ROI_width, unsigned long ROI_height, 
				 unsigned long ROI_start_row_this,  unsigned long ROI_start_col_this,
				 unsigned long ROI_start_row_other, unsigned long ROI_start_col_other);

        template <class T>
		bool mult(CUDA2DRealMatrix<T> &other, unsigned long ROI_width, unsigned long ROI_height, 
				 unsigned long ROI_start_row_this,  unsigned long ROI_start_col_this,
				 unsigned long ROI_start_row_other, unsigned long ROI_start_col_other);


		bool LogBase10(unsigned long ROI_width, unsigned long ROI_height, 
				 unsigned long ROI_start_row,  unsigned long ROI_start_col);

        bool LogBase10();

        template <class T>
        bool dot(CUDA2DRealMatrix<T> &rhs);

        template <class OTHER_TYPE>
        CUDA2DRealMatrix<DataType> operator+(const CUDA2DRealMatrix<OTHER_TYPE> &b) const;

        //template <class OTHER_TYPE>
        //CUDA2DRealMatrix<DataType> operator*(const CUDA2DRealMatrix<OTHER_TYPE> &b) const;

        //template <class OTHER_TYPE>
        //CUDA2DRealMatrix<DataType> operator-(const CUDA2DRealMatrix<OTHER_TYPE> &b) const;

        //template <class OTHER_TYPE>
        //CUDA2DRealMatrix<DataType>& operator+=(const CUDA2DRealMatrix<OTHER_TYPE> &b) const;

        //template <class OTHER_TYPE>
        //CUDA2DRealMatrix<DataType>& operator-=(const CUDA2DRealMatrix<OTHER_TYPE> &b) const;

        //template <class OTHER_TYPE>
        //CUDA2DRealMatrix<DataType> operator/(const CUDA2DRealMatrix<OTHER_TYPE> &b) const;

        //template <class OTHER_TYPE>
        //CUDA2DRealMatrix<DataType>& operator*=(const CUDA2DRealMatrix<OTHER_TYPE> &b) const;

        //template <class OTHER_TYPE>
        //CUDA2DRealMatrix<DataType>& operator/=(const CUDA2DRealMatrix<OTHER_TYPE> &b) const;

        //template <class OTHER_TYPE>
        //CUDA2DRealMatrix<DataType> dot(const CUDA2DRealMatrix<OTHER_TYPE> &b) const;

		bool AddScalar(float val, unsigned long ROI_width, unsigned long ROI_height, 
					   unsigned long ROI_start_row, unsigned long ROI_start_col);

		bool MultScalar(float val, unsigned long ROI_width, unsigned long ROI_height, 
					    unsigned long ROI_start_row, unsigned long ROI_start_col);

		bool Set(float val, unsigned long ROI_width, unsigned long ROI_height, 
			     unsigned long ROI_start_row, unsigned long ROI_start_col);
	
		/// Performs an affine transformation with the given transformation matrix.
		/// Places output data int the result matrix. The transformation may result in
		/// a matrix which does not have pixel values in every location (i.e. the result
		/// might not have a rectangular shape). To accomodate this, an image mask is 
		/// returned. This method will center result image in the new matrix (i.e.
		/// pixel value at location floor(n_rows_initial/2), floor(n_cols_initial/2) in
		/// the initial matrix will be equal to the pixel value at location 
		/// floor(n_rows_final/2), floor(n_cols_final/2) regardless of the transformation).
		/// If the result does not fit into the passed matrix, the image is cropped.
		/// A majority of the time, the result pixels will not map directly to integer
		/// locations on the original image, in this case, interpolation will be performed
		/// by the interpolation object passed.
		///
		/// \param affine_matrix
		///        Row major formated matrix that represents the affine transformation
		///
		/// \param result_matrix
		///        The transformed image will be centered in this matrix. Do not attempt 
		///        inplace transformations. If the result image does not fit, it will be cropped
		///
		/// \param image_mask
		///        The result matrix may not be rectangular, this image mask will indicate which
		///        pixels are valid. A value of 1 indicates the pixel value is valid here, a value
		///        of zero indicates that it is not. If the passed matrix is not the same 
		///        dimensions as the result matrix, it will be resized accordingly. 
		///
		/// \param interpolator
		///        A majority of the time, the result pixels will not map directly to integer
		///        locations on the original image, in this case interpolation will be performed
		///        by the interpolation object passed.
		bool PerformTransformation(float affine_matrix[4], CUDA2DRealMatrix<float> &result_matrix, 
								   CUDA2DRealMatrix<int> &image_mask, CUDA2DInterpolator * interpolator);
		
		/// Convenience method for performing only rotation transformations.
		/// This method generates an affine_matrix and calls PerfromTransformation
	    ///
		/// \param rotation_in_degrees
		///        The amount of rotation in clockwise direction
		///
		/// \param result_matrix
		///        The transformed image will be centered in this matrix. Do not attempt 
		///        inplace transformations. If the result image does not fit, it will be cropped
		///
		/// \param image_mask
		///        The result matrix may not be rectangular, this image mask will indicate which
		///        pixels are valid. A value of 1 indicates the pixel value is valid here, a value
		///        of zero indicates that it is not. If the passed matrix is not the same 
		///        dimensions as the result matrix, it will be resized accordingly. 
		///
		/// \param interpolator
		///        A majority of the time, the result pixels will not map directly to integer
		///        locations on the original image, in this case interpolation will be performed
		///        by the interpolation object passed.
		bool Rotate(double rotation_in_degrees, CUDA2DRealMatrix<float> &result_matrix, 
					CUDA2DRealMatrix<int> &image_mask, CUDA2DInterpolator * interpolator);

		/// Convenience method for performing only scaling transformations.
		/// This method generates an affine_matrix and calls PerfromTransformation
	    ///
		/// \param x_scaling_factor
		///        The amount to scale in the x directions
		///
		/// \param y_scaling_factor
		///        The amount to scale in the y directions
		///
		/// \param result
		///        The transformed image will be centered in this matrix. Do not attempt 
		///        inplace transformations. If the result image does not fit, it will be cropped
		///
		/// \param image_mask
		///        The result matrix may not be rectangular, this image mask will indicate which
		///        pixels are valid. A value of 1 indicates the pixel value is valid here, a value
		///        of zero indicates that it is not. If the passed matrix is not the same 
		///        dimensions as the result matrix, it will be resized accordingly. 
		///
		/// \param interpolator
		///        A majority of the time, the result pixels will not map directly to integer
		///        locations on the original image, in this case interpolation will be performed
		///        by the interpolation object passed.
		bool Scale(double x_scaling_factor, double y_scaling_factor, CUDA2DRealMatrix<float> &result,
			       CUDA2DRealMatrix<int> &image_mask, CUDA2DInterpolator * interpolator);


        double sum();

		double mean();

		bool ClampMax(double max_value);

		bool ClampMin(double min_value);

	private:
		
		// Making the default and copy constructors private so that users cannot
		// create instances of this class without specifying matrix dimensions
		// or by incorrectly. The default copy constructor would not allocate a
		// new matrix buffer and copy the matrix elements, it would only copy
		// the pointer to the buffer of the passed instance.
		CUDA2DRealMatrix();

		// Pointer to GPU data buffer. The pointer must be NULL when not allocated.
		DataType*		matrix_buffer;

		// Number of rows and columns in data buffer 
		unsigned long	n_rows,
						n_columns;

		//String representing a human readable version of 
		//the matrix
		char*			display_string;

		// String tell the type that this matrix stores
		char *          type;

};

/**
 * Default constructor
 **/
template <class DataType>
CUDA2DRealMatrix<DataType>::CUDA2DRealMatrix() {

    //Intialize
	matrix_buffer  = NULL;
	n_rows		   = 0;
	n_columns	   = 0;
	display_string = NULL;
	type           = NULL;

}

/**
 * Constructor
 **/
template <class DataType>
CUDA2DRealMatrix<DataType>::CUDA2DRealMatrix(unsigned long n_rows, unsigned long n_columns) {

	//Assign internal variables
	this->n_rows			= n_rows;
	this->n_columns			= n_columns;
	this->display_string	= NULL;
	this->type              = NULL;

	//Allocate space on the GPU
	cudaError_t result;
	result					= cudaMalloc((void **) &matrix_buffer, n_rows * n_columns * sizeof(DataType));

	if (result != cudaSuccess) {
		n_rows		  = 0;
		n_columns	  = 0;
		matrix_buffer = NULL;
	}

}

template <class DataType>
CUDA2DRealMatrix<DataType>::CUDA2DRealMatrix(CUDA2DRealMatrix<DataType> &other) {

	this->n_rows		 = other.n_rows;
	this->n_columns		 = other.n_columns;
	this->display_string = NULL;
	this->type           = NULL;

	cudaError_t res = cudaMalloc((void **) &this->matrix_buffer, sizeof(DataType)*n_rows*n_columns);
	if(res != cudaSuccess) {
		this->n_rows		 = 0;
		this->n_columns		 = 0;
		this->matrix_buffer = NULL;
	}
	else {
		res = cudaMemcpy(this->matrix_buffer, other.matrix_buffer, sizeof(DataType)*n_rows*n_columns, cudaMemcpyDeviceToDevice);
		if (res != cudaSuccess) {

			cudaFree(this->matrix_buffer);
			this->n_rows		 = 0;
			this->n_columns		 = 0;
			this->matrix_buffer = NULL;
		}

	}
}

/**
 * Destructor
 **/
template <class DataType>
CUDA2DRealMatrix<DataType>::~CUDA2DRealMatrix() {

	//Free the matrix buffer if it is not null
	if (matrix_buffer != NULL) {
		cudaFree(matrix_buffer);
	}

	if (display_string != NULL) {
		free(display_string);
	}
}


/***
 * GetSize
 ***/
template <class DataType>
bool CUDA2DRealMatrix<DataType>::GetSize(unsigned long &n_rows, unsigned long &n_columns) {

	if (this->n_rows != 0 && this->n_columns != 0) {
		n_rows		= this->n_rows;
		n_columns	= this->n_columns;
		return true;
	}
	else {
		return false;
	}
}

/***
 * GetMaximum
 ***/
template <class DataType>
bool CUDA2DRealMatrix<DataType>::GetMaximum(DataType &max_value, unsigned long &max_row, unsigned long &max_column) {

	//Ensure that this matrix has been initialized
	if ( n_rows == 0 || n_columns == 0 || matrix_buffer == NULL) {
		return false;
	}

	return CUDAFindMax<DataType>(matrix_buffer, n_rows, n_columns, max_row, max_column, max_value);

}

/***
 * Overloaded GetMaximum
 ***/
template <class DataType>
bool CUDA2DRealMatrix<DataType>::GetMaximum(DataType &max_val, unsigned long &max_row, unsigned long &max_column, unsigned long top_row, unsigned long left_column, unsigned long bottom_row, unsigned long right_column) {

	//Ensure that this matrix has been initialized
	if ( n_rows == 0 || n_columns == 0 || matrix_buffer == NULL) {
		return false;
	}

	return FindROIMax<DataType>(matrix_buffer, n_rows, n_columns, max_row, max_column, max_val, top_row, left_column, bottom_row, right_column);

}

/***
 * GetMinimum
 ***/

template <class DataType>
bool CUDA2DRealMatrix<DataType>::GetMinimum(DataType &min_value, unsigned long &min_row, unsigned long &min_column) {

	//Ensure that this matrix has been initialized
	if ( n_rows == 0 || n_columns == 0 || matrix_buffer == NULL) {
			return false;
	}

	return CUDAFindMin<DataType>(matrix_buffer, n_rows, n_columns, min_row, min_column, min_value);

}

/***
 * GetMinimum Overloaded
 ***/

template <class DataType>
bool CUDA2DRealMatrix<DataType>::GetMinimum(DataType &min_value, unsigned long &min_row, unsigned long &min_column, unsigned long top_row, unsigned long left_column, unsigned long bottom_row, unsigned long right_column) {

	//Ensure that this matrix has been initialized
	if ( n_rows == 0 || n_columns == 0 || matrix_buffer == NULL) {
			return false;
	}

	return FindROIMin(matrix_buffer, n_rows, n_columns, min_row, min_column, min_value, top_row, left_column, bottom_row, right_column);
}

/***
 * GetPointerToData
 ***/
template <class DataType>
bool CUDA2DRealMatrix<DataType>::GetPointerToData(DataType **data) {

	if ( n_rows == 0 || n_columns == 0 || matrix_buffer == NULL) {
		return false;
	}
	
	(*data) = matrix_buffer;
	return true;
}

/***
 * GetElement
 ***/
template <class DataType>
bool CUDA2DRealMatrix<DataType>::GetElement(unsigned long row, unsigned long column, DataType &value) {

	if( matrix_buffer == NULL || n_rows == 0 || n_columns == 0) {
		return false;
	}

	if (row >= n_rows || column >= n_columns) {
		return false;
	}

	//Find the offset in the matrix_buffer to the element location
	void *element_location = (void *) (((unsigned long long) matrix_buffer) + ((unsigned long long)(row * n_columns + column) * sizeof(DataType)));

	//Copy the element to the host
	cudaError_t result     = cudaMemcpy((void *) &value, element_location, sizeof(DataType), cudaMemcpyDeviceToHost);

	if (result != cudaSuccess) {
		return false;
	}

	return true;
}

/**
 * SetSize
 **/
template <class DataType>
bool CUDA2DRealMatrix<DataType>::SetSize(unsigned long n_rows, unsigned long n_columns) {

	if (n_rows	  == 0 || n_columns == 0) {
		return false;
	}

	if (n_rows == this->n_rows && n_columns == this->n_columns) {
		return true;
	}

	// determine if there is a need to reallocate the matrix buffer
	if (n_rows*n_columns == this->n_rows*this->n_columns) {
		this->n_rows	= n_rows;
		this->n_columns = n_columns;

		return true;
	}

	this->n_rows	= n_rows;
	this->n_columns = n_columns;

	cudaError_t result;

	//If there was data previously in the buffer free it
	if (matrix_buffer != NULL) {

		result = cudaFree(matrix_buffer);
		if (result != cudaSuccess) {
			return false;
		}

		matrix_buffer = NULL;
	}

	//Allocate space on the GPU for the matrix
	result = cudaMalloc((void **) &matrix_buffer, sizeof(DataType) * n_rows * n_columns);
	if (result != cudaSuccess) {
		matrix_buffer	= NULL;
		this->n_rows	= 0;
		this->n_columns = 0;
		return false;
	}

	return true;

}

/***
 * CopyDataFromHost
 ***/

template <class DataType> template <class DataTypeIn>
bool CUDA2DRealMatrix<DataType>::CopyDataFromHost(DataTypeIn *source_data, bool is_row_major) {

	//Ensure that source_data is valid
	if (source_data == NULL) {
		return false;
	}

	//Allocate space on the GPU for the uncasted data
	DataTypeIn * uncastedDeviceBuffer;

	cudaError_t result;

	//Because elements in the CUDA2DRealMatrix are stored in row major format
	//then if the host data is not in row major format, it must be transposed
	if (!is_row_major) {

		//Create a temporary array that will be transposed into the proper buffer
		DataTypeIn * tempBuffer;

		//Allocate space on GPU for temp buffer
		result = cudaMalloc((void **) &tempBuffer, sizeof(DataTypeIn) * n_rows * n_columns);
		if (result != cudaSuccess) {
			return false;
		}

		//Copy the untransposed buffer to the GPU
		result = cudaMemcpy((void *) tempBuffer, (void *) source_data, sizeof(DataTypeIn) * n_rows * n_columns, cudaMemcpyHostToDevice);
		if (result != cudaSuccess) {
			cudaFree(tempBuffer);
			return false;
		}

		//Determine the number of threads needed to transpose the matrix
		//NOTE: here we switch the rows and columns because the transpose kernel
		//assume row major format
		dim3 threadBlock(16, 16);
		dim3 threadGrid((threadBlock.x + n_rows    - 1)/threadBlock.x,
						(threadBlock.y + n_columns - 1)/threadBlock.y);

		//if the incomming data is the same type as the matrix, then just copy the data in
		if (typeid(DataType) == typeid(DataTypeIn)) {

			//Transpose the temp buffer
			//NOTE here the rows and columns need to switched because the transpose kernel
			//assumes row major format
			Transpose<DataType><<<threadGrid, threadBlock>>>((DataType *) matrix_buffer,(DataType *) tempBuffer, n_rows, n_columns);
			if (cudaGetLastError() != cudaSuccess) {
				cudaFree(tempBuffer);
				if (matrix_buffer != NULL) {
					cudaFree(matrix_buffer);
					matrix_buffer = NULL;
				}
				n_rows        = 0;
				n_columns     = 0;
				return false;
			}

		}
		else {

			//Transpose and cast the input data
			TransposeAndCast<DataTypeIn, DataType><<<threadGrid, threadBlock>>>(matrix_buffer, tempBuffer, n_rows, n_columns);
			if (cudaGetLastError() != cudaSuccess) {
				cudaFree(tempBuffer);
				if (matrix_buffer != NULL) {
					cudaFree(matrix_buffer);
					matrix_buffer = NULL;
				}
				n_rows        = 0;
				n_columns     = 0;
				return false;
			}

		}

		cudaFree(tempBuffer);

	}
	else {

		//If the data is of the right type, just copy right to the matrix_buffer
		if (typeid(DataTypeIn) == typeid(DataType)) {
			result = cudaMemcpy((void *) matrix_buffer, (void *) source_data, sizeof(DataType)*n_rows*n_columns, cudaMemcpyHostToDevice);
			if(result != cudaSuccess) {
				if (matrix_buffer != NULL) {
					cudaFree(matrix_buffer);
					matrix_buffer = NULL;
				}
				n_rows        = 0;
				n_columns     = 0;
				return false;
			}
		}
		else {

			//Data needs to be casted before it can be put into matrix_buffer
			result = cudaMalloc(&uncastedDeviceBuffer, sizeof(DataTypeIn)*n_rows*n_columns);
			if (result != cudaSuccess) {
				return false;
			}	

			//No need to transpose just copy data
			result = cudaMemcpy(uncastedDeviceBuffer, source_data, sizeof(DataTypeIn)*n_rows*n_columns, cudaMemcpyHostToDevice);
			if (result != cudaSuccess) {
				cudaFree(uncastedDeviceBuffer);
				return false;

			}

			dim3 threadBlock(16, 16);
			dim3 threadGrid( (threadBlock.x + n_columns - 1)/threadBlock.x, 
				             (threadBlock.y + n_rows    - 1)/threadBlock.y);
			//Cast the data
			Cast<DataTypeIn, DataType><<<threadGrid, threadBlock>>>(uncastedDeviceBuffer, matrix_buffer, n_rows, n_columns);
			if (cudaGetLastError() != cudaSuccess) {
				if (matrix_buffer != NULL) {
					cudaFree(matrix_buffer);
					matrix_buffer = NULL;
				}
				n_rows        = 0;
				n_columns     = 0;
				return false;
			}
			//Free the temp buffer and uncasted buffer
			cudaFree(uncastedDeviceBuffer);
		}
	}
	return true;
}


/**
 * CopyDataFromDevice
 **/

template <class DataType> template <class DataTypeIn>
bool CUDA2DRealMatrix<DataType>::CopyDataFromDevice(DataTypeIn *source_data, bool is_row_major) {

	//Ensure valid parameters
	if (source_data == NULL ) {
		return false;
	}

	//Matrices are stored in row major format, so we might need to transpose the data
	if (!is_row_major) {

		//Determine the number of threads needed to transpose the matrix
		//NOTE: here we switch the rows and columns because the transpose kernel
		//assume row major format
		dim3 threadBlock(16, 16);
		dim3 threadGrid((threadBlock.x + n_rows - 1)/threadBlock.x,
						(threadBlock.y + n_columns - 1)/threadBlock.y);

		//See if the source type is the same as the matrix type
		if (typeid(DataType) == typeid(DataTypeIn)) {

			//Transpose the temp buffer
			//NOTE here the rows and columns need to switched because the transpose kernel
			//assumes row major format
			Transpose<DataType><<<threadGrid, threadBlock>>>((DataType *) matrix_buffer, (DataType *) source_data, n_rows, n_columns);
			if (cudaGetLastError() != cudaSuccess) {
				if (matrix_buffer != NULL) {
					cudaFree(matrix_buffer);
					matrix_buffer = NULL;
				}
				n_rows        = 0;
				n_columns     = 0;
				return false;
			}

		}
		else {
			//Transpose and cast the input data
			TransposeAndCast<DataTypeIn, DataType><<<threadGrid, threadBlock>>>(matrix_buffer, source_data, n_rows, n_columns);
			if (cudaGetLastError() != cudaSuccess) {
				if (matrix_buffer != NULL) {
					cudaFree(matrix_buffer);
					matrix_buffer = NULL;
				}
				n_rows        = 0;
				n_columns     = 0;
				return false;
			}

		}

	}
	else {

		if (typeid(DataTypeIn) == typeid(DataType) ) {
			if (cudaMemcpy((void *) matrix_buffer, (void *) source_data, sizeof(DataType)*n_rows*n_columns, cudaMemcpyDeviceToDevice) != cudaSuccess) {
				if (matrix_buffer != NULL) {
					cudaFree(matrix_buffer);
					matrix_buffer = NULL;
				}
				n_rows        = 0;
				n_columns     = 0;
				return false;
			}
		}
		else {
			//Determine the number of threads needed to cast the data
			dim3 threadBlock(16, 16);
			dim3 threadGrid((threadBlock.x + n_columns - 1)/threadBlock.x,
							(threadBlock.y + n_rows    - 1)/threadBlock.y);

			//Cast the data
			Cast<DataTypeIn, DataType><<<threadGrid, threadBlock>>>(source_data, matrix_buffer, n_rows, n_columns);
			if (cudaGetLastError() != cudaSuccess) {
				if (matrix_buffer != NULL) {
					cudaFree(matrix_buffer);
					matrix_buffer = NULL;
				}
				n_rows        = 0;
				n_columns     = 0;
				return false;
			}
		}
	}

	return true;
}

/**
 * CopyDataToHost
 **/

template <class DataType> template <class DataTypeOut>
bool CUDA2DRealMatrix<DataType>::CopyDataToHost(DataTypeOut *destination_data, bool is_row_major) {

	cudaError_t result;

	//Ensure valid parameters
	if (destination_data == NULL) {
		return false;
	}

	if (!is_row_major) {
		//We must transpose
		DataTypeOut *transposedBuffer;
		result = cudaMalloc((void **) &transposedBuffer, sizeof(DataTypeOut)*n_rows*n_columns);
	
		if (result != cudaSuccess) {
			return false;
		}

		//Determine the number of threads needed to transpose the matrix
		dim3 threadBlock(16, 16);
		dim3 threadGrid((threadBlock.x + n_columns - 1)/threadBlock.x,
						(threadBlock.y + n_rows    - 1)/threadBlock.y);

		//If the destination type is the same as data type
		//then just copy the data
		if (typeid(DataType) == typeid(DataTypeOut)) {

			//Transpose the temp buffer
			Transpose<DataTypeOut><<<threadGrid, threadBlock>>>((DataTypeOut *) transposedBuffer,(DataTypeOut *) matrix_buffer, n_columns, n_rows);
			if (cudaGetLastError() != cudaSuccess) {
				cudaFree(transposedBuffer);
				return false;
			}

			result = cudaMemcpy(destination_data, transposedBuffer, n_rows*n_columns*sizeof(DataType), cudaMemcpyDeviceToHost);
			if (result != cudaSuccess) {
				cudaFree(transposedBuffer);
				return false;
			}

		}
		else { //we must cast

			TransposeAndCast<DataType, DataTypeOut><<<threadGrid, threadBlock>>>(transposedBuffer, matrix_buffer, n_columns, n_rows);
			if ( cudaGetLastError() != cudaSuccess) {
				return false;
			}

			result = cudaMemcpy((void *) destination_data, (void *) transposedBuffer, n_rows*n_columns*sizeof(DataTypeOut), cudaMemcpyDeviceToHost);

			if (result != cudaSuccess) {
				return false;
			}
		}

		cudaFree(transposedBuffer);
	}
	else {

		//If the destination type is the same as data type
		//then just copy the data
		if (typeid(DataType) == typeid(DataTypeOut)) {
			result = cudaMemcpy(destination_data, matrix_buffer, n_rows*n_columns*sizeof(DataType), cudaMemcpyDeviceToHost);

			if (result != cudaSuccess) {
				return false;
			}
		}
		else {//We must cast
			
			//Create a temporary buffer
			DataTypeOut * castedBuffer;

			result = cudaMalloc((void **) &castedBuffer, n_rows*n_columns*sizeof(DataTypeOut));
			if (result != cudaSuccess) {
				return false;
			}

			dim3 threadBlock(16, 16);
			dim3 threadGrid((threadBlock.x + n_columns - 1) /threadBlock.x,
							(threadBlock.y + n_rows - 1) /threadBlock.y);

			//Cast the data
			Cast<DataType, DataTypeOut><<<threadGrid, threadBlock>>>(matrix_buffer, castedBuffer, n_rows, n_columns);
			if (cudaGetLastError() != cudaSuccess) {
				return false;
			}

			//fill the destination buffer
			result = cudaMemcpy((void *) destination_data, (void *) castedBuffer, n_rows*n_columns*sizeof(DataTypeOut), cudaMemcpyDeviceToHost);

			//Free the casted data
			cudaFree(castedBuffer);

			if (result != cudaSuccess) {
				return false;
			}
		}
	}

	return true;

}

template <class DataType> template <class DataTypeOut>
bool CUDA2DRealMatrix<DataType>::CopyROIToHost(DataTypeOut * destination_data, bool is_row_major, 
						                       unsigned long top_row, unsigned long left_column, 
						                       unsigned long bottom_row, unsigned long right_column) {

	if (destination_data == NULL) {
		return false;
	}

	if (matrix_buffer == NULL || n_rows == 0 || n_columns == 0) {
		return false;
	}

	if (bottom_row >= n_rows || top_row > bottom_row || 
		right_column >= n_columns || left_column > right_column) {
		return false;
	}

	unsigned long ROI_rows    = bottom_row   - top_row + 1;
	unsigned long ROI_columns = right_column - left_column + 1;

	CUDA2DRealMatrix<DataType> temp_matrix(ROI_rows, ROI_columns);

	DataType * temp_pointer;
	if (!temp_matrix.GetPointerToData(&temp_pointer)) {
		return false;
	}

	dim3 block_dim(16, 16);
	dim3 grid_dim((block_dim.x + ROI_columns    - 1)/block_dim.x,
				  (block_dim.y + ROI_rows       - 1)/block_dim.y);

	GetROI<DataType><<<grid_dim, block_dim>>>(matrix_buffer, n_columns, temp_pointer, ROI_columns, ROI_rows, top_row, left_column);
	cudaError_t result = cudaGetLastError();
	if (result != cudaSuccess) {
		return false;
	}

	if (!temp_matrix.CopyDataToHost<DataTypeOut>(destination_data, is_row_major)) {
		return false;
	}
	
	return true;
}
/***
 * CopyDataToDevice
 ***/

template <class DataType> template <class DataTypeOut>
bool CUDA2DRealMatrix<DataType>::CopyDataToDevice(DataTypeOut *destination_data, bool is_row_major) {

	cudaError_t result;

	//Ensure valid parameters
	if (destination_data == NULL) {
		return false;
	}

	if (!is_row_major) {
		
		//Determine the number of threads needed to transpose the matrix
		dim3 threadBlock(16, 16);
		dim3 threadGrid((threadBlock.x + n_columns - 1)/threadBlock.x,
						(threadBlock.y + n_rows    - 1)/threadBlock.y);

		//If they the destination type is the same as the current type
		//just transpose and copy right into destination buffer
		if (typeid(DataType) == typeid(DataTypeOut)) {

			//Transpose the temp buffer
			Transpose<DataType><<<threadGrid, threadBlock>>>( (DataType *) destination_data, (DataType *) matrix_buffer, n_columns, n_rows);
			if (cudaGetLastError() != cudaSuccess) {
				return false;
			}

		}
		else {//We must cast

			TransposeAndCast<DataType, DataTypeOut><<<threadGrid, threadBlock>>>(destination_data, matrix_buffer, n_columns, n_rows);

		}
	}
	else {//No transposing needed

		//If the destination buffer and the source buffer are of the same type
		//the just use cudaMemcpy
		if (typeid(DataType) == typeid(DataTypeOut)) {

			result = cudaMemcpy(destination_data, matrix_buffer, n_rows*n_columns*sizeof(DataType), cudaMemcpyDeviceToDevice);

			if (result != cudaSuccess)
				return false;

		}//We must cast
		else {
			dim3 threadBlock(16, 16);
			dim3 threadGrid((threadBlock.x + n_columns - 1)/threadBlock.x,
							(threadBlock.y + n_rows - 1)/threadBlock.y);

			Cast<DataType, DataTypeOut><<<threadGrid, threadBlock>>>(matrix_buffer, destination_data, n_rows, n_columns);
			if (cudaGetLastError() != cudaSuccess) {
				return false;
			}
		}
	}
	
	return true;
}

template <class DataType> template <class DataTypeOut>
bool CUDA2DRealMatrix<DataType>::CopyROIToDevice(DataTypeOut * destination_data, bool is_row_major, 
				                                 unsigned long top_row, unsigned long left_column, 
					                             unsigned long bottom_row, unsigned long right_column) {

	if (destination_data == NULL) {
		return false;
	}

	if (matrix_buffer == NULL || n_rows == 0 || n_columns == 0) {
		return false;
	}

	if (bottom_row >= n_rows || top_row > bottom_row || 
		right_column >= n_columns || left_column > right_column) {
		return false;
	}

	unsigned long ROI_rows    = bottom_row   - top_row + 1;
	unsigned long ROI_columns = right_column - left_column + 1;

	dim3 block_dim(16, 16);
	dim3 grid_dim((block_dim.x + ROI_columns    - 1)/block_dim.x,
				  (block_dim.y + ROI_rows       - 1)/block_dim.y);

	if (typeid(DataTypeOut) != typeid(DataType)) {

		CUDA2DRealMatrix<DataType> temp_matrix(ROI_rows, ROI_columns);

		DataType * temp_pointer;

		if (!temp_matrix.GetPointerToData(&temp_pointer)) {
			return false;
		}

		GetROI<DataType><<<grid_dim, block_dim>>>(matrix_buffer, n_columns, temp_pointer, ROI_columns, ROI_rows, top_row, left_column);
		if (cudaGetLastError() != cudaSuccess) {
			return false;
		}

		if (!temp_matrix.CopyDataToDevice<DataTypeOut>(destination_data, is_row_major)) {
			return false;
		}
	}
	else {

		GetROI<DataType><<<grid_dim, block_dim>>>((DataType *) matrix_buffer, n_columns, (DataType *)destination_data, ROI_columns, ROI_rows, top_row, left_column);
		if (cudaGetLastError() != cudaSuccess) {
			return false;
		}

	}

	return true;

}


template <class DataType> 
template <class DataTypeIn>
bool CUDA2DRealMatrix<DataType>::AssignFromDevice(CUDA2DRealMatrix<DataTypeIn>  &input_data,
                                                  CUDA2DRealMatrix<int>         &row_locations, 
                                                  CUDA2DRealMatrix<int>         &col_locations) {

	if (matrix_buffer == NULL || n_rows == 0 || n_columns == 0) {
		return false;
	}

    unsigned long n_rows_input, n_cols_input;
    
    if (!input_data.GetSize(n_rows_input, n_cols_input)) {
        return false;
    }

    if (n_rows_input != 1) {
        return false;
    }

    unsigned long n_row_locs, n_col_locs, dummy;

    if (!row_locations.GetSize(dummy, n_row_locs)) {
        return false;
    }

    if (dummy != 1) {
        return false;
    }

    if (!col_locations.GetSize(dummy, n_col_locs)) {
        return false;
    }

    if (dummy != 1) {
        return false;
    }

    if (n_row_locs != n_col_locs) {
        return false;
    }

    DataTypeIn * input_data_pointer;
    int * row_locs_pointer;
    int * col_locs_pointer;

    if (!input_data.GetPointerToData(&input_data_pointer)) {
        return false;
    }

    if (!row_locations.GetPointerToData(&row_locs_pointer)) {
        return false;
    }

    if (!col_locations.GetPointerToData(&col_locs_pointer)) {
        return false;
    }


	dim3 block_dim(512);
	dim3 grid_dim((block_dim.x + n_row_locs - 1)/block_dim.x);

	Assign<DataType, DataTypeIn><<<grid_dim, block_dim>>>(matrix_buffer, n_rows,
                                                         n_columns, input_data_pointer,
                                                         n_row_locs, row_locs_pointer, col_locs_pointer);
	if (cudaGetLastError() != cudaSuccess) {
		return false;
	}

    return true;
}


template <class DataType> 
template <class DataTypeOut>
bool CUDA2DRealMatrix<DataType>::Get(CUDA2DRealMatrix<DataTypeOut>  &output_data,
                                     CUDA2DRealMatrix<int>         &row_locations, 
                                     CUDA2DRealMatrix<int>         &col_locations) {

	if (matrix_buffer == NULL || n_rows == 0 || n_columns == 0) {
		return false;
	}

    unsigned long n_rows_input, n_cols_input;
    
    if (!output_data.GetSize(n_rows_input, n_cols_input)) {
        return false;
    }

    if (n_rows_input != 1) {
        return false;
    }

    unsigned long n_row_locs, n_col_locs, dummy;

    if (!row_locations.GetSize(dummy, n_row_locs)) {
        return false;
    }

    if (dummy != 1) {
        return false;
    }

    if (!col_locations.GetSize(dummy, n_col_locs)) {
        return false;
    }

    if (dummy != 1) {
        return false;
    }

    if (n_row_locs != n_col_locs) {
        return false;
    }

    DataTypeOut * output_data_pointer;
    int * row_locs_pointer;
    int * col_locs_pointer;

    if (!output_data.GetPointerToData(&output_data_pointer)) {
        return false;
    }

    if (!row_locations.GetPointerToData(&row_locs_pointer)) {
        return false;
    }

    if (!col_locations.GetPointerToData(&col_locs_pointer)) {
        return false;
    }

	dim3 block_dim(512);
	dim3 grid_dim((block_dim.x + n_row_locs - 1)/block_dim.x);

	GetKernel<DataType, DataTypeOut><<<grid_dim, block_dim>>>(matrix_buffer, n_rows,
                                                       n_columns, output_data_pointer,
                                                       n_row_locs, row_locs_pointer, col_locs_pointer);

    cudaError_t cuda_error = cudaGetLastError();

	if (cuda_error != cudaSuccess) {
 //       std::cout << cudaGetErrorString(cuda_error) << std::endl;
 //       std::cout << cuda_error << std::endl;
 //       std::cout << grid_dim.x << " " << grid_dim.y << std::endl;
 //       std::cout << block_dim.x<< " " << block_dim.y << std::endl;

		return false;
	}

    return true;
}

/**
 * ToString
 ***/

template <class DataType>
std::string CUDA2DRealMatrix<DataType>::ToString() {

	std::stringstream str;

	double * matrix_data = (double *) calloc(n_rows*n_columns, sizeof(double)); 
	if (!CopyDataToHost<double>(matrix_data, true)) {
		str << "Could not retrieve data from device!\n";
	}
	else {
		for (unsigned long row = 0; row < n_rows; row++) {
			for (unsigned long col = 0; col < n_columns; col++) {

				if (col > 0) {
					str << ",\t";
				}

				str << (double) matrix_data[row * n_columns + col];
			}
			str << "\n";
		}
	}
	free(matrix_data);

	return str.str();
}

/**
 * ToString overloaded with for ROI
 ***/

template<class DataType>
std::string CUDA2DRealMatrix<DataType>::ToString(unsigned long min_row, unsigned long max_row, unsigned long min_column, unsigned long max_column) {

	std::stringstream	str;
	cudaError_t		result;

	if (min_row    >= max_row    || max_row    >= n_rows ||
		min_column >= max_column || max_column >= n_columns ||
		matrix_buffer == NULL) {
			str << "Invalid ToString parameters or matrix not initialized!\n"; 

			return str.str();
	}

	unsigned long ROI_rows    = max_row    - min_row    + 1;
	unsigned long ROI_columns = max_column - min_column + 1;
	DataType * host_buffer    = (DataType *) malloc(ROI_rows * ROI_columns * sizeof(DataType));

	void *copy_start_device   = (void *)(((unsigned long long) matrix_buffer) + ((unsigned long long)sizeof(DataType)*(min_row * n_columns + min_column)));
	void *copy_start_host     = (void *) host_buffer;

	for (unsigned long current_row = 0; current_row < ROI_rows; current_row++) {

		// copying one fraction of a line at a time
		result                = cudaMemcpy(copy_start_host, copy_start_device, sizeof(DataType) * ROI_columns, cudaMemcpyDeviceToHost);

		if (result != cudaSuccess) {
			free(host_buffer);

			str << "Invalid ToString parameters or matrix not initialized!\n"; 
			return str.str();
		}

		copy_start_device = (void *) (((unsigned long long) copy_start_device) + ((unsigned long long) sizeof(DataType) * n_columns));
		copy_start_host   = (void *) (((unsigned long long) copy_start_host)   + ((unsigned long long) sizeof(DataType) * ROI_columns));

	}

	for (unsigned long row = 0; row < ROI_rows; row++) {
		for (unsigned long col = 0; col < ROI_columns; col++) {

				if (col > 0) {
					str << ",\t";
				}


			str << host_buffer[row * ROI_columns + col];
		}
		str << "\n";
	}
	free(host_buffer);

	return str.str();
}

/**
 * Comparison operator
 **/

template<class DataType>
bool CUDA2DRealMatrix<DataType>::operator== (CUDA2DRealMatrix<DataType> &rhs) {

	unsigned long n_rows_rhs;
	unsigned long n_columns_rhs;

	if (!rhs.GetSize(n_rows_rhs, n_columns_rhs)) {
		return false;
	}

	if (n_rows_rhs != n_rows || n_columns_rhs != n_columns) {
		return false;
	}

	DataType * host_matrix     = (DataType *) malloc(sizeof(DataType)*n_rows*n_columns);
	DataType * host_matrix_rhs = (DataType *) malloc(sizeof(DataType)*n_rows*n_columns);

	if (!this->CopyDataToHost<DataType>(host_matrix, true)) {
		free(host_matrix);
		free(host_matrix_rhs);
		return false;
	}

	if (!rhs.CopyDataToHost<DataType>(host_matrix_rhs, true)) {
		free(host_matrix);
		free(host_matrix_rhs);
		return false;
	}

	if (memcmp(host_matrix, host_matrix_rhs, sizeof(DataType)*n_rows*n_columns) != 0) {
		free(host_matrix);
		free(host_matrix_rhs);
		return false;
	}

	free(host_matrix);
	free(host_matrix_rhs);

	return true;
}
template <class DataType>
template <class DataTypeIn>
bool CUDA2DRealMatrix<DataType>::Eq(CUDA2DRealMatrix<DataTypeIn> &other) {

    CUDA2DRealMatrix<DataType> * other_pointer;

    if (typeid(DataType) != typeid(DataTypeIn)) {
        other_pointer = new CUDA2DRealMatrix<DataType>(n_rows, n_columns);
        other.template Copy<DataType>(*other_pointer);
    }
    else {
        other_pointer = (CUDA2DRealMatrix<DataType> *) &other;
    }

    bool are_equal = *other_pointer == *this;

    if (typeid(DataType) != typeid(DataTypeIn)) {
        delete other_pointer;
    }
    return are_equal;
}
/**
 * Copy
 **/

template <class DataType> 
template <class DataTypeIn>
bool CUDA2DRealMatrix<DataType>::Copy(CUDA2DRealMatrix<DataTypeIn> &other) {
	if(!other.SetSize(this->n_rows, this->n_columns)) {
		return false;
	}
	
	if(!other.template CopyDataFromDevice<DataType>(this->matrix_buffer, true)) {
		return false;
	}

	return true;

}

template <class DataType>
bool CUDA2DRealMatrix<DataType>::IsValid() {

	if (matrix_buffer == NULL || n_rows == 0 || n_columns == 0) {
		return false;
	}

	return true;

}

		// Returns a string representing the type that this matrix stores
		// uint8  -> 8 bit unsigned integers
		// uint32 -> 32 bit unsigned integers
		// int32  -> 32 bit signed integers
		// int8   -> 8 bit signed integer
		// single -> 32 bit signed floating point numbers
		// double -> 64 bit signed floating point numbers
		// DO NOT free the char array that is returned, this class will handle that

template <class DataType>
char * CUDA2DRealMatrix<DataType>::GetType() {

	if (type == NULL) {

		const type_info & our_type = typeid(DataType);

		if (our_type == typeid(unsigned char)) {
			type = (char *) malloc(sizeof(char)*(strlen("uint8") + 1));
			strcpy(type, "uint8");
		}
		else if (our_type == typeid(unsigned int)) {
			type = (char *) malloc(sizeof(char)*(strlen("uint32") + 1));
			strcpy(type, "uint32");
		}
		else if (our_type == typeid(int)) {
			type = (char *) malloc(sizeof(char)*(strlen("int32") + 1));
			strcpy(type, "int32");
		}
		else if (our_type == typeid(char)) {
			type = (char *) malloc(sizeof(char)*(strlen("int8") + 1));
			strcpy(type, "int8");
		}
		else if (our_type == typeid(float)) {
			type = (char *) malloc(sizeof(char)*(strlen("single") + 1));
			strcpy(type, "single");
		}
		else if (our_type == typeid(double)) {
			type = (char *) malloc(sizeof(char)*(strlen("double") + 1));
			strcpy(type, "double");
		}
		else {
			type = (char *) malloc(sizeof(char)*(strlen("Unkown") + 1));
			strcpy(type, "Unkown");
		}
	}

	return type;
}	


template <class DataType> template <class T>
bool CUDA2DRealMatrix<DataType>::add(CUDA2DRealMatrix<T> &other, unsigned long ROI_width, unsigned long ROI_height, 
				 unsigned long ROI_start_row_this,  unsigned long ROI_start_col_this,
				 unsigned long ROI_start_row_other, unsigned long ROI_start_col_other) {

	unsigned long other_n_rows;
	unsigned long other_n_cols;

	if (!other.GetSize(other_n_rows, other_n_cols)) {
		return false;
	}

	if (ROI_start_row_other + ROI_height > other_n_rows ||
		ROI_start_col_other + ROI_width  > other_n_cols) {
		return false;
	}

	if (ROI_start_row_this + ROI_height > n_rows ||
		ROI_start_col_this + ROI_width  > n_columns) {
		return false;
	}

	T * other_data_pointer;
	if (!other.GetPointerToData(&other_data_pointer)) {
		return false;
	}

	// Launch the kernal that will add the ROI
	dim3 threadBlock(16, 16);
	dim3 threadGrid((threadBlock.x + ROI_width  - 1)/threadBlock.x,
				    (threadBlock.y + ROI_height - 1)/threadBlock.y);

	AddROI<DataType, T><<<threadGrid, threadBlock>>>(matrix_buffer, other_data_pointer, ROI_width, ROI_height, 
		   ROI_start_row_this, ROI_start_col_this, ROI_start_row_other, 
		   ROI_start_col_other, n_columns, other_n_cols);

	if (cudaGetLastError() != cudaSuccess) {
		return false;
	}

	return true;

}


template <class DataType> template <class T>
bool CUDA2DRealMatrix<DataType>::sub(CUDA2DRealMatrix<T> &other, unsigned long ROI_width, unsigned long ROI_height, 
				 unsigned long ROI_start_row_this,  unsigned long ROI_start_col_this,
				 unsigned long ROI_start_row_other, unsigned long ROI_start_col_other) {

	unsigned long other_n_rows;
	unsigned long other_n_cols;

	if (!other.GetSize(other_n_rows, other_n_cols)) {
		return false;
	}

	if (ROI_start_row_other + ROI_height > other_n_rows ||
		ROI_start_col_other + ROI_width  > other_n_cols) {
		return false;
	}

	if (ROI_start_row_this + ROI_height > n_rows ||
		ROI_start_col_this + ROI_width  > n_columns) {
		return false;
	}

	T * other_data_pointer;
	if (!other.GetPointerToData(&other_data_pointer)) {
		return false;
	}

	// Launch the kernal that will add the ROI
	dim3 threadBlock(16, 16);
	dim3 threadGrid((threadBlock.x + ROI_width  - 1)/threadBlock.x,
				    (threadBlock.y + ROI_height - 1)/threadBlock.y);

	SubROI<DataType, T><<<threadGrid, threadBlock>>>(matrix_buffer, other_data_pointer, ROI_width, ROI_height, 
		   ROI_start_row_this, ROI_start_col_this, ROI_start_row_other, 
		   ROI_start_col_other, n_columns, other_n_cols);

	if (cudaGetLastError() != cudaSuccess) {
		return false;
	}

	return true;

}


template <class DataType> template <class T>
bool CUDA2DRealMatrix<DataType>::div(CUDA2DRealMatrix<T> &other, unsigned long ROI_width, unsigned long ROI_height, 
				 unsigned long ROI_start_row_this,  unsigned long ROI_start_col_this,
				 unsigned long ROI_start_row_other, unsigned long ROI_start_col_other) {

	unsigned long other_n_rows;
	unsigned long other_n_cols;

	if (!other.GetSize(other_n_rows, other_n_cols)) {
		return false;
	}

	if (ROI_start_row_other + ROI_height > other_n_rows ||
		ROI_start_col_other + ROI_width  > other_n_cols) {
		return false;
	}

	if (ROI_start_row_this + ROI_height > n_rows ||
		ROI_start_col_this + ROI_width  > n_columns) {
		return false;
	}

	T * other_data_pointer;
	if (!other.GetPointerToData(&other_data_pointer)) {
		return false;
	}

	// Launch the kernal that will add the ROI
	dim3 threadBlock(16, 16);
	dim3 threadGrid((threadBlock.x + ROI_width  - 1)/threadBlock.x,
				    (threadBlock.y + ROI_height - 1)/threadBlock.y);

	DivROI<DataType, T><<<threadGrid, threadBlock>>>(matrix_buffer, other_data_pointer, ROI_width, ROI_height, 
		   ROI_start_row_this, ROI_start_col_this, ROI_start_row_other, 
		   ROI_start_col_other, n_columns, other_n_cols);

	if (cudaGetLastError() != cudaSuccess) {
		return false;
	}

	return true;

}


template <class DataType> template <class T>
bool CUDA2DRealMatrix<DataType>::mult(CUDA2DRealMatrix<T> &other, unsigned long ROI_width, unsigned long ROI_height, 
				 unsigned long ROI_start_row_this,  unsigned long ROI_start_col_this,
				 unsigned long ROI_start_row_other, unsigned long ROI_start_col_other) {

	unsigned long other_n_rows;
	unsigned long other_n_cols;

	if (!other.GetSize(other_n_rows, other_n_cols)) {
		return false;
	}

	if (ROI_start_row_other + ROI_height > other_n_rows ||
		ROI_start_col_other + ROI_width  > other_n_cols) {
		return false;
	}

	if (ROI_start_row_this + ROI_height > n_rows ||
		ROI_start_col_this + ROI_width  > n_columns) {
		return false;
	}

	T * other_data_pointer;
	if (!other.GetPointerToData(&other_data_pointer)) {
		return false;
	}

	// Launch the kernal that will add the ROI
	dim3 threadBlock(16, 16);
	dim3 threadGrid((threadBlock.x + ROI_width  - 1)/threadBlock.x,
				    (threadBlock.y + ROI_height - 1)/threadBlock.y);

	MultROI<DataType, T><<<threadGrid, threadBlock>>>(matrix_buffer, other_data_pointer, ROI_width, ROI_height, 
		   ROI_start_row_this, ROI_start_col_this, ROI_start_row_other, 
		   ROI_start_col_other, n_columns, other_n_cols);

	if (cudaGetLastError() != cudaSuccess) {
		return false;
	}

	return true;

}


template <class DataType>
bool CUDA2DRealMatrix<DataType>::LogBase10(unsigned long ROI_width, unsigned long ROI_height, 
									 unsigned long ROI_start_row, unsigned long ROI_start_col) {
	if (matrix_buffer == NULL) {
		return false;
	}

	if (ROI_start_row + ROI_height > n_rows ||
		ROI_start_col + ROI_width  > n_columns) {
		return false;
	}

	// Launch the kernal that will add the ROI
	dim3 threadBlock(16, 16);
	dim3 threadGrid((threadBlock.x + ROI_width  - 1)/threadBlock.x,
				    (threadBlock.y + ROI_height - 1)/threadBlock.y);

	LogROI<DataType> <<<threadGrid, threadBlock>>>(matrix_buffer,
												  ROI_width, ROI_height, 
												  ROI_start_row, ROI_start_col,
												  n_columns);

	if (cudaGetLastError() != cudaSuccess) {
		return false;
	}

	return true;
	

}


template <class DataType>
bool CUDA2DRealMatrix<DataType>::LogBase10() {
    return LogBase10(n_columns, n_rows, 0, 0);
}

template <class DataType>
template <class OTHER_TYPE>
CUDA2DRealMatrix<DataType> CUDA2DRealMatrix<DataType>::operator+(const CUDA2DRealMatrix<OTHER_TYPE> &b) const {

    CUDA2DRealMatrix<DataType> fail_matrix(0, 0);

	if (matrix_buffer == NULL || n_rows == 0 || n_columns == 0) {
		return fail_matrix;
	}

    unsigned n_rows_other, n_columns_other;

    if (!b.GetSize(n_rows_other, n_columns_other)) {
        return fail_matrix;
    }

    if (n_rows_other != n_rows || n_columns != n_columns_other) {
        return fail_matrix;
    }

    CUDA2DRealMatrix<DataType> result(*this);

    if (!result.add<OTHER_TYPE>(b, n_columns, n_rows, 0, 0, 0, 0)) {
        return fail_matrix;
    }

    return result;

}


template <class DataType>
bool CUDA2DRealMatrix<DataType>::AddScalar(float val, unsigned long ROI_width, unsigned long ROI_height, 
										   unsigned long ROI_start_row, unsigned long ROI_start_col) {
	if (matrix_buffer == NULL) {
		return false;
	}

	if (ROI_start_row + ROI_height > n_rows ||
		ROI_start_col + ROI_width  > n_columns) {
		return false;
	}

	// Launch the kernal that will add the ROI
	dim3 threadBlock(16, 16);
	dim3 threadGrid((threadBlock.x + ROI_width  - 1)/threadBlock.x,
				    (threadBlock.y + ROI_height - 1)/threadBlock.y);

	AddScalarROI<DataType> <<<threadGrid, threadBlock>>>(matrix_buffer, val, 
														ROI_width, ROI_height, 
														ROI_start_row, ROI_start_col,
														n_columns);

	if (cudaGetLastError() != cudaSuccess) {
		return false;
	}

	return true;
	

}


template <class DataType>
bool CUDA2DRealMatrix<DataType>::MultScalar(float val, unsigned long ROI_width, unsigned long ROI_height, 
										   unsigned long ROI_start_row, unsigned long ROI_start_col) {
	if (matrix_buffer == NULL) {
		return false;
	}

	if (ROI_start_row + ROI_height > n_rows ||
		ROI_start_col + ROI_width  > n_columns) {
		return false;
	}

	// Launch the kernal that will add the ROI
	dim3 threadBlock(16, 16);
	dim3 threadGrid((threadBlock.x + ROI_width  - 1)/threadBlock.x,
				    (threadBlock.y + ROI_height - 1)/threadBlock.y);

	MultScalarROI<DataType> <<<threadGrid, threadBlock>>>(matrix_buffer, val, 
														ROI_width, ROI_height, 
														ROI_start_row, ROI_start_col,
														n_columns);

	if (cudaGetLastError() != cudaSuccess) {
		return false;
	}

	return true;
	

}


template <class DataType>
bool CUDA2DRealMatrix<DataType>::Set(float val, unsigned long ROI_width, unsigned long ROI_height, 
										   unsigned long ROI_start_row, unsigned long ROI_start_col) {
	if (matrix_buffer == NULL) {
		return false;
	}

	if (ROI_start_row + ROI_height > n_rows ||
		ROI_start_col + ROI_width  > n_columns) {
		return false;
	}

	// Launch the kernal that will add the ROI
	dim3 threadBlock(16, 16);
	dim3 threadGrid((threadBlock.x + ROI_width  - 1)/threadBlock.x,
				    (threadBlock.y + ROI_height - 1)/threadBlock.y);

	SetROI<DataType><<<threadGrid, threadBlock>>>(matrix_buffer, val, 
														ROI_width, ROI_height, 
														ROI_start_row, ROI_start_col, n_columns);

	if (cudaGetLastError() != cudaSuccess) {
		return false;
	}

	return true;

}

//
// Perform affine transformation
//

template <class DataType>
bool CUDA2DRealMatrix<DataType>::PerformTransformation(float affine_matrix[4], CUDA2DRealMatrix<float> &result_matrix, 
						   CUDA2DRealMatrix<int> &image_mask, CUDA2DInterpolator * interpolator) {


	// Get the size of the result matrix
	unsigned long n_rows_result_matrix, n_columns_result_matrix;
	if (!result_matrix.GetSize(n_rows_result_matrix, n_columns_result_matrix)) {
		return false;
	}

	if (!image_mask.SetSize(n_rows_result_matrix, n_columns_result_matrix)) {
		return false;
	}

    // Create the 2D matrices that will contain the row coordinates and collumn coordinates
	CUDA2DRealMatrix<float> row_coordinates(1, n_rows_result_matrix*n_columns_result_matrix);
	CUDA2DRealMatrix<float> col_coordinates(1, n_rows_result_matrix*n_columns_result_matrix);

	// Get the pointers required to perform the computation
	int * image_mask_pointer;
	if (!image_mask.GetPointerToData(&image_mask_pointer)) {
		return false;
	}

	float * row_coordinates_pointer;
	if (!row_coordinates.GetPointerToData(&row_coordinates_pointer)) {
		return false;
	}

	float * col_coordinates_pointer;
	if (!col_coordinates.GetPointerToData(&col_coordinates_pointer)) {
		return false;
	}

	// copy the affine matrix to the gpu
	float * device_affine_matrix;
	if (cudaMalloc(&device_affine_matrix, sizeof(float)*4) != cudaSuccess) {
		return false;
	}

	if (cudaMemcpy(device_affine_matrix, affine_matrix, sizeof(float)*4, 
				   cudaMemcpyHostToDevice) != cudaSuccess) {
		cudaFree(device_affine_matrix);
		return false;
	}

	// Determine what coordinates we should interpolate on
	// calculate the grid structure
	dim3 threadBlock(16, 16);
	dim3 threadGrid((threadBlock.x + n_columns_result_matrix  - 1)/threadBlock.x,
				    (threadBlock.y + n_rows_result_matrix - 1)/threadBlock.y);

	// Launch ther kernel
	AffineTransformation<<<threadGrid, threadBlock>>>(image_mask_pointer, n_rows, n_columns, 
										n_rows_result_matrix, n_columns_result_matrix, 
										row_coordinates_pointer, col_coordinates_pointer, 
										device_affine_matrix);

	if (cudaGetLastError() != cudaSuccess) {

		return false;
	}

	// If the current type of this matrix is not float then we cast to float.
	// this reference variable will either be this or the casted data
	CUDA2DRealMatrix<float> * input_reference;
	// TODO should we support double precision?
	if (std::string(this->GetType()) != "single") {
		input_reference = new CUDA2DRealMatrix<float>(n_rows, n_columns);
		this->Copy<float>(*input_reference);
	}
	else {
		input_reference = (CUDA2DRealMatrix<float> *) this;
	}

	// Now interpolate on the coordinates determined by the AffineTransformation kernel
	if (!interpolator->CUDA2DInterpolate(*input_reference, result_matrix, row_coordinates, col_coordinates, 0.0)) {
		cudaFree(device_affine_matrix);
		return false;
	}

	// free the allocated device memory if we have done that
	if (std::string(this->GetType()) != "single") {

		delete input_reference;
	}

	// The interpolator object reorganizes the matrix shape, so we must put it back the way it was
	// Note, if the total number of elements didn't change when SetSize is called, then the matrix
	// data is retained
	if (!result_matrix.SetSize(n_rows_result_matrix, n_columns_result_matrix)) {
		cudaFree(device_affine_matrix);
		return false;
	}
	cudaFree(device_affine_matrix);

	return true;
}

//
// Rotate
//

template <class DataType>
bool CUDA2DRealMatrix<DataType>::Rotate(double rotation_in_degrees, CUDA2DRealMatrix<float> &result_matrix, 
			CUDA2DRealMatrix<int> &image_mask, CUDA2DInterpolator * interpolator) {
				
	double rotation_in_radians = PI*rotation_in_degrees/180.0;

	float affine_matrix[4];

	affine_matrix[0] = cos(rotation_in_radians);
	affine_matrix[1] = -sin(rotation_in_radians);
	affine_matrix[2] = sin(rotation_in_radians);
	affine_matrix[3] = cos(rotation_in_radians);

	return PerformTransformation(affine_matrix, result_matrix, image_mask, interpolator);
}

//
// Scale
//

template <class DataType>
bool CUDA2DRealMatrix<DataType>::Scale(double x_scaling_factor, double y_scaling_factor, CUDA2DRealMatrix<float> &result,
			       CUDA2DRealMatrix<int> &image_mask, CUDA2DInterpolator * interpolator) {

	float affine_matrix[4];
	affine_matrix[0] = x_scaling_factor;
	affine_matrix[1] = 0;
	affine_matrix[2] = 0;
	affine_matrix[3] = y_scaling_factor;

	return PerformTransformation(affine_matrix, result, image_mask, interpolator);
}

template <class DataType>
double CUDA2DRealMatrix<DataType>::sum() {

    if (matrix_buffer == NULL) {
		return 0;
	}

    unsigned long n_elements = n_rows * n_columns;

    const size_t block_size = 512; 
    const size_t num_blocks = (n_elements/block_size) + ((n_elements%block_size) ? 1 : 0);

    float * d_partial_sums_and_total;
    cudaMalloc((void**)&d_partial_sums_and_total, sizeof(float) * (num_blocks + 1)); // we don't need the + 1 at the end, its needed to use the kernel properly

    // launch one kernel to compute, per-block, a partial sum
    block_sum<DataType> <<<num_blocks,block_size,block_size * sizeof(float)>>>(matrix_buffer, d_partial_sums_and_total, n_elements);

    // We can't use this again, the dimension of the block will be too big for even
    // modest data set sizes on most GPU architectures. We could do it three times
    // to break up the problem even more, but all of the kernel launches might not
    // be faster than just copying the <1000 data points back to the host and counting 
    // it here
    //// launch a single block to compute the sum of the partial sums
    //block_sum<float> <<<1,num_blocks, num_blocks * sizeof(float)>>>(d_partial_sums_and_total, d_partial_sums_and_total + num_blocks, num_blocks);
    //// copy the result back to the host
    //DataType device_result = 0;
    //cudaMemcpy(&device_result, d_partial_sums_and_total + num_blocks, sizeof(float), cudaMemcpyDeviceToHost);

    float * host_partial_sums = (float *) malloc(sizeof(float)*num_blocks);
    cudaMemcpy(host_partial_sums, d_partial_sums_and_total, sizeof(float)*num_blocks, cudaMemcpyDeviceToHost);

    float sum = 0;

    for (int i = 0; i < num_blocks; i++) {
        sum += host_partial_sums[i];
    }

    // deallocate device memory
    cudaFree(d_partial_sums_and_total);
    free(host_partial_sums);

    return (double) sum;
}

template <class DataType>
double CUDA2DRealMatrix<DataType>::mean() {

	return sum() / ((double)(n_rows*n_columns));

}

template <class DataType>
bool CUDA2DRealMatrix<DataType>::ClampMax(double val) {

	dim3 threadBlock(16, 16);
	dim3 threadGrid((threadBlock.x + n_columns  - 1)/threadBlock.x,
				    (threadBlock.y + n_rows - 1)/threadBlock.y);

	ClampMaxKernel<DataType> <<<threadGrid, threadBlock>>>(matrix_buffer, (float) val, n_rows, n_columns);

	if (cudaGetLastError() != cudaSuccess) {

		return false;
	}

	return true;
}

template <class DataType>
bool CUDA2DRealMatrix<DataType>::ClampMin(double val) {

	dim3 threadBlock(16, 16);
	dim3 threadGrid((threadBlock.x + n_columns  - 1)/threadBlock.x,
				    (threadBlock.y + n_rows - 1)/threadBlock.y);

	ClampMinKernel<DataType> <<<threadGrid, threadBlock>>>(matrix_buffer, (float) val, n_rows, n_columns);

	if (cudaGetLastError() != cudaSuccess) {

		return false;
	}

	return true;

}