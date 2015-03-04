#pragma once
// Define this macro for debugging
#define DEBUG_SEARCH
#include "cuda_runtime.h"

// NOTE: the time required to find a maximum or a minimum in the GPU is 
//       comparable to that required by the CPU. The question is then, why
//       bother writing code to do it in the GPU???. In our case, we already
//       have the data in the GPU, and thus copying the data back to the CPU
//       and then finding the max (min) would be MUCH slower. We are better
//       off by finding the max (min) in the GPU and then copying only the
//       results back to the CPU. This might not be the case for other users.
//
//       Because of the comparable speed performance to that of the CPU
//       it is not worth it to generate Python or Matlab interfaces.

//////////////////////////////////////////
// Find max an min over a region of interest
//////////////////////////////////////////

template <class T>
__global__ void CUDAFindRowROIMin(T *dataSet,
								  unsigned long *column_min,
								  T             *result,
								  unsigned long  num_cols,
								  unsigned long  top_row,
								  unsigned long  left_column,
								  unsigned long  bottom_row,
								  unsigned long  right_column) {
 
    //Indexing variables
	unsigned long	index              = blockIdx.x * blockDim.x + threadIdx.x;
	unsigned long	num_rows_to_search = bottom_row   - top_row;
	unsigned long	num_cols_to_search = right_column - left_column;

	// auxiliary variables for faster access. This is a register, as opposed to
	// a variable in global memory (400/1 times lower latency access).
	unsigned long	tempCol = 0;
	T				temp_val;

	// pre-calculating multiplications
	unsigned long another_shift = (index + top_row) * num_cols + left_column;

	if (index <= num_rows_to_search) {

		// starting the search by assigning the first element of the ROI to the minimum value
		T               min_val   = dataSet[another_shift];

		for (unsigned long cur_col = 1; cur_col <= num_cols_to_search; cur_col++) {
			temp_val				= dataSet[another_shift + cur_col];

			if (temp_val <= min_val) {
				min_val = temp_val;
				tempCol = cur_col;
			}

		}

		//Write the values to global arrays
		result[index]     = min_val;
		column_min[index] = tempCol;
	}
}

template <class T>
__global__ void CUDAFindRowROIMax(T *dataSet,
								  unsigned long *column_max,
								  T *result,
								  unsigned long num_cols,
								  unsigned long top_row, 
								  unsigned long left_column, 
								  unsigned long bottom_row, 
								  unsigned long right_column) {

    //Indexing variables
	unsigned long index              = blockIdx.x*blockDim.x + threadIdx.x;
	unsigned long num_rows_to_search = bottom_row - top_row;
	unsigned long num_cols_to_search = right_column - left_column;
	
	// auxiliary variables for faster access. This is a register, as opposed to
	// a variable in global memory (400/1 times lower latency access).
	unsigned long	tempCol = 0;
	T				temp_val;
		
	// pre-calculating multiplications
	unsigned long another_shift = (index + top_row) * num_cols + left_column;

	if (index <= num_rows_to_search) {

		// starting the search by assigning the first element of the ROI to the maximum value
		T max_val = dataSet[another_shift];

		for (unsigned long cur_col = 1; cur_col <= num_cols_to_search; cur_col++) {
			temp_val			= dataSet[another_shift + cur_col];

			if (temp_val >= max_val) {
				max_val = temp_val;
				tempCol = cur_col;
			}
		}

		//Write the values to global arrays
		result[index]     = max_val;
		column_max[index] = tempCol;
	}
}

template <class T>
__global__ void CUDAFindRowMaxWithOverlappingPixelThreshold(T *dataSet,
													T * overlapping_pixels,
													unsigned long pixel_overlap_threshold,
												    unsigned long *column_max,
												    T *result,
												    unsigned long num_cols,
												    unsigned long top_row, 
												    unsigned long left_column, 
												    unsigned long bottom_row, 
												    unsigned long right_column,
                                                    bool          debug) {

    //Indexing variables
	unsigned long index              = blockIdx.x*blockDim.x + threadIdx.x;
	unsigned long num_rows_to_search = bottom_row - top_row;
	unsigned long num_cols_to_search = right_column - left_column;
	
	// auxiliary variables for faster access. This is a register, as opposed to
	// a variable in global memory (400/1 times lower latency access).
	unsigned long	tempCol = 0;
	T				temp_val;
		
	// pre-calculating multiplications
	unsigned long another_shift = (index + top_row) * num_cols + left_column;

	if (index <= num_rows_to_search) {

		// starting the search by assigning the first element of the ROI to the maximum value
		T max_val       = 0;
		
		for (unsigned long cur_col = 0; cur_col <= num_cols_to_search; cur_col++) {

			temp_val			= dataSet[another_shift + cur_col];
			T n_overlapping = overlapping_pixels[another_shift + cur_col];

			// determine if we can consider this pixel or not
			if (n_overlapping >= pixel_overlap_threshold) {

				if (temp_val >= max_val) {
					max_val = temp_val;
					tempCol = cur_col;
				}

			}

            if (debug) {
			    // determine if we can consider this pixel or not
			    if (n_overlapping >= pixel_overlap_threshold) {
                    overlapping_pixels[another_shift + cur_col] = 1.0;
			    }
                else {
                    overlapping_pixels[another_shift + cur_col] = 0.0;
                }
            }

		}

		//Write the values to global arrays
		result[index]     = max_val;
		column_max[index] = tempCol;
	}
}


/****
 * Cpp functions for calling ROI max and min kernels
 ****/

template <class T>
bool FindROIMax(T *dataSet,
				unsigned long numRows,
				unsigned long numCols, 
				unsigned long &rowMax, 
				unsigned long &colMax,
				T			  &maxVal,
				unsigned long top_row,
				unsigned long left_column, 
				unsigned long bottom_row, 
				unsigned long right_column,
				T * overlapping_pixels = NULL,
				unsigned long pixel_overlap_threshold = 1,
                bool debug = false) {

	// Ensure that the ROI coordinates passed are valid
	 if (bottom_row >= numRows || right_column >= numCols || top_row > bottom_row || left_column > right_column) {
		 return false;
	 }

	// Place to store the max values in each row
	T *				d_results;
	T *				h_results;

	// Place to store columns of max values in each row
	unsigned long *	d_column_buffer;
	unsigned long *	h_column_buffer;

	unsigned long	ROI_n_rows = bottom_row - top_row + 1;
	
	cudaError_t		result;

	// Allocating space on device for results and columns (there will be as 
	// many values as there are rows in the ROI)
	result          = cudaMalloc((void**) &d_results, ROI_n_rows * sizeof(T));
	if (result != cudaSuccess) {
		return false;
	}

	result          = cudaMalloc((void**) &d_column_buffer, ROI_n_rows * sizeof(unsigned long));
	if (result != cudaSuccess) {
		cudaFree(d_results);
		return false;
	}

	// Allocate space on host to get values. The comparison of the max values for each row
	// will be calculated in the host PC, rather than in the device, because global memory
	// access is slow.
	h_results                     = (T *)             malloc(ROI_n_rows * sizeof(T));
	h_column_buffer               = (unsigned long *) malloc(ROI_n_rows * sizeof(unsigned long));

	unsigned long threadsPerBlock = 256;
	unsigned long blocksPerGrid   = (threadsPerBlock + ROI_n_rows - 1) / threadsPerBlock;

	if (overlapping_pixels == NULL) {

		//Finding the maximum of each row using GPU kernels (i.e. one thread per row) 
		CUDAFindRowROIMax<T><<<blocksPerGrid, threadsPerBlock>>>(dataSet, d_column_buffer, d_results, numCols, top_row, left_column, bottom_row, right_column);
	}
	else {
		//Finding the maximum of each row using GPU kernels (i.e. one thread per row) 
		CUDAFindRowMaxWithOverlappingPixelThreshold<T><<<blocksPerGrid, threadsPerBlock>>>(dataSet, 
																						   overlapping_pixels,  
																						   pixel_overlap_threshold,
																						   d_column_buffer, 
																						   d_results, 
																						   numCols, 
																						   top_row, 
																						   left_column, 
																						   bottom_row, 
																						   right_column,
                                                                                           debug);
	}

	if (cudaGetLastError() != cudaSuccess) {
		cudaFree(d_column_buffer);
		cudaFree(d_results);
		free(    h_results);
		free(    h_column_buffer);
		return false;
	}	

	// copying the maximum values of each row from the device to the host PC for further processing
	result = cudaMemcpy(h_results, d_results, ROI_n_rows * sizeof(T), cudaMemcpyDeviceToHost);
	if(result != cudaSuccess) {
		cudaFree(d_results);
		cudaFree(d_column_buffer);
		free(    h_results);
		free(    h_column_buffer);
		return false;
	}

	// copying the column where each row maximum was found from the device to the host PC for further processing
	result = cudaMemcpy(h_column_buffer, d_column_buffer, ROI_n_rows *  sizeof(unsigned long), cudaMemcpyDeviceToHost);
	if (result != cudaSuccess) {
		cudaFree(d_results);
		cudaFree(d_column_buffer);
		free(    h_results);
		free(    h_column_buffer);
		return false;
	}

	// assigning the maximum of the first row to the maximum value as a starting point
	maxVal = h_results[0];
	rowMax = top_row;
	colMax = h_column_buffer[0] + left_column;

	//Find the column with the maximum value
	for (unsigned int index = 1; index < ROI_n_rows; index++) {
		if (h_results[index] >= maxVal) {
			maxVal = h_results[index];
			rowMax = index + top_row;
			colMax = h_column_buffer[index] + left_column;
		}
	}

	//Clean up buffers
	cudaFree(d_results);
	cudaFree(d_column_buffer);
	free(    h_results);
	free(    h_column_buffer);
	return true;

}

template <class T>
bool FindROIMin(T			 *dataSet,
				unsigned long numRows,
				unsigned long numCols,
				unsigned long &rowMin, 
				unsigned long &colMin, 
				T			  &minVal,
				unsigned long top_row,
				unsigned long left_column, 
				unsigned long bottom_row, 
				unsigned long right_column) {

	//Ensure that the parameters are valid
	 if (bottom_row >= numRows || right_column >= numCols || top_row > bottom_row || left_column > right_column) {
		 return false;
	 }

	//Place to store the max values in each row
	T				*d_results;
	T				*h_results;

	//Place to store columns of max values in each row
	unsigned long	*d_column_buffer;
	unsigned long	*h_column_buffer;

	unsigned long	 ROI_n_rows = bottom_row - top_row + 1;
	
	cudaError_t		result;

	// Allocating space on device for results and columns (there will be as 
	// many values as there are rows in the ROI)
	result      = cudaMalloc((void**) &d_results, ROI_n_rows * sizeof(T));
	if (result != cudaSuccess) {
		return false;
	}

	result      = cudaMalloc((void**) &d_column_buffer, ROI_n_rows * sizeof(unsigned long));
	if (result != cudaSuccess) {
		cudaFree(d_results);
		return false;
	}

	// Allocate space on host to get values. The comparison of the max values for each row
	// will be calculated in the host PC, rather than in the device, because global memory
	// access is slow.
	h_results       = (T *)             malloc(ROI_n_rows * sizeof(T));
	h_column_buffer = (unsigned long *) malloc(ROI_n_rows * sizeof(unsigned long));

	unsigned long threadsPerBlock = 256;
	unsigned long blocksPerGrid   = (ROI_n_rows + threadsPerBlock - 1) / threadsPerBlock;

	//Finding the minimum of each row using GPU kernels (i.e. one thread per row) 	
	CUDAFindRowROIMin<T><<<blocksPerGrid, threadsPerBlock>>>(dataSet, d_column_buffer, d_results, numCols, top_row, left_column, bottom_row, right_column);
	
	if (cudaGetLastError() != cudaSuccess) {
		cudaFree(d_column_buffer);
		cudaFree(d_results);
		free(    h_results);
		free(    h_column_buffer);
		return false;
	}	

	// copying the minimum values of each row from the device to the host PC for further processing
	result = cudaMemcpy(h_results, d_results, ROI_n_rows * sizeof(T), cudaMemcpyDeviceToHost);
	if(result != cudaSuccess) {
		cudaFree(d_results);
		cudaFree(d_column_buffer);
		free(    h_results);
		free(    h_column_buffer);
		return false;
	}

	// copying the column where each row minimum was found from the device to the host PC for further processing
	result = cudaMemcpy(h_column_buffer, d_column_buffer, ROI_n_rows * sizeof(unsigned long), cudaMemcpyDeviceToHost);
	if (result != cudaSuccess) {
		cudaFree(d_results);
		cudaFree(d_column_buffer);
		free(h_results);
		free(h_column_buffer);
		return false;
	}

	//Initialize to the first value
	minVal = h_results[0];
	rowMin = top_row;
	colMin = h_column_buffer[0] + left_column;

	//Find the column with the maximum value
	for (int index = 1; index < ROI_n_rows; index++) {
		if (h_results[index] <= minVal) {
			minVal = h_results[index];
			rowMin = index + top_row;
			colMin = h_column_buffer[index] + left_column;
		}
	}

	//Clean up buffers
	cudaFree(d_results);
	cudaFree(d_column_buffer);
	free(    h_results);
	free(    h_column_buffer);
	return true;

}

/****
 * Cpp functions for calling minimum and maximum kernels
 ***/
// these functions are just simple wrappers because they are particular cases in which the ROI covers the whole dataset


template <class T>
bool CUDAFindMax(T *dataSet, unsigned long numRows, unsigned long numCols, 
				unsigned long &rowMax, unsigned long &colMax, T &maxVal,
				T * overlapping_pixels = NULL,
				unsigned long pixel_overlap_threshold = 1) {

	return FindROIMax<T>(dataSet, numRows, numCols, rowMax, colMax, maxVal, 0, 0, numRows - 1, numCols-1, overlapping_pixels, pixel_overlap_threshold);
}

/**
 * CFindMinUint32
 **/

template <class T>
bool CUDAFindMin(T *dataSet, unsigned long numRows, unsigned long numCols, 
				unsigned long &rowMin, unsigned long &colMin, T &minVal) {

	return FindROIMin<T>(dataSet, numRows, numCols, rowMin, colMin, minVal, 0, 0, numRows - 1, numCols-1);
}