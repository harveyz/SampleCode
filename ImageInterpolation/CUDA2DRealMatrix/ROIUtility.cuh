#include "cuda_runtime.h"

#pragma once

template<class T>
__global__ void FillROI(T * buffer, unsigned long n_rows_buffer, unsigned long n_columns_buffer, T * ROI, 
					   unsigned long ROI_rows, unsigned long ROI_columns, unsigned long start_row, unsigned long start_col) {

    //Indexing variables
	unsigned long row = blockIdx.y*blockDim.y + threadIdx.y;
	unsigned long col = blockIdx.x*blockDim.x + threadIdx.x;
	unsigned long buffer_index  = (start_row + row)*n_columns_buffer + (col + start_col);
	unsigned long roi_index = row*ROI_columns + col;

	//check bounds
	if (row < ROI_rows && col < ROI_columns) {

		//Assign the ROI
		buffer[buffer_index] = ROI[roi_index];

	}
}

template <class T>
__global__ void FillROIClearRest(T * buffer, unsigned long n_rows_buffer, unsigned long n_columns_buffer, T * ROI, 
					   unsigned long ROI_rows, unsigned long ROI_columns, unsigned long start_row, unsigned long start_col) {

    //Indexing variables
	unsigned long row = blockIdx.y*blockDim.y + threadIdx.y;
	unsigned long col = blockIdx.x*blockDim.x + threadIdx.x;
	unsigned long buffer_index  = row*n_columns_buffer + col;
	long roi_index = (row - start_row)*ROI_columns + col - start_col;

	//check bounds
	if (row < n_rows_buffer && col < n_columns_buffer) {
		
		if (row >= start_row && row <= start_row + ROI_rows - 1 &&
			col >= start_col && col <= start_col + ROI_columns - 1) {
			//Assign to the ROI
			buffer[buffer_index] = ROI[roi_index];

		}
		else {
			//Clear this element
			buffer[buffer_index] = 0;
		}
	}
}

template<class T>
__global__ void SquareROI(T * buffer, unsigned long ROI_rows, unsigned long ROI_cols, 
						  unsigned long n_columns_buffer, unsigned long start_row, unsigned long start_col) {

    //Indexing variables
	unsigned long row = blockIdx.y*blockDim.y + threadIdx.y;
	unsigned long col = blockIdx.x*blockDim.x + threadIdx.x;
	unsigned long index  = (start_row + row)*n_columns_buffer + (col + start_col);

	if (row < ROI_rows && col < ROI_cols) {

		T val = buffer[index];

		//Assign the ROI
		buffer[index] = val*val;

	}

}

template<class T>
__global__ void GetROI(T * ROI_buffer, unsigned long columns_buffer, T * destination, unsigned long ROI_columns, 
					   unsigned long ROI_rows, unsigned long start_row, unsigned long start_col) {

    //Indexing variables
	unsigned long row = blockIdx.y*blockDim.y + threadIdx.y;
	unsigned long col = blockIdx.x*blockDim.x + threadIdx.x;

	//check bounds
	if (row < ROI_rows && col < ROI_columns) {
		//Assign the ROI
		destination[row*ROI_columns + col] = ROI_buffer[(row + start_row)*columns_buffer + (col + start_col)];
	}

}

template<class ThisType, class OtherType>

__global__ void AddROI(ThisType * this_buffer, OtherType * other_buffer,
				 unsigned long ROI_width, unsigned long ROI_height, 
				 unsigned long ROI_start_row_this,  unsigned long ROI_start_col_this,
				 unsigned long ROI_start_row_other, unsigned long ROI_start_col_other,
				 unsigned long n_cols_this,         unsigned long n_cols_other) {

	// Get the indexing variables
	unsigned long row = blockIdx.y*blockDim.y + threadIdx.y;
	unsigned long col = blockIdx.x*blockDim.x + threadIdx.x;

	unsigned long this_index  = (ROI_start_row_this  + row)*n_cols_this  + col + ROI_start_col_this;
	unsigned long other_index = (ROI_start_row_other + row)*n_cols_other + col + ROI_start_col_other;

	if (row < ROI_height && col < ROI_width) {
		
		ThisType val = this_buffer [this_index] + other_buffer[other_index];

		this_buffer [this_index] = val;
	}

}

template<class ThisType, class OtherType>

__global__ void SubROI(ThisType * this_buffer, OtherType * other_buffer,
				 unsigned long ROI_width, unsigned long ROI_height, 
				 unsigned long ROI_start_row_this,  unsigned long ROI_start_col_this,
				 unsigned long ROI_start_row_other, unsigned long ROI_start_col_other,
				 unsigned long n_cols_this,         unsigned long n_cols_other) {

	// Get the indexing variables
	unsigned long row = blockIdx.y*blockDim.y + threadIdx.y;
	unsigned long col = blockIdx.x*blockDim.x + threadIdx.x;

	unsigned long this_index  = (ROI_start_row_this  + row)*n_cols_this  + col + ROI_start_col_this;
	unsigned long other_index = (ROI_start_row_other + row)*n_cols_other + col + ROI_start_col_other;

	if (row < ROI_height && col < ROI_width) {
		
		ThisType val = this_buffer [this_index] - other_buffer[other_index];

		this_buffer [this_index] = val;
	}

}

template<class ThisType, class OtherType>

__global__ void MultROI(ThisType * this_buffer, OtherType * other_buffer,
				 unsigned long ROI_width, unsigned long ROI_height, 
				 unsigned long ROI_start_row_this,  unsigned long ROI_start_col_this,
				 unsigned long ROI_start_row_other, unsigned long ROI_start_col_other,
				 unsigned long n_cols_this,         unsigned long n_cols_other) {

	// Get the indexing variables
	unsigned long row = blockIdx.y*blockDim.y + threadIdx.y;
	unsigned long col = blockIdx.x*blockDim.x + threadIdx.x;

	unsigned long this_index  = (ROI_start_row_this  + row)*n_cols_this  + col + ROI_start_col_this;
	unsigned long other_index = (ROI_start_row_other + row)*n_cols_other + col + ROI_start_col_other;

	if (row < ROI_height && col < ROI_width) {
		
		ThisType val = this_buffer [this_index] * other_buffer[other_index];

		this_buffer [this_index] = val;
	}

}

template<class ThisType, class OtherType>

__global__ void DivROI(ThisType * this_buffer, OtherType * other_buffer,
				 unsigned long ROI_width, unsigned long ROI_height, 
				 unsigned long ROI_start_row_this,  unsigned long ROI_start_col_this,
				 unsigned long ROI_start_row_other, unsigned long ROI_start_col_other,
				 unsigned long n_cols_this,         unsigned long n_cols_other) {

	// Get the indexing variables
	unsigned long row = blockIdx.y*blockDim.y + threadIdx.y;
	unsigned long col = blockIdx.x*blockDim.x + threadIdx.x;

	unsigned long this_index  = (ROI_start_row_this  + row)*n_cols_this  + col + ROI_start_col_this;
	unsigned long other_index = (ROI_start_row_other + row)*n_cols_other + col + ROI_start_col_other;

	if (row < ROI_height && col < ROI_width) {
		
		ThisType val = this_buffer [this_index] / other_buffer[other_index];

		this_buffer [this_index] = val;
	}

}

template <class DataType>

__global__ void AddScalarROI(DataType * matrix_buffer, float val,
							 unsigned long ROI_width, unsigned long ROI_height,
							 unsigned long ROI_start_row, unsigned long ROI_start_col,
							 unsigned long n_cols) {

	// Get the indexing variables
	unsigned long row = blockIdx.y*blockDim.y + threadIdx.y;
	unsigned long col = blockIdx.x*blockDim.x + threadIdx.x;

	unsigned long this_index		= (ROI_start_row  + row)*n_cols  + col + ROI_start_col;

	if (row < ROI_height && col < ROI_width) {
		
		DataType result               = matrix_buffer [this_index] + val;

		matrix_buffer [this_index] = result;
	}

}

template <class DataType>

__global__ void MultScalarROI(DataType * matrix_buffer, float val,
							 unsigned long ROI_width, unsigned long ROI_height,
							 unsigned long ROI_start_row, unsigned long ROI_start_col,
							 unsigned long n_cols) {

	// Get the indexing variables
	unsigned long row = blockIdx.y*blockDim.y + threadIdx.y;
	unsigned long col = blockIdx.x*blockDim.x + threadIdx.x;

	unsigned long this_index		= (ROI_start_row  + row)*n_cols  + col + ROI_start_col;

	if (row < ROI_height && col < ROI_width) {
		
		DataType result               = matrix_buffer [this_index] * val;

		matrix_buffer [this_index] = result;
	}

}

template <class DataType>

__global__ void LogROI(DataType * matrix_buffer,
							 unsigned long ROI_width, unsigned long ROI_height,
							 unsigned long ROI_start_row, unsigned long ROI_start_col,
							 unsigned long n_cols) {

	// Get the indexing variables
	unsigned long row = blockIdx.y*blockDim.y + threadIdx.y;
	unsigned long col = blockIdx.x*blockDim.x + threadIdx.x;

	unsigned long this_index		= (ROI_start_row  + row)*n_cols  + col + ROI_start_col;

	if (row < ROI_height && col < ROI_width) {
		
		DataType result               = log((float)matrix_buffer [this_index]);

		matrix_buffer [this_index] = result;
	}

}

template <class DataType>

__global__ void SetROI(DataType * matrix_buffer, float val,
					   unsigned long ROI_width, unsigned long ROI_height,
					   unsigned long ROI_start_row, unsigned long ROI_start_col,
					   unsigned long n_cols) {

	// Get the indexing variables
	unsigned long row = blockIdx.y*blockDim.y + threadIdx.y;
	unsigned long col = blockIdx.x*blockDim.x + threadIdx.x;

	unsigned long this_index  = (ROI_start_row  + row)*n_cols  + col + ROI_start_col;

	if (row < ROI_height && col < ROI_width) {

		matrix_buffer [this_index] = val;
	}

}

// This kernel should be launched with 1 dimensional grid with 1 dimensional blocks

template<class OUTPUT_TYPE, class INPUT_TYPE>
__global__ void Assign(OUTPUT_TYPE * output_matrix, unsigned long n_rows_output, unsigned int n_cols_output, 
                       INPUT_TYPE * input_values,    unsigned int n_locations,
                       int * row_locations, int * col_locations) {

	unsigned long index = blockIdx.x*blockDim.x + threadIdx.x;

    if (index < n_locations) {

        int row_output = row_locations[index];
        int col_output = col_locations[index];

        if (row_output < n_rows_output && 
            col_output < n_cols_output && 
            row_output >= 0 && 
            col_output >= 0) {

            unsigned int output_index = row_output*n_cols_output + col_output;
            output_matrix[output_index] = input_values[index];

        }

    }

}

// This kernel should be launched with 1 dimensional grid with 1 dimensional blocks

template<class MATRIX_TYPE, class OUTPUT_TYPE>
__global__ void GetKernel(MATRIX_TYPE * matrix, 
                       unsigned long n_rows_matrix, unsigned int n_cols_matrix, 
                       OUTPUT_TYPE * output_values,  unsigned int n_locations,
                       int * row_locations, int * col_locations) {

	unsigned long index = blockIdx.x*blockDim.x + threadIdx.x;

    if (index < n_locations) {

        int row_output = row_locations[index];
        int col_output = col_locations[index];

        if (row_output < n_rows_matrix && 
            col_output < n_cols_matrix && 
            row_output >= 0 && 
            col_output >= 0) {

            unsigned int input_index = row_output*n_cols_matrix + col_output;
            output_values[index] = matrix[input_index];

        }

    }

}

template <class DataType>
__global__ void ClampMaxKernel(DataType * matrix_buffer, float max_value,
					   unsigned long n_rows,       unsigned long n_cols) {

	// Get the indexing variables
	unsigned long row = blockIdx.y*blockDim.y + threadIdx.y;
	unsigned long col = blockIdx.x*blockDim.x + threadIdx.x;
	unsigned long index = row*n_cols + col;
	float value;

	if (row < n_rows && col < n_cols) {

		value = (float) matrix_buffer [index];

		if (value > max_value) {
			matrix_buffer [index] = max_value;
		}

	}
}

template <class DataType>
__global__ void ClampMinKernel(DataType * matrix_buffer, float min_value,
					   unsigned long n_rows,       unsigned long n_cols) {

	// Get the indexing variables
	unsigned long row = blockIdx.y*blockDim.y + threadIdx.y;
	unsigned long col = blockIdx.x*blockDim.x + threadIdx.x;
	unsigned long index = row*n_cols + col;
	float value;

	if (row < n_rows && col < n_cols) {

		value = (float) matrix_buffer [index];

		if (value < min_value) {
			matrix_buffer [index] = min_value;
		}

	}
}
