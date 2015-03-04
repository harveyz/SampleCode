/******************************************************************************
 * Copyright 2014                                                             *
 * The Medical College of Wisconsin                                           *
 * All Rights Reserved                                                        *
 *                                                                            *
 * Zach Harvey   (zgh7555@rit.edu)                                            *
 * Alfredo Dubra (adubra@cvs.rochester.edu)                                   *
 ******************************************************************************/

#include "CUDAInterpolation.h"

#include <iostream>
#include <time.h>

//#define PRINT_MAT

#ifdef PRINT_MAT

#include <fstream>

#endif

using namespace std;

// The texture reference that we will bind our data to
texture<float, 2, cudaReadModeElementType> texture_reference;

// Texture for fast bspline interpolation
texture<float, 2, cudaReadModeElementType> b_spline_texture;

__global__ void LinearInterpolate(float * output_vector, unsigned long n_points, float * row_locations, float * col_locations, 
								  float out_of_bounds_value, unsigned long n_rows, unsigned long n_cols) {

	// Get the indexing variables
	unsigned long index = blockIdx.x*blockDim.x + threadIdx.x;

	if (index < n_points) {
		float y = row_locations[index];
		float x = col_locations[index];

		if (y > n_rows - 1 || y < 0 || x > n_cols - 1 || x < 0) {
			output_vector[index] = out_of_bounds_value;
		}
		else {
			output_vector[index] = tex2D(texture_reference, x + .5f, y + .5f);
		}
	}

}

__global__ void BiLinearInterpolate(float * output_vector, unsigned long n_points, float * row_locations, float * col_locations, float * regular_data,
								   float out_of_bounds_value, unsigned long n_rows, unsigned long n_cols) {

	// Get the indexing variables
	unsigned int index = blockIdx.x*blockDim.x + threadIdx.x;

	if (index < n_points) {
		float y = row_locations[index];
		float x = col_locations[index];

		if (y > n_rows - 1 || y < 0 || x > n_cols - 1 || x < 0) {
			output_vector[index] = out_of_bounds_value;
		}
		else {

			int k     = (int) floorf(x);
			int j     = (int) floorf(y);

			float ux[2];

			ux[1]  = x - ((float) k);
			ux[0]  = 1.0f - ux[1];

			float uy[2];

			uy[1]  = y - ((float) j);
			uy[0]  = 1.0f - uy[1];
	        
			float interpolated_value = 0;

			for (int cur_col = 0; cur_col < 2; cur_col++) {
				for (int cur_row = 0; cur_row < 2; cur_row++) {

					if (((j+cur_row) < n_rows) && ((k+cur_col) < n_cols)) {
						interpolated_value += regular_data[ (j + cur_row)*(n_cols) + k + cur_col]*ux[cur_col]*uy[cur_row];
					}

				}
			}

			output_vector[index] = interpolated_value;
		}
	}

}

__global__ void BiCubicInterpolateClampBoundary(float * output_vector, unsigned long n_points, float * row_locations, float * col_locations, float * regular_data,
								   float out_of_bounds_value, unsigned long n_rows, unsigned long n_cols) {

	// Get the indexing variables
	unsigned long index = blockIdx.x*blockDim.x + threadIdx.x;

	if (index < n_points) {
		float y = row_locations[index];
		float x = col_locations[index];

		if (y > n_rows - 2 || y < 1 || x > n_cols - 2 || x < 1) {
			output_vector[index] = out_of_bounds_value;
		}
		else {

			int k     = (int) floorf(x);
			int j     = (int) floorf(y);

			float s2   = x - ((float) k);
			float s1   = s2 + 1.0f;
			float s3   = 1.0f - s2;
			float s4   = s3 + 1.0f;

			float ux[4];
	    
			ux[0] = -.5*s1*s1*s1 + 2.5f*s1*s1 - 4*s1 + 2;
			ux[1] = 1.5f*s2*s2*s2 - 2.5f*s2*s2 + 1;
			ux[2] = 1.5f*s3*s3*s3 - 2.5f*s3*s3 + 1;
			ux[3] = -.5*s4*s4*s4 + 2.5f*s4*s4 - 4*s4 + 2;
			s2   = y - ((float)j);
			s1   = s2 + 1.0f;
			s3   = 1.0f - s2;
			s4   = s3 + 1.0f;

			float uy[4];
	        
			uy[0] = -.5*s1*s1*s1 + 2.5f*s1*s1 - 4*s1 + 2;
			uy[1] = 1.5f*s2*s2*s2 - 2.5f*s2*s2 + 1;
			uy[2] = 1.5f*s3*s3*s3 - 2.5f*s3*s3 + 1;
			uy[3] = -.5*s4*s4*s4 + 2.5f*s4*s4 - 4*s4 + 2;

			float interpolated_value = 0;

			for (int cur_row = 0; cur_row < 4; cur_row++) {
				for (int cur_col = 0; cur_col < 4; cur_col++) {

				   interpolated_value += regular_data[ (j + cur_row - 1)*(n_cols) + k + cur_col - 1]*ux[cur_col]*uy[cur_row];

				}
			}
			output_vector[index] = interpolated_value;
		}
	}

}

__global__ void BiCubicInterpolateBiLinearBoundary(float * output_vector, unsigned long n_points, float * row_locations, float * col_locations, float * regular_data,
								   float out_of_bounds_value, unsigned long n_rows, unsigned long n_cols) {

	// Get the indexing variables
	unsigned long index = blockIdx.x*blockDim.x + threadIdx.x;

	if (index < n_points) {
		float y = row_locations[index];
		float x = col_locations[index];

		if (y > n_rows - 1 || y < 0 || x > n_cols - 1 || x < 0) {
			output_vector[index] = out_of_bounds_value;
		}
		else if (y > n_rows - 2 || y < 1 || x > n_cols - 2 || x < 1) {
			int k     = (int) floorf(x);
			int j     = (int) floorf(y);

			float r1 = ((((float)k) + 1.0f) - x)* regular_data[j*n_cols + k] + (x - ((float)k))*regular_data[(j)*n_cols + k + 1];
			float r2 = ((((float)k) + 1.0f) - x)* regular_data[(j+1)*n_cols + k] + (x - ((float)k))*regular_data[(j+1)*n_cols + k + 1];

			output_vector[index] = ((((float)j) + 1.0f) - y)* r1 + (y - ((float)j))*r2;
		}
		else {

			int k     = (int) floorf(x);
			int j     = (int) floorf(y);

			float s2   = x - ((float) k);
			float s1   = s2 + 1.0f;
			float s3   = 1.0f - s2;
			float s4   = s3 + 1.0f;

			float ux[4];
	    
			ux[0] = -.5*s1*s1*s1 + 2.5f*s1*s1 - 4*s1 + 2;
			ux[1] = 1.5f*s2*s2*s2 - 2.5f*s2*s2 + 1;
			ux[2] = 1.5f*s3*s3*s3 - 2.5f*s3*s3 + 1;
			ux[3] = -.5*s4*s4*s4 + 2.5f*s4*s4 - 4*s4 + 2;
			s2   = y - ((float)j);
			s1   = s2 + 1.0f;
			s3   = 1.0f - s2;
			s4   = s3 + 1.0f;

			float uy[4];
	        
			uy[0] = -.5*s1*s1*s1 + 2.5f*s1*s1 - 4*s1 + 2;
			uy[1] = 1.5f*s2*s2*s2 - 2.5f*s2*s2 + 1;
			uy[2] = 1.5f*s3*s3*s3 - 2.5f*s3*s3 + 1;
			uy[3] = -.5*s4*s4*s4 + 2.5f*s4*s4 - 4*s4 + 2;
	        
			float interpolated_value = 0;

			for (int cur_row = 0; cur_row < 4; cur_row++) {
				for (int cur_col = 0; cur_col < 4; cur_col++) {

				   interpolated_value += regular_data[ (j + cur_row - 1)*(n_cols) + k + cur_col - 1]*ux[cur_col]*uy[cur_row];

				}
			}
			output_vector[index] = interpolated_value;
		}
	}

}


__global__ void BiCubicInterpolate(float * output_vector, unsigned long n_points, float * row_locations, float * col_locations, float * regular_data,
								   float out_of_bounds_value, unsigned long n_rows, unsigned long n_cols) {

	// Get the indexing variables
	unsigned long index = blockIdx.x*blockDim.x + threadIdx.x;

	if (index < n_points) {
		float y = row_locations[index];
		float x = col_locations[index];

		if (y > n_rows - 1 || y < 0 || x > n_cols - 1 || x < 0) {
			output_vector[index] = out_of_bounds_value;
		}
		else {

			int k     = (int) floorf(x);
			int j     = (int) floorf(y);

			float s2   = x - ((float) k);
			float s1   = s2 + 1.0f;
			float s3   = 1.0f - s2;
			float s4   = s3 + 1.0f;

			float ux[4];
	    
			ux[0] = -.5*s1*s1*s1 + 2.5f*s1*s1 - 4*s1 + 2;
			ux[1] = 1.5f*s2*s2*s2 - 2.5f*s2*s2 + 1;
			ux[2] = 1.5f*s3*s3*s3 - 2.5f*s3*s3 + 1;
			ux[3] = -.5*s4*s4*s4 + 2.5f*s4*s4 - 4*s4 + 2;
			s2   = y - ((float)j);
			s1   = s2 + 1.0f;
			s3   = 1.0f - s2;
			s4   = s3 + 1.0f;

			float uy[4];
	        
			uy[0] = -.5*s1*s1*s1 + 2.5f*s1*s1 - 4*s1 + 2;
			uy[1] = 1.5f*s2*s2*s2 - 2.5f*s2*s2 + 1;
			uy[2] = 1.5f*s3*s3*s3 - 2.5f*s3*s3 + 1;
			uy[3] = -.5*s4*s4*s4 + 2.5f*s4*s4 - 4*s4 + 2;
	        
			float interpolated_value = 0;

			for (int cur_row = 0; cur_row < 4; cur_row++) {
				for (int cur_col = 0; cur_col < 4; cur_col++) {

				   interpolated_value += regular_data[ (j + cur_row)*(n_cols + 3) + k + cur_col]*ux[cur_col]*uy[cur_row];

				}
			}
			output_vector[index] = interpolated_value;
		}
	}

}
// As the convolution kernel has support over four interpolation nodes, the point at x = -1 must be 
// determined. In order to retain the third order approximation established in the selection of the 
// coeeficients of the convolution kernel, the s^3 terms must vanish. ( see equations 18-25 of R. Keys)
__global__ void PadBiCubic(float * input_buffer, float * output_buffer, unsigned long n_rows, unsigned long n_cols) {


	unsigned long int col     = blockIdx.x * blockDim.x + threadIdx.x;
	unsigned long int row     = blockIdx.y * blockDim.y + threadIdx.y;

	if (col < n_cols + 3 && row < n_rows + 3) {

		if (row == 0 && col > 0 && col < n_cols + 1) {
			// top row
			output_buffer[row*(n_cols + 3) + col] = 3*input_buffer[col - 1] - 3*input_buffer[n_cols + col - 1] + input_buffer[2*n_cols + col - 1];
		}
		else if (col == 0 && row > 0 && row < n_rows + 1) {
			// left col
			output_buffer[row*(n_cols + 3) + col] = 3*input_buffer[(row - 1)*n_cols] - 3*input_buffer[(row-1)*n_cols + 1] + input_buffer[(row-1)*n_cols + 2];
		}
		else if (col == n_cols + 1 && row > 0 && row < n_rows + 1) {
			// right col
			output_buffer[row*(n_cols + 3) + col] = 3*input_buffer[(row-1)*n_cols + n_cols - 1] - 3*input_buffer[(row-1)*n_cols + n_cols - 2] + input_buffer[(row - 1)*n_cols + n_cols - 3];
		}
		else if (row == n_rows + 1 && col > 0 && col < n_cols + 1) {
			// bottom row
			output_buffer[row*(n_cols + 3) + col] = 3*input_buffer[(n_rows - 1)*n_cols + col - 1] - 3*input_buffer[(n_rows - 2)*n_cols + col - 1] + input_buffer[(n_rows - 3)*n_cols + col - 1];
		}
		else if (row > 0 && row < n_rows + 1 && col > 0 && col < n_cols+1) {
			// middle
			output_buffer[row*(n_cols + 3) + col] = input_buffer[(row - 1)*n_cols + col - 1];
		}
		else if (row == n_rows + 2) {
			// far bottom
			output_buffer[row*(n_cols + 3) + col] = 0;
		}
		else if (col == n_cols + 2) {
			// far right
			output_buffer[row*(n_cols + 3) + col] = 0;
		}

	}

	__syncthreads(); 

	if (col == 0 && row == 0) {
		// top left
		output_buffer[0] = 3*output_buffer[1] - 3*output_buffer[2] + output_buffer[3];
	}
	else if (col == 0 && row == n_rows+1) {
		//bottom left
		output_buffer[row*(n_cols + 3) + col] = 3*output_buffer[row*(n_cols + 3) + 1] - 3*output_buffer[row*(n_cols + 3) +2] + output_buffer[row*(n_cols + 3) + 3];
	}
	else if (col == n_cols+1 && row == 0) {
		//top right
		output_buffer[row*(n_cols + 3) + col] = 3*output_buffer[n_cols - 2] - 3*output_buffer[n_cols - 1] + output_buffer[n_cols];
	}
	else if (col == n_cols + 1 && row == n_rows + 1) {
		// bottom right
		output_buffer[row*(n_cols + 3) + col] = 3*output_buffer[row*(n_cols + 3) + n_cols - 2] - 3*output_buffer[row*(n_cols + 3) + n_cols - 1] + output_buffer[row*(n_cols+3) + n_cols];
	}
}

__global__ void CubicBSplineInterpolate(float * output_vector, unsigned long n_points, float * row_locations, 
										float * col_locations, float * coeff_matrix, float out_of_bounds_value, 
										unsigned long n_rows, unsigned long n_cols) {

	// Get the indexing variables
	unsigned long index = blockIdx.x*blockDim.x + threadIdx.x;

	if (index < n_points) {
		float y = row_locations[index];
		float x = col_locations[index];

		if (y > n_rows - 1 || y < 0 || x > n_cols - 1 || x < 0) {
			output_vector[index] = out_of_bounds_value;
		}
		else {

			int k     = (int) floorf(x);
			int j     = (int) floorf(y);

			float b2   = x - ((float) k);
			float b1   = b2 + 1.0f;
			float b3   = 1.0f - b2;
			float b4   = b3 + 1.0f;

			float bx[4];

			bx[0] = (2.0f-b1)*(2.0f-b1)*(2.0f-b1)/6.0f;
			bx[1] = 2.0f/3.0f - b2*b2 + b2*b2*b2/2.0f;
			bx[2] = 2.0f/3.0f - b3*b3 + b3*b3*b3/2.0f;
			bx[3] = (2.0f-b4)*(2.0f-b4)*(2.0f-b4)/6.0f;
	    
			b2   = y - ((float)j);
			b1   = b2 + 1.0f;
			b3   = 1.0f - b2;
			b4   = b3 + 1.0f;

			float by[4];

			by[0] = (2.0f-b1)*(2.0f-b1)*(2.0f-b1)/6.0f;
			by[1] = 2.0f/3.0f - b2*b2 + b2*b2*b2/2.0f;
			by[2] = 2.0f/3.0f - b3*b3 + b3*b3*b3/2.0f;
			by[3] = (2.0f-b4)*(2.0f-b4)*(2.0f-b4)/6.0f;

			float interpolated_value = 0;

			for (int cur_row = 0; cur_row < 4; cur_row++) {
				for (int cur_col = 0; cur_col < 4; cur_col++) {

				   interpolated_value += 6.0f*coeff_matrix[ (j + cur_row)*(n_cols+3) + k + cur_col]*bx[cur_col]*by[cur_row];

				}
			}
			output_vector[index] =interpolated_value;
		}
	}

}

__global__ void CubicBSplineInterpolateFast(float * output_vector, unsigned long n_points, float * row_locations, 
										float * col_locations, float out_of_bounds_value, 
										unsigned long n_rows, unsigned long n_cols) {

	// Get the indexing variables
	unsigned int index = blockIdx.x*blockDim.x + threadIdx.x;

	if (index < n_points) {
		float y = row_locations[index];
		float x = col_locations[index];

		if (y > n_rows - 1 || y < 0 || x > n_cols - 1 || x < 0) {
			output_vector[index] = out_of_bounds_value;
		}
		else {


			float k     = floorf(x);
			float j     = floorf(y);

			//float b2   = x - ((float) k);
			//float b1   = b2 + 1.0f;

			//float b[2];
			float sx[2], sy[2], shifted_x[2], shifted_y[2];

			float b2   = x - k;
			float b1   = b2 + 1.0f;
			float b3   = 1.0f - b2;
			float b4   = b3 + 1.0f;

			float bx[4];

			bx[0] = (2.0f-b1)*(2.0f-b1)*(2.0f-b1)/6.0f;
			bx[1] = 2.0f/3.0f - b2*b2 + b2*b2*b2/2.0f;
			bx[2] = 2.0f/3.0f - b3*b3 + b3*b3*b3/2.0f;
			bx[3] = (2.0f-b4)*(2.0f-b4)*(2.0f-b4)/6.0f;

			shifted_x[0] = k - bx[0]/(bx[1] + bx[0]);
			sx[0] = (k - shifted_x[0])/bx[0];
			shifted_x[1] = k + (2.0f*bx[3] + bx[2])/(bx[2] + bx[3]);
			sx[1] = (k + 2.0f - shifted_x[1])/bx[2];

			b2   = y - j;
			b1   = b2 + 1.0f;
			b3   = 1.0f - b2;
			b4   = b3 + 1.0f;

			float by[4];

			by[0] = (2.0f-b1)*(2.0f-b1)*(2.0f-b1)/6.0f;
			by[1] = 2.0f/3.0f - b2*b2 + b2*b2*b2/2.0f;
			by[2] = 2.0f/3.0f - b3*b3 + b3*b3*b3/2.0f;
			by[3] = (2.0f-b4)*(2.0f-b4)*(2.0f-b4)/6.0f;

			shifted_y[0] = j - by[0]/(by[1] + by[0]);
			sy[0] = (j - shifted_y[0])/by[0];

			shifted_y[1] = j+ (2.0f*by[3] + by[2])/(by[2] + by[3]);
			sy[1]        = (j + 2.0f - shifted_y[1])/by[2];

			//b[0] = (2.0f-b1)*(2.0f-b1)*(2.0f-b1)/6.0f;
			//b[1] = 2.0f/3.0f - b2*b2 + b2*b2*b2/2.0f;

			//shifted_x[0] = ((float) k) - b[0]/(b[1] + b[0]);
			//sx[0] = (((float) k) - shifted_x[0])/b[0];

			//b1   = 1.0f - b2;
			//b2   = b1 + 1.0f;

			//b[0] = 2.0f/3.0f - b1*b1 + b1*b1*b1/2.0f;
			//b[1] = (2.0f-b2)*(2.0f-b2)*(2.0f-b2)/6.0f;

			//shifted_x[1] = ((float) k) + (2.0f*b[1] + b[0])/(b[0] + b[1]);
			//sx[1] = (((float) k) + 2.0f - shifted_x[1])/b[0];

			//b2   = y - ((float)j);
			//b1   = b2 + 1.0f;

			//b[0] = (2.0f-b1)*(2.0f-b1)*(2.0f-b1)/6.0f;
			//b[1] = 2.0f/3.0f - b2*b2 + b2*b2*b2/2.0f;

			//shifted_y[0] = ((float) j) - b[0]/(b[1] + b[0]);
			//sy[0] = (((float) j) - shifted_y[0])/b[0];

			//b1   = 1.0f - b2;
			//b2   = b1 + 1.0f;

			//b[0] = 2.0f/3.0f - b1*b1 + b1*b1*b1/2.0f;
			//b[1] = (2.0f-b2)*(2.0f-b2)*(2.0f-b2)/6.0f;

			//shifted_y[1] = ((float) j) + (2.0f*b[1] + b[0])/(b[0] + b[1]);
			//sy[1]        = (((float) j) + 2.0f - shifted_y[1])/b[0];

			float interpolated_value = 0;

			// perform the interpolation using texture lookups
			// if we unroll the loop extra registers wont be used for the counting variables?
			#pragma unroll 2
			for (int cur_row = 0; cur_row < 2; cur_row++) {
				#pragma unroll 2
				for (int cur_col = 0; cur_col < 2; cur_col++) {

					interpolated_value += 6*tex2D(b_spline_texture, shifted_x[cur_col] + 1.5f, shifted_y[cur_row] + 1.5f)  /  (sx[cur_col]*sy[cur_row]);
				}
			}

			output_vector[index] = interpolated_value;
		}
	}

}

// To determine the coefficints of the bspline interpolation we first filter down the columns of the image and then filter accros the rows
// of the image. This can be done because the two dimensional version of the filters in Unser M 1999 are seperable

__global__ void CalcBSplineCoeffColumnWise(float * regular_data, unsigned long n_rows, unsigned long n_cols, float * coeff_matrix, unsigned int k0, float z1) {

	unsigned long col_index = blockIdx.x*blockDim.x + threadIdx.x;

	int cur_row = 0;

	if (col_index < n_cols) {

		float c_plus_0 = 0.0f;

		// Initialize the first value in the coefficient matrix
		for (cur_row = 0; cur_row <= k0; cur_row++) {
			float inc_val = regular_data[cur_row*n_cols + col_index];
			for (unsigned int cur_pow = 0; cur_pow < cur_row; cur_pow++) {
				inc_val *= z1;
			}
			c_plus_0 += inc_val;
		}

		coeff_matrix[n_cols + 3 + col_index + 1] = c_plus_0;

		// Filter in the forward direction
		for (cur_row = 1; cur_row < n_rows; cur_row++) {
			coeff_matrix[(cur_row + 1)*(n_cols + 3) + col_index + 1] = regular_data[cur_row*n_cols + col_index] + z1*coeff_matrix[cur_row*(n_cols + 3) + col_index + 1];
		}

		// Calculate the final value of the coefficients
		coeff_matrix[n_rows*(n_cols + 3) + col_index + 1] = (2*coeff_matrix[n_rows*(n_cols + 3) + col_index + 1] - regular_data[(n_rows - 1)*n_cols + col_index])*(-z1/(1-z1*z1));

		// Filter in the reverse direction
		for (cur_row = n_rows - 1; cur_row > 0; cur_row--) {
			coeff_matrix[(cur_row)*(n_cols + 3) + col_index + 1] = (coeff_matrix[(cur_row + 1)*(n_cols + 3) + col_index + 1] - coeff_matrix[(cur_row)*(n_cols + 3) + col_index + 1])*z1;
		}

		// Use mirror-symmetric boundry conditions to determine the coefficients outside of the interval [0, N-1] ie -1, N, N+1
		coeff_matrix[col_index + 1]                             = coeff_matrix[2*(n_cols + 3) + col_index + 1];

		coeff_matrix[(n_rows + 1)*(n_cols + 3) + col_index + 1] = coeff_matrix[(n_rows - 1)*(n_cols + 3) + col_index + 1];
		coeff_matrix[(n_rows + 2)*(n_cols + 3) + col_index + 1] = coeff_matrix[(n_rows - 2)*(n_cols + 3) + col_index + 1];

	}

}

// Note: the coefficients calculated down the columns were not scaled, therefore, before they are used they must be scaled by six

__global__ void CalcBSplineCoeffRowWise(unsigned long n_rows, unsigned long n_cols, float * coeff_matrix, unsigned int k0, float z1) {

	unsigned long row_index = blockIdx.x*blockDim.x + threadIdx.x;

	int cur_col = 0;

	if (row_index < n_rows + 3) {

		float c_plus_0 = 0.0f;

		// Initialize the first value in the coefficient matrix
		for (cur_col = 0; cur_col <= k0; cur_col++) {
			float inc_val = 6*coeff_matrix[row_index*(n_cols + 3) + cur_col + 1];
			for (unsigned int cur_pow = 0; cur_pow < cur_col; cur_pow++) {
				inc_val *= z1;
			}
			c_plus_0 += inc_val;
		}

		coeff_matrix[row_index*(n_cols + 3) + 1] = c_plus_0;

		// Grab the final data point before it is overwritten
		float final_val = 6*coeff_matrix[row_index*(n_cols + 3) + n_cols];

		// Filter in the forward direction
		for (cur_col = 1; cur_col < n_cols; cur_col++) {
			coeff_matrix[row_index*(n_cols + 3) + cur_col + 1] = 6*coeff_matrix[row_index*(n_cols + 3) + cur_col + 1] + 
																	z1*coeff_matrix[row_index*(n_cols + 3) + cur_col];
		}

		// Calculate the final value of the coefficients
		coeff_matrix[row_index*(n_cols + 3) + n_cols] = (2*coeff_matrix[row_index*(n_cols + 3) + n_cols] - final_val)*(-z1/(1-z1*z1));

		// Filter in the reverse direction
		for (cur_col = n_cols - 1; cur_col > 0; cur_col--) {
			coeff_matrix[row_index*(n_cols + 3) + cur_col] = (coeff_matrix[row_index*(n_cols + 3) + cur_col + 1] - coeff_matrix[row_index*(n_cols + 3) + cur_col])*z1;
		}

		// Use mirror-symmetric boundry conditions to determine the coefficients outside of the interval [0, N-1] ie -1, N, N+1
		coeff_matrix[row_index*(n_cols + 3)]              = coeff_matrix[row_index*(n_cols + 3) + 2];
		coeff_matrix[row_index*(n_cols + 3) + n_cols + 1] = coeff_matrix[row_index*(n_cols + 3) + n_cols - 1];
		coeff_matrix[row_index*(n_cols + 3) + n_cols + 2] = coeff_matrix[row_index*(n_cols + 3) + n_cols - 2];

	}

}

// Note: the coefficients calculated down the columns were not scaled, therefore, before they are used they must be scaled by six

__global__ void CalcBSplineCoeffRowWiseTransposed(unsigned long n_rows, unsigned long n_cols, float * coeff_matrix, unsigned int k0, float z1) {

	unsigned long row_index = blockIdx.x*blockDim.x + threadIdx.x;

	int cur_col = 0;

	if (row_index < n_rows + 3) {

		float c_plus_0 = 0.0f;

		// Initialize the first value in the coefficient matrix
		for (cur_col = 0; cur_col <= k0; cur_col++) {
			float inc_val = 6*coeff_matrix[(cur_col + 1)*(n_rows + 3) + row_index];
			for (unsigned int cur_pow = 0; cur_pow < cur_col; cur_pow++) {
				inc_val *= z1;
			}
			c_plus_0 += inc_val;
		}

		coeff_matrix[(n_rows + 3) + row_index] = c_plus_0;

		// Grab the final data point before it is overwritten
		float final_val = 6*coeff_matrix[n_cols*(n_rows + 3) + row_index];

		// Filter in the forward direction
		for (cur_col = 1; cur_col < n_cols; cur_col++) {
			coeff_matrix[(cur_col + 1)*(n_rows + 3) + row_index] = 6*coeff_matrix[(cur_col + 1)*(n_rows + 3) + row_index] + 
																	z1*coeff_matrix[cur_col*(n_rows + 3) + row_index];
		}

		// Calculate the final value of the coefficients
		coeff_matrix[n_cols*(n_rows + 3) + row_index] = (2*coeff_matrix[n_cols*(n_rows + 3) + row_index] - final_val)*(-z1/(1-z1*z1));

		// Filter in the reverse direction
		for (cur_col = n_cols - 1; cur_col > 0; cur_col--) {
			coeff_matrix[cur_col*(n_rows + 3) + row_index] = (coeff_matrix[(cur_col + 1)*(n_rows + 3) + row_index] - coeff_matrix[cur_col*(n_rows + 3) + row_index])*z1;
		}

		// Use mirror-symmetric boundry conditions to determine the coefficients outside of the interval [0, N-1] ie -1, N, N+1
		coeff_matrix[row_index]                             = coeff_matrix[2*(n_rows + 3) + row_index];
		coeff_matrix[(n_cols + 1)*(n_rows + 3) + row_index] = coeff_matrix[(n_cols - 1)*(n_rows + 3) + row_index];
		coeff_matrix[(n_cols + 2)*(n_rows + 3) + row_index] = coeff_matrix[(n_cols - 2)*(n_rows + 3) + row_index];

	}

}


bool CUDALinearInterpolation(CUDA2DRealMatrix<float> &regular_data, 
							 CUDA2DRealMatrix<float> &interpolated_data, 
							 CUDA2DRealMatrix<float> &row_coordinates, 
							 CUDA2DRealMatrix<float> &col_coordinates,
							 float out_of_bounds_value) {

	// Get the number of rows and columns in the regular data
    unsigned long n_rows;
	unsigned long n_cols;

	if (!regular_data.GetSize(n_rows, n_cols)) {
		return false;
	}

	// Get the number of points that we wish to interpolate on and ensure
	// the the data sent is in the proper format
	unsigned long n_rows_desired;
	unsigned long n_cols_desired;
	unsigned long should_be_one;

	if (!row_coordinates.GetSize(should_be_one, n_rows_desired)) {
		return false;
	}

	if (should_be_one != 1) {
		return false;
	}

	if (!col_coordinates.GetSize(should_be_one, n_cols_desired)) {
		return false;
	}

	if (should_be_one != 1) {
		return false;
	}

	if (n_rows_desired != n_cols_desired) {
		return false;
	}

	if (!interpolated_data.SetSize(1, n_rows_desired)) {
		return false;
	}

	// Ensure that we can launch enough CUDA threads to do the interpolation
	// TODO is there a way we don't have to hard code
	if (n_rows_desired > 33553920) {
		return false;
	}


	// Set up the texture reference
	texture_reference.normalized     = false;
	texture_reference.filterMode     = cudaFilterModeLinear;
	texture_reference.addressMode[0] = cudaAddressModeClamp;
	texture_reference.addressMode[1] = cudaAddressModeClamp;

	// Create a texture description to determine how the data will be returned
	cudaChannelFormatDesc channel_description = cudaCreateChannelDesc(32, 0, 0, 0, cudaChannelFormatKindFloat);

	// Create a memory buffer that we can bind to the texture
	cudaArray* texture_data;
	if (cudaMallocArray(&texture_data, &channel_description, n_cols, n_rows) != cudaSuccess) {
		return false;
	}

	// get a pointer to the regular data so that it can be copied to the cu array
	float * regular_data_pointer;
	if (!regular_data.GetPointerToData(&regular_data_pointer)) {
		cudaFreeArray(texture_data);
		return false;
	}

	// copy the data to the cuda array
	if (cudaMemcpyToArray(texture_data, 0, 0, regular_data_pointer, n_cols*n_rows*sizeof(float), cudaMemcpyDeviceToDevice) != cudaSuccess) {
		cudaFreeArray(texture_data);
		return false;
	}

	// Bind the regular data to the texture
	if (cudaBindTextureToArray(texture_reference, texture_data, channel_description) != cudaSuccess) {
		cudaFreeArray(texture_data);
		return false;
	}

	

	// Get the pointers to the input arguments
	float * interpolated_data_pointer;
	float * col_loc_pointer;
	float * row_loc_pointer;

	if (!interpolated_data.GetPointerToData(&interpolated_data_pointer)) {
		cudaFreeArray(texture_data);
		return false;
	}

	if (!row_coordinates.GetPointerToData(&row_loc_pointer)) {
		cudaFreeArray(texture_data);
		return false;
	}

	if (!col_coordinates.GetPointerToData(&col_loc_pointer)) {
		cudaFreeArray(texture_data);
		return false;
	}


	// Launch the kernel to do the interpolation at the desired locations
	dim3 threadBlock(512);
	dim3 threadGrid((threadBlock.x + n_rows_desired - 1)/threadBlock.x);

	LinearInterpolate<<<threadGrid, threadBlock>>>(interpolated_data_pointer, n_rows_desired, row_loc_pointer, col_loc_pointer, out_of_bounds_value, n_rows, n_cols);

	if (cudaGetLastError() != cudaSuccess) {
		cudaFreeArray(texture_data);
		return false;
	}


	// Unbind the texture
	cudaUnbindTexture(&texture_reference);
	cudaFreeArray(texture_data);
	return true;
}




bool CUDA2DNearestNeighborInterpolator::CUDA2DInterpolate(CUDA2DRealMatrix<float> &regular_data, 
							 CUDA2DRealMatrix<float> &interpolated_data, 
							 CUDA2DRealMatrix<float> &row_coordinates, 
							 CUDA2DRealMatrix<float> &col_coordinates,
							 float out_of_bounds_value) {

	// Get the number of rows and columns in the regular data
    unsigned long n_rows;
	unsigned long n_cols;

	if (!regular_data.GetSize(n_rows, n_cols)) {
		return false;
	}

	// Get the number of points that we wish to interpolate on and ensure
	// the the data sent is in the proper format
	unsigned long n_rows_desired;
	unsigned long n_cols_desired;
	unsigned long should_be_one;

	if (!row_coordinates.GetSize(should_be_one, n_rows_desired)) {
		return false;
	}

	if (should_be_one != 1) {
		return false;
	}

	if (!col_coordinates.GetSize(should_be_one, n_cols_desired)) {
		return false;
	}

	if (should_be_one != 1) {
		return false;
	}

	if (n_rows_desired != n_cols_desired) {
		return false;
	}

	if (!interpolated_data.SetSize(1, n_rows_desired)) {
		return false;
	}

	// Ensure that we can launch enough CUDA threads to do the interpolation
	// TODO is there a way we don't have to hard code
	if (n_rows_desired > 33553920) {
		return false;
	}

	// Set up the texture reference
	texture_reference.normalized     = false;
	texture_reference.filterMode     = cudaFilterModePoint;
	texture_reference.addressMode[0] = cudaAddressModeClamp;
	texture_reference.addressMode[1] = cudaAddressModeClamp;

	// Create a texture description to determine how the data will be returned
	cudaChannelFormatDesc channel_description = cudaCreateChannelDesc(32, 0, 0, 0, cudaChannelFormatKindFloat);

	// Create a memory buffer that we can bind to the texture
	cudaArray* texture_data;
	if (cudaMallocArray(&texture_data, &channel_description, n_cols, n_rows) != cudaSuccess) {
		return false;
	}

	// get a pointer to the regular data so that it can be copied to the cu array
	float * regular_data_pointer;
	if (!regular_data.GetPointerToData(&regular_data_pointer)) {
		cudaFreeArray(texture_data);
		return false;
	}

	// copy the data to the cuda array
	if (cudaMemcpyToArray(texture_data, 0, 0, regular_data_pointer, n_cols*n_rows*sizeof(float), cudaMemcpyDeviceToDevice) != cudaSuccess) {
		cudaFreeArray(texture_data);
		return false;
	}

	// Bind the regular data to the texture
	if (cudaBindTextureToArray(texture_reference, texture_data, channel_description) != cudaSuccess) {
		cudaFreeArray(texture_data);
		return false;
	}

	// Get the pointers to the input arguments
	float * interpolated_data_pointer;
	float * col_loc_pointer;
	float * row_loc_pointer;

	if (!interpolated_data.GetPointerToData(&interpolated_data_pointer)) {
		cudaFreeArray(texture_data);
		return false;
	}

	if (!row_coordinates.GetPointerToData(&row_loc_pointer)) {
		cudaFreeArray(texture_data);
		return false;
	}

	if (!col_coordinates.GetPointerToData(&col_loc_pointer)) {
		cudaFreeArray(texture_data);
		return false;
	}

	// Launch the kernel to do the interpolation at the desired locations
	dim3 threadBlock(512);
	dim3 threadGrid((threadBlock.x + n_rows_desired - 1)/threadBlock.x);

	LinearInterpolate<<<threadGrid, threadBlock>>>(interpolated_data_pointer, n_rows_desired, row_loc_pointer, col_loc_pointer, out_of_bounds_value, n_rows, n_cols);

	if (cudaGetLastError() != cudaSuccess) {
		cudaFreeArray(texture_data);
		return false;
	}

	// Unbind the texture
	cudaUnbindTexture(&texture_reference);
	cudaFreeArray(texture_data);
	return true;
}





bool CUDA2DLinearInterpolator::CUDA2DInterpolate(CUDA2DRealMatrix<float> &regular_data, 
							 CUDA2DRealMatrix<float> &interpolated_data, 
							 CUDA2DRealMatrix<float> &row_coordinates, 
							 CUDA2DRealMatrix<float> &col_coordinates,
							 float out_of_bounds_value) {

	// Get the number of rows and columns in the regular data
    unsigned long n_rows;
	unsigned long n_cols;

	if (!regular_data.GetSize(n_rows, n_cols)) {
		return false;
	}

	// Get the number of points that we wish to interpolate on and ensure
	// the the data sent is in the proper format
	unsigned long n_rows_desired;
	unsigned long n_cols_desired;
	unsigned long should_be_one;

	if (!row_coordinates.GetSize(should_be_one, n_rows_desired)) {
		return false;
	}

	if (should_be_one != 1) {
		return false;
	}

	if (!col_coordinates.GetSize(should_be_one, n_cols_desired)) {
		return false;
	}

	if (should_be_one != 1) {
		return false;
	}

	if (n_rows_desired != n_cols_desired) {
		return false;
	}

	if (!interpolated_data.SetSize(1, n_rows_desired)) {
		return false;
	}

	// Ensure that we can launch enough CUDA threads to do the interpolation
	// TODO is there a way we don't have to hard code
	if (n_rows_desired > 33553920) {
		return false;
	}

	// Set up the texture reference
	texture_reference.normalized     = false;
	texture_reference.filterMode     = cudaFilterModeLinear;
	texture_reference.addressMode[0] = cudaAddressModeClamp;
	texture_reference.addressMode[1] = cudaAddressModeClamp;

	// Create a texture description to determine how the data will be returned
	cudaChannelFormatDesc channel_description = cudaCreateChannelDesc(32, 0, 0, 0, cudaChannelFormatKindFloat);

	// Create a memory buffer that we can bind to the texture
	cudaArray* texture_data;
	if (cudaMallocArray(&texture_data, &channel_description, n_cols, n_rows) != cudaSuccess) {
		return false;
	}

	// get a pointer to the regular data so that it can be copied to the cu array
	float * regular_data_pointer;
	if (!regular_data.GetPointerToData(&regular_data_pointer)) {
		cudaFreeArray(texture_data);
		return false;
	}

	// copy the data to the cuda array
	if (cudaMemcpyToArray(texture_data, 0, 0, regular_data_pointer, n_cols*n_rows*sizeof(float), cudaMemcpyDeviceToDevice) != cudaSuccess) {
		cudaFreeArray(texture_data);
		return false;
	}

	// Bind the regular data to the texture
	if (cudaBindTextureToArray(texture_reference, texture_data, channel_description) != cudaSuccess) {
		cudaFreeArray(texture_data);
		return false;
	}

	// Get the pointers to the input arguments
	float * interpolated_data_pointer;
	float * col_loc_pointer;
	float * row_loc_pointer;

	if (!interpolated_data.GetPointerToData(&interpolated_data_pointer)) {
		cudaFreeArray(texture_data);
		return false;
	}

	if (!row_coordinates.GetPointerToData(&row_loc_pointer)) {
		cudaFreeArray(texture_data);
		return false;
	}

	if (!col_coordinates.GetPointerToData(&col_loc_pointer)) {
		cudaFreeArray(texture_data);
		return false;
	}

	// Launch the kernel to do the interpolation at the desired locations
	dim3 threadBlock(512);
	dim3 threadGrid((threadBlock.x + n_rows_desired - 1)/threadBlock.x);

	LinearInterpolate<<<threadGrid, threadBlock>>>(interpolated_data_pointer, n_rows_desired, row_loc_pointer, col_loc_pointer, out_of_bounds_value, n_rows, n_cols);

	if (cudaGetLastError() != cudaSuccess) {
		cudaFreeArray(texture_data);
		return false;
	}

	// Unbind the texture
	cudaUnbindTexture(&texture_reference);
	cudaFreeArray(texture_data);
	return true;
}


bool CUDA2DLinearInterpolator::CUDA2DInterpolateFast(CUDA2DRealMatrix<float> &regular_data, 
							 CUDA2DRealMatrix<float> &interpolated_data, 
							 CUDA2DRealMatrix<float> &row_coordinates, 
							 CUDA2DRealMatrix<float> &col_coordinates,
							 float out_of_bounds_value) {

	// Get the number of rows and columns in the regular data
    unsigned long n_rows;
	unsigned long n_cols;

	if (!regular_data.GetSize(n_rows, n_cols)) {
		return false;
	}

	float * regular_data_pointer;
	if (!regular_data.GetPointerToData(&regular_data_pointer)) {
		return false;
	}

	// Get the number of points that we wish to interpolate on and ensure
	// the the data sent is in the proper format
	unsigned long n_rows_desired;
	unsigned long n_cols_desired;
	unsigned long should_be_one;

	if (!row_coordinates.GetSize(should_be_one, n_rows_desired)) {
		return false;
	}

	if (should_be_one != 1) {
		return false;
	}

	if (!col_coordinates.GetSize(should_be_one, n_cols_desired)) {
		return false;
	}

	if (should_be_one != 1) {
		return false;
	}

	if (n_rows_desired != n_cols_desired) {
		return false;
	}

	if (!interpolated_data.SetSize(1, n_rows_desired)) {
		return false;
	}

	// Ensure that we can launch enough CUDA threads to do the interpolation
	// TODO is there a way we don't have to hard code
	// TODO handle this in a smarter way (do part of it first then iterate)
	if (n_rows_desired > 33553920) {
		return false;
	}

	// Get the pointers to the input arguments
	float * interpolated_data_pointer;
	float * col_loc_pointer;
	float * row_loc_pointer;

	if (!interpolated_data.GetPointerToData(&interpolated_data_pointer)) {
		return false;
	}

	if (!row_coordinates.GetPointerToData(&row_loc_pointer)) {
		return false;
	}

	if (!col_coordinates.GetPointerToData(&col_loc_pointer)) {
		return false;
	}

	// Launch the kernel to do the interpolation at the desired locations
	dim3 other_block(512);
	dim3 other_grid((other_block.x + n_rows_desired - 1)/other_block.x);

	BiLinearInterpolate<<<other_grid, other_block>>>(interpolated_data_pointer, n_rows_desired, row_loc_pointer, col_loc_pointer, regular_data_pointer, out_of_bounds_value, n_rows, n_cols);

	if (cudaGetLastError() != cudaSuccess) {
		return false;
	}

	return true;
}

CUDA2DCubicInterpolator::CUDA2DCubicInterpolator(unsigned long n_rows, unsigned long n_cols) {

	padded = new CUDA2DRealMatrix<float>(n_rows + 3, n_cols + 3);

	if (!padded->IsValid()) {
		error = true;
		padded = NULL;
	}
	else {
		error = false;
	}
}

CUDA2DCubicInterpolator::CUDA2DCubicInterpolator(unsigned long n_rows, unsigned long n_cols, string boundary_condition) {


	if (tolower(boundary_condition[0]) == 'o') {
		this->boundary_condition = OUT_OF_BOUNDS;
	}
	else if (tolower(boundary_condition[0]) == 'l') {
		this->boundary_condition = BILINEAR;
	}
	else {
		this->boundary_condition = DEFAULT;
	}

	padded = new CUDA2DRealMatrix<float>(n_rows + 3, n_cols + 3);

	if (!padded->IsValid()) {
		error = true;
		padded = NULL;
		this->n_rows = n_rows;
		this->n_cols = n_cols;
	}
	else {
		error = false;
		this->n_rows = 0;
		this->n_cols = 0;
	}
}

CUDA2DCubicInterpolator::~CUDA2DCubicInterpolator() {

	if (padded != NULL) {
		delete padded;
	}

}

bool CUDA2DCubicInterpolator::CUDA2DInterpolate(CUDA2DRealMatrix<float> &regular_data, 
										  CUDA2DRealMatrix<float> &interpolated_data, 
										  CUDA2DRealMatrix<float> &row_coordinates, 
										  CUDA2DRealMatrix<float> &col_coordinates,
										  float out_of_bounds_value) {
	
	if (error) {
		cout << "1" << endl;
        return false;
        
	}	

	// Get the number of rows and columns in the regular data
    unsigned long n_rows;
	unsigned long n_cols;

	if (!regular_data.GetSize(n_rows, n_cols)) {
		cout << "2" << endl;
        return false;
        
	}

	if (n_rows != this->n_rows || n_cols != this->n_cols) {
		if (padded != NULL) {
			delete padded;
		}

		padded = new CUDA2DRealMatrix<float>(n_rows+3, n_cols+3);
		this->n_rows = n_rows;
		this->n_cols = n_cols;

	}

	float * regular_data_pointer;
	if (!regular_data.GetPointerToData(&regular_data_pointer)) {
		cout << "3" << endl;
        return false;
        
	}

	// Get the number of points that we wish to interpolate on and ensure
	// the the data sent is in the proper format
	unsigned long n_rows_desired;
	unsigned long n_cols_desired;
	unsigned long should_be_one;

	if (!row_coordinates.GetSize(should_be_one, n_rows_desired)) {
		cout << "4" << endl;
        return false;
        
	}

	if (should_be_one != 1) {
		cout << "5" << endl;
        return false;
        
	}

	if (!col_coordinates.GetSize(should_be_one, n_cols_desired)) {
		cout << "6" << endl;
        return false;
        
	}

	if (should_be_one != 1) {
		cout << "7" << endl;
        return false;
        
	}

	if (n_rows_desired != n_cols_desired) {
		cout << "8" << endl;
        return false;
        
	}

	if (!interpolated_data.SetSize(1, n_rows_desired)) {
		cout << "9" << endl;
        return false;
        
	}

	// Ensure that we can launch enough CUDA threads to do the interpolation
	// TODO is there a way we don't have to hard code
	// TODO handle this in a smarter way (do part of it first then iterate)
	if (n_rows_desired > 33553920) {
		cout << "10" << endl;
        return false;
        
	}

	// Get the pointers to the input arguments
	float * interpolated_data_pointer;
	float * col_loc_pointer;
	float * row_loc_pointer;

	if (!interpolated_data.GetPointerToData(&interpolated_data_pointer)) {
		cout << "11" << endl;
        return false;
        
	}

	if (!row_coordinates.GetPointerToData(&row_loc_pointer)) {
		cout << "12" << endl;
        return false;
        
	}

	if (!col_coordinates.GetPointerToData(&col_loc_pointer)) {
		cout << "13" << endl;
        return false;
        
	}


	if (boundary_condition == DEFAULT) {

		float * padded_data_pointer;
		if (!padded->GetPointerToData(&padded_data_pointer)) {
			cout << "14" << endl;
            return false;
            
		}

		// pad the data
		dim3 threadBlock(16, 16);
		dim3 threadGrid((threadBlock.x + n_cols + 3 - 1)/threadBlock.x,
						(threadBlock.y + n_rows + 3 - 1)/threadBlock.y);

		PadBiCubic<<<threadGrid, threadBlock>>>(regular_data_pointer, padded_data_pointer, n_rows, n_cols);
		if (cudaGetLastError() != cudaSuccess) {
			cout << "15" << endl;
            return false;
            
		}


		// Launch the kernel to do the interpolation at the desired locations
		dim3 other_block(512);
		dim3 other_grid((other_block.x + n_rows_desired - 1)/other_block.x);

		BiCubicInterpolate<<<other_grid, other_block>>>(interpolated_data_pointer, n_rows_desired, row_loc_pointer, col_loc_pointer, padded_data_pointer, out_of_bounds_value, n_rows, n_cols);

		cudaError_t e = cudaGetLastError();

		if (e != cudaSuccess) {
			cout << "16" << endl;
            return false;
            
		}

	}
	else if (boundary_condition == OUT_OF_BOUNDS) {

		// Launch the kernel to do the interpolation at the desired locations
		dim3 other_block(512);
		dim3 other_grid((other_block.x + n_rows_desired - 1)/other_block.x);

		BiCubicInterpolateClampBoundary<<<other_grid, other_block>>>(interpolated_data_pointer, n_rows_desired, row_loc_pointer, col_loc_pointer, regular_data_pointer, out_of_bounds_value, n_rows, n_cols);

		if (cudaGetLastError() != cudaSuccess) {
            cout << "17" << endl;
			return false;
		}
	}
	else {
		// Launch the kernel to do the interpolation at the desired locations
		dim3 other_block(512);
		dim3 other_grid((other_block.x + n_rows_desired - 1)/other_block.x);

		BiCubicInterpolateBiLinearBoundary<<<other_grid, other_block>>>(interpolated_data_pointer, n_rows_desired, row_loc_pointer, col_loc_pointer, regular_data_pointer, out_of_bounds_value, n_rows, n_cols);

		if (cudaGetLastError() != cudaSuccess) {
            cout << "18" << endl;
			return false;
		}
	}

	return true;

}

CUDA2DCubicSplineInterpolator::CUDA2DCubicSplineInterpolator() {

	coeff_matrix    = NULL;
	n_rows_reg_grid = 0;
	n_cols_reg_grid = 0;

}

CUDA2DCubicSplineInterpolator::CUDA2DCubicSplineInterpolator(unsigned long n_rows_reg_grid, unsigned long n_cols_reg_grid) {

	error = false;

	if (n_rows_reg_grid != 0 && n_cols_reg_grid != 0) {
		coeff_matrix = new CUDA2DRealMatrix<float>(n_rows_reg_grid + 3, n_cols_reg_grid + 3);
		this->n_cols_reg_grid = n_cols_reg_grid;
		this->n_rows_reg_grid = n_rows_reg_grid;

		transposed = new CUDA2DRealMatrix<float>(n_cols_reg_grid + 3, n_rows_reg_grid + 3);

		if (!coeff_matrix->IsValid() || !transposed->IsValid()) {
			delete coeff_matrix;
			delete transposed;
			coeff_matrix = NULL;
			transposed = NULL;
			this->n_cols_reg_grid = 0;
			this->n_rows_reg_grid = 0;
			error = true;
		}

		else {
			coeff_matrix->Set(0, n_cols_reg_grid + 3, n_rows_reg_grid + 3, 0, 0);
		}
	}
	else {
		coeff_matrix = NULL;
		transposed = NULL;
		this->n_cols_reg_grid = 0;
		this->n_rows_reg_grid = 0;
		error = true;
	}

}

CUDA2DCubicSplineInterpolator::~CUDA2DCubicSplineInterpolator() {

	if (coeff_matrix != NULL) {
		delete coeff_matrix;
	}

	if (transposed != NULL) {
		delete transposed;
	}

}

bool CUDA2DCubicSplineInterpolator::CUDA2DInterpolate(CUDA2DRealMatrix<float> &regular_data, CUDA2DRealMatrix<float> &interpolated_data, 
										 CUDA2DRealMatrix<float> &row_coordinates, CUDA2DRealMatrix<float> &col_coordinates, 
										 float out_of_bounds_value) {

	if (error) {
		return false;
	}	

	// Get the number of rows and columns in the regular data
    unsigned long n_rows;
	unsigned long n_cols;

	if (!regular_data.GetSize(n_rows, n_cols)) {
		return false;
	}

	// Get the number of points that we wish to interpolate on and ensure
	// the the data sent is in the proper format
	unsigned long n_rows_desired;
	unsigned long n_cols_desired;
	unsigned long should_be_one;

	if (!row_coordinates.GetSize(should_be_one, n_rows_desired)) {
		return false;
	}

	if (should_be_one != 1) {
		return false;
	}

	if (!col_coordinates.GetSize(should_be_one, n_cols_desired)) {
		return false;
	}

	if (should_be_one != 1) {
		return false;
	}

	if (n_rows_desired != n_cols_desired) {
		return false;
	}

	if (!interpolated_data.SetSize(1, n_rows_desired)) {
		return false;
	}

	// Ensure that we can launch enough CUDA threads to do the interpolation
	// TODO is there a way we don't have to hard code
	if (n_rows_desired > 33553920) {

		// break in up into N interpolations
		unsigned long N = (n_rows_desired - 33553919)/33553920;  

		CUDA2DRealMatrix<float> sub_rows(1, 33553920);
		CUDA2DRealMatrix<float> sub_cols(1, 33553920);	

		CUDA2DRealMatrix<float> sub_interp(1, 33553920);

		float * row_loc_pointer;
		float * col_loc_pointer;
		float * output_pointer;

		if (!interpolated_data.GetPointerToData(&output_pointer)) {
			return false;
		}

		if (!sub_rows.GetPointerToData(&row_loc_pointer)) {
			return false;
		}

		if (!sub_cols.GetPointerToData(&col_loc_pointer)) {
			return false;
		}

		for (unsigned long index = 0; index < N; index++) {

			if (index == N - 1) {
				if (!sub_rows.SetSize(1, (n_rows_desired % 33553920))) {
					return false;
				}
				
				if (!sub_cols.SetSize(1, (n_rows_desired % 33553920))) {
					return false;
				}

				if (!sub_interp.SetSize(1, (n_rows_desired % 33553920))) {
					return false;
				}

				if (!row_coordinates.CopyROIToDevice<float>(row_loc_pointer, true, 0, N*33553920, 0, (N*33553920 + (n_rows_desired % 33553920)))) {
					return false;
				}

				if (!col_coordinates.CopyROIToDevice<float>(col_loc_pointer, true, 0, N*33553920, 0, (N*33553920 + (n_rows_desired % 33553920)))) {
					return false;
				}
			}

			if (!this->CUDA2DInterpolate(regular_data, sub_interp, sub_rows, sub_cols, out_of_bounds_value)) {
				return false;
			}

			if (!sub_interp.CopyDataToDevice<float>(output_pointer + sizeof(float)*33553920*N, true)) {
				return false;
			}

		}

		return true;
	}

	// Ensure the sizes are correct
	if (n_rows != n_rows_reg_grid || n_cols != n_cols_reg_grid) {
		
		if (!coeff_matrix->SetSize(n_rows + 3, n_cols + 3)) {
			return false;
		}


		if (!transposed->SetSize(n_rows + 3, n_cols + 3)) {
			return false;
		}

		if (!coeff_matrix->Set(0, n_cols_reg_grid + 3, n_rows_reg_grid + 3, 0, 0)) {
			return false;
		}

		n_rows_reg_grid = n_rows;
		n_cols_reg_grid = n_cols;
	}

	float * regular_data_pointer;
	if (!regular_data.GetPointerToData(&regular_data_pointer)) {
		return false;
	}

	// Get the pointers to the input arguments
	float * interpolated_data_pointer;
	float * col_loc_pointer;
	float * row_loc_pointer;
	float * coeff_data_pointer;
	
	if (!coeff_matrix->GetPointerToData(&coeff_data_pointer)) {
		return false;
	}

	if (!interpolated_data.GetPointerToData(&interpolated_data_pointer)) {
		return false;
	}

	if (!row_coordinates.GetPointerToData(&row_loc_pointer)) {
		return false;
	}

	if (!col_coordinates.GetPointerToData(&col_loc_pointer)) {
		return false;
	}

#ifdef PRINT_MAT
	std::ofstream file;
	file.open("..\\bin\\data.dat");
	file << regular_data.ToString();
	file.close();
#endif

	// Evaluate the coefficients for the bspline interpolation
	dim3 threadBlock(512);
	dim3 threadGrid((threadBlock.x + n_cols_reg_grid - 1)/threadBlock.x);
	
	CalcBSplineCoeffColumnWise<<<threadGrid, threadBlock>>>(regular_data_pointer, n_rows_reg_grid, n_cols_reg_grid, coeff_data_pointer, K0, -2.0f+sqrt(3.0f));

	cudaError_t e = cudaGetLastError();

	if (e != cudaSuccess) {
		return false;
	}

#ifdef PRINT_MAT
	file.open("..\\bin\\col_coeff.dat");
	file << coeff_matrix->ToString();
	file.close();
#endif

	// Now transpose the data so that we can have efficient memory accesses

	float * output_matrix_pointer;
	if (!transposed->GetPointerToData(&output_matrix_pointer)) {
		return false;
	}

	dim3 transpose_thread_block(16, 16);
	dim3 transpose_thread_grid ((transpose_thread_block.x + n_cols_reg_grid + 3 - 1)/transpose_thread_block.x,
								(transpose_thread_block.y + n_rows_reg_grid + 3 - 1)/transpose_thread_block.y);

	Transpose<float><<<transpose_thread_grid, transpose_thread_block>>>(output_matrix_pointer, coeff_data_pointer, n_cols_reg_grid + 3, n_rows_reg_grid + 3);
	if (cudaGetLastError() != cudaSuccess) {
		return false;
	}

	threadGrid.x = (threadBlock.x + n_rows_reg_grid + 3 - 1)/threadBlock.x;

	CalcBSplineCoeffRowWiseTransposed<<<threadGrid, threadBlock>>>(n_rows_reg_grid, n_cols_reg_grid, output_matrix_pointer, K0, -2.0f+sqrt(3.0f));

	if (cudaGetLastError() != cudaSuccess) {
		return false;
	}

	// now transpose back into the original matrix
	// Transpose kernel assumes row major format and we have col major format so we must invert the dimensions of the thread grid

	transpose_thread_grid.x = (transpose_thread_block.x + n_rows_reg_grid + 3 - 1)/transpose_thread_block.x;
	transpose_thread_grid.y = (transpose_thread_block.y + n_cols_reg_grid + 3 - 1)/transpose_thread_block.y;

	// also we reverse the order of the dimensions of the matrix

	Transpose<float><<<transpose_thread_grid, transpose_thread_block>>>(coeff_data_pointer, output_matrix_pointer, n_rows_reg_grid + 3, n_cols_reg_grid + 3);
	if (cudaGetLastError() != cudaSuccess) {
		return false;
	}

#ifdef PRINT_MAT
	file.open("..\\bin\\row_coeff.dat");
	file << coeff_matrix->ToString();
	file.close();
#endif

	threadGrid.x = (threadBlock.x + n_cols_desired - 1)/threadBlock.x;
	clock_t start = clock();

	CubicBSplineInterpolate<<<threadGrid, threadBlock>>>(interpolated_data_pointer, n_rows_desired, row_loc_pointer, col_loc_pointer, coeff_data_pointer, out_of_bounds_value, n_rows, n_cols);
	if (cudaGetLastError() != cudaSuccess) {
		return false;
	}

	return true;

}

///////// START fast version

CUDA2DCubicSplineInterpolatorFaster::CUDA2DCubicSplineInterpolatorFaster() {

	coeff_matrix    = NULL;
	transposed      = NULL;
	n_rows_reg_grid = 0;
	n_cols_reg_grid = 0;

}

CUDA2DCubicSplineInterpolatorFaster::CUDA2DCubicSplineInterpolatorFaster(unsigned long n_rows_reg_grid, unsigned long n_cols_reg_grid) {

	error = false;

	// Set up texturing stuff
	b_spline_texture.normalized     = false;
	b_spline_texture.filterMode     = cudaFilterModeLinear;
	b_spline_texture.addressMode[0] = cudaAddressModeClamp;
	b_spline_texture.addressMode[1] = cudaAddressModeClamp;

	// Create a texture description to determine how the data will be returned

	channel_description = cudaCreateChannelDesc(32, 0, 0, 0, cudaChannelFormatKindFloat);

	error = false;

	if (n_rows_reg_grid != 0 && n_cols_reg_grid != 0) {
		coeff_matrix = new CUDA2DRealMatrix<float>(n_rows_reg_grid + 3, n_cols_reg_grid + 3);
		this->n_cols_reg_grid = n_cols_reg_grid;
		this->n_rows_reg_grid = n_rows_reg_grid;

		transposed = new CUDA2DRealMatrix<float>(n_cols_reg_grid + 3, n_rows_reg_grid + 3);

		// Create a memory buffer that we can bind to the texture
		if (cudaMallocArray(&texture_data, &channel_description,(n_cols_reg_grid + 3), n_rows_reg_grid + 3) != cudaSuccess) {
			error = true;
		}

		if (!coeff_matrix->IsValid() || !transposed->IsValid()) {
			delete coeff_matrix;
			delete transposed;
			coeff_matrix = NULL;
			transposed = NULL;
			this->n_cols_reg_grid = 0;
			this->n_rows_reg_grid = 0;
			error = true;
		}

		else {
			coeff_matrix->Set(0, n_cols_reg_grid + 3, n_rows_reg_grid + 3, 0, 0);
		}
	}
	else {
		coeff_matrix = NULL;
		transposed = NULL;
		this->n_cols_reg_grid = 0;
		this->n_rows_reg_grid = 0;
		error = true;
	}

}

CUDA2DCubicSplineInterpolatorFaster::~CUDA2DCubicSplineInterpolatorFaster() {

	if (coeff_matrix != NULL) {
		delete coeff_matrix;
	}

	if (transposed != NULL) {
		delete transposed;
	}
	if (texture_data != NULL) {
		cudaFreeArray(texture_data);
	}

}

bool CUDA2DCubicSplineInterpolatorFaster::CUDA2DInterpolate(CUDA2DRealMatrix<float> &regular_data, CUDA2DRealMatrix<float> &interpolated_data, 
										 CUDA2DRealMatrix<float> &row_coordinates, CUDA2DRealMatrix<float> &col_coordinates, 
										 float out_of_bounds_value) {

	if (error) {
		return false;
	}	

	// Get the number of rows and columns in the regular data
    unsigned long n_rows;
	unsigned long n_cols;

	if (!regular_data.GetSize(n_rows, n_cols)) {
		return false;
	}

	// Get the number of points that we wish to interpolate on and ensure
	// the the data sent is in the proper format
	unsigned long n_rows_desired;
	unsigned long n_cols_desired;
	unsigned long should_be_one;

	if (!row_coordinates.GetSize(should_be_one, n_rows_desired)) {
		return false;
	}

	if (should_be_one != 1) {
		return false;
	}

	if (!col_coordinates.GetSize(should_be_one, n_cols_desired)) {
		return false;
	}

	if (should_be_one != 1) {
		return false;
	}

	if (n_rows_desired != n_cols_desired) {
		return false;
	}

	if (!interpolated_data.SetSize(1, n_rows_desired)) {
		return false;
	}

	// Ensure that we can launch enough CUDA threads to do the interpolation
	// TODO is there a way we don't have to hard code
	if (n_rows_desired > 33553920) {

		// don't worry about this for now
		// TODO
		return false;
		// break in up into N interpolations
		//unsigned long N = (n_rows_desired - 33553919)/33553920;  

		//CUDA2DRealMatrix<float> sub_rows(1, 33553920);
		//CUDA2DRealMatrix<float> sub_cols(1, 33553920);	

		//CUDA2DRealMatrix<float> sub_interp(1, 33553920);

		//float * row_loc_pointer;
		//float * col_loc_pointer;
		//float * output_pointer;

		//if (!interpolated_data.GetPointerToData(&output_pointer)) {
		//	return false;
		//}

		//if (!sub_rows.GetPointerToData(&row_loc_pointer)) {
		//	return false;
		//}

		//if (!sub_cols.GetPointerToData(&col_loc_pointer)) {
		//	return false;
		//}

		//for (unsigned long index = 0; index < N; index++) {

		//	if (index == N - 1) {
		//		if (!sub_rows.SetSize(1, (n_rows_desired % 33553920))) {
		//			return false;
		//		}
		//		
		//		if (!sub_cols.SetSize(1, (n_rows_desired % 33553920))) {
		//			return false;
		//		}

		//		if (!sub_interp.SetSize(1, (n_rows_desired % 33553920))) {
		//			return false;
		//		}

		//		if (!row_coordinates.CopyROIToDevice<float>(row_loc_pointer, true, 0, N*33553920, 0, (N*33553920 + (n_rows_desired % 33553920)))) {
		//			return false;
		//		}

		//		if (!col_coordinates.CopyROIToDevice<float>(col_loc_pointer, true, 0, N*33553920, 0, (N*33553920 + (n_rows_desired % 33553920)))) {
		//			return false;
		//		}
		//	}

		//	if (!this->CUDA2DInterpolate(regular_data, sub_interp, sub_rows, sub_cols, out_of_bounds_value)) {
		//		return false;
		//	}

		//	if (!sub_interp.CopyDataToDevice<float>(output_pointer + sizeof(float)*33553920*N, true)) {
		//		return false;
		//	}

		//}

		//return true;
	}

	// Ensure the sizes are correct
	if (n_rows != n_rows_reg_grid || n_cols != n_cols_reg_grid) {

		if (!coeff_matrix->SetSize(n_rows + 3, n_cols + 3)) {
			return false;
		}


		if (!transposed->SetSize(n_rows + 3, n_cols + 3)) {
			return false;
		}

		if (!coeff_matrix->Set(0, n_cols_reg_grid + 3, n_rows_reg_grid + 3, 0, 0)) {
			return false;
		}

		n_rows_reg_grid = n_rows;
		n_cols_reg_grid = n_cols;
	}

	float * regular_data_pointer;
	if (!regular_data.GetPointerToData(&regular_data_pointer)) {

		return false;
	}

	// Get the pointers to the input arguments
	float * interpolated_data_pointer;
	float * col_loc_pointer;
	float * row_loc_pointer;
	float * coeff_data_pointer;

	if (!coeff_matrix->GetPointerToData(&coeff_data_pointer)) {
		return false;
	}

	if (!interpolated_data.GetPointerToData(&interpolated_data_pointer)) {
		return false;
	}

	if (!row_coordinates.GetPointerToData(&row_loc_pointer)) {
		return false;
	}

	if (!col_coordinates.GetPointerToData(&col_loc_pointer)) {
		return false;
	}

#ifdef PRINT_MAT
	std::ofstream file;
	file.open("..\\bin\\data.dat");
	file << regular_data.ToString();
	file.close();
#endif

	// Evaluate the coefficients for the bspline interpolation
	dim3 threadBlock(512);
	dim3 threadGrid((threadBlock.x + n_cols_reg_grid - 1)/threadBlock.x);
	
	CalcBSplineCoeffColumnWise<<<threadGrid, threadBlock>>>(regular_data_pointer, n_rows_reg_grid, n_cols_reg_grid, coeff_data_pointer, K0, -2.0f+sqrt(3.0f));

	cudaError_t e = cudaGetLastError();

	if (e != cudaSuccess) {
		return false;
	}

#ifdef PRINT_MAT
	file.open("..\\bin\\col_coeff.dat");
	file << coeff_matrix->ToString();
	file.close();
#endif

	// Now transpose the data so that we can have efficient memory accesses
	float * output_matrix_pointer;
	if (!transposed->GetPointerToData(&output_matrix_pointer)) {
		return false;
	}

	dim3 transpose_thread_block(16, 16);
	dim3 transpose_thread_grid ((transpose_thread_block.x + n_cols_reg_grid + 3 - 1)/transpose_thread_block.x,
								(transpose_thread_block.y + n_rows_reg_grid + 3 - 1)/transpose_thread_block.y);

	Transpose<float><<<transpose_thread_grid, transpose_thread_block>>>(output_matrix_pointer, coeff_data_pointer, n_cols_reg_grid + 3, n_rows_reg_grid + 3);
	e = cudaGetLastError();

	if (e != cudaSuccess) {
		return false;
	}

	threadGrid.x = (threadBlock.x + n_rows_reg_grid + 3 - 1)/threadBlock.x;

	CalcBSplineCoeffRowWiseTransposed<<<threadGrid, threadBlock>>>(n_rows_reg_grid, n_cols_reg_grid, output_matrix_pointer, K0, -2.0f+sqrt(3.0f));

	e = cudaGetLastError();

	if (e != cudaSuccess) {
		return false;
	}

	// now transpose back into the original matrix
	// Transpose kernel assumes row major format and we have col major format so we must invert the dimensions of the thread grid

	transpose_thread_grid.x = (transpose_thread_block.x + n_rows_reg_grid + 3 - 1)/transpose_thread_block.x;
	transpose_thread_grid.y = (transpose_thread_block.y + n_cols_reg_grid + 3 - 1)/transpose_thread_block.y;

	// also we reverse the order of the dimensions of the matrix

	Transpose<float><<<transpose_thread_grid, transpose_thread_block>>>(coeff_data_pointer, output_matrix_pointer, n_rows_reg_grid + 3, n_cols_reg_grid + 3);
	e = cudaGetLastError();
	if (e != cudaSuccess) {
		return false;
	}

#ifdef PRINT_MAT
	file.open("..\\bin\\row_coeff.dat");
	file << coeff_matrix->ToString();
	file.close();
#endif
	threadGrid.x = (threadBlock.x + n_cols_desired - 1)/threadBlock.x;

	// bind the coefficients

	// copy the data to the cuda array
	if (( e = cudaMemcpyToArray(texture_data, 0, 0, coeff_data_pointer, (n_cols_reg_grid + 3)*(n_rows_reg_grid + 3)*sizeof(float), cudaMemcpyDeviceToDevice)) != cudaSuccess) {
		return false;
	}

	// Bind the regular data to the texture
	if (cudaBindTextureToArray(b_spline_texture, texture_data, channel_description) != cudaSuccess) {
		return false;
	}

	//if ((e = cudaBindTexture2D(NULL, &b_spline_texture, coeff_data_pointer, &channel_description, (n_cols_reg_grid + 3), n_rows_reg_grid + 3, sizeof(float)*(n_cols_reg_grid + 3) ) ) != cudaSuccess) {
	//		return false;
	//}


	CubicBSplineInterpolateFast<<<threadGrid, threadBlock>>>(interpolated_data_pointer, n_rows_desired, row_loc_pointer, col_loc_pointer, out_of_bounds_value, n_rows, n_cols);
	if (e != cudaSuccess) {
		return false;
	}

	e = cudaGetLastError();


	cudaUnbindTexture(&b_spline_texture);
	
	return true;

}



