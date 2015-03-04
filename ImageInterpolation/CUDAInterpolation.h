#pragma once

// Forward definition of classes
class CUDA2DInterpolator;

template <class DataType>
class CUDA2DRealMatrix;


#include <cuda.h>
#include <math.h>
#include <string>

#define DEFAULT		  1
#define OUT_OF_BOUNDS 2
#define BILINEAR	  3

#define N_U_BINS 1000

#define N_B_BINS 1000

#define K0 28

// Does linear interpolation

// Deprecated 

bool CUDALinearInterpolation(CUDA2DRealMatrix<float> &regular_data, 
							 CUDA2DRealMatrix<float> &interpolated_data, 
							 CUDA2DRealMatrix<float> &row_coordinates, 
							 CUDA2DRealMatrix<float> &col_coordinates,
							  float out_of_bounds_value);

/********************************
 * This class provides an interface for a generic interpolation
 * class. Creating a common interface in a superclass is convienient
 * as only one python interface will be required. All subclasses
 * will inherit the interface
 ********************************/

// TODO: Smarter error checking (returning strings)

class CUDA2DInterpolator {

public:

	// Assuming regularly spaced data in both the x and y directions in which the 
	// first element of input_regularly_spaced_data corresponds to location (0,0) and 
	// the spacing between samples is one, if an output row and col coordinate correspond
	// to a location outside of the regularly spaced grid, then the output value
	// at that location will be out_of_bounds_value

	virtual bool CUDA2DInterpolate(CUDA2DRealMatrix<float> &input_regularly_spaced_data, 
								   CUDA2DRealMatrix<float> &output_interpolated_data, 
								   CUDA2DRealMatrix<float> &output_row_coordinates, 
								   CUDA2DRealMatrix<float> &output_col_coordinates,
								   float out_of_bounds_value) = 0;

};


/********************************
 * Does linear interpolation when the CUDA2DInterpolate method is called
 * it uses the interpolation hardware present on NVidia graphics
 * cards to do this.
 ********************************/

class CUDA2DLinearInterpolator : public CUDA2DInterpolator {

public:

	virtual bool CUDA2DInterpolate(CUDA2DRealMatrix<float> &input_regularly_spaced_data, 
								   CUDA2DRealMatrix<float> &output_interpolated_data, 
								   CUDA2DRealMatrix<float> &output_row_coordinates, 
								   CUDA2DRealMatrix<float> &output_col_coordinates,
								   float out_of_bounds_value);

	virtual bool CUDA2DInterpolateFast(CUDA2DRealMatrix<float> &input_regularly_spaced_data, 
								   CUDA2DRealMatrix<float> &output_interpolated_data, 
								   CUDA2DRealMatrix<float> &output_row_coordinates, 
								   CUDA2DRealMatrix<float> &output_col_coordinates,
								   float out_of_bounds_value);


};

class CUDA2DNearestNeighborInterpolator : public CUDA2DInterpolator {

public:

	virtual bool CUDA2DInterpolate(CUDA2DRealMatrix<float> &input_regularly_spaced_data, 
								   CUDA2DRealMatrix<float> &output_interpolated_data, 
								   CUDA2DRealMatrix<float> &output_row_coordinates, 
								   CUDA2DRealMatrix<float> &output_col_coordinates,
								   float out_of_bounds_value);


};


/********************************************
 * This class performs Cubic Convolution Interpolation as described
 * in:	
 *		R. Keys, (1981). "Cubic convolution interpolation for digital image processing". 
 *		IEEE Transactions on Signal Processing, Acoustics, Speech, and Signal Processing 29: 1153.
 * ********************************************/

class CUDA2DCubicInterpolator : public CUDA2DInterpolator {

public:

	// Calling this constructor will use the boundary conditions proposed in the 
	// reference above

	CUDA2DCubicInterpolator(unsigned long n_rows, unsigned long n_cols);

	// Calling this constructor will use the boundary conditions specified
	// the boundary condition string. This string will be interpereted 
	// as follows
	//
	// boundary_condition_method = default         ('d')
	// ""					     = out_of_bounds   ('o')
	// ""						 = bilinear_interp ('l')
	//
	// In order to save computation time, the convolution kernel is precalculated
	// over the interval 0 to 2. The interval is devided into n_u_points bins and
	// linear interpolation is done between the bins
	 
	CUDA2DCubicInterpolator(unsigned long n_rows, unsigned long n_cols, std::string boundary_condition_method);

	~CUDA2DCubicInterpolator();

	virtual bool CUDA2DInterpolate(CUDA2DRealMatrix<float> &input_regularly_spaced_data, 
								   CUDA2DRealMatrix<float> &output_interpolated_data, 
								   CUDA2DRealMatrix<float> &output_row_coordinates, 
								   CUDA2DRealMatrix<float> &output_col_coordinates,
								   float out_of_bounds_value);

private:
	
	bool        error;

	CUDA2DRealMatrix<float> * padded;
	unsigned long n_rows;
	unsigned long n_cols;

	// 1 => default
	// 2 => out_of_bounds
	// 3 => bilinear
	int boundary_condition;

};

/************************************************
 * This class performs cubic b-spline interpolation as defined in
 * Unser M 1999 Splines: a perfect fit for signal and image processing IEEE Signal Process. Mag. 16 22–38
 */

class CUDA2DCubicSplineInterpolator : public CUDA2DInterpolator {

public:

	CUDA2DCubicSplineInterpolator(unsigned long n_rows_reg_grid, unsigned long n_cols_reg_grid);

	~CUDA2DCubicSplineInterpolator();


	virtual bool CUDA2DInterpolate(CUDA2DRealMatrix<float> &input_regularly_spaced_data, 
								   CUDA2DRealMatrix<float> &output_interpolated_data, 
								   CUDA2DRealMatrix<float> &output_row_coordinates, 
								   CUDA2DRealMatrix<float> &output_col_coordinates,
								   float out_of_bounds_value);
private:

	// Do not call the default constructor
	CUDA2DCubicSplineInterpolator();

	CUDA2DRealMatrix<float> * coeff_matrix;
	CUDA2DRealMatrix<float> * transposed;

	unsigned long n_rows_reg_grid;
	unsigned long n_cols_reg_grid;

	bool          error;
};

class CUDA2DCubicSplineInterpolatorFaster : public CUDA2DInterpolator {

public:

	CUDA2DCubicSplineInterpolatorFaster(unsigned long n_rows_reg_grid, unsigned long n_cols_reg_grid);

	~CUDA2DCubicSplineInterpolatorFaster();


	virtual bool CUDA2DInterpolate(CUDA2DRealMatrix<float> &input_regularly_spaced_data, 
								   CUDA2DRealMatrix<float> &output_interpolated_data, 
								   CUDA2DRealMatrix<float> &output_row_coordinates, 
								   CUDA2DRealMatrix<float> &output_col_coordinates,
								   float out_of_bounds_value);
private:

	// Do not call the default constructor
	CUDA2DCubicSplineInterpolatorFaster();

	CUDA2DRealMatrix<float> * coeff_matrix;
	CUDA2DRealMatrix<float> * transposed;
	cudaArray * texture_data;

	unsigned long n_rows_reg_grid;
	unsigned long n_cols_reg_grid;

	cudaChannelFormatDesc channel_description;

	bool          error;
};

#include <CUDAImageProcessing/CUDA2DRealMatrix/Cpp/CUDA2DRealMatrix.h>