#pragma once

#include <fftw3.h>
#include <string>
#include <vector>
#include <fstream>

/// These #define statements will return the number of columns and
/// the number of elements actually stored
#define NUM_COLS_COMPLEX(n_cols)			(n_cols / 2 + 1)
#define NUM_ELEMENTS_COMPLEX(n_rows, n_cols) n_rows * NUM_COLS_COMPLEX(n_cols)

/// Used to tell the class what type of padding to use
enum PaddingType {NONE          = 0, 
			      POWER_OF_2    = 1, 
				  MULTIPLE_OF_2 = 2,
				  MULTIPLE_OF_8 = 3};


/// CPU based implementation of the normalized cross correlation.
///
/// This class provides an interface to perform the template matching
/// algorithm described by JP lewis in 
/// "Fast Template Matching", Vision Interface 1995. It relies heavily
/// on the FFTW library (fftw.org) to perform fast fourrier transforms

class NormalizedCrossCorrelator {

	// private members
	private:

		fftwf_complex * padded_complex_buffer_1;
		fftwf_complex * padded_complex_buffer_2;
		fftwf_complex * padded_complex_buffer_3;
		fftwf_complex * padded_complex_buffer_4;
		fftwf_complex * padded_complex_buffer_5;

		float *         padded_real_buffer_1;
		float *         padded_real_buffer_2;
		float *         padded_real_buffer_3;

		// staging areas for the reference and current frame
		float *         reference_stage;
		float *         current_stage;

		// pointers to the buffers that are currently being used
		// to compute NCC
		float *         active_reference_norm_factor;
		float *         active_current_norm_factor;

		fftwf_complex * active_reference_fft;
		fftwf_complex * active_current_fft;

		// these vectors store pointers to the cached reference and current fames
		std::vector<float *>         reference_norm_factors_cache;
		std::vector<fftwf_complex *> reference_fft_cache;

		std::vector<float *>         current_norm_factors_cache;
		std::vector<fftwf_complex *> current_fft_cache;

		//Size of the reference frame
		unsigned long	n_rows_reference;
		unsigned long	n_columns_reference;

		//Size of the current frame
		unsigned long	n_rows_current;
		unsigned long	n_columns_current;

		//Padded matrix sizes
		unsigned long	rows_padded;
		unsigned long	columns_padded;

		//Extra padding needed to ensure that the padded buffer
		//dimensions are a power of two, if the padded buffers are
		//powers <some extra padding here> runs much faster
		// Im not sure if this is needed
		unsigned long   rows_extra;
		unsigned long   columns_extra;

		//Indicators to determine if cross correlation can begin
		bool			reference_set;
		bool			current_set;
		bool			correlated;

		//threshold value
		float			overlap_norm_threshold;

		// FFTW fft plans
		fftwf_plan      forward_plan;
		fftwf_plan      reverse_plan;
		bool            plans_initialized;

	public:

		/// Imports wisdom from file
		///
		/// \param fftw_wisdom_file
		///        wisdom file to import
		static bool ImportWisdomFile(std::string fftw_wisdom_file);

		/// Exports current wisdom to file
		///
		/// \param fftw_wisdom_file
		///        wisdom file name to export
		static bool ExportWisdomFile(std::string fftw_wisdom_file);

		/// Constructor.
		/// 
		/// \param overlap_norm_threshold
		///        When the norm of the overlapping area of either
		///        the reference or current frame is less than this
		///        value, then the correlation is set to 0
		NormalizedCrossCorrelator(float overlap_norm_threshold);



		/// Destructor.
		~NormalizedCrossCorrelator();

		/// Initializes internal structures with the given sizes.
		/// To accomodate patient follow ups that may have had previouslly registered data, we
		/// allow the ability to set arbitrary images shapes. This is done by using
		/// the image mask parameter. The dimensions set here should be the full size of 
		/// the image mask. For subsequent calls to SetReference or SetCurrent, the passed
		/// data should have the dimensions set here.
		///
		/// \param n_rows_reference
		///        the number of rows in the reference image mask
		///
		/// \param n_columns_reference
		///        the number of columns in the reference image_mask
		///
		/// \param reference_image_mask
		///        The image image mask that will be applied to every reference frame
		///        set. The image mask should have the dimensions specified by the
		///        above mentioned parameters. A non-zero value indicates that a
		///        a pixel exists in this index, while a zero value indicates it does
		///        not exist.
		///
		/// \param n_rows_reference
		///        the number of rows in the reference image mask
		///
		/// \param n_columns_reference
		///        the number of columns in the reference image_mask
		///
		/// \param current_image_mask
		///        The image image mask that will be applied to every reference frame
		///        set. The image mask should have the dimensions specified by the
		///        above mentioned parameters. A non-zero value indicates that a
		///        a pixel exists in this index, while a zero value indicates it does
		///        not exist.
		///
		/// \param padding_scheme
		///        This value determines what kind of padding will be used before the FFTs are
		///        performed use one of the following values
		///        0 -> no padding
		///        1 -> pad to power of two in both dimensions
		///        2 -> pad to multiple of two in both directions
		///        3 -> pad to multiple of eight in both directions
		///
		template <class MASK_TYPE>
		bool SetFrameSizes(unsigned long n_rows_reference,     unsigned long n_columns_reference, 
						   MASK_TYPE *   reference_image_mask, unsigned long n_rows_current,   
						   unsigned long n_columns_current,    MASK_TYPE *   current_image_mask,
						   unsigned int  padding_scheme);

		template <class MASK_TYPE>
		bool SetReferencePupil(MASK_TYPE * reference_pupil);
		
		template <class MASK_TYPE>
		bool SetCurrentPupil(MASK_TYPE * current_pupil);

		/// Sets the frame sizes with no image mask.
		///
		/// \param n_rows_reference
		///        the number of rows in the reference image mask
		///
		/// \param n_columns_reference
		///        the number of columns in the reference image_mask
		///
		/// \param n_rows_reference
		///        the number of rows in the reference image mask
		///
		/// \param n_columns_reference
		///        the number of columns in the reference image_mask
		bool SetFrameSizes(unsigned long n_rows_reference, unsigned long n_columns_reference,
						   unsigned long n_rows_current,   unsigned long n_columns_current,
						   PaddingType padding);



		/// Specify the reference frame of the correlation.
		///
		/// \param reference_buffer
		///        pointer to buffer containing the reference frame. It
		///        is assumed row major contiguous data with dimensions
		///        as set by the SetFrameSizes methods
		bool SetReference(float * reference_buffer);	

		/// Specify the reference frame of the correlation. And an associated
		/// image mask
		///
		/// \param reference_buffer
		///        pointer to buffer containing the reference frame. It
		///        is assumed row major contiguous data with dimensions
		///        as set by the SetFrameSizes methods
		///
		/// \param reference_mask
		///        pointer to a buffer containing the mask. A non-zero value
		///        in a pixel location indicates a valid pixel.
		bool SetReference(float * reference_buffer, int * reference_mask);	

		/// This method allows the user to pass only a ROI to the reference frame. 
		/// The ROI should be the same dimensions as the reference frame
		/// This method will return false if the ROI dimension is not correct or if
		/// there is an error copying
		///
		/// \param reference_buffer
		///        row major buffer containing the reference image ROI,
		///
		/// \param stride_x
		///        The number of floating point values in one line of the reference buffer
		///
		/// \param top_row, left_column, bottom_row, right_column
		///        Describes the ROI inside the reference_buffer. The ROI dimensions
		///        should match the dimensions set by the SetFrameSizes method
		
		bool SetReferenceROI(float * reference_buffer, unsigned long stride_x,
							unsigned long top_row,     unsigned long left_column, 
							unsigned long bottom_row,  unsigned long right_column);

		/// This method allows the user to pass only a ROI to the reference frame. 
		/// The ROI should be the same dimensions as the reference frame
		/// This method will return false if the ROI dimension is not correct or if
		/// there is an error copying
		///
		/// \param reference_buffer
		///        row major buffer containing the reference image ROI,
		///
		/// \param reference_mask
		///        A pointer to an integer array. A non-zero value in an index
		///        indicates a valid pixel there.
		///
		/// \param stride_x
		///        The number of floating point values in one line of the reference buffer
		///
		/// \param top_row, left_column, bottom_row, right_column
		///        Describes the ROI inside the reference_buffer. The ROI dimensions
		///        should match the dimensions set by the SetFrameSizes method
		
		bool SetReferenceROI(float * reference_buffer, int * reference_mask,
							unsigned long stride_x,
							unsigned long top_row,     unsigned long left_column, 
							unsigned long bottom_row,  unsigned long right_column);

		/// Caches the reference frame into a local buffer.
		/// After caching the frame, the precomputed normalization factor and 
		/// FFT of the frame can be recalled in constant time by calling 
		/// SetReferenceFrame(int handle). The CacheReferenceFrame returns a
		/// non negative handle that can be used in subsequent calls to 
		/// SetReferenceFrame. If the return value is negative, then the memory
		/// allocation failed or the frame sizes have not been set up correctly
		///
		int CacheReferenceFrame();
		
		/// Recals a cached reference cached using the CacheReferenceFrame method.
		/// Previous calls to CacheReferenceFrame return a handle that can be passed
		/// here. This method sets the active normalization factor and reference FFT
		/// to that of the reference as indicated by the handle object.
		///
		/// \param handle
		///        Integer representing the cache you wish to recal.
		bool SetReference(int handle);

		template <class REFERENCE_TYPE>
		bool SetReference(REFERENCE_TYPE * current_buffer);


		/// Specify the current frame of the correlation.
		///
		/// \param current_bufferr
		///        pointer to buffer containing the current frame. It
		///        is assumed row major contiguous data with dimensions
		///        as set by the SetFrameSizes methods
		bool SetCurrent(float * current_buffer);

		/// Specify the reference frame of the correlation. And an associated
		/// image mask
		///
		/// \param current_buffer
		///        pointer to buffer containing the current frame. It
		///        is assumed row major contiguous data with dimensions
		///        as set by the SetFrameSizes methods
		///
		/// \param current_mask
		///        pointer to a buffer containing the mask. A non-zero value
		///        in a pixel location indicates a valid pixel.
		bool SetCurrent(float * current_buffer, int * current_mask);	

		template <class FLOATING_TYPE>
		bool SetCurrent(FLOATING_TYPE * current_buffer);

		/// This method allows the user to pass only a ROI to the reference frame. 
		/// The ROI should be the same dimensions as the reference frame
		/// This method will return false if the ROI dimension is not correct or if
		/// there is an error copying
		///
		/// \param current_buffer
		///        row major buffer containing the current image ROI,
		///
		/// \param stride_x
		///        The number of floating point values in one line of the reference buffer
		///
		/// \param top_row, left_column, bottom_row, right_column
		///        Describes the ROI inside the reference_buffer. The ROI dimensions
		///        should match the dimensions set by the SetFrameSizes method
		bool SetCurrentROI(float * current_buffer,   unsigned long stride_x,
						   unsigned long top_row,    unsigned long left_column, 
						   unsigned long bottom_row, unsigned long right_column);

		/// This method allows the user to pass only a ROI to the reference frame. 
		/// The ROI should be the same dimensions as the reference frame
		/// This method will return false if the ROI dimension is not correct or if
		/// there is an error copying
		///
		/// \param current_buffer
		///        row major buffer containing the current image ROI,
		///
		/// \param stride_x
		///        The number of floating point values in one line of the reference buffer
		///
		/// \param top_row, left_column, bottom_row, right_column
		///        Describes the ROI inside the reference_buffer. The ROI dimensions
		///        should match the dimensions set by the SetFrameSizes method
		bool SetCurrentROI(float * current_buffer,   int * current_mask,
						   unsigned long stride_x,
						   unsigned long top_row,    unsigned long left_column, 
						   unsigned long bottom_row, unsigned long right_column);

		/// Caches the current frame into a local buffer.
		/// After caching the frame, the precomputed normalization factor and 
		/// FFT of the frame can be recalled in constant time by calling 
		/// SetCurrentFrame(int handle). The CacheCurrentFrame returns a
		/// non negative handle that can be used in subsequent calls to 
		/// SetCurrentFrame. If the return value is negative, then the memory
		/// allocation failed or the frame sizes have not been set up correctly
		///
		int CacheCurrentFrame();
		
		/// Recals a cached current cached using the CacheCurrentFrame method.
		/// Previous calls to CacheCurrentFrame return a handle that can be passed
		/// here. This method sets the active normalization factor and current FFT
		/// to that of the current as indicated by the handle object.
		///
		/// \param handle
		///        Integer representing the cache you wish to recall.
		bool SetCurrent(int handle);

		/// Performs the normalized cross correlation.
		/// Only computes if the reference and the current have been set
		bool NormalizedCrossCorrelate();

		/// Returns the maximum correlation value and the location of it.
		///
		/// \param val
		///        A reference variable for output of the maximum correlation value
		/// \param row_max
		///        A reference varaible for output of the row of the correlation matrix
		///        where the maximum is located
		/// \param column_max
		///        A reference varaible for output of the column of the correlation matrix
		///        where the maximum is located
		bool GetCrossCorrelationMax(float &val, unsigned long &row_max,   unsigned long &column_max);

		/// Returns the maximum correlation value within a ROI and the location of it.
		/// Note, the location will be an absolute location within the entire NCC matrix
		/// \param val
		///        A reference variable for output of the maximum correlation value
		/// \param row_max
		///        A reference varaible for output of the row of the correlation matrix
		///        where the maximum is located
		/// \param column_max
		///        A reference varaible for output of the column of the correlation matrix
		///        where the maximum is located
		///
		/// \param top_row, left_column, bottom_row, right_column
		///        variables used to determine the ROI
		bool GetCrossCorrelationMaxROI(float &val, unsigned long &row_max,   unsigned long &column_max, 
									              unsigned long top_row,    unsigned long left_column,
												  unsigned long bottom_row, unsigned long right_column);

		/// Returns the maximum in the cross correlation matrix.
		/// The search area will only include pixels that garentee a certain number
		/// of overlapping pixels
		///
		/// \param val is the maximum value found
		/// \param row_max will contain the row where the maximum existed
		/// \param column_max will contian the column where the maximum existed
		/// \param n_overlapping_pixels is the minimum number of pixels required 
		///                             to overlap at the corresponding translation
		bool GetCrossCorrelationMax(float & val, unsigned long &row_max, unsigned long &column_max, 
									unsigned long n_overlapping_pixels);

		/// Copies the entier NCC matrix to the provided buffer.
		///
		/// \param output_host_buffer
		///        row major contiguous buffer where the NCC will be stored. 
		bool GetCorrelationMatrix(float * output_host_buffer);

		/// \return returns the number of columns in the reference frame
		//          as specified by the SetFrameSizes method
		unsigned long GetNumberOfColumnsReference();


		/// \return returns the number of columns in the current frame
		//          as specified by the SetFrameSizes method
		unsigned long GetNumberOfColumnsCurrent();
		
		/// \return returns the number of rows in the reference frame
		//          as specified by the SetFrameSizes method
		unsigned long GetNumberOfRowsReference();


		/// \return returns the number of rows in the current frame
		//          as specified by the SetFrameSizes method
		unsigned long GetNumberOfRowsCurrent();

	private:

		/// Default constructor, private so you can't use it.
		NormalizedCrossCorrelator();

		/// initializes internal structures.
		void InitializeData();

		/// Sets the sizes for the intermediate buffers.
		/// additionally it allocates fft object and other required resrouces
		/// Use SetFrameSizes
		void InitializeDimensions(unsigned long n_rows_reference,         unsigned long n_columns_reference, 
								   unsigned long n_rows_current,          unsigned long n_columns_current,
								   unsigned int padding_scheme);

		/// Creates the pupil given the image mask.
		/// use SetFrameSizes(...) 
		template <class MASK_TYPE>
		bool CreatePupils(MASK_TYPE * reference_mask, MASK_TYPE * current_mask);

		/// deallocates the intermediate buffers.
		void ClearBuffers();

		/// calculates norm factors.
		bool CalculateNormFactorOne();
		bool CalculateNormFactorTwo();

};

/////////////////
// Templated methods and functions
/////////////////

/////////////////
// Helper functions
/////////////////
template<class T>
void DumpMatrix(std::string file_name, unsigned long n_rows, unsigned long n_cols, T * matrix_buffer) {

	std::ofstream output_file(file_name);

	for (unsigned long cur_row = 0; cur_row < n_rows; cur_row++) {
		for (unsigned long cur_col = 0; cur_col <n_cols; cur_col++) {

			if (cur_col > 0) {
				output_file << ",\t";
			}

			output_file << matrix_buffer[cur_row*n_cols+cur_col];
		}

		output_file << endl;
	}

	output_file.close();
}

// copies a ROI from a larger buffer (input) to a different buffer (output) whos dimensions are the exact size
// of the ROI
template<class T>
void SetROIFromBuffer(T * input,                  unsigned long long input_stride_in_elements,           
                      unsigned long long right_column, unsigned long long left_column,  
					  unsigned long long bottom_row,   unsigned long long top_row,      
					  T * output,                   unsigned long long output_stride_in_elements) {

	// dimensions of 
	unsigned long long ROI_columns = right_column  - left_column + 1;
	unsigned long long ROI_rows    = bottom_row    - top_row     + 1;
	
	unsigned long long cur_input_row_pointer  = ((unsigned long long)input) + top_row*input_stride_in_elements*sizeof(T);
	unsigned long long cur_output_row_pointer = (unsigned long long) output;


	for (unsigned long cur_row = 0; cur_row < ROI_rows; cur_row++) {

		memcpy((void *) cur_output_row_pointer, (void *) cur_input_row_pointer, sizeof(T)*ROI_columns);

		cur_input_row_pointer  += input_stride_in_elements*sizeof(T);
		cur_output_row_pointer += output_stride_in_elements*sizeof(T);

	}

}


// Squares (element by element) a ROI in a matrix

template<class T>
void SquareROI(T * buffer, unsigned long ROI_rows, unsigned long ROI_cols, 
			   unsigned long n_columns_buffer, unsigned long start_row, unsigned long start_col) {

    for (unsigned long row = 0; row < ROI_rows; row++) {
		for (unsigned long col = 0; col < ROI_cols; col++) {
			unsigned long index  = (start_row + row)*n_columns_buffer + (col + start_col);

			T val = buffer[index];

			//Assign the ROI
			buffer[index] = val*val;

		}
	}

}


template<class MASK_TYPE>
void CreatePupil(unsigned long buffer_rows, unsigned long buffer_columns, unsigned long pupil_columns, 
				 unsigned long pupil_rows, float * buffer, unsigned long start_row, unsigned long start_col,
				 MASK_TYPE * image_mask) {

	//Indexing variables
	for (unsigned long row = 0; row < buffer_rows; row++) {
		for (unsigned long col = 0; col < buffer_columns; col++) {

			unsigned long index = row*buffer_columns + col;
		
			if (row >= start_row && row < start_row + pupil_rows &&
				col >= start_col && col < start_col + pupil_columns) {

					// inside of image mask. Note that the pupil is significantly
					// bigger than the image mask. The indexing variables refer to 
					// the location in the pupil (paramater named buffer). The pupil
					// rows and pupil_columns refer to the dimensions of the image mask
				
					unsigned long mask_row = row - start_row;
					unsigned long mask_col = col - start_col;

					// index into the image mask
					unsigned long mask_index = (mask_row)*pupil_columns + mask_col;

					// a non zero value indicates a pixel exists here so we check
					float value = (float) image_mask[mask_index];	

					if (value != 0) {
						value = (float) 1;
					}

					buffer[index] = value;	
			}
			else {
					// outside of image mask, just set to zero
					buffer[index] = 0;
			}
		}
	}
}

/////////////////
// Methods
/////////////////

template <class MASK_TYPE>
bool NormalizedCrossCorrelator::SetFrameSizes(unsigned long n_rows_reference,     unsigned long n_columns_reference, 
										  	  MASK_TYPE *   reference_image_mask, unsigned long n_rows_current,   
											  unsigned long n_columns_current,    MASK_TYPE *   current_image_mask,
											  unsigned int  padding_scheme) {

	InitializeDimensions(n_rows_reference, n_columns_reference, 
						 n_rows_current, n_columns_current, padding_scheme);


	CreatePupils(reference_image_mask, current_image_mask);

	return true;
}


// Creates pupils
template <class MASK_TYPE>
bool NormalizedCrossCorrelator::CreatePupils(MASK_TYPE * reference_mask, MASK_TYPE * current_mask) {

	// now create the pupil with the given image mask
	CreatePupil<MASK_TYPE>(rows_padded, columns_padded, n_columns_current,   
				n_rows_current, padded_real_buffer_1, rows_extra + n_rows_reference - 1,
				n_columns_reference - 1, current_mask);
#ifdef PRINT_PUPIL
	DumpMatrix("current_pupil.dat", rows_padded, columns_padded, padded_real_buffer_1);
#endif


	// Execute the forward FFT on the pupil
	fftwf_execute_dft_r2c(forward_plan, padded_real_buffer_1, padded_complex_buffer_1);

#ifdef PRINT_PUPIL
	DumpComplexMatrix("current_pupil_fft_real.dat", "current_pupil_fft_imag.dat", 
			    rows_padded, NUM_COLS_COMPLEX(columns_padded), padded_complex_buffer_1);
#endif 

	CreatePupil<MASK_TYPE>(rows_padded, columns_padded, 
				n_columns_reference, n_rows_reference, 
				padded_real_buffer_1, rows_extra, 0, reference_mask);

#ifdef PRINT_PUPIL
	DumpMatrix("reference_pupil.dat", rows_padded, columns_padded, padded_real_buffer_1);
#endif

	// Execute the forward FFT on the pupil
	fftwf_execute_dft_r2c(forward_plan, padded_real_buffer_1, padded_complex_buffer_4);

#ifdef PRINT_PUPIL
	DumpComplexMatrix("reference_pupil_fft_real.dat", "reference_pupil_fft_imag.dat", 
			    rows_padded, NUM_COLS_COMPLEX(columns_padded), padded_complex_buffer_4);
#endif 

	return true;

}

template <class MASK_TYPE>
bool NormalizedCrossCorrelator::SetReferencePupil(MASK_TYPE * reference_pupil) {

	//Ensure that the sizes have been set
	if (n_rows_reference  == 0 || n_columns_reference == 0 || n_rows_current == 0 || 
		n_columns_current == 0 || rows_padded == 0         || columns_padded == 0) {
		return false;
	}

	CreatePupil<MASK_TYPE>(rows_padded, columns_padded, 
				n_columns_reference, n_rows_reference, 
				padded_real_buffer_1, rows_extra, 0, reference_pupil);

#ifdef PRINT_PUPIL
	DumpMatrix("reference_pupil.dat", rows_padded, columns_padded, padded_real_buffer_1);
#endif

	// Execute the forward FFT on the pupil
	fftwf_execute_dft_r2c(forward_plan, padded_real_buffer_1, padded_complex_buffer_4);

#ifdef PRINT_PUPIL
	DumpComplexMatrix("reference_pupil_fft_real.dat", "reference_pupil_fft_imag.dat", 
			    rows_padded, NUM_COLS_COMPLEX(columns_padded), padded_complex_buffer_4);
#endif 

	return true;

}
		
template <class MASK_TYPE>
bool NormalizedCrossCorrelator::SetCurrentPupil(MASK_TYPE * current_pupil) {

	//Ensure that the sizes have been set
	if (n_rows_reference  == 0 || n_columns_reference == 0 || n_rows_current == 0 || 
		n_columns_current == 0 || rows_padded == 0         || columns_padded == 0) {
		return false;
	}

	// now create the pupil with the given image mask
	CreatePupil<MASK_TYPE>(rows_padded, columns_padded, n_columns_current,   
				n_rows_current, padded_real_buffer_1, rows_extra + n_rows_reference - 1,
				n_columns_reference - 1, current_pupil);
#ifdef PRINT_PUPIL
	DumpMatrix("current_pupil.dat", rows_padded, columns_padded, padded_real_buffer_1);
#endif


	// Execute the forward FFT on the pupil
	fftwf_execute_dft_r2c(forward_plan, padded_real_buffer_1, padded_complex_buffer_1);

#ifdef PRINT_PUPIL
	DumpComplexMatrix("current_pupil_fft_real.dat", "current_pupil_fft_imag.dat", 
			    rows_padded, NUM_COLS_COMPLEX(columns_padded), padded_complex_buffer_1);
#endif 

	return true;
}

template <class FLOATING_TYPE>
bool NormalizedCrossCorrelator::SetCurrent(FLOATING_TYPE * current_buffer) {

	if (current_buffer == NULL) {
		return false;
	}

	//Ensure that the sizes have been set
	if (n_rows_reference  == 0 || n_columns_reference == 0 || n_rows_current == 0 || 
		n_columns_current == 0 || rows_padded == 0         || columns_padded == 0) {
		return false;
	}

	float * temp_output_buffer = NULL;
	bool    success;

	if (typeid(FLOATING_TYPE) != typeid(float)) {

		temp_output_buffer = (float *) malloc(sizeof(float)*
											  n_rows_current*
											  n_columns_current);

		for (unsigned int current_index = 0;
			current_index < n_rows_current*n_columns_current;
			current_index++) {

				temp_output_buffer[current_index] = (float) current_buffer[current_index];
		}
		success = this->SetCurrent(temp_output_buffer);

		free(temp_output_buffer);
	}
	else {
		success = this->SetCurrent(current_buffer);
	}

	return success;

}

template <class REFERENCE_TYPE>
bool NormalizedCrossCorrelator::SetReference(REFERENCE_TYPE * reference_buffer) {

	if (reference_buffer == NULL) {
		return false;
	}

	//Ensure that the sizes have been set
	if (n_rows_reference  == 0 || n_columns_reference == 0 || n_rows_current == 0 || 
		n_columns_current == 0 || rows_padded == 0         || columns_padded == 0) {
		return false;
	}

	float * temp_output_buffer = NULL;
	bool    success;

	if (typeid(REFERENCE_TYPE) != typeid(float)) {

		temp_output_buffer = (float *) malloc(sizeof(float)*
											  n_rows_reference*
											  n_columns_reference);

		for (unsigned int current_index = 0;
			current_index < n_rows_reference*n_columns_reference;
			current_index++) {

				temp_output_buffer[current_index] = (float) reference_buffer[current_index];
		}
		success = this->SetReference(temp_output_buffer);

		free(temp_output_buffer);
	}
	else {
		success = this->SetReference(reference_buffer);
	}

	return success;
}