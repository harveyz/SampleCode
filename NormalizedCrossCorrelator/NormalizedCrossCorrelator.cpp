#include "NormalizedCrossCorrelator.h"

using namespace std;

// Determines the number of threads that the FFTW library will use
#define N_FFTW_THREADS 7

// When these preprocessor statements are defined,
// intermediate buffers will be dumped to file
// for debugging purposes.
//#define PRINT_NORMFACTORS
//#define PRINT_PUPIL
//#define PRINT_NCC_MATRIX

/////////////
// Some helper functions to perform matrix operations
/////////////

// Prints real matrix to file for debugging purposes



// dumps a complext matrix to tow files for debugging purposes.

void DumpComplexMatrix(string real_file_name, string imag_file_name, 
	unsigned long n_rows, unsigned long n_cols, fftwf_complex * matrix_buffer) {

	ofstream real_output_file(real_file_name);
	ofstream imag_output_file(imag_file_name);

	for (unsigned long cur_row = 0; cur_row < n_rows; cur_row++) {
		for (unsigned long cur_col = 0; cur_col <n_cols; cur_col++) {

			if (cur_col > 0) {
				real_output_file << ",\t";
				imag_output_file << ",\t";
			}

			real_output_file << matrix_buffer[cur_row*n_cols+cur_col][0];
			imag_output_file << matrix_buffer[cur_row*n_cols+cur_col][1];
		}

		real_output_file << endl;
		imag_output_file << endl;
	}

	real_output_file.close();
	imag_output_file.close();
}


// Element by element complex conjugate and multiply

void ElemComplexConjugateAndMultiply(fftwf_complex * A, fftwf_complex * B, fftwf_complex * C, 
									 unsigned long n_rows, unsigned long n_columns) {

    for (unsigned long row = 0; row < n_rows; row++) {
		for (unsigned long col = 0; col < n_columns; col++) {

			unsigned long index = row*n_columns + col;

			// Note, the user could be using inplace matrix multiplication
			// i.e. A == C or B == C could be true
			// We need to use a temporary value to ensure we don't
			// trip over our selves. 
			fftwf_complex c_val;

			c_val[0] = A[index][0]*B[index][0] + A[index][1]*B[index][1];
			c_val[1] = A[index][0]*B[index][1] - A[index][1]*B[index][0];

			C[index][0] = c_val[0];
			C[index][1] = c_val[1];

		}
	}

}

// Performs the final step in the normalized cross correlation
template <class T>
void NormCrossCorr(T *       numerator, T *       norm_factor_1, T * norm_factor_2,
				   unsigned long n_rows,    unsigned long n_columns,     float   overlap_norm_threshold) {

	//Indexing variables
    for (unsigned long row = 0; row < n_rows; row++) {
		for (unsigned long col = 0; col < n_columns; col++) {
			unsigned long	    current_index		= row * n_columns + col;

			T			norm_factor_1_val	= norm_factor_1[current_index];
			T			norm_factor_2_val	= norm_factor_2[current_index];
			T			num					= numerator[current_index];

			if (norm_factor_1_val < overlap_norm_threshold || 
				norm_factor_2_val < overlap_norm_threshold) {
				num = 0.0;
			}
			else {
				num = num / sqrtf(norm_factor_1_val * norm_factor_2_val);
			}

			numerator[current_index] =  num;
		}
	}

}

// fills an ROI and clears the rest
template <class T>
void FillROIClearRest(T * buffer, unsigned long n_rows_buffer, unsigned long n_columns_buffer, T * ROI, 
					  unsigned long ROI_rows, unsigned long ROI_columns, unsigned long start_row, unsigned long start_col) {

	//Indexing variables
    for (unsigned long row = 0; row < n_rows_buffer; row++) {
		for (unsigned long col = 0; col < n_columns_buffer; col++) {
		    //Indexing variables
			unsigned long buffer_index  = row*n_columns_buffer + col;
			long roi_index = (row - start_row)*ROI_columns + col - start_col;

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
}

// determines the closest power of two.

unsigned long NearestPowerOfTwo(unsigned long val) {

	if (val == 0) {
		return 1;
	}
	unsigned long start_val = 2;
	for (int i = 0; i < sizeof(unsigned long)*8; i++) {
		if (val <= start_val << i) {
			return start_val << i;
		}
	}

	return 1;
}


//////////////////////////////
// Normalized cross correlator class
//////////////////////////////
bool NormalizedCrossCorrelator::ImportWisdomFile(string fftw_wisdom_file) {

	// Attempt to load the wisdom file.
	return fftwf_import_wisdom_from_filename(fftw_wisdom_file.c_str()) != 0;

}

bool NormalizedCrossCorrelator::ExportWisdomFile(string fftw_wisdom_file) {

	// Attempt to export the wisdom file.
	return fftwf_export_wisdom_to_filename(fftw_wisdom_file.c_str()) != 0;

}

NormalizedCrossCorrelator::NormalizedCrossCorrelator() {
	// does nothing and is declared private so it 
	// should never be used
}

NormalizedCrossCorrelator::~NormalizedCrossCorrelator() {
	ClearBuffers();
}

void NormalizedCrossCorrelator::InitializeData() {
	// Initialize some stuff
	reference_set           = false;
	current_set             = false;
	correlated              = false;
	n_rows_reference        = 0;
	n_columns_reference     = 0;
	n_rows_current          = 0;
	n_columns_current       = 0;

	rows_padded             = 0;
	columns_padded          = 0;

	rows_extra              = 0;
	columns_extra           = 0;

	padded_complex_buffer_1 = NULL;
	padded_complex_buffer_2 = NULL;
	padded_complex_buffer_3 = NULL;
	padded_complex_buffer_4 = NULL;
	padded_complex_buffer_5 = NULL;

	padded_real_buffer_1    = NULL;
	padded_real_buffer_2    = NULL;
	padded_real_buffer_3    = NULL;

	reference_stage         = NULL;
	current_stage           = NULL;

	// FFTW fft plans
	plans_initialized       = false;
}

// allocates buffers
void NormalizedCrossCorrelator::InitializeDimensions(unsigned long n_rows_reference,         unsigned long n_columns_reference, 
								   unsigned long n_rows_current,          unsigned long n_columns_current,
								   unsigned int padding) {

	// clear anything that is still in memory
	this->ClearBuffers();

	//assign the matrix sizes
	this->n_rows_current		= n_rows_current;
	this->n_columns_current		= n_columns_current;
	this->n_rows_reference		= n_rows_reference;
	this->n_columns_reference	= n_columns_reference;

	//The padded matrix size (we will expand if they are not a power of two
	rows_padded					= n_rows_current    + n_rows_reference    - 1;
	columns_padded				= n_columns_current + n_columns_reference - 1;

	// TODO: Figure out what other padding schemes make sense
	//       and implement them
	switch (padding) {

	case NONE:
		rows_extra    = 0;
		columns_extra = 0;
		break;

	case POWER_OF_2:
		//Determine the nearest power of two for the given sizes
		rows_extra					= NearestPowerOfTwo(rows_padded)    - rows_padded;
		columns_extra				= NearestPowerOfTwo(columns_padded) - columns_padded; 
		break;

	default:
		// TODO add extra cases here to handle different padding methods
		rows_extra    = 0;
		columns_extra = 0;
		break;

	}

	//Assign the power of two to the rows_padded and columns_padded
	rows_padded    += rows_extra;
	columns_padded += columns_extra;

	//indicators to determine if the correct buffers have been set
	reference_set           = false;
	current_set             = false;
	correlated              = false;

	// allocate real memory buffers
	// note we use fftwf_malloc to ensure proper alignment with so that FFTW can use SIMD instructions
	padded_real_buffer_1 = (float *) fftwf_malloc(sizeof(float)*rows_padded*columns_padded);
	padded_real_buffer_2 = (float *) fftwf_malloc(sizeof(float)*rows_padded*columns_padded);
	padded_real_buffer_3 = (float *) fftwf_malloc(sizeof(float)*rows_padded*columns_padded);

	current_stage        = (float *) fftwf_malloc(sizeof(float)*n_rows_current*n_columns_current);
	reference_stage      = (float *) fftwf_malloc(sizeof(float)*n_rows_reference*n_columns_reference);

	// allocate complex buffers using FFTW malloc function
	padded_complex_buffer_1 = (fftwf_complex*) fftwf_malloc(sizeof(fftwf_complex)*NUM_ELEMENTS_COMPLEX(rows_padded, columns_padded));
	padded_complex_buffer_2 = (fftwf_complex*) fftwf_malloc(sizeof(fftwf_complex)*NUM_ELEMENTS_COMPLEX(rows_padded, columns_padded));
	padded_complex_buffer_3 = (fftwf_complex*) fftwf_malloc(sizeof(fftwf_complex)*NUM_ELEMENTS_COMPLEX(rows_padded, columns_padded));
	padded_complex_buffer_4 = (fftwf_complex*) fftwf_malloc(sizeof(fftwf_complex)*NUM_ELEMENTS_COMPLEX(rows_padded, columns_padded));
	padded_complex_buffer_5 = (fftwf_complex*) fftwf_malloc(sizeof(fftwf_complex)*NUM_ELEMENTS_COMPLEX(rows_padded, columns_padded));

	// These pointers are used to determine which buffers to compute
	// the NCC on. 
	active_reference_norm_factor = padded_real_buffer_2;
	active_current_norm_factor   = padded_real_buffer_3;

	active_reference_fft         = padded_complex_buffer_3;
	active_current_fft           = padded_complex_buffer_5;

	// Telling FFTW library that we want to use 4 threads for ffts
	fftwf_plan_with_nthreads(N_FFTW_THREADS);

	// Create the FFTW plans here
	// note, later we reasign what arrays we use when we execute the plan
	// so the arrays passed here are just to satisfy the API and set up the resources
	forward_plan = fftwf_plan_dft_r2c_2d(rows_padded, columns_padded,
                                padded_real_buffer_1, padded_complex_buffer_1,
								FFTW_ESTIMATE);
                                //FFTW_PATIENT);

	// Telling FFTW library that we want to use 4 threads for ffts
	// TODO do I need to call this before every plan creation?
	fftwf_plan_with_nthreads(N_FFTW_THREADS);

	reverse_plan = fftwf_plan_dft_c2r_2d(rows_padded, columns_padded,
                                padded_complex_buffer_2, padded_real_buffer_2,
								FFTW_ESTIMATE);
                                //FFTW_PATIENT);

	plans_initialized = true;
}

// clears memory
void NormalizedCrossCorrelator::ClearBuffers() {

	// allocate imaginary buffers
	if (padded_complex_buffer_1 != NULL) {
		fftwf_free(padded_complex_buffer_1);
		padded_complex_buffer_1 = NULL;
	}
	if (padded_complex_buffer_2 != NULL) {
		fftwf_free(padded_complex_buffer_2);
		padded_complex_buffer_2 = NULL;
	}
	if (padded_complex_buffer_3 != NULL) {
		fftwf_free(padded_complex_buffer_3);
		padded_complex_buffer_3 = NULL;
	}
	if (padded_complex_buffer_4 != NULL) {
		fftwf_free(padded_complex_buffer_4);
		padded_complex_buffer_4 = NULL;
	}
	if (padded_complex_buffer_5 != NULL) {
		fftwf_free(padded_complex_buffer_5);
		padded_complex_buffer_5 = NULL;
	}

	// allocate real buffers
	if (padded_real_buffer_1 != NULL) {
		fftwf_free(padded_real_buffer_1);
		padded_real_buffer_1 = NULL;
	}
	if (padded_real_buffer_2 != NULL) {
		fftwf_free(padded_real_buffer_2);
		padded_real_buffer_2 = NULL;
	}
	if (padded_real_buffer_3 != NULL) {
		fftwf_free(padded_real_buffer_3);
		padded_real_buffer_3 = NULL;
	}
	if (current_stage != NULL) {
		fftwf_free(current_stage);
		current_stage = NULL;
	}
	if (reference_stage != NULL) {
		fftwf_free(reference_stage);
		reference_stage = NULL;
	}

	// release resouces associated with the FFTW plans
	if (plans_initialized) {
		fftwf_destroy_plan(forward_plan);
		fftwf_destroy_plan(reverse_plan);
	}

	// clear the caches
	for (int i = 0; i < reference_norm_factors_cache.size(); i++) {
		fftwf_free(reference_norm_factors_cache[i]);
	}
	for (int i = 0; i < reference_fft_cache.size(); i++) {
		fftwf_free(reference_fft_cache[i]);
	}
	for (int i = 0; i < current_norm_factors_cache.size(); i++) {
		fftwf_free(current_norm_factors_cache[i]);
	}
	for (int i = 0; i < current_fft_cache.size(); i++) {
		fftwf_free(current_fft_cache[i]);
	}

	reference_norm_factors_cache.clear();
	reference_fft_cache.clear();
	current_norm_factors_cache.clear();
	current_fft_cache.clear();

	plans_initialized = false;

}

bool NormalizedCrossCorrelator::CalculateNormFactorOne() {

	correlated = false;

	FillROIClearRest<float>(padded_real_buffer_1, rows_padded,      columns_padded, 
							reference_stage,      n_rows_reference, n_columns_reference, 
							rows_extra, 0);

	// Compute the DFT of the reference frame
	fftwf_execute_dft_r2c(forward_plan, padded_real_buffer_1, padded_complex_buffer_3);

	// The DFT has been computed so square the real_buffer_1 buffer
	SquareROI<float>(padded_real_buffer_1, n_rows_reference, n_columns_reference, 
						 columns_padded,   rows_extra,       0);

    // Compute the DFT of the reference squared
	fftwf_execute_dft_r2c(forward_plan, padded_real_buffer_1, padded_complex_buffer_2);

	ElemComplexConjugateAndMultiply(padded_complex_buffer_2, padded_complex_buffer_1,
									padded_complex_buffer_2, rows_padded, 
									NUM_COLS_COMPLEX(columns_padded));

    // Compute the DFT of the reference squared
	fftwf_execute_dft_c2r(reverse_plan, padded_complex_buffer_2, padded_real_buffer_2);

#ifdef PRINT_NORMFACTORS
	DumpMatrix("reference_norm_factor.dat", rows_padded, columns_padded, padded_real_buffer_2);
#endif

	// Reset the active buffers so the pointer are pointing to the newly calculated
	// buffers
	active_reference_fft         = padded_complex_buffer_3;
	active_reference_norm_factor = padded_real_buffer_2;

	reference_set = true;
	return true;

}


bool NormalizedCrossCorrelator::CalculateNormFactorTwo() {

	correlated = false;

	FillROIClearRest<float>(padded_real_buffer_1,            rows_padded,    columns_padded, 
							current_stage,                   n_rows_current, n_columns_current, 
							n_rows_reference + rows_extra-1, n_columns_reference-1);

	//Compute the DFT of the reference frame
	fftwf_execute_dft_r2c(forward_plan, padded_real_buffer_1, padded_complex_buffer_5);



	//The DFT has been computed so square the real_buffer_1 buffer
	SquareROI<float>(padded_real_buffer_1, n_rows_current, n_columns_current, 
						 columns_padded, n_rows_reference+rows_extra-1, n_columns_reference-1);

    // Compute the DFT of the reference squared
	fftwf_execute_dft_r2c(forward_plan, padded_real_buffer_1, padded_complex_buffer_2);

	/////////////////////////////////////////
	//Calculate the second normalization factor
	/////////////////////////////////////////

	ElemComplexConjugateAndMultiply(padded_complex_buffer_4, padded_complex_buffer_2, 
		padded_complex_buffer_2, rows_padded, NUM_COLS_COMPLEX(columns_padded));

    // Compute the reverse DFT
	fftwf_execute_dft_c2r(reverse_plan, padded_complex_buffer_2, padded_real_buffer_3);

#ifdef PRINT_NORMFACTORS
	DumpMatrix("current_norm_factor.dat", rows_padded, columns_padded, padded_real_buffer_3);
#endif

	// Reset the active buffers so the pointer are pointing to the newly calculated
	// buffers
	active_current_fft         = padded_complex_buffer_5;
	active_current_norm_factor = padded_real_buffer_3;

	current_set = true;
	return true;
}

// Constructor

NormalizedCrossCorrelator::NormalizedCrossCorrelator(float overlap_norm_threshold) {
	// Tell FFTW that we are going to want multithreading
	fftwf_init_threads();
	this->overlap_norm_threshold = overlap_norm_threshold;
	InitializeData();
}


bool NormalizedCrossCorrelator::SetFrameSizes(unsigned long n_rows_reference, unsigned long n_columns_reference, 
											  unsigned long n_rows_current,   unsigned long n_columns_current, 
											  PaddingType padding) {

	InitializeDimensions(n_rows_reference, n_columns_reference, 
						 n_rows_current, n_columns_current, padding);


	// create default image masks
	int * reference_image_mask = (int *) malloc(sizeof(int)*n_rows_reference*n_columns_reference);
	int * current_image_mask   = (int *) malloc(sizeof(int)*n_rows_current*n_columns_current);

	int * reference_image_mask_end = reference_image_mask + n_rows_reference*n_columns_reference;
	int * current_image_mask_end   = current_image_mask   + n_rows_current*n_columns_current;

	fill<int *, int>(reference_image_mask, reference_image_mask_end, 1);
	fill<int *, int>(current_image_mask,   current_image_mask_end, 1);

	CreatePupils<int>(reference_image_mask, current_image_mask);

	return true;
}

bool NormalizedCrossCorrelator::SetReference(float * reference_buffer) {

	if (reference_buffer == NULL) {
		return false;
	}

	memcpy(reference_stage, 
		reference_buffer, 
		sizeof(float)*n_rows_reference*n_columns_reference);

	return CalculateNormFactorOne();

}

bool NormalizedCrossCorrelator::SetReference(float * reference_buffer, int * reference_mask) {

	bool success = this->SetReferencePupil<int>(reference_mask);

	if (success) {
		return SetReference(reference_buffer);
	} else {
		return success;
	}

}

bool NormalizedCrossCorrelator::SetReferenceROI(float * reference_buffer, unsigned long stride_x,
					unsigned long top_row,     unsigned long left_column, 
					unsigned long bottom_row,  unsigned long right_column) {

	//Determine the dimensions of the ROI to ensure that it is not larger than the reference frame
	unsigned long ROI_columns = right_column  - left_column + 1;
	unsigned long ROI_rows    = bottom_row    - top_row     + 1;

	if (ROI_columns !=n_columns_reference ||
		ROI_rows    != n_rows_reference) {
		return false;
	}

	SetROIFromBuffer(reference_buffer, stride_x, right_column,    left_column,  
   				     bottom_row,       top_row,  reference_stage, n_columns_reference); 

	return CalculateNormFactorOne();

}

bool NormalizedCrossCorrelator::SetReferenceROI(float * reference_buffer, int * reference_mask,
												unsigned long stride_x,
												unsigned long top_row,     unsigned long left_column, 
												unsigned long bottom_row,  unsigned long right_column) {

	//Determine the dimensions of the ROI to ensure that it is not larger than the reference frame
	unsigned long ROI_columns = right_column  - left_column + 1;
	unsigned long ROI_rows    = bottom_row    - top_row     + 1;

	if (ROI_columns != n_columns_reference ||
		ROI_rows    != n_rows_reference) {
		return false;
	}

	// Copy the ROI of the mask to a temporary buffer
	int * temp_buffer = (int *) malloc(sizeof(int)*n_columns_reference*n_rows_reference);
	SetROIFromBuffer(reference_mask, stride_x, right_column, left_column, 
					 bottom_row, top_row, temp_buffer, n_columns_reference);


	CreatePupil<int>(rows_padded, columns_padded, 
				       n_columns_reference, n_rows_reference, 
				       padded_real_buffer_1, rows_extra, 0, temp_buffer);

	SetROIFromBuffer(reference_buffer, stride_x, right_column,    left_column,  
   				     bottom_row,       top_row,  reference_stage, n_columns_reference); 

	free(temp_buffer);

	return CalculateNormFactorOne();

}


//
// Caches reference frame
//

int NormalizedCrossCorrelator::CacheReferenceFrame() {

	// verify that there is reference frame to cache
	if (!reference_set) {
		return -1;
	}

	// allocate memory for the buffers
	fftwf_complex * fft_cache  = (fftwf_complex*) fftwf_malloc(sizeof(fftwf_complex)*NUM_ELEMENTS_COMPLEX(rows_padded, columns_padded));
	float *         norm_cache = (float *) fftwf_malloc(sizeof(float)*rows_padded*columns_padded);

	//copy the active buffers to the cache
	memcpy(fft_cache,  padded_complex_buffer_3, sizeof(fftwf_complex)*NUM_ELEMENTS_COMPLEX(rows_padded, columns_padded));
	memcpy(norm_cache, padded_real_buffer_2, sizeof(float)*rows_padded*columns_padded);

	// push the buffers onto the cache vectors
	reference_norm_factors_cache.push_back(norm_cache);
	reference_fft_cache.push_back(fft_cache);

	return reference_norm_factors_cache.size() - 1;
}

//
// Recalls cache
//
bool NormalizedCrossCorrelator::SetReference(int handle) {

	if (handle >= reference_norm_factors_cache.size()) {
		return false;
	}

	// reassign the active pointers
	active_reference_norm_factor = reference_norm_factors_cache[handle];
	active_reference_fft         = reference_fft_cache[handle];

	reference_set = true;
	correlated    = false;

	return true;

}

//
// Caches current frame
//

int NormalizedCrossCorrelator::CacheCurrentFrame() {

	// verify that there is reference frame to cache
	if (!current_set) {
		return -1;
	}

	// allocate memory for the buffers
	fftwf_complex * fft_cache  = (fftwf_complex*) fftwf_malloc(sizeof(fftwf_complex)*NUM_ELEMENTS_COMPLEX(rows_padded, columns_padded));
	float *         norm_cache = (float *) fftwf_malloc(sizeof(float)*rows_padded*columns_padded);

	//copy the active buffers to the cache
	memcpy(fft_cache,  padded_complex_buffer_5, sizeof(fftwf_complex)*NUM_ELEMENTS_COMPLEX(rows_padded, columns_padded));
	memcpy(norm_cache, padded_real_buffer_3, sizeof(float)*rows_padded*columns_padded);

	// push the buffers onto the cache vectors
	current_norm_factors_cache.push_back(norm_cache);
	current_fft_cache.push_back(fft_cache);

	return current_norm_factors_cache.size() - 1;
}

//
// Recalls cache
//
bool NormalizedCrossCorrelator::SetCurrent(int handle) {

	if (handle >= current_norm_factors_cache.size()) {
		return false;
	}

	// reassign the active pointers
	active_current_norm_factor = current_norm_factors_cache[handle];
	active_current_fft         = current_fft_cache[handle];

	current_set = true;
	correlated    = false;


	return true;

}

bool NormalizedCrossCorrelator::SetCurrent(float * current_buffer) {

	if (current_buffer == NULL) {
		return false;
	}

	//Ensure that the sizes have been set
	if (n_rows_reference  == 0 || n_columns_reference == 0 || n_rows_current == 0 || 
		n_columns_current == 0 || rows_padded == 0         || columns_padded == 0) {
		return false;
	}

	memcpy(current_stage, 
		current_buffer, 
		sizeof(float)*n_rows_current*n_columns_current);

	return CalculateNormFactorTwo();

}

bool NormalizedCrossCorrelator::SetCurrent(float * current_buffer, int * current_mask) {

	bool success = SetCurrentPupil<int>(current_mask);

	if (success) {
		return SetCurrent(current_buffer);
	} else {
		return false;
	}
}

bool NormalizedCrossCorrelator::SetCurrentROI(float * current_buffer,   unsigned long stride_x,
						                      unsigned long top_row,    unsigned long left_column, 
						                      unsigned long bottom_row, unsigned long right_column) {

	//Determine the dimensions of the ROI to ensure that it is not larger than the reference frame
	unsigned long ROI_columns = right_column  - left_column + 1;
	unsigned long ROI_rows    = bottom_row    - top_row     + 1;
	

	if (ROI_columns != n_columns_current ||
		 ROI_rows   != n_rows_current) {
		return false;
	}

	SetROIFromBuffer(current_buffer, stride_x, right_column,    left_column,  
   				     bottom_row,       top_row,  current_stage, n_columns_current); 

	return CalculateNormFactorTwo();
}


bool NormalizedCrossCorrelator::SetCurrentROI(float * current_buffer,   int * current_mask,
											  unsigned long stride_x,
						                      unsigned long top_row,    unsigned long left_column, 
						                      unsigned long bottom_row, unsigned long right_column) {

	//Determine the dimensions of the ROI to ensure that it is not larger than the reference frame
	unsigned long ROI_columns = right_column  - left_column + 1;
	unsigned long ROI_rows    = bottom_row    - top_row     + 1;
	
	if (ROI_columns != n_columns_current ||
		ROI_rows    != n_rows_current) {
		return false;
	}

	// Copy the ROI of the mask to a temporary buffer
	int * temp_buffer = (int *) malloc(sizeof(int)*n_columns_current*n_rows_current);
	SetROIFromBuffer(current_mask, stride_x, right_column, left_column, 
					 bottom_row, top_row, temp_buffer, n_columns_current);

	CreatePupil<int>(rows_padded, columns_padded, n_columns_current,   
				n_rows_current, padded_real_buffer_1, rows_extra + n_rows_reference - 1,
				n_columns_reference - 1, current_mask);

	SetROIFromBuffer(current_buffer, stride_x, right_column,    left_column,  
   				     bottom_row,       top_row,  current_stage, n_columns_current); 

	free(temp_buffer);

	return CalculateNormFactorTwo();
}


bool NormalizedCrossCorrelator::NormalizedCrossCorrelate() {

	if (correlated) {
		return true;
	}

	//Ensure that the proper parameters have been set
	if (!current_set || !reference_set) {
		return false;
	}

	//////////////////////////////////////////////
	//Calculate the numerator of the cross correlation
	//////////////////////////////////////////////
	ElemComplexConjugateAndMultiply(active_reference_fft, active_current_fft, 
									padded_complex_buffer_2, rows_padded, NUM_COLS_COMPLEX(columns_padded));

    // Compute the reverse DFT
	fftwf_execute_dft_c2r(reverse_plan, padded_complex_buffer_2, padded_real_buffer_1);

#ifdef PRINT_NCC_MATRIX
	DumpMatrix("numerator.dat", rows_padded, columns_padded, padded_real_buffer_1);
#endif

	//////////////////////////////////////
	//Calculate normalized cross correlation
	//////////////////////////////////////
	float overlap_norm_threshold_scaled = overlap_norm_threshold * columns_padded * rows_padded;

	NormCrossCorr<float>(padded_real_buffer_1, active_reference_norm_factor, active_current_norm_factor,
				         rows_padded,          columns_padded,       overlap_norm_threshold_scaled);

	correlated = true;
	return true;
}

//
// Find the peak in the cross correlation matrix
//
// (just a wrapper for the ROI version)

bool NormalizedCrossCorrelator::GetCrossCorrelationMax(float &val, unsigned long &row_max,   unsigned long &column_max) {

	return GetCrossCorrelationMaxROI(val, row_max, column_max, 0, 0, rows_padded-1, columns_padded - 1);

}

//
// finds the peak of the cross correaltion matrix over a ROI
//
bool NormalizedCrossCorrelator::GetCrossCorrelationMaxROI(float &val, unsigned long &row_max,   unsigned long &column_max, 
														  unsigned long top_row,    unsigned long left_column,
														  unsigned long bottom_row, unsigned long right_column) {
	// Note, the NCC matrix is zero padded to prevent periodic artifacts caused by circular convolution, 
	// there could also be extra padding to improve the performance of the FFT. We only look over the top
	// left hand corner
	// dimensions of 
	unsigned long ROI_columns = right_column  - left_column + 1;
	unsigned long ROI_rows    = bottom_row    - top_row     + 1;
	
	unsigned long n_rows_NCC_actual = n_rows_reference + n_rows_current - 1;
	unsigned long n_columns_NCC_actual = n_columns_reference + n_columns_current - 1;

	// make sure the ROI is over the actual data in the NCC matrix
	if (left_column + ROI_columns > n_columns_NCC_actual ||
		top_row     + ROI_rows    > n_rows_NCC_actual) {
		return false;
	}

	if (!correlated) {
		return false;
	}

	float * row_pointer =(float *) (((unsigned long long) padded_real_buffer_1) + top_row*columns_padded*sizeof(float) + left_column*sizeof(float));

	for (unsigned long cur_row = 0; cur_row < ROI_rows; cur_row++) {

		if (cur_row == 0){
			val        = row_pointer[0];
			row_max    = top_row;
			column_max = left_column;
		}

		for (unsigned long cur_col = 0; cur_col < ROI_columns; cur_col++) {

			if (row_pointer[cur_col] > val) {
				val        = row_pointer[cur_col];
				row_max    = top_row + cur_row;
				column_max = left_column + cur_col;
			}

		}

		row_pointer = (float *) (((unsigned long long) row_pointer) + columns_padded*sizeof(float));

	}
	return true;
}

bool NormalizedCrossCorrelator::GetCrossCorrelationMax(float & val, 
													  unsigned long &row_max, 
													  unsigned long &column_max, 
													  unsigned long n_overlapping_pixels) {
	

	fftwf_complex * temp_buffer = (fftwf_complex *) malloc(sizeof(fftwf_complex)*rows_padded*
															NUM_COLS_COMPLEX(columns_padded));
	ElemComplexConjugateAndMultiply(padded_complex_buffer_4, padded_complex_buffer_2, 
		temp_buffer, rows_padded, NUM_COLS_COMPLEX(columns_padded));

	

}

//
// Copies the entier NCC matrix to the provided buffer.
//
bool NormalizedCrossCorrelator::GetCorrelationMatrix(float * output_host_buffer) {

	// verify that the output pointer is not null
	// and that there is data to return
	if (output_host_buffer == NULL ||
		!correlated) {
		return false;
	}

	// Note, the NCC matrix is zero padded to prevent periodic artifacts caused by circular convolution, 
	// there could also be extra padding to improve the performance of the FFT. Therefore the stride on
	// the ncc matrix is different from the stride of the output matrix
	unsigned long n_rows_ncc = n_rows_reference + n_rows_current - 1;
	unsigned long n_cols_ncc = n_columns_reference + n_columns_current - 1;

	unsigned long long current_ncc_pointer = (unsigned long long) padded_real_buffer_1;

	for (unsigned long current_row = 0; current_row < n_rows_ncc; current_row++) {
		memcpy(output_host_buffer, (void *) current_ncc_pointer, n_cols_ncc*sizeof(float));
		current_ncc_pointer += columns_padded*sizeof(float);
		output_host_buffer  += n_cols_ncc;
	}

	return true;
}

unsigned long NormalizedCrossCorrelator::GetNumberOfColumnsReference() {
	return n_columns_reference;
} 

unsigned long NormalizedCrossCorrelator::GetNumberOfColumnsCurrent() {
	return n_columns_current;
}

unsigned long NormalizedCrossCorrelator::GetNumberOfRowsReference() {
	return n_rows_reference;
}

unsigned long NormalizedCrossCorrelator::GetNumberOfRowsCurrent() {
	return n_rows_current;
}
