%module NormalizedCrossCorrelator

%{
#include <Python.h>
#include <numpy/arrayobject.h>
#include <NormalizedCrossCorrelator/Cpp/NormalizedCrossCorrelator.h>
#include<vector>
%}

%include "std_vector.i"
%include "std_string.i"

%init %{
import_array();
%}

%typemap(in) std::string {
    if (PyString_Check($input)) {
         $1 = std::string(PyString_AsString($input));
     } else {
         PyErr_SetString(PyExc_TypeError, "string expected");
         return NULL;
     }
}

%feature("autodoc", "0");


namespace std {
   %template(IntVector) vector<int>;
}



/// Used to tell the class what type of padding to use
enum PaddingType {NONE          = 0, 
			      POWER_OF_2    = 1, 
				  MULTIPLE_OF_2 = 2,
				  MULTIPLE_OF_8 = 3};



%feature("docstring")
"""
This class provides a Python interface to class that calculates normalized
cross correlations using the CPI

Zach Harvey (zgh7555@gmail.com)
Medical College of Wisconsin
September 2012

"""
class NormalizedCrossCorrelator {


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
		//bool SetFrameSizes(unsigned long n_rows_reference, unsigned long n_columns_reference,
		//				   unsigned long n_rows_current,   unsigned long n_columns_current,
		//				   PaddingType padding);



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

		/// Specify the current frame of the correlation.
		///
		/// \param current_bufferr
		///        pointer to buffer containing the current frame. It
		///        is assumed row major contiguous data with dimensions
		///        as set by the SetFrameSizes methods
		bool SetCurrent(float * current_bufferr);

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


};
%extend NormalizedCrossCorrelator{

	
	%feature("docstring")
	"""
	This provides a way for the user to set the referecne frame from a numpy array. The internal
	data type of the numpy array must be 'single', 'double', or 'uint8'. Pass one of these strings
	to the method to indicate what type it is. This method will return a boolean indicating the 
	success or failure
	"""
		
	PyObject * SetReferenceFromNumpyArray(PyObject * reference) {
	
		// making sure the passed object is a numpy array	
		if (PyArray_Check(reference)) {
		
			bool			success   = false;
			
			PyArrayObject * reference_array = (PyArrayObject *) reference;
			
			int reference_type_num    = reference_array->descr->type_num;
			
			if (!PyArray_ISCARRAY(reference_array)) {
				return PyBool_FromLong(0);
			}
			
			if(reference_array->nd != 2) {
				return PyBool_FromLong(0);
			}
			
			unsigned long n_rows    = self->GetNumberOfRowsReference();
			unsigned long n_columns = self->GetNumberOfColumnsReference();

			if (reference_array->dimensions[0] != n_rows ||
				reference_array->dimensions[1] != n_columns) {
				return PyBool_FromLong(0);
			}
			
			if ((reference_type_num == NPY_FLOAT32) || (reference_type_num == NPY_FLOAT)) {
				success = self->SetReference((float *) reference_array->data);
			}
			else if (reference_type_num == NPY_FLOAT64) {
				success = self->SetReference<double>((double *) reference_array->data);
			}
			else if (reference_type_num == NPY_UINT8) {
				success = self->SetReference<unsigned char>((unsigned char *) reference_array->data);
			}
			else if (reference_type_num == NPY_UINT32) {
				success = self->SetReference<unsigned int>((unsigned int *) reference_array->data);
			}
			else if (reference_type_num == NPY_INT32) {
				success = self->SetReference<int>((int *) reference_array->data);
			}

			else {
				return PyBool_FromLong(0);
			}

			return PyBool_FromLong(success);
		
		}

		return (PyBool_FromLong(0));
	}
	
		
	%feature("docstring")
	"""
	This provides a way for the user to set the referecne frame from a numpy array as well as an 
	image mask for the reference frame. The internal
	data type of the numpy array must be 'single', 'double', or 'uint8'. 
	This method will return a boolean indicating the 
	success or failure
	"""
		
	PyObject * SetReferenceFromNumpyArray(PyObject * reference, PyObject * reference_mask) {
	
		// making sure the passed object is a numpy array	
		if (PyArray_Check(reference) &&
			PyArray_Check(reference_mask)) {
		
			bool			success   = false;
			
			PyArrayObject * reference_array      = (PyArrayObject *) reference;
			PyArrayObject * reference_mask_array = (PyArrayObject *) reference_mask;
			int reference_type_num               = reference_array->descr->type_num;
			int reference_mask_type_num          = reference_mask_array->descr->type_num;
			
			if (!PyArray_ISCARRAY(reference_array)) {
				return PyBool_FromLong(0);
			}

			if (!PyArray_ISCARRAY(reference_mask_array)) {
				return PyBool_FromLong(0);
			}
			
			if(reference_array->nd != 2) {
				return PyBool_FromLong(0);
			}
			
			if(reference_mask_array->nd != 2) {
				return PyBool_FromLong(0);
			}

			unsigned long n_rows    = self->GetNumberOfRowsReference();
			unsigned long n_columns = self->GetNumberOfColumnsReference();

			if (reference_array->dimensions[0] != n_rows ||
				reference_array->dimensions[1] != n_columns) {
				return PyBool_FromLong(0);
			}
			
			if (reference_mask_array->dimensions[0] != n_rows ||
				reference_mask_array->dimensions[1] != n_columns) {
				return PyBool_FromLong(0);
			}

			if ((reference_type_num == NPY_FLOAT32) || (reference_type_num == NPY_FLOAT) &&
			    reference_mask_type_num == NPY_INT32) {
				success = self->SetReference((float *) reference_array->data,
											 (int *)   reference_mask_array->data);
			}
			else {
				return PyBool_FromLong(0);
			}

			return PyBool_FromLong(success);
		
		}

		return (PyBool_FromLong(0));
	}

	%feature("docstring")
	"""
	This provides a way for the user to set the referecne frame from a numpy array as well as an 
	image mask for the reference frame. The internal
	data type of the numpy array must be 'single', 'double', or 'uint8'. 
	This method will return a boolean indicating the 
	success or failure
	"""
		
	PyObject * SetReferenceFromNumpyArrayROI(PyObject * reference, PyObject * reference_mask,
						                     unsigned long top_row,    unsigned long left_column, 
						                     unsigned long bottom_row, unsigned long right_column) {
	
		// making sure the passed object is a numpy array	
		if (PyArray_Check(reference) &&
			PyArray_Check(reference_mask)) {
		
			bool			success   = false;
			
			PyArrayObject * reference_array      = (PyArrayObject *) reference;
			PyArrayObject * reference_mask_array = (PyArrayObject *) reference_mask;
			int reference_type_num               = reference_array->descr->type_num;
			int reference_mask_type_num          = reference_mask_array->descr->type_num;
			
			if (!PyArray_ISCARRAY(reference_array)) {
				return PyBool_FromLong(0);
			}

			if (!PyArray_ISCARRAY(reference_mask_array)) {
				return PyBool_FromLong(0);
			}
			
			if(reference_array->nd != 2) {
				return PyBool_FromLong(0);
			}
			
			if(reference_mask_array->nd != 2) {
				return PyBool_FromLong(0);
			}

			if ((reference_type_num == NPY_FLOAT32) || (reference_type_num == NPY_FLOAT) &&
			    reference_mask_type_num == NPY_INT32) {
				success = self->SetReferenceROI((float *) reference_array->data,
											 (int *)   reference_mask_array->data,
											 reference_mask_array->dimensions[1],
											 top_row,    left_column, 
						                     bottom_row, right_column);
			}
			else {
				return PyBool_FromLong(0);
			}

			return PyBool_FromLong(success);
		
		}

		return (PyBool_FromLong(0));
	}

	%feature("docstring")
	"""
	This provides a way for the user to set the referecne frame from a numpy array as well as an 
	image mask for the reference frame. The internal
	data type of the numpy array must be 'single', 'double', or 'uint8'. 
	This method will return a boolean indicating the 
	success or failure
	"""
		
	PyObject * SetReferenceFromNumpyArrayROI(PyObject * reference,
						                     unsigned long top_row,    unsigned long left_column, 
						                     unsigned long bottom_row, unsigned long right_column) {
	
		// making sure the passed object is a numpy array	
		if (PyArray_Check(reference)) {
		
			bool			success   = false;
			
			PyArrayObject * reference_array      = (PyArrayObject *) reference;
			int reference_type_num               = reference_array->descr->type_num;
			
			if (!PyArray_ISCARRAY(reference_array)) {
				return PyBool_FromLong(0);
			}
			
			if(reference_array->nd != 2) {
				return PyBool_FromLong(0);
			}

			if ((reference_type_num == NPY_FLOAT32) || (reference_type_num == NPY_FLOAT)) {
				success = self->SetReferenceROI((float *) reference_array->data,
											 reference_array->dimensions[1],
											 top_row,    left_column, 
						                     bottom_row, right_column);
			}
			else {
				return PyBool_FromLong(0);
			}

			return PyBool_FromLong(success);
		
		}

		return (PyBool_FromLong(0));
	}

	%feature("docstring")
	"""
	This provides a way for the user to set the current frame from a numpy array. The internal
	data type of the numpy array must be 'single', 'double', or 'uint8'. Pass one of these strings
	to the method to indicate what type it is. This method will return a boolean indicating the 
	success or failure
	"""
	
	PyObject * SetCurrentFromNumpyArray(PyObject * current) {
	
		// making sure the passed object is a numpy array	
		if (PyArray_Check(current)) {
		
			bool			success		  = false;
			
			PyArrayObject * current_array = (PyArrayObject *) current;
			
			int current_type_num          = current_array->descr->type_num;

			if (!PyArray_ISCARRAY(current_array)) {
				return PyBool_FromLong(0);
			}
			
			if(current_array->nd != 2) {
				return PyBool_FromLong(0);
			}
			
			unsigned long n_rows    = self->GetNumberOfRowsCurrent();
			unsigned long n_columns = self->GetNumberOfColumnsCurrent();

			if (current_array->dimensions[0] != n_rows ||
				current_array->dimensions[1] != n_columns) {
				return PyBool_FromLong(0);
			}
			
			if ((current_type_num == NPY_FLOAT32) || (current_type_num == NPY_FLOAT)) {
				success = self->SetCurrent((float *) current_array->data);
			}
			else if (current_type_num == NPY_FLOAT64) {
				success = self->SetCurrent<double>((double *) current_array->data);
			}
			else if (current_type_num == NPY_UINT8) {
				success = self->SetCurrent<unsigned char>((unsigned char *) current_array->data);
			}
			else if (current_type_num == NPY_UINT32) {
				success = self->SetCurrent<unsigned int>((unsigned int *) current_array->data);
			}
			else if (current_type_num == NPY_INT32) {
				success = self->SetCurrent<int>((int *) current_array->data);
			}

			else {
				return PyBool_FromLong(0);
			}

			return PyBool_FromLong(success);
		
		}

		return (PyBool_FromLong(0));
	}
	
	%feature("docstring")
	"""
	This provides a way for the user to set the current frame from a numpy array. The internal
	data type of the numpy array must be 'single', 'double', or 'uint8'. Pass one of these strings
	to the method to indicate what type it is. This method will return a boolean indicating the 
	success or failure
	"""
	
	PyObject * SetCurrentFromNumpyArray(PyObject * current, PyObject * current_mask) {
	
		// making sure the passed object is a numpy array	
		if (PyArray_Check(current) &&
			PyArray_Check(current_mask)) {
		
			bool			success		  = false;
			
			PyArrayObject * current_array      = (PyArrayObject *) current;
			PyArrayObject * current_mask_array = (PyArrayObject *) current_mask;
			
			int current_type_num          = current_array->descr->type_num;
			int current_mask_type_num     = current_mask_array->descr->type_num;

			if (!PyArray_ISCARRAY(current_array)) {
				return PyBool_FromLong(0);
			}
			if (!PyArray_ISCARRAY(current_mask_array)) {
				return PyBool_FromLong(0);
			}
			
			if(current_array->nd != 2) {
				return PyBool_FromLong(0);
			}
			
			if(current_mask_array->nd != 2) {
				return PyBool_FromLong(0);
			}

			unsigned long n_rows    = self->GetNumberOfRowsCurrent();
			unsigned long n_columns = self->GetNumberOfColumnsCurrent();

			if (current_array->dimensions[0] != n_rows ||
				current_array->dimensions[1] != n_columns) {
				return PyBool_FromLong(0);
			}
			
			if (current_mask_array->dimensions[0] != n_rows ||
				current_mask_array->dimensions[1] != n_columns) {
				return PyBool_FromLong(0);
			}

			if ((current_type_num == NPY_FLOAT32) || (current_type_num == NPY_FLOAT) &&
			     current_mask_type_num == NPY_INT32) {
				success = self->SetCurrent((float *) current_array->data,
										   (int *) current_mask_array->data);
			}
			else {
				return PyBool_FromLong(0);
			}

			return PyBool_FromLong(success);
		
		}

		return (PyBool_FromLong(0));
	}

	%feature("docstring")
	"""
	This provides a way for the user to set the current frame from a numpy array. The internal
	data type of the numpy array must be 'single', 'double', or 'uint8'. Pass one of these strings
	to the method to indicate what type it is. This method will return a boolean indicating the 
	success or failure
	"""
	
	PyObject * SetCurrentFromNumpyArrayROI(PyObject * current,
										   unsigned long top_row,    unsigned long left_column, 
										   unsigned long bottom_row, unsigned long right_column) {
	
		// making sure the passed object is a numpy array	
		if (PyArray_Check(current)) {
		
			bool			success		  = false;
			
			PyArrayObject * current_array      = (PyArrayObject *) current;
			
			int current_type_num          = current_array->descr->type_num;

			if (!PyArray_ISCARRAY(current_array)) {
				return PyBool_FromLong(0);
			}

			if(current_array->nd != 2) {
				return PyBool_FromLong(0);
			}

			if ((current_type_num == NPY_FLOAT32) || (current_type_num == NPY_FLOAT)) {
				success = self->SetCurrentROI((float *)   current_array->data,
										   current_array->dimensions[1],
										   top_row,    left_column, 
										   bottom_row, right_column);
			}
			else {
				return PyBool_FromLong(0);
			}

			return PyBool_FromLong(success);
		
		}

		return (PyBool_FromLong(0));
	}


	%feature("docstring")
	"""
	This provides a way for the user to set the current frame from a numpy array. The internal
	data type of the numpy array must be 'single', 'double', or 'uint8'. Pass one of these strings
	to the method to indicate what type it is. This method will return a boolean indicating the 
	success or failure
	"""
	
	PyObject * SetCurrentFromNumpyArrayROI(PyObject * current, PyObject * current_mask,
										   unsigned long top_row,    unsigned long left_column, 
										   unsigned long bottom_row, unsigned long right_column) {
	
		// making sure the passed object is a numpy array	
		if (PyArray_Check(current) &&
			PyArray_Check(current_mask)) {
		
			bool			success		  = false;
			
			PyArrayObject * current_array      = (PyArrayObject *) current;
			PyArrayObject * current_mask_array = (PyArrayObject *) current_mask;
			
			int current_type_num          = current_array->descr->type_num;
			int current_mask_type_num     = current_mask_array->descr->type_num;

			if (!PyArray_ISCARRAY(current_array)) {
				return PyBool_FromLong(0);
			}
			if (!PyArray_ISCARRAY(current_mask_array)) {
				return PyBool_FromLong(0);
			}
			
			if(current_array->nd != 2) {
				return PyBool_FromLong(0);
			}
			
			if(current_mask_array->nd != 2) {
				return PyBool_FromLong(0);
			}

			if ((current_type_num == NPY_FLOAT32) || (current_type_num == NPY_FLOAT) &&
			     current_mask_type_num == NPY_INT32) {
				success = self->SetCurrentROI((float *)   current_array->data,
										   (int *)        current_mask_array->data,
										   current_mask_array->dimensions[1],
										   top_row,    left_column, 
										   bottom_row, right_column);
			}
			else {
				return PyBool_FromLong(0);
			}

			return PyBool_FromLong(success);
		
		}

		return (PyBool_FromLong(0));
	}



	%feature("docstring")
	"""
	This method provides a way to copy the cross correlation matrix to a numpy array
	The numpy array will be created and have either double or single precision
	as indicated by the double_precsion parameter. This method will return false if there is a 
	runtime CUDA error or if the user has not already set the frame sizes, reference frame, 
	and current frame.
	"""
	
	PyObject * CopyNCCMatrixToNumpyArray () {
	
		//Get the output dimensions
		unsigned long n_rows;
		unsigned long n_columns;
		
		n_rows    = self->GetNumberOfRowsReference() + self->GetNumberOfRowsCurrent() - 1;
		n_columns = self->GetNumberOfColumnsReference() + self->GetNumberOfColumnsCurrent() - 1;
	
		npy_intp dims[2];
		dims[0] = n_rows;
		dims[1] = n_columns;
	
		//Create a result array
		
		PyObject * res;
		res = PyArray_SimpleNew(2, dims, PyArray_FLOAT32);
		
		bool success = self->GetCorrelationMatrix((float *)((PyArrayObject *)res)->data);
		
		if (!success) {
		
			memset( ((PyArrayObject *)res)->data, 0, n_rows*n_columns*sizeof(float) );
	
		}
		
		return res;
	}	

	
	%feature("docstring")
	"""
	This method provides a way to Cross correlate the reference and current frame and returns it
	in a numpy array passed as a parameter. The numpy array will be created and have either double 
	or single precision as indicated by the double_precsion parameter. This method will return 
	false if there is a runtime CUDA error or if the user has not already set the frame sizes, 
	reference frame, and current frame.
	"""
	
	PyObject * CopyNCCMatrixToNumpyArray(PyObject * output) {
		
		if (PyArray_Check(output)) {	
		
			PyArrayObject * out_array = (PyArrayObject *) output;
			if(out_array->nd != 2) {
				return PyBool_FromLong(0);
			}
			
			//Ensure it is of C type (row-major) contiguous
			if (!PyArray_ISCARRAY(out_array)) {
				return PyBool_FromLong(0);
			}
			
			unsigned long n_rows;
			unsigned long n_columns;
			
			//Getting dimensions
			n_rows    = self->GetNumberOfRowsReference() + self->GetNumberOfRowsCurrent() - 1;
			n_columns = self->GetNumberOfColumnsReference() + self->GetNumberOfColumnsCurrent() - 1;
			
			//Checking dimensions
			if (out_array->dimensions[0] != n_rows ||
				out_array->dimensions[1] != n_columns) {
				return PyBool_FromLong(0);
			}
			if (out_array->descr->type_num != PyArray_FLOAT32 ||
				out_array->descr->type_num != PyArray_FLOAT) {
				return PyBool_FromLong(0);
			}
			
			
			bool success = self->GetCorrelationMatrix((float *) out_array->data);
			
			if (success) {
				return PyBool_FromLong(1);
			}
			
			return PyBool_FromLong(0);
			
		}

		return (PyBool_FromLong(0));
		

	}
	

	%feature("docstring")
	"""
	This method provides a way to Cross correlate the reference and current frame and determine
	the maximum over the entire cross correlation matrix. The maximum value, row of the max, the column
	of max, and a boolean indicating the success or failure of the operation in a list in that order.
	The boolean indicating success will be false if there was a CUDA runtime error or if the user
	had not previously set the frame sizes, reference frame, or current frame.
	"""
	
	PyObject * CrossCorrelationMax() {
		
		float max_val;
		unsigned long max_row;
		unsigned long max_col;
		
		bool success = self->GetCrossCorrelationMax(max_val, max_row, max_col);
		
		PyObject * return_list = PyList_New(4);
		
		if (success) {
			if (PyList_SetItem(return_list, 0, PyFloat_FromDouble(max_val)) != 0){
				success = false;
			}
			else if (PyList_SetItem(return_list, 1, PyLong_FromUnsignedLong(max_row)) != 0) {
				success = false;
			}
			else if (PyList_SetItem(return_list, 2, PyLong_FromUnsignedLong(max_col)) != 0) {
				success = false;
			}
			
			PyList_SetItem(return_list, 3, PyBool_FromLong(1));
		}
		else {		
			PyList_SetItem(return_list, 0, PyFloat_FromDouble(0));
			PyList_SetItem(return_list, 1, PyFloat_FromDouble(0));
			PyList_SetItem(return_list, 2, PyFloat_FromDouble(0));	
			PyList_SetItem(return_list, 3, PyBool_FromLong(0));
		}
		
		return return_list;
	}
	
	%feature("docstring")
	"""
	This method provides a way to Cross correlate the reference and current frame and determine
	the maximum over the a subset of the cross correlation matrix. The region of interest is 
	determined by the rectangle with top left corner (top_row, left_column) and bottom right corner
	(bottom_row, right_column). Note these indices are zero based. The maximum value, row of the max, t
	he column of max, and a boolean indicating the success or failure of the operation in a 
	list in that order. The boolean indicating success will be false if there was a CUDA runtime 
	error or if the user had not previously set the frame sizes, reference frame, or current frame.
	"""
	
	PyObject * CrossCorrelationMax(unsigned long top_row,    unsigned long left_column,
								   unsigned long bottom_row, unsigned long right_column) {
	
		float max_val;
		unsigned long max_row;
		unsigned long max_col;
		
		bool success = self->GetCrossCorrelationMaxROI(max_val, max_row, max_col, top_row, left_column, bottom_row, right_column);
		
		PyObject * return_list = PyList_New(4);
		
		if (success) {
			if (PyList_SetItem(return_list, 0, PyFloat_FromDouble(max_val)) != 0){
				success = false;
			}
			else if (PyList_SetItem(return_list, 1, PyLong_FromUnsignedLong(max_row)) != 0) {
				success = false;
			}
			else if (PyList_SetItem(return_list, 2, PyLong_FromUnsignedLong(max_col)) != 0) {
				success = false;
			}
			
			PyList_SetItem(return_list, 3, PyBool_FromLong(1));
		}
		else {		
			PyList_SetItem(return_list, 0, PyFloat_FromDouble(0));
			PyList_SetItem(return_list, 1, PyFloat_FromDouble(0));
			PyList_SetItem(return_list, 2, PyFloat_FromDouble(0));	
			PyList_SetItem(return_list, 3, PyBool_FromLong(0));
		}
		
		return return_list;
	}

	bool SetFrameSizes(unsigned long n_rows_reference, unsigned long n_columns_reference,
						   unsigned long n_rows_current,   unsigned long n_columns_current,
						   PaddingType padding) {

		bool result;
		Py_BEGIN_ALLOW_THREADS
		result = self->SetFrameSizes(n_rows_reference, n_columns_reference,
						    n_rows_current,   n_columns_current,
						    padding);
		Py_END_ALLOW_THREADS

		return result;
	}

	%feature("docstring")
	"""
	Sets the frame szies for the NCC object. This method also allows you to pass NumPy arrays as 
	parameters for the image masks
	"""

	bool SetFrameSizes(unsigned long n_rows_reference,         unsigned long n_columns_reference, 
						PyObject *   reference_image_mask,     unsigned long n_rows_current,   
						unsigned long n_columns_current,        PyObject *    current_image_mask,
						unsigned int padding_scheme) {

		// Verify input
		if (!PyArray_Check(reference_image_mask) || !PyArray_Check(current_image_mask)) {
			return false;
		}

		if (!PyArray_ISCARRAY(current_image_mask) || !PyArray_ISCARRAY(reference_image_mask)) {
				return false;
		}

		PyArrayObject * reference_image_mask_array = (PyArrayObject *) reference_image_mask;
		PyArrayObject * current_image_mask_array   = (PyArrayObject *) current_image_mask;

		if (reference_image_mask_array->nd != 2 || current_image_mask_array->nd != 2) {
			return false;
		}
			
		if (reference_image_mask_array->dimensions[0] != n_rows_reference ||
			reference_image_mask_array->dimensions[1] != n_columns_reference) {
			return false;
		}

		if (current_image_mask_array->dimensions[0] != n_rows_current ||
			current_image_mask_array->dimensions[1] != n_columns_current) {
			return false;
		}

		int reference_type_num    = reference_image_mask_array->descr->type_num;
		int current_type_num      = current_image_mask_array->descr->type_num;
			
		if (reference_type_num != current_type_num) {
			return false;
		}

		///////////
		// Check the type of the reference array, if it is different than int cast it
		///////////
			
		if (reference_type_num == NPY_UINT8) {
			return self->SetFrameSizes<unsigned char>(n_rows_reference, n_columns_reference, (unsigned char *) reference_image_mask_array->data,
													  n_rows_current,   n_columns_current,   (unsigned char *) current_image_mask_array->data,
													  padding_scheme);				   
		}
		else if (reference_type_num == NPY_UINT16) {
			return self->SetFrameSizes<unsigned short>(n_rows_reference, n_columns_reference, (unsigned short *) reference_image_mask_array->data,
													  n_rows_current,   n_columns_current,   (unsigned short *) current_image_mask_array->data,
													  padding_scheme);	
		}
		else if (reference_type_num == NPY_INT32) {
			return self->SetFrameSizes<int>(n_rows_reference, n_columns_reference, (int *) reference_image_mask_array->data,
											n_rows_current,   n_columns_current,   (int *) current_image_mask_array->data,
											padding_scheme);	
		}
		else if ((reference_type_num == NPY_FLOAT32) || (reference_type_num == NPY_FLOAT)) {
			return self->SetFrameSizes<float>(n_rows_reference, n_columns_reference, (float *) reference_image_mask_array->data,
											n_rows_current,   n_columns_current,   (float *) current_image_mask_array->data,
											padding_scheme);	
		}

		else if (reference_type_num == NPY_FLOAT64) {
			return self->SetFrameSizes<double>(n_rows_reference, n_columns_reference, (double *) reference_image_mask_array->data,
													  n_rows_current,   n_columns_current,   (double *) current_image_mask_array->data,
													  padding_scheme);	
		}
		else {
			return false;
		}

	}

	%feature("docstring")
	"""
	
	"""
	
	PyObject * SetCurrentMaskFromNumpy(PyObject * current_mask) {
	
		// making sure the passed object is a numpy array	
		if (PyArray_Check(current_mask)) {
		
			bool			success		  = false;
			
			PyArrayObject * current_mask_array = (PyArrayObject *) current_mask;
			
			int current_type_num          = current_mask_array->descr->type_num;

			if (!PyArray_ISCARRAY(current_mask_array)) {
				return PyBool_FromLong(0);
			}
			
			if(current_mask_array->nd != 2) {
				return PyBool_FromLong(0);
			}
			
			unsigned long n_rows    = self->GetNumberOfRowsCurrent();
			unsigned long n_columns = self->GetNumberOfColumnsCurrent();

			if (current_mask_array->dimensions[0] != n_rows ||
				current_mask_array->dimensions[1] != n_columns) {
				return PyBool_FromLong(0);
			}
			
			if ((current_type_num == NPY_FLOAT32) || (current_type_num == NPY_FLOAT)) {
				success = self->SetCurrentPupil<float>((float *) current_mask_array->data);
			}
			else if (current_type_num == NPY_FLOAT64) {
				success = self->SetCurrentPupil<double>((double *) current_mask_array->data);
			}
			else if (current_type_num == NPY_UINT8) {
				success = self->SetCurrentPupil<unsigned char>((unsigned char *) current_mask_array->data);
			}
			else if (current_type_num == NPY_UINT32) {
				success = self->SetCurrentPupil<unsigned int>((unsigned int *) current_mask_array->data);
			}
			else if (current_type_num == NPY_INT32) {
				success = self->SetCurrentPupil<int>((int *) current_mask_array->data);
			}

			else {
				return PyBool_FromLong(0);
			}

			return PyBool_FromLong(success);
		
		}

		return (PyBool_FromLong(0));
	}

	%feature("docstring")
	"""
	
	"""
	
	PyObject * SetReferenceMaskFromNumpy(PyObject * reference_mask) {
	
		// making sure the passed object is a numpy array	
		if (PyArray_Check(reference_mask)) {
		
			bool			success		  = false;
			
			PyArrayObject * reference_mask_array = (PyArrayObject *) reference_mask;
			
			int current_type_num          = reference_mask_array->descr->type_num;

			if (!PyArray_ISCARRAY(reference_mask_array)) {
				return PyBool_FromLong(0);
			}
			
			if(reference_mask_array->nd != 2) {
				return PyBool_FromLong(0);
			}
			
			unsigned long n_rows    = self->GetNumberOfRowsCurrent();
			unsigned long n_columns = self->GetNumberOfColumnsCurrent();

			if (reference_mask_array->dimensions[0] != n_rows ||
				reference_mask_array->dimensions[1] != n_columns) {
				return PyBool_FromLong(0);
			}
			
			if ((current_type_num == NPY_FLOAT32) || (current_type_num == NPY_FLOAT)) {
				success = self->SetReferencePupil<float>((float *) reference_mask_array->data);
			}
			else if (current_type_num == NPY_FLOAT64) {
				success = self->SetReferencePupil<double>((double *) reference_mask_array->data);
			}
			else if (current_type_num == NPY_UINT8) {
				success = self->SetReferencePupil<unsigned char>((unsigned char *) reference_mask_array->data);
			}
			else if (current_type_num == NPY_UINT32) {
				success = self->SetReferencePupil<unsigned int>((unsigned int *) reference_mask_array->data);
			}
			else if (current_type_num == NPY_INT32) {
				success = self->SetReferencePupil<int>((int *) reference_mask_array->data);
			}

			else {
				return PyBool_FromLong(0);
			}

			return PyBool_FromLong(success);
		
		}

		return (PyBool_FromLong(0));
	}

}
