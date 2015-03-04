%module CUDAInterpolation

%{
#include <Python.h>
#include <numpy/arrayobject.h>

#include <CUDAImageProcessing/CUDAInterpolation/Cpp/CUDAInterpolation.h>
#include <Utilities/Cpp/NumpyHelperFunctions.h>
#include <cuda_runtime.h>
%}

%init %{
import_array();
%}

%feature("autodoc", "0");

%typemap(in) std::string {
    if (PyString_Check($input)) {
         $1 = std::string(PyString_AsString($input));
     } else {
         PyErr_SetString(PyExc_TypeError, "string expected");
         return NULL;
     }
}

%typemap(typecheck,precedence=SWIG_TYPECHECK_STRING) std::string {
   $1 = PyString_Check($input) ? 1 : 0;
}

%feature("docstring")
"""
This function does a bi-linear interpolation over the 2 dimensional grid passed as 
a uint8 or float32 2d numpy array (regular_data) at the locations specified by
(row_coordinates, col_coordinates). The locations that are outside the grid will be 
set to the value passed by out_of_bouds_value
"""

%inline %{

PyObject * CUDAMapCoordinatesFromNumpyArray(PyObject *  regular_data,  // uint8, float32 
											PyObject *  row_coordinates, 
											PyObject *  col_coordinates,
											float out_of_bounds_value) {
	// GPU copy of the regular data
	CUDA2DRealMatrix<float> GPU_regular_data(1,1);

	// GPU copy of the row coordinates
	CUDA2DRealMatrix<float> GPU_row_coordinates(1,1);

	// GPU copy of the column coordinates
	CUDA2DRealMatrix<float> GPU_col_coordinates(1,1);

	// Returned touple
	PyObject * ret_ptr = PyTuple_New(2);

	// Number of coordinates to interpolate on
	unsigned long n_rows_desired;
	unsigned long n_cols_desired;

	npy_intp dims_fail[1] = {0};

	if (!Copy2DNumpyArrayToCUDA2DRealMatrix<float>(regular_data, GPU_regular_data)) {
		PyTuple_SET_ITEM(ret_ptr, 1, PyBool_FromLong(0));
		PyTuple_SET_ITEM(ret_ptr, 0, PyArray_SimpleNew(1, dims_fail, PyArray_FLOAT32));
		return ret_ptr;
	}

	if (!Copy1DNumpyArrayToCUDA2DRealMatrix<float>(row_coordinates, GPU_row_coordinates)) {
		PyTuple_SET_ITEM(ret_ptr, 1, PyBool_FromLong(0));
		PyTuple_SET_ITEM(ret_ptr, 0, PyArray_SimpleNew(1, dims_fail, PyArray_FLOAT32));
		return ret_ptr;
	}

	if (!Copy1DNumpyArrayToCUDA2DRealMatrix<float>(col_coordinates, GPU_col_coordinates)) {
		PyTuple_SET_ITEM(ret_ptr, 1, PyBool_FromLong(0));
		PyTuple_SET_ITEM(ret_ptr, 0, PyArray_SimpleNew(1, dims_fail, PyArray_FLOAT32));
		return ret_ptr;
	}

	PyArrayObject * rows_array_casted = (PyArrayObject*) row_coordinates;
	n_rows_desired = rows_array_casted->dimensions[0];

	PyArrayObject * cols_array_casted = (PyArrayObject*) col_coordinates;
	n_cols_desired = cols_array_casted->dimensions[0];

	if (n_rows_desired != n_cols_desired) {
		PyTuple_SET_ITEM(ret_ptr, 1, PyBool_FromLong(0));
		PyTuple_SET_ITEM(ret_ptr, 0, PyArray_SimpleNew(1, dims_fail, PyArray_FLOAT32));
		return ret_ptr;	
	}

	npy_intp success_dims[1] = {n_rows_desired};

	CUDA2DRealMatrix<float> GPU_interpolated_data(1, n_rows_desired);

	// CUDA2DInterpolate
	if (!CUDALinearInterpolation(GPU_regular_data, GPU_interpolated_data, GPU_row_coordinates, 
									GPU_col_coordinates, out_of_bounds_value)) {
		PyTuple_SET_ITEM(ret_ptr, 1, PyBool_FromLong(0));
		PyTuple_SET_ITEM(ret_ptr, 0, PyArray_SimpleNew(1, dims_fail, PyArray_FLOAT32));
		return ret_ptr;	
	}

	PyArrayObject * returned_numpy_interpolated_data;
	
	returned_numpy_interpolated_data = (PyArrayObject *) PyArray_SimpleNew(1, success_dims, PyArray_FLOAT32);

	if (!GPU_interpolated_data.CopyDataToHost<float>((float *)returned_numpy_interpolated_data->data, true)) {
		PyTuple_SET_ITEM(ret_ptr, 1, PyBool_FromLong(0));
		PyTuple_SET_ITEM(ret_ptr, 0, PyArray_SimpleNew(1, dims_fail, PyArray_FLOAT32));
		Py_XDECREF(returned_numpy_interpolated_data);
		return ret_ptr;	
	}

	PyTuple_SET_ITEM(ret_ptr, 1, PyBool_FromLong(1));
	PyTuple_SET_ITEM(ret_ptr, 0, (PyObject*)returned_numpy_interpolated_data);
	return ret_ptr;	

}

PyObject *  CUDAMapCoordinatesFromSingleCUDA2DRealMatrix(CUDA2DRealMatrix<float> &regular_data,
														 PyObject *  row_coordinates, 
														 PyObject *  col_coordinates,
														 float out_of_bounds_value) {

	// GPU copy of the row coordinates
	CUDA2DRealMatrix<float> GPU_row_coordinates(1,1);

	// GPU copy of the column coordinates
	CUDA2DRealMatrix<float> GPU_col_coordinates(1,1);

	// Returned touple
	PyObject * ret_ptr = PyTuple_New(2);

	// Number of coordinates to interpolate on
	unsigned long n_rows_desired;
	unsigned long n_cols_desired;

	npy_intp dims_fail[1] = {0};

	if (!Copy1DNumpyArrayToCUDA2DRealMatrix<float>(row_coordinates, GPU_row_coordinates)) {
		PyTuple_SET_ITEM(ret_ptr, 1, PyBool_FromLong(0));
		PyTuple_SET_ITEM(ret_ptr, 0, PyArray_SimpleNew(1, dims_fail, PyArray_FLOAT32));
		return ret_ptr;
	}

	if (!Copy1DNumpyArrayToCUDA2DRealMatrix<float>(col_coordinates, GPU_col_coordinates)) {
		PyTuple_SET_ITEM(ret_ptr, 1, PyBool_FromLong(0));
		PyTuple_SET_ITEM(ret_ptr, 0, PyArray_SimpleNew(1, dims_fail, PyArray_FLOAT32));
		return ret_ptr;
	}

	PyArrayObject * rows_array_casted = (PyArrayObject*) row_coordinates;
	n_rows_desired = rows_array_casted->dimensions[0];

	PyArrayObject * cols_array_casted = (PyArrayObject*) col_coordinates;
	n_cols_desired = cols_array_casted->dimensions[0];

	if (n_rows_desired != n_cols_desired) {
		PyTuple_SET_ITEM(ret_ptr, 1, PyBool_FromLong(0));
		PyTuple_SET_ITEM(ret_ptr, 0, PyArray_SimpleNew(1, dims_fail, PyArray_FLOAT32));
		return ret_ptr;	
	}

	npy_intp success_dims[1] = {n_rows_desired};

	CUDA2DRealMatrix<float> GPU_interpolated_data(1, n_rows_desired);

	// CUDA2DInterpolate
	if (!CUDALinearInterpolation(regular_data, GPU_interpolated_data, 
								GPU_row_coordinates, GPU_col_coordinates, out_of_bounds_value)) {
		PyTuple_SET_ITEM(ret_ptr, 1, PyBool_FromLong(0));
		PyTuple_SET_ITEM(ret_ptr, 0, PyArray_SimpleNew(1, dims_fail, PyArray_FLOAT32));
		return ret_ptr;	
	}

	PyArrayObject * returned_numpy_interpolated_data;
	returned_numpy_interpolated_data = (PyArrayObject *) PyArray_SimpleNew(1, success_dims, PyArray_FLOAT32);

	if (!GPU_interpolated_data.CopyDataToHost<float>((float *)returned_numpy_interpolated_data->data, true)) {
		PyTuple_SET_ITEM(ret_ptr, 1, PyBool_FromLong(0));
		PyTuple_SET_ITEM(ret_ptr, 0, PyArray_SimpleNew(1, dims_fail, PyArray_FLOAT32));
		Py_XDECREF(returned_numpy_interpolated_data);
		return ret_ptr;	
	}

	PyTuple_SET_ITEM(ret_ptr, 1, PyBool_FromLong(1));
	PyTuple_SET_ITEM(ret_ptr, 0, (PyObject*)returned_numpy_interpolated_data);
	return ret_ptr;	

}

PyObject *  CUDAMapCoordinatesFromUint8CUDA2DRealMatrix(CUDA2DRealMatrix<unsigned char> &regular_data,
														PyObject *  row_coordinates, 
														PyObject *  col_coordinates,
														float out_of_bounds_value) {

	// GPU copy of the regular data
	CUDA2DRealMatrix<float> GPU_regular_data(1,1);

	// GPU copy of the row coordinates
	CUDA2DRealMatrix<float> GPU_row_coordinates(1,1);

	// GPU copy of the column coordinates
	CUDA2DRealMatrix<float> GPU_col_coordinates(1,1);

	// Returned touple
	PyObject * ret_ptr = PyTuple_New(2);

	// Number of coordinates to interpolate on
	unsigned long n_rows_desired;
	unsigned long n_cols_desired;

	npy_intp dims_fail[1] = {0};

	unsigned long n_rows;
	unsigned long n_cols;
	
	if (!regular_data.GetSize(n_rows, n_cols)) {
		PyTuple_SET_ITEM(ret_ptr, 1, PyBool_FromLong(0));
		PyTuple_SET_ITEM(ret_ptr, 0, PyArray_SimpleNew(1, dims_fail, PyArray_FLOAT32));
		return ret_ptr;
	}

	if (!GPU_regular_data.SetSize(n_rows, n_cols)) {
		PyTuple_SET_ITEM(ret_ptr, 1, PyBool_FromLong(0));
		PyTuple_SET_ITEM(ret_ptr, 0, PyArray_SimpleNew(1, dims_fail, PyArray_FLOAT32));
		return ret_ptr;
	}

	// We must cast the input data
	float * GPU_regular_data_ptr;
	if (!GPU_regular_data.GetPointerToData(&GPU_regular_data_ptr)) {
		PyTuple_SET_ITEM(ret_ptr, 1, PyBool_FromLong(0));
		PyTuple_SET_ITEM(ret_ptr, 0, PyArray_SimpleNew(1, dims_fail, PyArray_FLOAT32));
		return ret_ptr;
	}	

	if (!regular_data.CopyDataToDevice<float>(GPU_regular_data_ptr, true)) {
		PyTuple_SET_ITEM(ret_ptr, 1, PyBool_FromLong(0));
		PyTuple_SET_ITEM(ret_ptr, 0, PyArray_SimpleNew(1, dims_fail, PyArray_FLOAT32));
		return ret_ptr;
	}

	if (!Copy1DNumpyArrayToCUDA2DRealMatrix<float>(row_coordinates, GPU_row_coordinates)) {
		PyTuple_SET_ITEM(ret_ptr, 1, PyBool_FromLong(0));
		PyTuple_SET_ITEM(ret_ptr, 0, PyArray_SimpleNew(1, dims_fail, PyArray_FLOAT32));
		return ret_ptr;
	}

	if (!Copy1DNumpyArrayToCUDA2DRealMatrix<float>(col_coordinates, GPU_col_coordinates)) {
		PyTuple_SET_ITEM(ret_ptr, 1, PyBool_FromLong(0));
		PyTuple_SET_ITEM(ret_ptr, 0, PyArray_SimpleNew(1, dims_fail, PyArray_FLOAT32));
		return ret_ptr;
	}

	PyArrayObject * rows_array_casted = (PyArrayObject*) row_coordinates;
	n_rows_desired = rows_array_casted->dimensions[0];

	PyArrayObject * cols_array_casted = (PyArrayObject*) col_coordinates;
	n_cols_desired = cols_array_casted->dimensions[0];

	if (n_rows_desired != n_cols_desired) {
		PyTuple_SET_ITEM(ret_ptr, 1, PyBool_FromLong(0));
		PyTuple_SET_ITEM(ret_ptr, 0, PyArray_SimpleNew(1, dims_fail, PyArray_FLOAT32));
		return ret_ptr;	
	}

	npy_intp success_dims[1] = {n_rows_desired};

	CUDA2DRealMatrix<float> GPU_interpolated_data(1, n_rows_desired);

	// CUDA2DInterpolate
	if (!CUDALinearInterpolation(GPU_regular_data, GPU_interpolated_data, 
								GPU_row_coordinates, GPU_col_coordinates, out_of_bounds_value)) {
		PyTuple_SET_ITEM(ret_ptr, 1, PyBool_FromLong(0));
		PyTuple_SET_ITEM(ret_ptr, 0, PyArray_SimpleNew(1, dims_fail, PyArray_FLOAT32));
		return ret_ptr;	
	}

	PyArrayObject * returned_numpy_interpolated_data;
	returned_numpy_interpolated_data = (PyArrayObject *) PyArray_SimpleNew(1, success_dims, PyArray_FLOAT32);

	if (!GPU_interpolated_data.CopyDataToHost<float>((float *)returned_numpy_interpolated_data->data, true)) {
		PyTuple_SET_ITEM(ret_ptr, 1, PyBool_FromLong(0));
		PyTuple_SET_ITEM(ret_ptr, 0, PyArray_SimpleNew(1, dims_fail, PyArray_FLOAT32));
		Py_XDECREF(returned_numpy_interpolated_data);
		return ret_ptr;	
	}

	PyTuple_SET_ITEM(ret_ptr, 1, PyBool_FromLong(1));
	PyTuple_SET_ITEM(ret_ptr, 0, (PyObject*)returned_numpy_interpolated_data);
	return ret_ptr;	

}

%}
														
bool CUDALinearInterpolation(CUDA2DRealMatrix<float> &regular_data, 
							 CUDA2DRealMatrix<float> &interpolated_data, 
							 CUDA2DRealMatrix<float> &row_coordinates, 
							 CUDA2DRealMatrix<float> &col_coordinates,
							  float out_of_bounds_value);												

class CUDA2DInterpolator {

	virtual bool CUDA2DInterpolate(CUDA2DRealMatrix<float> &regular_data, 
							 CUDA2DRealMatrix<float> &interpolated_data, 
							 CUDA2DRealMatrix<float> &row_coordinates, 
							 CUDA2DRealMatrix<float> &col_coordinates,
							  float out_of_bounds_value) = 0;

};

%extend CUDA2DInterpolator {


	PyObject * MapCoordinatesFromNumpyArray(PyObject *  regular_data,  // uint8, float32 
											PyObject *  row_coordinates, 
											PyObject *  col_coordinates,
											float out_of_bounds_value) {
		// GPU copy of the regular data
		CUDA2DRealMatrix<float> GPU_regular_data(1,1);

		// GPU copy of the row coordinates
		CUDA2DRealMatrix<float> GPU_row_coordinates(1,1);

		// GPU copy of the column coordinates
		CUDA2DRealMatrix<float> GPU_col_coordinates(1,1);

		// Returned touple
		PyObject * ret_ptr = PyTuple_New(2);

		// Number of coordinates to interpolate on
		unsigned long n_rows_desired;
		unsigned long n_cols_desired;

		npy_intp dims_fail[1] = {0};

		if (!Copy2DNumpyArrayToCUDA2DRealMatrix<float>(regular_data, GPU_regular_data)) {
			PyTuple_SET_ITEM(ret_ptr, 1, PyBool_FromLong(0));
			PyTuple_SET_ITEM(ret_ptr, 0, PyArray_SimpleNew(1, dims_fail, PyArray_FLOAT32));
			return ret_ptr;
		}

		if (!Copy1DNumpyArrayToCUDA2DRealMatrix<float>(row_coordinates, GPU_row_coordinates)) {
			PyTuple_SET_ITEM(ret_ptr, 1, PyBool_FromLong(0));
			PyTuple_SET_ITEM(ret_ptr, 0, PyArray_SimpleNew(1, dims_fail, PyArray_FLOAT32));
			return ret_ptr;
		}

		if (!Copy1DNumpyArrayToCUDA2DRealMatrix<float>(col_coordinates, GPU_col_coordinates)) {
			PyTuple_SET_ITEM(ret_ptr, 1, PyBool_FromLong(0));
			PyTuple_SET_ITEM(ret_ptr, 0, PyArray_SimpleNew(1, dims_fail, PyArray_FLOAT32));
			return ret_ptr;
		}

		PyArrayObject * rows_array_casted = (PyArrayObject*) row_coordinates;
		n_rows_desired = rows_array_casted->dimensions[0];

		PyArrayObject * cols_array_casted = (PyArrayObject*) col_coordinates;
		n_cols_desired = cols_array_casted->dimensions[0];

		if (n_rows_desired != n_cols_desired) {
			PyTuple_SET_ITEM(ret_ptr, 1, PyBool_FromLong(0));
			PyTuple_SET_ITEM(ret_ptr, 0, PyArray_SimpleNew(1, dims_fail, PyArray_FLOAT32));
			return ret_ptr;	
		}

		npy_intp success_dims[1] = {n_rows_desired};

		CUDA2DRealMatrix<float> GPU_interpolated_data(1, n_rows_desired);

		// CUDA2DInterpolate
		if (!self->CUDA2DInterpolate(GPU_regular_data, GPU_interpolated_data, GPU_row_coordinates, 
										GPU_col_coordinates, out_of_bounds_value)) {
			PyTuple_SET_ITEM(ret_ptr, 1, PyBool_FromLong(0));
			PyTuple_SET_ITEM(ret_ptr, 0, PyArray_SimpleNew(1, dims_fail, PyArray_FLOAT32));
			return ret_ptr;	
		}

		PyArrayObject * returned_numpy_interpolated_data;
		
		returned_numpy_interpolated_data = (PyArrayObject *) PyArray_SimpleNew(1, success_dims, PyArray_FLOAT32);

		if (!GPU_interpolated_data.CopyDataToHost<float>((float *)returned_numpy_interpolated_data->data, true)) {
			PyTuple_SET_ITEM(ret_ptr, 1, PyBool_FromLong(0));
			PyTuple_SET_ITEM(ret_ptr, 0, PyArray_SimpleNew(1, dims_fail, PyArray_FLOAT32));
			Py_XDECREF(returned_numpy_interpolated_data);
			return ret_ptr;	
		}

		PyTuple_SET_ITEM(ret_ptr, 1, PyBool_FromLong(1));
		PyTuple_SET_ITEM(ret_ptr, 0, (PyObject*)returned_numpy_interpolated_data);
		return ret_ptr;	

	}

	PyObject *  MapCoordinatesFromSingleCUDA2DRealMatrix(CUDA2DRealMatrix<float> &regular_data,
															 PyObject *  row_coordinates, 
															 PyObject *  col_coordinates,
															 float out_of_bounds_value) {

		// GPU copy of the row coordinates
		CUDA2DRealMatrix<float> GPU_row_coordinates(1,1);

		// GPU copy of the column coordinates
		CUDA2DRealMatrix<float> GPU_col_coordinates(1,1);

		// Returned touple
		PyObject * ret_ptr = PyTuple_New(2);

		// Number of coordinates to interpolate on
		unsigned long n_rows_desired;
		unsigned long n_cols_desired;

		npy_intp dims_fail[1] = {0};

		if (!Copy1DNumpyArrayToCUDA2DRealMatrix<float>(row_coordinates, GPU_row_coordinates)) {
			PyTuple_SET_ITEM(ret_ptr, 1, PyBool_FromLong(0));
			PyTuple_SET_ITEM(ret_ptr, 0, PyArray_SimpleNew(1, dims_fail, PyArray_FLOAT32));
			return ret_ptr;
		}

		if (!Copy1DNumpyArrayToCUDA2DRealMatrix<float>(col_coordinates, GPU_col_coordinates)) {
			PyTuple_SET_ITEM(ret_ptr, 1, PyBool_FromLong(0));
			PyTuple_SET_ITEM(ret_ptr, 0, PyArray_SimpleNew(1, dims_fail, PyArray_FLOAT32));
			return ret_ptr;
		}

		PyArrayObject * rows_array_casted = (PyArrayObject*) row_coordinates;
		n_rows_desired = rows_array_casted->dimensions[0];

		PyArrayObject * cols_array_casted = (PyArrayObject*) col_coordinates;
		n_cols_desired = cols_array_casted->dimensions[0];

		if (n_rows_desired != n_cols_desired) {
			PyTuple_SET_ITEM(ret_ptr, 1, PyBool_FromLong(0));
			PyTuple_SET_ITEM(ret_ptr, 0, PyArray_SimpleNew(1, dims_fail, PyArray_FLOAT32));
			return ret_ptr;	
		}

		npy_intp success_dims[1] = {n_rows_desired};

		CUDA2DRealMatrix<float> GPU_interpolated_data(1, n_rows_desired);

		// CUDA2DInterpolate
		if (!self->CUDA2DInterpolate(regular_data, GPU_interpolated_data, 
									GPU_row_coordinates, GPU_col_coordinates, out_of_bounds_value)) {
			PyTuple_SET_ITEM(ret_ptr, 1, PyBool_FromLong(0));
			PyTuple_SET_ITEM(ret_ptr, 0, PyArray_SimpleNew(1, dims_fail, PyArray_FLOAT32));
			return ret_ptr;	
		}

		PyArrayObject * returned_numpy_interpolated_data;
		returned_numpy_interpolated_data = (PyArrayObject *) PyArray_SimpleNew(1, success_dims, PyArray_FLOAT32);

		if (!GPU_interpolated_data.CopyDataToHost<float>((float *)returned_numpy_interpolated_data->data, true)) {
			PyTuple_SET_ITEM(ret_ptr, 1, PyBool_FromLong(0));
			PyTuple_SET_ITEM(ret_ptr, 0, PyArray_SimpleNew(1, dims_fail, PyArray_FLOAT32));
			Py_XDECREF(returned_numpy_interpolated_data);
			return ret_ptr;	
		}

		PyTuple_SET_ITEM(ret_ptr, 1, PyBool_FromLong(1));
		PyTuple_SET_ITEM(ret_ptr, 0, (PyObject*)returned_numpy_interpolated_data);
		return ret_ptr;	

	}

	PyObject *  MapCoordinatesFromUint8CUDA2DRealMatrix(CUDA2DRealMatrix<unsigned char> &regular_data,
														PyObject *  row_coordinates, 
														PyObject *  col_coordinates,
														float out_of_bounds_value) {

		// GPU copy of the regular data
		CUDA2DRealMatrix<float> GPU_regular_data(1,1);

		// GPU copy of the row coordinates
		CUDA2DRealMatrix<float> GPU_row_coordinates(1,1);

		// GPU copy of the column coordinates
		CUDA2DRealMatrix<float> GPU_col_coordinates(1,1);

		// Returned touple
		PyObject * ret_ptr = PyTuple_New(2);

		// Number of coordinates to interpolate on
		unsigned long n_rows_desired;
		unsigned long n_cols_desired;

		npy_intp dims_fail[1] = {0};

		unsigned long n_rows;
		unsigned long n_cols;
		
		if (!regular_data.GetSize(n_rows, n_cols)) {
			PyTuple_SET_ITEM(ret_ptr, 1, PyBool_FromLong(0));
			PyTuple_SET_ITEM(ret_ptr, 0, PyArray_SimpleNew(1, dims_fail, PyArray_FLOAT32));
			return ret_ptr;
		}

		if (!GPU_regular_data.SetSize(n_rows, n_cols)) {
			PyTuple_SET_ITEM(ret_ptr, 1, PyBool_FromLong(0));
			PyTuple_SET_ITEM(ret_ptr, 0, PyArray_SimpleNew(1, dims_fail, PyArray_FLOAT32));
			return ret_ptr;
		}

		// We must cast the input data
		float * GPU_regular_data_ptr;
		if (!GPU_regular_data.GetPointerToData(&GPU_regular_data_ptr)) {
			PyTuple_SET_ITEM(ret_ptr, 1, PyBool_FromLong(0));
			PyTuple_SET_ITEM(ret_ptr, 0, PyArray_SimpleNew(1, dims_fail, PyArray_FLOAT32));
			return ret_ptr;
		}	

		if (!regular_data.CopyDataToDevice<float>(GPU_regular_data_ptr, true)) {
			PyTuple_SET_ITEM(ret_ptr, 1, PyBool_FromLong(0));
			PyTuple_SET_ITEM(ret_ptr, 0, PyArray_SimpleNew(1, dims_fail, PyArray_FLOAT32));
			return ret_ptr;
		}

		if (!Copy1DNumpyArrayToCUDA2DRealMatrix<float>(row_coordinates, GPU_row_coordinates)) {
			PyTuple_SET_ITEM(ret_ptr, 1, PyBool_FromLong(0));
			PyTuple_SET_ITEM(ret_ptr, 0, PyArray_SimpleNew(1, dims_fail, PyArray_FLOAT32));
			return ret_ptr;
		}

		if (!Copy1DNumpyArrayToCUDA2DRealMatrix<float>(col_coordinates, GPU_col_coordinates)) {
			PyTuple_SET_ITEM(ret_ptr, 1, PyBool_FromLong(0));
			PyTuple_SET_ITEM(ret_ptr, 0, PyArray_SimpleNew(1, dims_fail, PyArray_FLOAT32));
			return ret_ptr;
		}

		PyArrayObject * rows_array_casted = (PyArrayObject*) row_coordinates;
		n_rows_desired = rows_array_casted->dimensions[0];

		PyArrayObject * cols_array_casted = (PyArrayObject*) col_coordinates;
		n_cols_desired = cols_array_casted->dimensions[0];

		if (n_rows_desired != n_cols_desired) {
			PyTuple_SET_ITEM(ret_ptr, 1, PyBool_FromLong(0));
			PyTuple_SET_ITEM(ret_ptr, 0, PyArray_SimpleNew(1, dims_fail, PyArray_FLOAT32));
			return ret_ptr;	
		}

		npy_intp success_dims[1] = {n_rows_desired};

		CUDA2DRealMatrix<float> GPU_interpolated_data(1, n_rows_desired);

		// CUDA2DInterpolate
		if (!self->CUDA2DInterpolate(GPU_regular_data, GPU_interpolated_data, 
									GPU_row_coordinates, GPU_col_coordinates, out_of_bounds_value)) {
			PyTuple_SET_ITEM(ret_ptr, 1, PyBool_FromLong(0));
			PyTuple_SET_ITEM(ret_ptr, 0, PyArray_SimpleNew(1, dims_fail, PyArray_FLOAT32));
			return ret_ptr;	
		}

		PyArrayObject * returned_numpy_interpolated_data;
		returned_numpy_interpolated_data = (PyArrayObject *) PyArray_SimpleNew(1, success_dims, PyArray_FLOAT32);

		if (!GPU_interpolated_data.CopyDataToHost<float>((float *)returned_numpy_interpolated_data->data, true)) {
			PyTuple_SET_ITEM(ret_ptr, 1, PyBool_FromLong(0));
			PyTuple_SET_ITEM(ret_ptr, 0, PyArray_SimpleNew(1, dims_fail, PyArray_FLOAT32));
			Py_XDECREF(returned_numpy_interpolated_data);
			return ret_ptr;	
		}

		PyTuple_SET_ITEM(ret_ptr, 1, PyBool_FromLong(1));
		PyTuple_SET_ITEM(ret_ptr, 0, (PyObject*)returned_numpy_interpolated_data);
		return ret_ptr;	

	}

}


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
};