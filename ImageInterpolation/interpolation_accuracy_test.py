import numpy as np
import time
import CUDA2DRealMatrix
import CUDAInterpolation
from scipy import ndimage
import matplotlib.pyplot as plt

def WriteNumPyArrayToFile(filename, data):
    FILE = open(filename, "w")

    (n_rows, n_cols) = data.shape
    
    for cur_row in range(n_rows):

        for cur_col in range(n_cols):

            FILE.write(str(data[cur_row, cur_col]) + "\t")

        FILE.write("\n")

    FILE.close();

n_rows = 1000
n_cols = 1000

out_of_bounds_val = -1000;

X, Y = np.float32(np.meshgrid(range(n_cols), range(n_rows)))

regular_data = np.float32(127*np.sin(2*np.pi*X/20)*np.sin(2*np.pi*Y/20)+255)

cuda_regular_data = CUDA2DRealMatrix.CUDA2DRealMatrixSingle(n_rows, n_cols)

cubic_interpolator = CUDAInterpolation.CUDABiCubicInterpolator();

if not cuda_regular_data.CopyFromNumpyArray(regular_data):
    print "Could not copy data to GPU!"

X_irregular = np.float32(X + np.random.rand(n_rows, n_cols))
Y_irregular = np.float32(Y + np.random.rand(n_rows, n_cols))

irregular_data = np.float32(127*np.sin(2*np.pi*X_irregular/20)*np.sin(2*np.pi*Y_irregular/20)+255)

X_irregular = np.float32(X_irregular.reshape([n_rows*n_cols]));
Y_irregular = np.float32(Y_irregular.reshape([n_rows*n_cols]));

# interpolate using cuda
(cuda_linear_interpolated, success)       = CUDAInterpolation.CUDAMapCoordinatesFromSingleCUDA2DRealMatrix(cuda_regular_data, Y_irregular, X_irregular, out_of_bounds_val)
(cuda_cubic_interpolated,  cubic_success) = cubic_interpolator.MapCoordinatesFromNumpyArray(regular_data, Y_irregular, X_irregular, out_of_bounds_val)

if not success or not cubic_success:
    print "CUDA Interpoaltion failed!"

# interpoalte using numpy    
numpy_linear_spline_interpolated = ndimage.map_coordinates(regular_data, [X_irregular, Y_irregular], order = 1, cval = out_of_bounds_val, prefilter = False)
numpy_cubic_spline_interpolated  = ndimage.map_coordinates(regular_data, [X_irregular, Y_irregular], order = 3, cval = out_of_bounds_val, prefilter = True)

# reshape the arrays so that they can be displayed
cuda_linear_interpolated         = cuda_linear_interpolated.reshape([n_rows, n_cols])
cuda_cubic_interpolated          = cuda_cubic_interpolated.reshape([n_rows, n_cols])

numpy_linear_spline_interpolated = numpy_linear_spline_interpolated.reshape([n_rows, n_cols])
numpy_cubic_spline_interpolated  = numpy_cubic_spline_interpolated.reshape([n_rows, n_cols])

error_cuda_linear                = np.absolute(cuda_linear_interpolated - irregular_data)*(cuda_cubic_interpolated != out_of_bounds_val) 
error_cuda_cubic                 = np.absolute(cuda_cubic_interpolated - irregular_data)*(cuda_cubic_interpolated != out_of_bounds_val) 
error_numpy_linear               = np.absolute(numpy_linear_spline_interpolated - irregular_data)*(cuda_cubic_interpolated != out_of_bounds_val)
error_numpy_cubic                = np.absolute(numpy_cubic_spline_interpolated - irregular_data)*(cuda_cubic_interpolated != out_of_bounds_val)

error_figure = plt.figure()

irreg_norm = np.linalg.norm(irregular_data)

cur_axis     = error_figure.add_subplot(1, 4, 1)
cax          = cur_axis.imshow(error_cuda_linear)
cur_axis.set_title('Error CUDA Linear (error = ' + "%.3f" % (100*np.linalg.norm(error_cuda_linear) / irreg_norm) + '%)')
cbar         = error_figure.colorbar(cax)

cur_axis     = error_figure.add_subplot(1, 4, 2)
cax          = cur_axis.imshow(error_cuda_cubic)
cur_axis.set_title('Error CUDA Cubic (error = ' + "%.3f" % (100*np.linalg.norm(error_cuda_cubic) / irreg_norm) + '%)')
cbar         = error_figure.colorbar(cax)

cur_axis     = error_figure.add_subplot(1, 4, 3)
cax          = cur_axis.imshow(error_numpy_linear)
cur_axis.set_title('Error NumPy Linear (error = ' + "%.3f" % (100*np.linalg.norm(error_numpy_linear) / irreg_norm) + '%)')
cbar         = error_figure.colorbar(cax)

cur_axis     = error_figure.add_subplot(1, 4, 4)
cax          = cur_axis.imshow(error_numpy_cubic)
cur_axis.set_title('Error NumPy Cubic Spline (error = ' + "%.3f" % (100*np.linalg.norm(error_numpy_cubic) / irreg_norm) + '%)')
cbar         = error_figure.colorbar(cax)

plt.show()

##WriteNumPyArrayToFile("regular_data.dat", regular_data)
##WriteNumPyArrayToFile("irregular_data.dat", irregular_data)
##WriteNumPyArrayToFile("interpolated_gold.dat", interpolated_gold)
##WriteNumPyArrayToFile("interpolated_values.dat", interpolated_values)
##WriteNumPyArrayToFile("cubic_interpolated_values.dat", cubic_interpolated_values)

