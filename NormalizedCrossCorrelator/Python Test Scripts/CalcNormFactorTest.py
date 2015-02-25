import numpy as np
import NormalizedCrossCorrelator
import PythonTools
from pylab import *
import CreatePupilTest

def NextPowerOfTwo(val):

    return_val = 1

    while return_val <= val:
        return_val = return_val*2

    return return_val

def CalcNormFactors(current, reference, matrix_sizes, padding, image_masks = None):
    
    # extract the dimensioins
    n_rows_reference    = matrix_sizes[0]
    n_columns_reference = matrix_sizes[1]
    n_rows_current      = matrix_sizes[2]
    n_columns_current   = matrix_sizes[3]

    # determine the size of the padded matrix
    n_rows_padded       = n_rows_reference + n_rows_current - 1
    n_columns_padded    = n_columns_reference + n_columns_current - 1

    rows_extra = 0
    cols_extra = 0

    # if we pad to apower of two, we add a little more
    if padding == 1:
        rows_extra        = NextPowerOfTwo(n_rows_padded)   - n_rows_padded
        cols_extra        = NextPowerOfTwo(n_columns_padded) - n_columns_padded
    else:
        rows_extra        = 0
        cols_extra        = 0
    
    n_rows_padded    += rows_extra
    n_columns_padded += cols_extra

    padded_reference    = np.zeros([n_rows_padded, n_columns_padded])
    padded_current      = np.zeros([n_rows_padded, n_columns_padded])

    # create the pupil
    pupil_reference     = np.zeros([n_rows_padded, n_columns_padded])
    pupil_current       = np.zeros([n_rows_padded, n_columns_padded])

    # starting location for the current data
    shift_rows          = n_rows_reference - 1
    shift_columns       = n_columns_reference - 1

    if image_masks == None:

        # set the standard pupil
        pupil_reference[rows_extra:(rows_extra + n_rows_reference), 0:n_columns_reference]                                               = 1.0;
        pupil_current[(rows_extra + shift_rows):(rows_extra + shift_rows+n_rows_current), shift_columns:shift_columns+n_columns_current] = 1.0;

    else :

        # set the masks
        pupil_reference[rows_extra:(rows_extra + n_rows_reference), 0:n_columns_reference]                                               = np.float64(image_masks[0]);
        pupil_current[(rows_extra + shift_rows):(rows_extra + shift_rows+n_rows_current), shift_columns:shift_columns+n_columns_current] = np.float64(image_masks[1]);


    # calculate the FFT of the pupil
    pupil_reference_fft = np.fft.rfft2(pupil_reference)
    pupil_current_fft   = np.fft.rfft2(pupil_current)

    padded_reference[rows_extra:(rows_extra + n_rows_reference), 0:n_columns_reference]                                               = reference
    padded_current[(rows_extra + shift_rows):(rows_extra + shift_rows+n_rows_current), shift_columns:shift_columns+n_columns_current] = current

    norm_factor_ref  = np.real(np.fft.irfft2(np.conj(np.fft.rfft2(padded_reference*padded_reference))*pupil_current_fft, s=(n_rows_padded,n_columns_padded)))
    norm_factor_cur  = np.real(np.fft.irfft2(np.conj(pupil_reference_fft)*np.fft.rfft2(padded_current*padded_current), s=(n_rows_padded,n_columns_padded)))

    return (norm_factor_ref, norm_factor_cur)

if __name__ == '__main__':        
        
    # order, (n_rows_reference, n_cols_reference, n_rows_current, n_cols_current)
    #matrix_test_sizes = [(150, 300, 230, 512), (512, 230, 150, 300), (1341, 1223, 412, 211), \
    #                     (512, 1024, 1024, 512), (632, 231, 526, 213)]

    matrix_test_sizes = [(547, 381, 617, 583)]
    padding = 0
    padding_enum = NormalizedCrossCorrelator.NONE

    # create the cross corrleator
    cross_correlator = NormalizedCrossCorrelator.NormalizedCrossCorrelator(1.0)

    # run through all test sizes and set the pupil or template
    for current_size in matrix_test_sizes:

        reference = np.float32(np.mod(np.ceil(np.random.rand(current_size[0], current_size[1])*1000), 255))
        current   = np.float32(np.mod(np.ceil(np.random.rand(current_size[2], current_size[3])*1000), 255))

        # create some arbitrary image masks
        reference_mask = np.ones((current_size[0], current_size[1]), dtype='int32')
        current_mask = np.ones((current_size[2], current_size[3]), dtype='int32')
        #reference_mask = np.int32((np.random.random_sample((current_size[0], current_size[1])) >= .5))
        #current_mask   = np.int32((np.random.random_sample((current_size[2], current_size[3])) >= .5))

        #reference_mask = np.zeros((current_size[0], current_size[1]), dtype = 'int32')
        #current_mask   = np.zeros((current_size[2], current_size[3]), dtype = 'int32')

        #reference_mask[0:current_size[0]*2/3, 0:current_size[1]*2/3] = 1
        #current_mask[0:current_size[2]*2/3, 0:current_size[3]*2/3] = 1

        reference_mask_as_list = reference_mask.reshape((reference_mask.shape[0]*reference_mask.shape[1])).tolist()
        current_mask_as_list   = current_mask.reshape((current_mask.shape[0]*current_mask.shape[1])).tolist()
        reference_mask_as_list = NormalizedCrossCorrelator.IntVector(reference_mask_as_list)
        current_mask_as_list   = NormalizedCrossCorrelator.IntVector(current_mask_as_list)
        
        # setting the cross-correlation size
        if not cross_correlator.SetFrameSizes(long(current_size[0]), long(current_size[1]),   \
                                              reference_mask_as_list, long(current_size[2]),
                                              long(current_size[3]), current_mask_as_list,  \
                                              padding_enum):
            print "Cross correlator could not be sized on size " + str(current_size)

        
        if not cross_correlator.SetReferenceFromNumpyArray(reference):
            print "Error setting reference!"

        if not cross_correlator.SetCurrentFromNumpyArray(current):
            print "Error setting Current!"

        norm_factors = CalcNormFactors(current, reference, current_size, padding, (reference_mask, current_mask))

        # load the result matricies 
        test_current_norm_factor   = np.loadtxt('current_norm_factor.dat', delimiter = ",\t")
        test_reference_norm_factor = np.loadtxt('reference_norm_factor.dat', delimiter = ",\t")

        # The C++ FFTs are not normalized, we must normalize them here
        # to compare
        test_current_norm_factor   = test_current_norm_factor/(test_current_norm_factor.shape[0]*test_current_norm_factor.shape[1])
        test_reference_norm_factor = test_reference_norm_factor/(test_reference_norm_factor.shape[0]*test_reference_norm_factor.shape[1])

        eps = .000000001
        subplot(3, 2, 1)
        imshow(test_reference_norm_factor)
        colorbar()
        title('FFTW computed reference norm')
        axis('tight')

        subplot(3, 2, 2)
        imshow(test_current_norm_factor)
        colorbar()
        title('FFTW computed current norm')
        axis('tight')

        subplot(3, 2, 3)
        imshow(np.abs(norm_factors[0]))
        colorbar()
        title('numpy computed reference norm')
        axis('tight')

        subplot(3, 2, 4)
        imshow(np.abs(norm_factors[1]))
        colorbar()
        title('numpy computed current norm')
        axis('tight')

        subplot(3, 2, 5)
        error = np.log10((np.abs(test_reference_norm_factor - norm_factors[0])/(np.abs(norm_factors[0]) + eps)) + eps)
        imshow(error)
        colorbar()
        title('Relative difference in reference norm (log scale)')
        axis('tight')

        subplot(3, 2, 6)
        error = np.log10((np.abs(test_current_norm_factor - norm_factors[1])/(np.abs( norm_factors[1]) + eps)) + eps)
        imshow(error)
        colorbar()
        title('Relative difference in current norm (log scale)')
        axis('tight')


        show()
        
