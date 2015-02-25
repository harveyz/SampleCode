import numpy as np
import NormalizedCrossCorrelator
import PythonTools
from pylab import *

def NextPowerOfTwo(val):

    return_val = 1

    while return_val <= val:
        return_val = return_val*2

    return return_val

def CreatePupil(matrix_sizes, padding, image_masks = None):

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

        subplot(1, 2, 1)
        imshow(pupil_reference)
        colorbar()

        subplot(1, 2, 2)
        imshow(pupil_current)
        colorbar()

        show()


    # calculate the FFT of the pupil
    pupil_reference_fft = np.fft.rfft2(pupil_reference)
    pupil_current_fft   = np.fft.rfft2(pupil_current)

    imshow(np.log10(np.abs(pupil_reference_fft)))
    colorbar()
    show()

    # note, FFTs of real data result in complex symmetric matrices
    # they are symmetric in the x direction, so we only need the left
    # half of the matrix

    #x_max = np.floor(pupil_reference_fft.shape[1]/2.0) + 1
    
    #pupil_reference_fft = pupil_reference_fft[:, 0:x_max]
    #pupil_current_fft   = pupil_current_fft[:, 0:x_max]
    
    return(pupil_reference, pupil_current, pupil_reference_fft, pupil_current_fft)

            
if __name__ == '__main__':        
        
    # order, (n_rows_reference, n_cols_reference, n_rows_current, n_cols_current)
    #matrix_test_sizes = [(150, 300, 230, 512), (512, 230, 150, 300), (1341, 1223, 412, 211), \
    #                     (512, 1024, 1024, 512), (632, 231, 526, 213)]

    matrix_test_sizes = [(630, 700, 650, 512)]
    
    padding = 0
    padding_enum = NormalizedCrossCorrelator.NONE

    # create the cross corrleator
    cross_correlator = NormalizedCrossCorrelator.NormalizedCrossCorrelator(1.0)

    # run through all test sizes and set the pupil or template
    for current_size in matrix_test_sizes:

        # create some arbitrary image masks
        #reference_mask = np.ones((current_size[0], current_size[1]), dtype='int32')
        #current_mask = np.ones((current_size[2], current_size[3]), dtype='int32')
        reference_mask = np.int32((np.random.random_sample((current_size[0], current_size[1])) >= .5))
        current_mask   = np.int32((np.random.random_sample((current_size[2], current_size[3])) >= .5))

        #reference_mask = np.zeros((current_size[0], current_size[1]), dtype = 'int32')
        #current_mask   = np.zeros((current_size[2], current_size[3]), dtype = 'int32')

        #reference_mask[0:current_size[0]*2/3, 0:current_size[1]*2/3] = 1
        #current_mask[0:current_size[2]*2/3, 0:current_size[3]*2/3] = 1

        reference_mask_as_list = reference_mask.reshape((reference_mask.shape[0]*reference_mask.shape[1])).tolist()
        current_mask_as_list   = current_mask.reshape((current_mask.shape[0]*current_mask.shape[1])).tolist()

        current_mask_as_list = NormalizedCrossCorrelator.IntVector(current_mask_as_list)
        reference_mask_as_list = NormalizedCrossCorrelator.IntVector(reference_mask_as_list)
        
        # setting the cross-correlation size
        if not cross_correlator.SetFrameSizes(long(current_size[0]), long(current_size[1]),   \
                                              reference_mask_as_list, long(current_size[2]),
                                              long(current_size[3]), current_mask_as_list,  \
                                              padding_enum):
            print "Cross correlator could not be sized on size " + str(current_size)


        gold_matrices = CreatePupil(current_size, padding, (reference_mask, current_mask))

        # compare all of the matrices
        # load the matrices that the NCC module should have spit out. Note these
        # will only be created if the PRINT_MAT preprocessor definition exisits at
        # compile time
        current_template            = np.loadtxt('current_pupil.dat',            delimiter = ",\t")
        current_template_fft_real   = np.loadtxt('current_pupil_fft_real.dat',   delimiter = ",\t")
        current_template_fft_imag   = np.loadtxt('current_pupil_fft_imag.dat',   delimiter = ",\t")
        reference_template          = np.loadtxt('reference_pupil.dat',          delimiter = ",\t")
        reference_template_fft_real = np.loadtxt('reference_pupil_fft_real.dat', delimiter = ",\t")
        reference_template_fft_imag = np.loadtxt('reference_pupil_fft_imag.dat', delimiter = ",\t")

        # normalize  ffts
        #current_template_fft_real = current_template_fft_real/current_template_fft_real.size
        #current_template_fft_imag = current_template_fft_imag/current_template_fft_imag.size
        #reference_template_fft_real = reference_template_fft_real/reference_template_fft_real.size
        #reference_template_fft_imag = reference_template_fft_imag/reference_template_fft_imag.size

        # order of returned values from gold calculations
        # return(pupil_reference, pupil_current, pupil_reference_fft, pupil_current_fft)

        # absolute pupil error
        subplot(3, 2, 1)
        error = np.abs(reference_template - gold_matrices[0])
        imshow(error)
        colorbar()
        title('Error in reference template')
        axis('tight')

        subplot(3, 2, 2)
        error = np.abs(current_template - gold_matrices[1])
        imshow(error)
        colorbar()
        title('Error in current template')
        axis('tight')

        eps = .000000001

        # error in real component of FFT of template
        subplot(3, 2, 3)
        #error = np.log10(np.abs(reference_template_fft_real - gold_matrices[2].real)/(np.abs(gold_matrices[2].real) + eps) + eps)
        error = np.abs(reference_template_fft_real - gold_matrices[2].real)
        imshow(error)
        colorbar()
        title('Relative Error in reference template fft real')
        axis('tight')

        subplot(3, 2, 4)
        #error = np.log10(np.abs(current_template_fft_real - gold_matrices[3].real)/(np.abs(gold_matrices[3].real) + eps) + eps)
        error = np.abs(current_template_fft_real - gold_matrices[3].real)
        imshow(error)
        colorbar()
        title('Relative Error in current template fft real ')
        axis('tight')

        # error in complex component of FFT of template
        subplot(3, 2, 5)
        #error = np.log10(np.abs(reference_template_fft_imag - gold_matrices[2].imag)/(np.abs(gold_matrices[2].imag) + eps) + eps)
        error = np.abs(reference_template_fft_imag - gold_matrices[2].imag)
        imshow(error)
        colorbar()
        title('Relative Error in reference template fft imag')
        axis('tight')

        subplot(3, 2, 6)
        #error = np.log10(np.abs(current_template_fft_imag - gold_matrices[3].imag)/(np.abs(gold_matrices[3].imag) + eps) + eps)
        error = np.abs(current_template_fft_imag - gold_matrices[3].imag)
        imshow(error)
        colorbar()
        title('Relative Error in current template fft imag')
        axis('tight')

        show()
        
