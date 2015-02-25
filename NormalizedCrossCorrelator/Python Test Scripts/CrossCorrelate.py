import numpy as np

    

def CrossCorrelate(reference, current, return_other_matrices=False):
    threshold = 1

    n_rows_reference    = reference.shape[0]
    n_columns_reference = reference.shape[1]

    n_rows_current      = current.shape[0]
    n_columns_current   = current.shape[1]

    n_rows_padded       = n_rows_reference + n_rows_current - 1
    n_columns_padded    = n_columns_reference + n_columns_current - 1

    padded_reference    = np.zeros([n_rows_padded, n_columns_padded])
    padded_current      = np.zeros([n_rows_padded, n_columns_padded])

    shift_rows          = n_rows_reference - 1
    shift_columns       = n_columns_reference - 1

    padded_reference[0:n_rows_reference, 0:n_columns_reference]                                            = reference
    padded_current[shift_rows:shift_rows + n_rows_current, shift_columns:shift_columns + n_columns_current] = current

    cross_correlation   = np.real(np.fft.ifft2(np.conj(np.fft.fft2(padded_reference))*np.fft.fft2(padded_current)))

    pupil_reference     = np.zeros([n_rows_padded, n_columns_padded])
    pupil_current       = np.zeros([n_rows_padded, n_columns_padded])

    pupil_reference[0:n_rows_reference, 0:n_columns_reference]                                         = 1;
    pupil_current[shift_rows:shift_rows+n_rows_current, shift_columns:shift_columns+n_columns_current] = 1;

    norm_factor_ref  = np.real(np.fft.ifft2(np.conj(np.fft.fft2(padded_reference*padded_reference))*np.fft.fft2(pupil_current)))
    norm_factor_cur  = np.real(np.fft.ifft2(np.conj(np.fft.fft2(pupil_reference))*np.fft.fft2(padded_current*padded_current)))

    normalized_cross_corr = cross_correlation/np.sqrt((norm_factor_ref + np.finfo(float).eps)*(norm_factor_cur + np.finfo(float).eps))

    normalized_cross_corr[norm_factor_ref < threshold] = 0
    normalized_cross_corr[norm_factor_cur < threshold] = 0

    if return_other_matrices:
        return (normalized_cross_corr, cross_correlation)
    else:
        return normalized_cross_corr


