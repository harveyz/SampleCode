import NormalizedCrossCorrelator
import CrossCorrelate
from pylab import *
import time
import numpy as np

a = raw_input("Press enter to continue")

n_rows_reference    = 350
n_columns_reference = 510

n_rows_current      = 420
n_columns_current   = 650 

trashed = 10

reference = np.float32(np.mod(np.ceil(np.random.rand(n_rows_reference, n_columns_reference)*1000), 255))
current   = np.float32(np.mod(np.ceil(np.random.rand(n_rows_current, n_columns_current)*1000), 255))

cross_correlator = NormalizedCrossCorrelator.NormalizedCrossCorrelator(1)

start = time.clock()

if not cross_correlator.SetFrameSizes(long(n_rows_reference), long(n_columns_reference), long(n_rows_current), long(n_columns_current), NormalizedCrossCorrelator.NONE):
    print "Could not set frame sizes!"

if not cross_correlator.SetReferenceFromNumpyArray(reference):
    print "Could not set reference!"
    
if not cross_correlator.SetCurrentFromNumpyArray(current):
    print "Could not set current!"

if not cross_correlator.NormalizedCrossCorrelate():
    print "Could not correlate"

numerator = np.loadtxt('numerator.dat', delimiter = ",\t")

# normalize the DFTs
numerator = numerator/numerator.size

cpp_result = cross_correlator.CopyNCCMatrixToNumpyArray()

stop = time.clock()

#print "Time to cross correlate with CUDA: " + str(stop - start) + " seconds"

start = time.clock()

(numpy_res, numpy_numerator) = CrossCorrelate.CrossCorrelate(reference, current, True)

stop = time.clock()

#print "Time to cross correlate with numpy: " + str(stop - start) + " seconds"
eps = 0.00000001
#diff_total = np.abs(numpy_numerator - numerator)/(np.abs(numpy_numerator)+eps)# - numpy_res)
##
##diff_inner = np.log10(np.abs(cuda_res[trashed:n_rows_reference + n_rows_current - 1 - trashed, trashed:n_columns_reference + n_columns_current - 1 - trashed] -  
##                      numpy_res[trashed:n_rows_reference + n_rows_current - 1 - trashed, trashed:n_columns_reference + n_columns_current - 1 - trashed]))
##
diff_total = np.abs(cpp_result - numpy_res)
#print "Maximum error over entire matrix: " + str(diff_total.max())
#print "Maximum error over matrix with " + str(trashed) + " rows and columns removed: " + str(diff_inner.max())                              

# plot the error on a cool graph

imshow(diff_total)
title("Total Error over entire correlation matrix (log scale)")
colorbar()
show()
