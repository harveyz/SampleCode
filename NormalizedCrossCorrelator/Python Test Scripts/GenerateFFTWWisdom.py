import numpy

# basic sizes
matrix_sizes = range(50, 1000, 50)

# adding some powers of two
matrix_sizes.extend([128, 256, 512, 1024, 2048])

matrix_sizes = sorted(matrix_sizes)

sizes_file_name = 'wisdom_sizes.txt'

file_object = open(sizes_file_name, 'w')

for cur_size in matrix_sizes:
    file_object.write('rof' + str(cur_size) + 'x' + str(cur_size) + '\n')

file_object.close()

