
##############################################################
# This is an implementation of Matrix and Vector datatypes 
# in Python. 
# It is pretty unncecessary given the fact that there are 
# already much better solutions available such as pandas, numpy
##############################################################

import math 
from copy import deepcopy
from random import uniform, randrange, seed

#################################################
################## class Common #################
#################################################
class Common:

    # returns the signs for a list of numbers
    # for example, [2, -6, 0, 3] => [1, -1, 0, 1]
    def sign(list):
        if list ==  []: 
            return [] 
        else:
            result = [] 
            for elem in list:
                if elem == 0:
                    result.append(0)
                elif elem > 0:
                    result.append(1)
                else:
                    result.append(-1)
            return result

    # delete elements from an array with indices of elements           
    # to delete given in indices
    def delete(array, indices):
        new_array = []
        for i in range(0, len(array)):
            if not i in indices:
                new_array.append(array[i])
        return new_array
        
        
#################################################
################## class Matrix #################
#################################################


class Matrix:
    
    fstring = '{:4}'
    left_sep = ""
    right_sep = ""
    
    # create size1 x size2 matrix with optional initial values
    # dtype specifies the desired data type of matrix elements
    def __init__(self, size1, size2, dtype = float, init_value = 0):
        if size1 <= 0 or size2 <= 0: raise ValueError("a matrix must have positive dimensions")
        self.m = [[] for r in range(0,size1)]
        self.dim1 = size1
        self.dim2 = size2
        self.dtype = dtype
        for i in range(0, size1):
            for j in range(0, size2):
                self.m[i].append(dtype(init_value))
                
    # get the shape of the matrix, i.e. its numbers of rows and columns
    def shape(self):
        return (self.dim1, self.dim2)
    
    # change format string. used by __str__            
    def set_format(s = '{:4}'):
        Matrix.fstring = s
        
    # determines how a matrix will be represented in __str__
    def set_separators(left, right):
        Matrix.left_sep = left
        Matrix.right_sep = right
        
    # clone matrix
    def clone(self):
        return deepcopy(self)
                
    # transpose the matrix
    def T(self):
        m = Matrix(self.dim2, self.dim1, dtype = self.dtype)
        for r in range(0, self.dim2):
            for c in range(0, self.dim1):
                m.m[r][c] = self.m[c][r]
        return m
    
    # get number of rows (dim1)
    def dim1(self):
        return self.dim1
        
    # get number of columns (dim2)
    def dim2(self):
        return self.dim2
        
    # expects a slice such as matrix[1,2] or matrix[5] or matrix[4:6,7:9]
    # matrix[1,2] returns the matrix element in row 1 and column 2 
    # matrix[5] returns all elements of the fith row vector
    # matrix[4:6,7:9] returns a matrix that spans the rows 4,5 of  
    # the given matrix and the columns 7,8 of these rows
    def __getitem__(self, arg):
        if isinstance(arg, slice): # we got instance[i:j]
            # get specified rows of matrix as 2-dim array
            result = []
            if arg.start == None:
                first = 0
            else:
                first = arg.start
            if arg.stop == None:
                last = len(self.v)
            else:
                last = arg.stop
            if arg.step == None:
                step = 1
            else:
                step = arg.step
            
            for i in range(first, last, step):
                result.append(self.m[i])
                
            return result
            
        elif isinstance(arg, tuple):
            if len(arg) > 2: 
                raise ValueError("maximum of 2 slices allowed")
            else:
                s1 = arg[0]
                s2 = arg[1]
                (x,y) = (isinstance(s1 ,int), isinstance(s2, int))
                if (x,y) == (True, True): # we got instance[i,j]
                    return self.m[s1][s2]
                elif (x,y) == (True, False): # we got instance[i,k:l]
                    tmp = self.m[s1]
                    return tmp[s2]
                elif (x,y) == (False, True): # we got instance[k:l, j]
                    res = []
                    for row in self.m[s1]:
                        res.append(row[s2])
                    return res
                else: # (False, False) # we got instance[i1:2,j1:j2]
                    res = []
                    for row in self.m[s1]:
                        res.append(row[s2])
                    return res       
        else:
            # get single row of matrix as a list
            return self.m[arg] 
            
    
    
            
    # sets a value in matrix. raises ValueError if indices are not in range
    def change_item(self,i,j, val):
        if not i in range(0, self.dim1) or not j in range(0, self.dim2):
            raise ValueError("indices out of range")
        else:
            self.m[i][j] = self.dtype(val)
        
    # multiply all matrix elements with a scalar
    def mult_with_scalar(self, val):
        m = Matrix(self.dim1, self.dim2)
        for r in range (0, self.dim1):
            for c in range(0, self.dim2):
                m.m[r][c] = self.m[r][c] * val
        return m
               
    
    # string representation of matrix 
    def __str__(self):
        s = "\n"
        for r in range(0, self.dim1):
            line = Matrix.left_sep
            for c in range(0, self.dim2):
                line += " " + Matrix.fstring.format(self.m[r][c])
                if c == self.dim2-1:
                    line += Matrix.right_sep + "\n"
            s += line
        return s
    
        
    # get the by leaving out all elements from row i and col j 
    # raises a ValueError if indices i,j are out of range
    def minor(self, i, j):
        if not i in range(0, self.dim1) or not j in range(0, self.dim2):
            raise ValueError("indices out of range")
        m = Matrix(self.dim1-1, self.dim2-1, dtype = self.dtype)
        for r in range(0, self.dim1):
            for c in range(0, self.dim2):
                if r == i or c == j:
                    continue
                if r < i:
                    r_sub = r
                elif r > i:
                    r_sub = r - 1
                if c < j:
                    c_sub = c
                elif c > j:
                    c_sub = c - 1
                m.m[r_sub][c_sub] = self.m[r][c]
        return m               
        
    # recursive calculation of the determinant using sub-matrices
    # returns a ValueError is matrix is not quadratic
    def det(self):
        if self.dim1 != self.dim2:
            raise ValueError("Determinants can only be calculated for quadratic matrices")
        if self.dim1 == 1:
            return self.m[0][0]
        else: # developing around 0,0 
            det = self.dtype(0)
            for c in range(0, self.dim1):
                if c % 2 == 0:
                    factor =  self.dtype(1)
                else:
                    factor = self.dtype(-1)
                det += factor * self.m[0][c] * self.minor(0, c).det()
            return det 
    
    # calculates cofactor of position i,j
    def cofactor(self, i, j):
        cof_ij = self.minor(i,j).det()
        if (i+j) % 2 == 0:
            return cof_ij
        else:
            return -cof_ij
            
    # calculates all co-factors and returns the co-factor matrix
    def cofactor_matrix(self):
        m = Matrix(self.dim1, self.dim2, dtype = self.dtype)
        for r in range(0, self.dim1):
            for c in range(0, self.dim2):
                m[r][c] = self.cofactor(r,c)
        return m
        
    # calculates the adjoint matrix by transposng the 
    # co-factor matrix
    def adjoint_matrix(self):
        return self.cofactor_matrix().T()
        
    # creates the inverse matrix iff det != 0. Raises a  
    # ValueError if that is not the case
    def inverse_matrix(self):
        if self.det() == 0:
            raise ValueError("matrix with det == 0 has no inverse")
        else:
            return self.adjoint_matrix().mult_with_scalar(self.dtype(1) / self.det())
            
    # calculates the equation matrix * <x1,x2, ..., xn> = <v1, v2, ..., vn>
    # raises ValueError if det(matrix) == 0, <x1, x2, ..., xn> being 
    # not transposed, matris is not quadratic, length of vector does 
    # not comply with dimension of matrix
    def solve_equation(self, vector):
        if self.det() == 0:
            raise ValueError("det == 0")
        if (vector.is_transposed()):
            raise ValueError("vector must not be transposed")
        if not self.is_square():
            raise ValueError("matrix must be quadratic")
        if len(vector) != self.dim2:
            raise ValueError("dimensions of matrix and vector do not match")
        return self.inverse_matrix() * vector
            
        
    # get row vector for row
    def row_vector(self, row):
        v = Vector(self.dim2, dtype = self.dtype, transposed = True)
        v.v = self.m[row]
        return v
        
    # get all row vectors at once
    def all_row_vectors(self):
        shape = self.shape()
        result = []
        for i in range(0, shape[0]):
            result.append(self.row_vector(i))
        return result
            
    # get column vector for column
    def column_vector(self, col):
        v = Vector(self.dim1, dtype = self.dtype, transposed = False)
        for r in range(0, self.dim1):
            v[r] = self.m[r][col]
        return v
        
    # get all column vectors at once
    def all_column_vectors(self):
        shape = self.shape()
        result = []
        for j in range(0, shape[1]):
            result.append(self.column_vector(j))
        return result
        
    # get Unit/identity matrix matrix for given size
    def identity(size, dtype = float):
        dim1 = size
        dim2 = size
        m = Matrix(size,size, dtype)
        for r in range(0, size):
            for c in range(0, size):
                if r == c:
                    m.m[r][c] = dtype(1)
                else:
                    m.m[r][c] = dtype(0)
        return m   
        
    # returns the diagonal of a square matrix as a list
    def diagonal(self):
        shp = self.shape()
        if shp[1] > shp[0]:
            raise ValueError("diagonal only available in square matrix")
        list = []
        for i in range(0, shp[1]):
            list.append(self.m[i][i])
        return list
        
    # add two matrices with each other
    #  if their sizes are not the same, a ValueError is raised
    def __add__(self, other):
        if self.dim1 != other.dim1 and self.dim2 != other.dim2:
            raise ValueError("matrices must have similar sizes")
        else:
            m = Matrix(self.dim1, self.dim2, dtype = self.dtype)
            for r in range(0, self.dim1):
                for c in range (0, self.dim2):
                    m.m[r][c] = self.m[r][c] + other.m[r][c]
            return m
            
    # matrix multiplication self * other. Raises ValueError if 
    # object passed as argument is neither a matrix nor a vector,  
    # or when self and other have incompatible dimensions
    def __mul__(self, other):
        if isinstance(other, Matrix):
            # self.dim2 must be equal to other.dim1
            # the result is a self.dim1 x other.dim2 matrix
            if self.dim2 != other.dim1:
                msg = "can not multiply a "
                msg += str(self.dim1) + " x " + str(self.dim2) + " matrix "
                msg += "with a " + str(other.dim1) + " x " + str(other.dim2) + " matrix"
                raise ValueError(msg)
            else:
                m = Matrix(self.dim1, other.dim2, dtype=self.dtype)
                for r in range(0, self.dim1):
                    for c in range(0, other.dim2):
                        m.m[r][c] = self.row_vector(r) * other.column_vector(c)
                return m
        elif isinstance(other, Vector):
            if (not other.is_transposed() and self.dim1 != len(other)) or (other.is_transposed() and self.dim1 != 1):
                raise ValueError("incompatible dimensions of matrix and vector")
            else:
                if not other.is_transposed():
                    v = Vector(self.dim1, dtype = self.dtype, transposed = False)
                    for r in range(0, self.dim1):
                        value = self.dtype(0)
                        for k in range(0, self.dim2):
                            value += self.m[r][k] * other[k]
                        v[r] = value
                    return v
                else: # other.transposed and self.dim2 == 1
                    sum = self.dtype(0)
                    for k in range(0, len(other)):
                        sum += self.m[0][k] * other[k]
                    return sum
        elif isinstance(other, self.dtype):
            return self.mult_with_scalar(other)
        else:
            raise TypeError("second argument must be matrix or vector")
            
    def __matmul__(self, other):
        return self * other
            
            
    # subtracting one matrix from the other. Raises ValueError if sizes are
    # not equal
    def __sub__(self, other):
        if self.dim1 != other.dim1 and self.dim2 != other.dim2:
            raise ValueError("matrices must have similar sizes")
        else:
            m = Matrix(self.dim1, self.dim2, dtype = self.dtype)
            for r in range(0, self.dim1):
                for c in range (0, self.dim2):
                    m.m[r][c] = self.m[r][c] - other.m[r][c]
            return m
            
    # check matrices for equality
    def __eq__(self, other):
        if not isinstance(other, Matrix):
            return False
        if self.dim1 != other.dim1 or self.dim2 != other.dim2:
            return False
        for r in range(0, self.dim1):
            for c in range(0, self.dim2):
                if self.m[r][c] != other.m[r][c]:
                    return False
        return True
        
    # check two matrices for inequality    
    def __ne__(self, other):
        return not (self == other)
        
    # check whether matrix is symmetric, i.e., whether m == m.T. Raises
    # ValueError if dimensions are not equal
    def is_symmetric(self):
        if self.dim1 != self.dim2:
            raise ValueError("symmetry not defined for non-quadratic matrices")
        else:
            return self == self.T()
            
    # check for square matrix
    def is_square(self):
        return self.dim1 == self.dim2
            
    # calculate the standard norm
    def norm(self):
        n = self.dtype(0)
        for c in range(0, self.dim2):
            n = max(n, self.column_vector(c).norm())
        return n
        
    def frobenius_norm(self):
        sum = self.dtype(0)
        for r in range(0, self.dim1):
            for c in range(0, self.dim2):
                sum += self.m[r][c] * self.m[r][c]
        return math.sqrt(sum)
            
    # create a matrix from a vector
    def from_vector(v):
        if v.is_transposed():
            m = Matrix(1, len(v), dtype = v.dtype)
            for i in range(0, len(v)): m.m[0][i] = v.dtype(v[i])
        else:
            m = Matrix(len(v),1, dtype = v.dtype)
            for i in range(0, len(v)): m.m[i][0] = v.dtype(v[i])
        return m
        
    # build absolute values of all matrix elements
    def __abs__(self):
        m = Matrix(self.dim1, self.dim2, dtype = self.dtype)
        for r in range(0, self.dim1):
            for c in range(0, self.dim2):
                m.m[r][c] = abs(self.m[r][c])
        return m
                
    # multiply all matrix elements with n
    def mult_n_times(self, n):
        if n <= 1:
            return self
        m = deepcopy(self)
        for i in range(0, n): 
            m = m * self
        return m
    
    # create and initialize a matrix from a list
    def from_list(list, dtype = float):
        dim1 = len(list)
        dim2 = len(list[0])
        m = Matrix(dim1,dim2,dtype = dtype)
        value_2D = []
        for r in range(0, len(list)):
            value_1D = []
            for c in range(0, len(list[r])):
                value_1D.append(dtype(list[r][c]))
            value_2D.append(value_1D)
                
        m.m = value_2D
        return m
        
    # returns 2d array of of all matrix elements
    def to_list(self):
        return self.m
        
    # converts matrix row-by-row elements to one-dimensional array    
    def to_flat_list(self):
        list = []
        for r in range(0, self.dim1):
            for c in range(0, self.dim2):
                list.append(self.m[r][c])
        return list
        
    # creates a matrix from a flat list using the shape 
    # (shape[0], shape[1]). Raises a ValueError if list 
    # length does not suffice specified shape
    def from_flat_list(list, shape, dtype = float):
        if len(list) != shape[0] * shape[1]:
            raise ValueError("len(list) <> shape_0 * shape_1")
        elif shape == None:
            raise ValueError("shape must not be None")
        else:
            m = Matrix(shape[0], shape[1], dtype = dtype)
            for r in range(0, shape[0]):
                for c in range(0, shape[1]):
                    m.m[r][c] = dtype(list[r * shape[1] + c])
            return m
        
    def reshape(self, shape, dtype = float):
        if shape == None : 
            raise ValueError("shape must not be None")
        elif shape[0] == self.dim1 and shape[1] == self.dim2:
            return self.clone()
        elif self.dim1 * self.dim2 != shape[0] * shape[1]:
            raise ValueError("shape does not correspond to dim1*dim2")
        else:
            list = self.to_flat_list()
            return Matrix.from_flat_list(list, shape, dtype = self.dtype)
            
        
    # applies lambda to each element of matrix
    def apply(self, lambda_f):
        m = Matrix(self.dim1, self.dim2, dtype = self.dtype)
        for r in range(0, self.dim1):
            for c in range(0, self.dim2):
                m.m[r][c] = lambda_f(self.m[r][c])
        return m
        
    # like apply, but with lambda getting called with
    # row, col, value at (row,col) 
    def apply2(self, lambda_f):
        m = Matrix(self.dim1, self.dim2, dtype = self.dtype)
        for r in range(0, self.dim1):
            for c in range(0, self.dim2):
                m.m[r][c] = lambda_f(r, c, self.m[r][c])
        return m
        
    def swap_rows(self, i1, i2):
        if not i1 in range(0, self.dim1) or not i2 in range(0, self.dim1):
            raise ValueError("indices out of range")
        else:
            m = self.clone()
            if i1 == i2: 
                return m
            else:
                m = self.clone()
                tmp = m.m[i1]
                m.m[i1] = m.m[i2]
                m.m[i2] = tmp
                return m
            
    def swap_columns(self, j1, j2):
        if not j1 in range(0, self.dim2) or not j2 in range(0, self.dim2):
            raise ValueError("indices out of range")
        else:
            m = self.clone()
            if j1 == j2: 
                return m
            else:
                for row in range(0, m.dim1):
                    tmp = m[row][j1]
                    m[row][j1] = m[row][j2]
                    m[row][j2] = tmp
                return m        
                
    # row <row> is multiplied with <factor>
    def mult_with_factor(self, row, factor):
        m = self.clone()
        for col in range(0, m.dim2):
            m.m[row][col] = factor * m.m[row][col]
        return m
        
    # M[touched_row] += factor * M[untouched_row]
    def add_multiple_of_row(self, touched_row, untouched_row, factor):
        m = self.clone()
        for c in range(0, m.dim2):
            m.m[touched_row][c] += m.m[untouched_row][col] * factor


    # creates  the row echolon form of the matrix
    def echolon(self):
        def find_pivot(matrix, start_row, col):
            for row in range(start_row, matrix.dim1):
                if matrix.m[row][col] != 0:
                    return (matrix.m[row][col], row)
            return (-1,-1)           
            
        m = Matrix.clone(self)
        col = 0
        if m.dim1 < m.dim2:
            upper_limit = m.dim1
        else:
            upper_limit = m.dim2
        for row in range(0, upper_limit):
            col = row
            if m.m[row][col] == 1: continue
            (pivot, r_pivot) = find_pivot(m, row + 1, col)
            if (pivot,r_pivot) == (-1,-1):
                continue
            else:
                m = m.swap_rows(row, r_pivot)
                if m.m[row][col] != 1:
                    m = m.mult_with_scalar(row, 1/pivot)
                for r in range(row+1, m.dim1):
                    if m[r][col] == 0:
                        continue
                    else:
                        factor = -m.m[r][col]
                        if factor != 0:
                            for c in range(0, m.dim2):
                                m[r][c] = m[r][c] + factor * m[row][c]
        return m
        
    
    # calculates the reduced row echolon form of the matrix    
    def reduced_echolon(self):
        m = self.clone()
        lead = 0
        for r in range(0, self.dim1):
            if (m.dim2 <= lead): break
            i = r
            while m.m[i][lead] == 0:
                i += 1
                if (i == m.dim1):
                    i = r
                    lead += 1
                    if m.dim2 == lead:
                        return m
            m = m.swap_rows(i,r)
            val = m.m[r][lead]
            for j in range(0, m.dim2):
                m.m[r][j] /= val
            for i in range(0, m.dim1):
                if (i == r): 
                    continue
                val = m.m[i][lead]
                for j in range(0, m.dim2):
                    m[i][j] -= val * m.m[r][j]
            lead += 1
        return m
        
    # set up a matrix by stacking up the row vectors
    # [ ---- r1 ----] 
    # [ ---- r2 ----] 
    #       .... 
    # [ ---- rn ----]
    def from_row_vectors(vec_array):
        for arg in vec_array:
            if not isinstance(arg, Vector):
                raise TypeError("vectors expected as arguments")
            elif (len(vec_array) == 0): 
                raise ValueError("empty argument list")
            else:
                shape = vec_array[0].shape()
                m = Matrix(len(vec_array), shape[0])
                i = 0
                for vec in vec_array:
                    v_shape = vec.shape()
                    if v_shape != shape or v_shape[1] != True:
                        raise ValueError("vectors must be transposed and share the same shape")    
                    m.m[i] = vec.to_list()
                    i += 1
                return m
    
    # create a matrix from column vectors   
    # | c1 c2 c3 .... cn | 
    # |  |  |  |       | | 
    # |  |  |  |       | | 
                   
    def from_column_vectors(vec_array):
        if not isinstance(vec_array, list):
            raise TypeError("unexpected argument type")
        for vec in vec_array:
            if not isinstance(vec, Vector):
                raise TypeError("vectors expected as arguments")
            elif (len(vec_array) == 0): 
                raise ValueError("empty argument list")
            else:
                shape = vec_array[0].shape()
                m = Matrix(shape[0], len(vec_array))
                i = 0
                for col in range(0, len(vec_array)):
                    vec = vec_array[col]
                    v_shape = vec.shape()
                    if v_shape != shape or v_shape[1] != False:
                        raise ValueError("vectors must be transposed and share the same shape")    
                    for j in range(0, v_shape[0]):
                        m.m[j][i] = vec[j]
                    i += 1
                return m
                
    def qr_decomposition(self):
        u = []
        e = []
        a = self.all_column_vectors()
        shape = self.shape()
        e = [None for i in range(0, shape[1])]
        u = [None for i in range(0, shape[1])]
        if not shape[1] >= shape[0]:
            raise ValueError("number of columns must be >= number of rows")
        u[0] = a[0]
        e[0] = u[0].mult_with_scalar(1 / u[0].euclidean_norm())
        u[1] = a[1] - e[0] * (a[1].T()*e[0])
        e[1] = u[1].mult_with_scalar(1 / u[1].euclidean_norm())
        for k in range(2, shape[1]):
            u[k] = a[k]
            for i in range(0,k):
                u[k] -= e[i] * (a[k].T() * e[i])
            e[k] = u[k].mult_with_scalar(1/u[k].euclidean_norm())
        Q = Matrix.from_column_vectors(e)
        R = Matrix(shape[1], shape[0], dtype = self.dtype)
        for i in range(0, shape[1]):
            for j in range(0, shape[0]):
                if i > j: R.m[i][j] = 0
                else:
                    R.m[i][j] = a[j].T() * e[i]
        return (Q,R)
        
    # eigenvalues calculation using QR decomposition 
    def eigenvalues(self):
        epsilon = 1E-12
        i_max = 1000
        if self.dtype == int:
            diff = int("inf")
        else:
            diff = self.dtype("inf")
        i = 0
        A_new = self.clone()
        A_orig = self.clone()
        while (diff > epsilon) and (i < i_max):
            A_orig = A_new
            (Q,R) = A_orig.qr_decomposition()
            A_new = R @ Q
            diff = abs(A_new - A_orig).frobenius_norm()
            i += 1
        return A_new.diagonal()
        
    # creates a diagonal matrix with the list 
    # elements populating the diagonal
    def diagonal_matrix(list, dtype = float):
        m = Matrix(len(list), len(list), dtype)
        for i in range(0, len(list)):
            m[i][i] = list[i]
        return m
        
    # creates a list with all entries from start to stop 
    # using step as increment. The values created are in 
    # [start, stop[
    def arange(start = 0, stop = 1, step = 1, dtype = float):
        result = []
        act_value = start
        while act_value < stop:
            result.append(act_value)
            act_value += step
        return result
        
    # get a matrix filled with random numbers
    def random_matrix(shp, fromvalue, tovalue, dtype, seedval = None):
        if seedval != None:
            seed(seedval)
        rows = shp[0]
        cols = shp[1]
        m = Matrix(rows, cols, dtype)
        for r in range(0, rows):
            for c in range(0, cols):
                if dtype == int:
                    m.m[r][c] = int(randrange(fromvalue, tovalue))
                else:
                    m.m[r][c] = dtype(uniform(fromvalue, tovalue))
        return m
            
 #################################################              
 ################## class Vector #################
 #################################################
                
                                              
class Vector:
    fstring = '{:4}'
    left_sep = ""
    right_sep = ""
    
    _transposed = False
    
    # Initialize vector with size, initial values for
    # its elements and initial transposition-state
    def __init__(self, size, transposed = False, dtype = float, init_value = 0):
        self.v = [dtype(init_value) for i in range(0,size)]
        self._transposed = transposed
        self.dtype = dtype
        
    # get the number of elements in the vector
    # if vector is transposed => (len(v), True)
    #                    else => (len(v), False)
    def shape(self):
        if self.is_transposed():
            return (len(self), True)
        else:
            return (len(self), False)
            
    # change format string. used by __str__            
    def set_format(s = '{:4}'):
        Matrix.fstring = s
        
    # determines how a matrix will be represented in __str__
    def set_separators(left, right):
        Vector.left_sep = left
        Vector.right_sep = right
        
    # clone vector
    def clone(self):
        return deepcopy(self)
        
    # check whether vector is transposed
    def is_transposed(self):
        return self._transposed
    
        
    # string representation of vector
    def __str__(self):
        if self._transposed:
            res = Vector.left_sep
            for i in range(0, len(self.v)): 
                res += " " + Vector.fstring.format(self.v[i])
            res += Vector.right_sep
            return res
        else:
            res = "\n"
            for i in range(0, len(self.v)):
                res +=  Vector.left_sep + Vector.fstring.format(self.v[i]) + Vector.right_sep + "\n"
            return res
    
    
    # vector transposition        
    def T(self):
        v = Vector(len(self), dtype = self.dtype, transposed = not self.is_transposed())
        for i in range(0, len(self)): 
            v[i] = self[i]
        return v
        
    # return length of vector
    def __len__(self):
        return len(self.v)
        
    # calculate absolute value of all elements
    def __abs__(self):
        res = Vector(len(self), dtype = self.dtype)
        for i in range(0, len(self)):
            res[i] = abs(self[i])
        return res
                   
    # access vector element, for example, x = v[6]
    # also allows slicing such as vec[0:] or v[0:2, 4:]. 
    # Raises a ValueError if index is out of range
    def __getitem__(self, arg):
        result = []
        if isinstance(arg, slice):
            result = []
            if arg.start == None:
                first = 0
            else:
                first = arg.start
            if arg.stop == None:
                last = len(self.v)
            else:
                last = arg.stop
            if arg.step == None:
                step = 1
            else:
                step = arg.step
            
            for i in range(first, last, step):
                result.append(self.v[i])
        if isinstance(arg, tuple):
            result = []
            for x, y in enumerate(arg):
                result += self[y]
            return result
        return self.v[arg]
        if i < 0 or i >= len(self.v):
            raise ValueError("index out of range")
        return self.v[i]
        
    # possible usages:
    # vector[8] = 9  => assigns value to singlevector position
    # vector[0:3] = [0,1,2] = set vector[0:3] to list elements
    # vector[2:4, 6:8] => sets vector positions 2:4 and 6:8 
    # to elements of list
    def __setitem__(self, arg, val):
        if isinstance(arg, slice):
            if arg.start == None: first = 0
            else: first = arg.start
            if arg.stop == None: last = len(v.v)
            else: last = arg.stop
            if arg.step == None: step = 1
            else: step = arg.step
            j = 0
            for i in range(first, last, step):
                self.v[i] = self.dtype(val[j])
                j += 1
        elif isinstance(arg, tuple):
            j = 0
            for x,y in enumerate(arg):
                if y.start == None: start = 0
                else: start = y.start
                if y.stop == None: stop = len(v.v)
                else: stop = y.stop
                if y.step == None: step = 1
                else: step = y.step
                for i in range(start, stop, step):
                    self.v[i] = self.dtype(val[j])
                    j+= 1
        else:
            self.v[arg] = self.dtype(val)
        
    # multiplication of vectors 
    # vector multiplication with matrices is delegated.
    # raises ValuError if other is not a vector, if length 
    # of vectors is not equal, 
    # to class Matrix
    def __mul__(self, other):
        if isinstance(other, Matrix):
            m = Matrix.from_vector(self)
            return m * other
        elif isinstance(other, self.dtype):
            return self.mult_with_scalar(other)
        if not isinstance(other, Vector):
            raise TypeError("other object must also be a vector")
        if len(other) != len(self):
            raise ValueError("incompatible lengths of vectors")
        else:
            if self._transposed:
                if not other._transposed:
                    return self.scalar_product(other)
                else:
                    v = Vector(len(self), dtype = self.dtype, transposed = self._transposed)
                    for i in range(0, len(self)):
                        v[i] = self.dtype(self[i] * other[i])
                    return v
            else: # not self.transposed  
                if other._transposed:
                    mat = Matrix(len(self), len(other), dtype = self.dtype)
                    for r in range(0, len(self)):
                        for c in range(0, len(other)):
                            mat.m[r][c] = self[r]*other[c]
                    return mat       
                else: 
                    v = Vector(len(self), dtype = self.dtype, transposed = self._transposed)
                    for i in range(0, len(self)):
                        v[i] = self[i] * other[i]
                    return v
    
    # add two vectors with each other. Raises ValueError of lengths
    # are not equal or when trying to multiply a transposed with a 
    # non-transpised vector
    def __add__(self, other):
        if len(other) != len(self):
            raise ValueError("incompatible lengths of vectors")
        elif self._transposed != other._transposed:
            raise ValueError("transposed and not transposed vectors cannot be added")
        else:
            res = Vector(len(self), dtype = self.dtype, transposed = self._transposed)
            for i in range(0, len(self)): res[i] = self[i] + other[i]
            return res
            
    # negative vector: all elements switch their sign
    def __neg__(self):
        res = Vector(len(self), dtype = self.dtype, transposed = self._transposed)
        for i in range(0, len(self)): res[i] = -self[i]
        return res
        
    # positive vector: nothing changes
    def __pos__(self):
        res = Vector(len(self), dtype = self.dtype, transposed = self._transposed)
        for i in range(0, len(self)): res[i] = self[i]
        return res
        
    # build scalar product of two vectors
    def scalar_product(self, other):
        if len(self) != len(other):
            raise ValueError("incompatible lengths of vectors")
        else:
            res = self.dtype(0)
            for i in range(0,len(self)): res += self[i]*other[i]
            return res
            
    # subtract one vector from the other. Raises ValueError when
    # vector lengths do not match, or when one vector is transposed while 
    # the other isn't
    def __sub__(self, other):
        if len(other) != len(self):
            raise ValueError("incompatible lengths of vectors")
        elif self._transposed != other._transposed:
            raise ValueError("transposed and not transposed vectors cannot be subtracted from each other")
        else:
            res = Vector(len(self), dtype = self.dtype)
            for i in range(0, len(self)): res[i] = self[i] - other[i]
            return res
            
    # test vectors for equality
    def __eq__(self, other):
        if not isinstance(other, Vector) or len(self) != len(other): return False
        if self._transposed != other._transposed: return False
        for i in range(0, len(self)):
            if self[i] != other[i]: return False
        return True
        
    # test vectors for non equality
    def __ne__(self, other):
        return not (self == other)
        
    # get euclidean norm of vector
    def euclidean_norm(self):
        res = self.dtype(0.0)
        for i in range(0,len(self)):
            res += self[i] * self[i]
        return math.sqrt(res)
        
    # get regular norm of vector 
    def norm(self):
        res = self.dtype(0.0)
        for i in range(0,len(self)):
            res += abs(self[i])
        return res
            
    # multiply all vector elements with a scalar
    def mult_with_scalar(self, scalar):
        res = Vector(len(self))
        for i in range(0, len(self)): res[i] = self[i] * self.dtype(scalar)
        return res
        
    # check whether one vector is orthogonal to the other
    def is_orthogonal_to(self, other):
        if len(other) != len(self):
            raise ValueError("incompatible lengths of vectors")
        if self.is_transposed() or other.is_transposed():
            raise ValueError("vectors must not be transposed")
        else:
            return self.scalar_product(other) == 0
    
    # get ith unit/base vector for dimension = size
    def unit_vector(size, i, dtype = float):
        v = Vector(size)
        for j in range(0,i): v[j] = dtype(0)
        v[i] = 1
        for j in range(i+1, size): v[j] = dtype(0)
        return v
        
    # retrieve all unit/base vectors for dimension = size
    def all_unit_vectors(size, dtype = float):
        vec_arr = []
        for i in range(0, size):
                vec_arr.append(Vector.unit_vector(size,i, dtype))
        return vec_arr
        
    # create a vector from a list
    def from_list(list, dtype = float, transposed = False):
        v = Vector(len(list), transposed = transposed)
        for i in range(0, len(v)): v[i] = dtype(list[i])
        return v


    # get a vector filled with random numbers
    def random_vector(length, fromvalue, tovalue, dtype, seedval = None):
        if seedval != None:
            seed(seedval)
        v = Vector(length, dtype)
        if dtype == int:
            for i in range(0, length):
                v[i] = int(randrange(fromvalue, tovalue))
        else:
            for i in range(0, length):
                v[i] = dtype(uniform(fromvalue,tovalue))
        return v
        
    
    # return list of all vector elements
    def to_list(self):
        return self.v
        
    # apply lambda to each element of vector
    def apply(self, lambda_f):
        v = Vector(len(self), dtype = self.dtype)
        for i in range(0, len(self)):
            v[i] = lambda_f(self[i])
        return v
        
    # same as apply, but with additional vector position passed to 
    # lambda
    def apply2(self, lambda_f):
        v = Vector(len(self), dtype = self.dtype)
        for i in range(0, len(self)):
            v[i] = lambda_f(i, self[i])
        return v


