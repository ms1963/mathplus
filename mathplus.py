
##############################################################
# This is an implementation of Matrix and Vector datatypes 
# in Python. In addition, classes for rational numbers and 
# Polynomials are included.
# It is pretty unncecessary given the fact that there are 
# already much better solutions available such as pandas, numpy
##############################################################

from __future__ import print_function
from __future__ import division
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
        
    # erase values from array so that that none of 
    # these numbers is left in the array 
    def erase(array, values):
        new_array = deepcopy(array)
        for val in values:
            while val in new_array:
                new_array.remove(val)
        return new_array
        

    # checks the shape of an array a, for example,
    # [1,2] -> (1,2) 
    # [[1,2,3][4,5,6]] -> (2,3)
    def shape(a):
        if not isinstance(a, list):
            raise ValueError("only arrays are allowed")
        if isinstance(a[0], list):
            length = len(a[0])
            for i in range(1, len(a)):
                if len(a[i]) != length:
                    raise ValueError("only rectangular arrays are allowed")
            return(len(a), length)
        else: 
            return(1,len(a))
        
    # determines how many items are covered by a 
    # slice slc applied to array        
    def slice_length(slc, array):
        return len(array[slc])
        
    # print for matrices
    def print_matrix(m):   
        for r in range(0, m.dim1):
            print("[ ", end="")
            for c in range(0, m.dim2):
                print(" " + str(m.m[r][c]), end ="")
                if c < m.dim2-1: print("\t",end="")
            print("  ]")
        
    # print for vectors: based upon __str__
    def print_vector(v):
        if v.is_transposed():
            print("[ ", end="")
            for r in range(0, len(v)):
                print("" + str(v[r]), end ="") 
                if r < len(v)-1: 
                    print("\t", end ="")
                else:
                    print("", end="")
            print(" ]")
        else:
            print("[")
            for r in range(0, len(v)):
                print(" " + str(v[r]) + "\n", end = "")
            print("]")
            
    # concatenate two rectangular arrays on axis 0 or 1 
    def concatenate(arr1, arr2, axis = 0):
        shp1 = Common.shape(arr1)
        shp2 = Common.shape(arr2)
        if axis == 0:
            if shp1[1] != shp2[1]:
                raise ValueError("cannot concatenate array with different number of columns on axis 0")
            else:
                result = []
                for r in range(0, shp1[0]): 
                    result.append(arr1[r])
                for r in range(0, shp2[0]):
                    result.append(arr2[r])
                return result
        else: # axis <> 0
            if shp1[0] != shp2[0]:
                raise ValueError("cannot concatenate array with different number of rows on axis 1")
            else:
                result = [[0 for i in range(0, shp1[1]+shp2[1])] for j in range(0, shp1[0])]
                for c in range(0, shp1[1]):
                    for r in range(0, shp1[0]):
                        result[r][c] = arr1[r][c]
                for c in range(shp1[1], shp1[1] + shp2[1]):
                    for r in range(0, shp2[0]):
                        result[r][c] = arr2[r][c-shp1[1]]
                return result
                
    # summing all array elements on axis 0 or 1
    def sum(arr, axis = 0):
        shp = Common.shape(arr)
        result = []
        if axis == 0:
            for c in range(0, shp[1]):
                sum = 0
                for r in range(0, shp[0]):
                    sum += arr[r][c]   
                result.append(sum)
        else: # axis <= 0 
            for r in range(0, shp[0]):
                sum = 0
                for c in range (0, shp[1]):
                    sum += arr[r][c]
                result.append(sum)
        return result
        
    # calculates the mean of array elements
    def mean(arr):
        sum = 0
        for elem in arr:
            sum += elem
        return sum / len(arr)
        
    # calculates the standard deviation of array elemments
    def std_dev(arr):
        sum = 0
        mu = Common.mean(arr)
        for i in range(0, len(arr)):
            sum += (arr[i] - mu) ** 2
        res = math.sqrt(sum / len(arr))
        return res 
        
    # calculates the variance of array elements
    def variance(arr):
        return Common.std_dev(arr) ** 2
        
    # calculates the median of array elements
    def median(arr):
        a = deepcopy(arr)
        a.sort()
        len_a = len(a)
        if len(a) % 2 == 1:
            return a[len_a // 2]
        else:
            return (a[(len_a - 1)//2]+a[(len_a+1)//2])/2
        
            
        
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
        
    # tr() denotes the trace of a matrix which is the elements
    # on the main diagonal summed up. tr() is only defines for
    # square matrices
    def tr(self):
        if not self.is_square():
            raise ValueError("trace only defined for square matrices")
        else:
            sum = 0
            diag = self.diagonal()
            for elem in diag: sum += elem
            return sum
    
    # change format string. used by __str__        
    def set_format(s):
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
        
    def H(self):
        if self.dtype != complex: return self.T()
        m = Matrix(self.dim2, self.dim1, dtype = self.dtype)
        for r in range(0, self.dim2):
            for c in range(0, self.dim1):
                m.m[r][c] = self.m[c][r].conjugate()
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
            
    # usage examples for __setitem__:
    # m[4,3] = 0 (could also be done with m[4][3] = 0 
    # m[0] = [1,2,3] used to set a whole line 
    # m[1,1:3] = [4,8] sets columns 1 & 2 in line 1 
    # m[0:2,2] = [4,2] sets column 2 in rows 0 & 1 to 4 resp. 2 
    # m[2:4,1:3] = [[21, 22],[31, 32]] changes multiple rows
    # and columns
    def __setitem__(self, arg, val):
        if isinstance(arg, slice): # we got instance[i:j:s]
            if not isinstance(val, list):
                raise TypeError("argument must be a list")
            slclen =   Common.slice_length(arg, [0 for i in range(0,self.dim1)])
            dim1, dim2  = Common.shape(val)
            if (slclen != dim1) or (dim2 != self.dim2):
                raise ValueError("an array with " + str(slclen) + " rows and " + str(self.dim2) + " columns expected as argument")
            # set specified rows of matrix as 2-dim 
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
            
            r = 0
            for i in range(first, last, step):
                self.m[i] = val[r]
                r += 1
        elif isinstance(arg, tuple):
            if len(arg) > 2: 
                raise ValueError("maximum of 2 slices allowed")
            else:
                s1 = arg[0]
                s2 = arg[1]
                (x,y) = (isinstance(s1 ,int), isinstance(s2, int))
                if (x,y) == (True, True): # we got instance[i,j]
                    self.m[s1][s2] = val
                elif (x,y) == (True, False): # we got instance[i,k:l]
                    if not isinstance(val, list):
                        raise TypeError("list expected as argument")
                    length_required = Common.slice_length(s2, self.m[s1])
                    dim1, dim2 = Common.shape(val)
                    if dim1 != 1:
                        raise ValueError("one dimensional array required as argument")
                    if length_required != dim2:
                        raise ValueError("argument must be a list with length = " + str(length_required))
                    if s2.start == None:
                        first = 0
                    else:
                        first = s2.start
                    if s2.stop == None:
                        last = len(self.m[s1])
                    else:
                        last = s2.stop
                    if s2.step == None:
                        step = 1
                    else:
                        step = s2.step
                    r = 0
                    for i in range(first, last, step):
                        self.m[s1][i] = val[r]
                        r += 1
                elif (x,y) == (False, True): # we got instance[k:l, j]
                    length_required = Common.slice_length(s1,self.column_vector(0))
                    if not isinstance(val, list):
                        raise TypeError("list expected as argument")
                    dim1, dim2 = Common.shape(val)
                    if dim1 != 1:
                        raise ValueError("one dimensional array required as argument")
                    if dim2 != length_required:
                        raise ValueError("argument must be a list with length = " + str(length_required))
                    if s1.start == None:
                        first1 = 0
                    else:
                        first1 = s1.start
                    if s1.stop == None:
                        last1 = len(self.m[s1])
                    else:
                        last1 = s1.stop
                    if s1.step == None:
                        step1 = 1
                    else:
                        step1 = s1.step
                        
                    if s2.start == None:
                        first2 = 0
                    else:
                        first2 = s2.start
                    if s2.stop == None:
                        last2 = len(self.m[s1])
                    else:
                        last2 = s2.stop
                    if s2.step == None:
                        step2 = 1
                    else:
                        step2 = s2.step
                    r = 0
                    for i1 in range(first1, last1, step1):
                        c = 0
                        for i2 in range(first2, last2, step3):
                            self.m[i1][i2] = val[r,c]
                            c += 1
                        r += 1
                else: # (False, False) # we got instance[i1:2,j1:j2]
                    if not isinstance(val, list):
                        raise TypeError("list expected as argument")
                    length1_required = Common.slice_length(s1, self.column_vector(0).v)
                    length2_required = Common.slice_length(s2, self.row_vector(0).v)
                    if (length1_required, length2_required) != Common.shape(val):
                        raise ValueError("list with shape " + str(length1_required) + " x " + str(length2_required) + " expected as argument")                            
                        
                    if s1 == None:
                        first = 0
                    else:
                        first = s1.start
                    if s1.stop == None:
                        last = len(self.m[s1])
                    else:
                        last = s1.stop
                    if s1.step == None:
                        step = 1
                    else:
                        step = s1.step
                        
                    r = 0
                    for row in self.m[s1]: 
                        row[s2] = val[r]
                        r += 1       
        else:
            # set single row of matrix as a list
            if not isinstance(val, list):
                raise ValueError("list expected as argument")
            dim1, dim2 = Common.shape(val)
            if dim1 != 1 or dim2 != self.dim2:
                raise ValueError("argument has not the right dimensions (1," + str(self.dim2))
            self.m[arg] = val
            
    # sets a value in matrix. raises ValueError if indices are not in range
    def change_item(self,i,j, val):
        if not i in range(0, self.dim1) or not j in range(0, self.dim2):
            raise ValueError("indices out of range")
        else:
            self.m[i][j] = self.dtype(val)
        
    # multiply all matrix elements with a scalar
    def scalar_product(self, val):
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
    
    def __repr__(self):
        return "dim1 = " + str(self.dim1) + "\ndim2 = " + str(self.dim2) + "\ndtype = " + str(self.dtype) + "\nContent of inner array self.m = " + str(self.m)
        
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
            return self.adjoint_matrix().scalar_product(self.dtype(1) / self.det())
            
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
        if shp[1] != shp[0]:
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
            if (self.dim2 != len(other)) or (other.is_transposed() and self.dim1 != 1):
                raise ValueError("incompatible dimensions of matrix and vector")
            else:
                if not other.is_transposed():
                    v = Vector(self.dim1, dtype = self.dtype, transposed = False)
                    for r in range(0, self.dim1):
                        v[r] = self.row_vector(r) * other
                    return v
                else: # other.transposed and self.dim2 == 1 
                    return self.column_vector(0) * other
        elif isinstance(other, self.dtype):
            return self.scalar_product(other)
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
        
    def is_hermitian(self):
        if self.dim1 != self.dim2:
            raise ValueError("only a squre matrix can be hermitian")
        if self.dtype != complex:
            return self.is_symmetric()
        else:
            for r in range(self.dim1):
                for c in range(self.dim2):
                    if self.m[r][c] != self.m[c][r].conjugate():
                        return False
            return True
            
    # calculate the standard norm for column vectors
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
            
    # create a matrix from vector 
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
        if n == 0:
            return Matrix.identity(len(self), dtype = self.dtype)
        elif n == 1:
            return self
        else:
            m = deepcopy(self)
            for i in range(0, n-1): 
                m = m * self
            return m
    
    # create and initialize a matrix from a list of lists
    def from_list(_list, dtype = float):
        dim1 = len(_list)
        if (dim1 == 0):
            raise ValueError("Initialization list must not be empty")
            
        dim2 = len(_list[0])
        
        value_2D = []
        
        m = Matrix(dim1,dim2,dtype = dtype)
        if (dim1 == 1):
            for i in range(0, len(_list)):
                value_2D.append(_list[i]) 
        else:
            for r in range(0, dim1):
                value_1D = []
                for c in range(0, dim2):
                    value_1D.append(dtype(_list[r][c]))
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
            
        
    # map applies lambda to each element of matrix
    def map(self, lambda_f):
        m = Matrix(self.dim1, self.dim2, dtype = self.dtype)
        for r in range(0, self.dim1):
            for c in range(0, self.dim2):
                m.m[r][c] = lambda_f(self.m[r][c])
        return m
        
    # like map, but with lambda getting called with
    # row, col, value at (row,col) 
    def map2(self, lambda_f):
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
                    m = m.scalar_product(row, 1/pivot)
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
        
    # to determine the rank of a matrix, we need to compute its row echolon form 
    # and count all row vectors that are no null vectors (<=> euclidean norm == 0)
    def rank(self):
        row_echolon = self.echolon()
        count = 0
        for i in range(0, row_echolon.dim1):
            if row_echolon.row_vector(i).euclidean_norm() != 0:
                count += 1
        return count
        
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
        e[0] = u[0].scalar_product(1 / u[0].euclidean_norm())
        u[1] = a[1] - e[0] * (a[1].T()*e[0])
        e[1] = u[1].scalar_product(1 / u[1].euclidean_norm())
        for k in range(2, shape[1]):
            u[k] = a[k]
            for i in range(0,k):
                u[k] -= e[i] * (a[k].T() * e[i])
            e[k] = u[k].scalar_product(1/u[k].euclidean_norm())
        Q = Matrix.from_column_vectors(e)
        R = Matrix(shape[1], shape[0], dtype = self.dtype)
        for i in range(0, shape[1]):
            for j in range(0, shape[0]):
                if i > j: R.m[i][j] = 0
                else:
                    R.m[i][j] = a[j].T() * e[i]
        return (Q,R)
        
    # eigenvalues and eigenvectors calculation using QR decomposition 
    # epsilon is the precision respectively the tolerance
    # i_max defines the maximum number of iterations
    def eigen(self, epsilon = 1E-14, i_max = 1000):
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
        return (A_new.diagonal(), A_new.all_column_vectors())
        
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
        
    # rotation matrices for 2D and 3d 
    # for 3d: rotation around x, y, z, general rotation
    def rotation2D(angle, dtype = float):
        m = Matrix(2,2,dtype)
        m.m[0][0] =  dtype(math.cos(angle))
        m.m[0][1] =  dtype(-math.sin(angle))
        m.m[1][0] =  dtype(math.sin(angle))
        m.m[1][1] =  dtype(math.cos(angle))
        return m
        
    def rotation3d_x(angle_x, dtype=float):
        m1 = Matrix(3,3,dtype)
        m1.m[0][0] = dtype(1)
        m1.m[0][1] = dtype(0)
        m1.m[0][2] = dtype(0)
        m1.m[1][0] = dtype(0)
        m1.m[1][1] = dtype(math.cos(angle_x))
        m1.m[1][2] = dtype(-math.sin(angle_x))
        m1.m[2][0] = dtype(0)
        m1.m[2][1] = dtype(math.sin(angle_x))
        m1.m[2][2] = dtype(math.cos[angle_x])
        return m1
        
    def rotation3d_y(angle_y, dtype=float):
        m2 = Matrix(3,3,dtype)
        m2.m[0][0] = dtype(math.cos(angle_y))
        m2.m[0][1] = dtype(0)
        m2.m[0][2] = dtype(math.sin(angle_y))
        m2.m[1][0] = dtype(0)
        m2.m[1][1] = dtype(1)
        m2.m[1][2] = dtype(0)
        m2.m[2][0] = dtype(-math.sin(angle_y))
        m2.m[2][1] = dtype(0)
        m2.m[2][2] = dtype(math.cos(angle_y))
        return m2
        
    def rotation3d_z(angle_z, dtype = float):
        m3 = Matrix(3,3,dtype)
        m3.m[0][0] = dtype(math.cos(angle_z))
        m3.m[0][1] = dtype(-math.sin(angle_z))
        m3.m[0][2] = dtype(0)
        m3.m[1][0] = dtype(math.sin(angle_z))
        m3.m[1][1] = dtype(math.cos(angle_z))
        m3.m[1][2] = dtype(0)
        m3.m[2][0] = dtype(0)
        m3.m[2][1] = dtype(0)
        m3.m[2][2] = dtype(1)
        return m3
        
    def rotation3D(angle_x, angle_y, angle_z, dtype = float):
        rot_x = Matrix.rotation3d_z(angle_x, dtype)
        rot_y = Matrix.rotation3d_y(angle_y, dtype)
        rot_z = Matrix.rotation3d_z(angle_z, dtype)
        return rot_z @ rot_y @ rot_x
    
    # remove a row from self and return result as a new matrix 
    def remove_row(self, i):
        (dim1, dim2) = self.shape()
        if not i in range(0, dim1):
            raise ValueError("row does not exist")
        else:
            dim1 -= 1
            row_vectors = self.all_row_vectors()
            row_vectors.remove(row_vectors[i])
            return Matrix.from_row_vectors(row_vectors)
            
    # remove a column from self and return result as a new matrix
    def remove_column(self, j):
        (dim1, dim2) = self.shape()
        if not j in range(0, dim2):
            raise ValueError("column does not exist")
        else:
            dim2 -= 1
            column_vectors = self.all_column_vectors()
            column_vectors.remove(column_vectors[j])
            return Matrix.from_column_vectors(column_vectors)
            
    # remove multiple rows from a matrix
    def remove_rows(self, row_list):
        m = deepcopy(self)
        if row_list == []: return m
        rl = row_list
        rl.sort()
        completed = False
        while not completed:
            r = rl[0]
            m = m.remove_row(r)
            rl.remove(r)
            if rl == []: completed = True
            else:
                for i in range(0,len(rl)):
                    rl[i] -= 1
        return m
        
    # remove multiple columns from a matrix
    def remove_columns(self, col_list):
        m = deepcopy(self)
        if col_list == []: return m
        cl = col_list
        cl.sort()
        completed = False
        while not completed:
            c = cl[0]
            m = m.remove_column(c)
            cl.remove(c)
            if cl == []: completed = True
            else:
                for i in range(0,len(cl)):
                    cl[i] -= 1
        return m
        
    # calculate the power of matrix m - pow(m,1) == m,  
    # pow(m,2) = m @ m, ...
    def __pow__(self, arg):
        if not isinstance(arg, int):
            raise TypeError("only ints are allow in Matrix.__pow__")
        if arg < 0:
            raise ValueError("argument to __pow__ must be >= 0")
        return self.mult_n_times(arg)
        
    # methods to add transposed      vectors to all or a subset of rows 
    # methods to add not transposed  vectors to all or a subset of cols
        
    def add_vector_to_all_rows(self, vector):
        dim1, dim2 = self.shape()
        if len(vector) != dim2:
            raise ValueError("size of vector and dim2 do not match")
        elif not vector.is_transposed():
            raise ValueError("vector must be transposed to be added to matrix")
        else:
            result = deepcopy(self)
            for r in range(0, dim1):
                for c in range(0, dim2):
                    result.m[r][c] += vector[c]
            return result
        
    # rows must be iterable
    def add_vector_to_rows(self, rows, vector):
        dim1, dim2 = self.shape()
        if len(vector) != dim2:
            raise ValueError("size of vector and dim2 do not match")
        elif not vector.is_transposed():
            raise ValueError("vector must be transposed to be added to matrix")
        else:
            result = deepcopy(self)
            for r in rows:
                for c in range(0, dim2):
                    result.m[r][c] += vector[c]
            return result
            
    def add_vector_to_all_columns(self, vector):
        dim1, dim2 = self.shape()
        if len(vector) != dim1:
            raise ValueError("size of vector and dim1 do not match")
        elif vector.is_transposed():
            raise ValueError("vector must not be transposed to be added to matrix")
        else:
            result = deepcopy(self)
            for c in range(0, dim2):
                for r in range(0, dim1):
                    result.m[r][c] += vector[r]
            return result
    
    # cols mut be iterable    
    def add_vector_to_columns(self, cols, vector):
        dim1, dim2 = self.shape()
        if len(vector) != dim1:
            raise ValueError("size of vector and dim1 do not match")
        elif not vector.is_transposed():
            raise ValueError("vector must be transposed to be added to matrix")
        else:
            result = deepcopy(self)
            for c in cols:
                for r in range(0, dim1):
                    result.m[r][c] += vector[c]
            return result           
            
    # generators for rows and columns
    # they return row or column vectors
    def iter_rows(self):
        for r in range(0, self.dim1):
            yield deepcopy(self.row_vector(r))
            
    def iter_cols(self):
        for c in range(0, self.dim2):
            yield deepcopy(self.column_vector(c))
            
    def iter_rows_in_range(self, rowrng):
        for r in rowrng:
            yield deepcopy(self.row_vector(r))
            
    def iter_cols_in_range(self, colrng):
        for c in colrng:
            yield deepcopy(self.column_vector(c))
            
    # computes the mean of all column vectors if axis == 0, 
    # and the mean of all row vectors, otherwise
    # returns the list of means
    def mean(self, axis = 0):
        res = []
        if axis == 0:
            for c in range(0, self.dim2):
                res.append(self.column_vector(c).mean())
        else: # axis == 1 
            for r in range(0, self.dim1):
                res.append(self.row_vector(r).mean())
        return res
         
    # this methods determines the sum of all columns or 
    # rows depending on axis
    def sum(self, axis = 0):
        return Common.sum(self.m, axis)
        
    # concatenation of the rows or columns of matrix other
    # to matrix self depending on axis
    def concatenate(self, other, axis = 0):
        array = Common.concatenate(self.m, other.m, axis)
        m = Matrix.from_list(array, dtype = self.dtype)
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
    def set_format(s):
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
            
    def __repr__(self):
        return "len = " + str(len(self)) + "\ntransposed = " + str(self.is_transposed()) + "\ndtype = " + str(self.dtype) + "\nContent of inner array self.v = " + str(self.v)
    
    
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
    def __mul__(self, other):
        if isinstance(other, Matrix):
            m = Matrix.from_vector(self)
            return m * other
        elif isinstance(other, self.dtype):
            return self.scalar_product(other)
        if not isinstance(other, Vector):
            raise TypeError("other object must also be a vector")
        if len(other) != len(self):
            raise ValueError("incompatible lengths of vectors")
        else:
            if self._transposed:
                if not other._transposed:
                    return self.cross_product(other)
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
    # non-transposed vector
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
    def cross_product(self, other):
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
    
    # normalizing a vector
    def normalize(self):
        v = deepcopy(self)
        norm = self.dtype(v.euclidean_norm())
        if norm > 0:
            v = v.scalar_product(1/norm)
        else:
            raise ValueError("vector with euclidean norm 0 cannot be normalized")
        return v
            
    # multiply all vector elements with a scalar
    def scalar_product(self, scalar):
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
            return self.cross_product(other) == 0
    
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
        
    # get a new Vector by removing elements from self
    # indices to remove are passed as the indices list
    def remove(self, indices):
        v = deepcopy(self)
        if indices == []: 
            return v
        else:
            arr = []
            for i in range(0, len(self)):
                if i in indices: continue
                else: arr.append(self.v[i])
            m = Vector.from_list(arr, dtype = self.dtype) 
            return m
                
    # generator for vector elements
    def iter_vec(self):
        for i in range(0, len(self)):
            yield self.v[i]
            
    def iter_vec_in_range(self, rng):
        for i in rng:
            yield self.v[i]
    
    def mean(self):
        return Common.mean(self.v)
        
    # map applies lambda to each element of vector
    def map(self, lambda_f):
        v = Vector(len(self), dtype = self.dtype)
        for i in range(0, len(self)):
            v[i] = lambda_f(self[i])
        return v
        
    # same as map, but with additional vector position passed to 
    # lambda
    def map2(self, lambda_f):
        v = Vector(len(self), dtype = self.dtype)
        for i in range(0, len(self)):
            v[i] = lambda_f(i, self[i])
        return v
        
#################################################
################ class Polynomial ###############
#################################################       

class Polynomial:
    def _gcd(a, b):
        while b != 0:
            t = b
            b = a % b
            a = t
        return a
    
    # calculate gcd for multiple numbers
    def _gcdmult(args):
        if len(args) == 1:
            return args[0]
        elif len(args) == 2:
            return Polynomial._gcd(args[0], args[1])
        elif len(args) == 0:
            return None
        else:
            g = Polynomial._gcd(args[0], args[1])
            for i in range(2, len(args)):
                g = Polynomial._gcd(g, args[i])
            return g
    
    # the list a contains the coefficients of the 
    # polynomial starting with a0, a1, ... 
    # p(x) is then a0 + a1*x + a2*x^2 + a3*x^3 + ...
    def __init__(self, a):
        if a == None:
            raise TypeError("None is not permitted as initializer")
        elif a == []: 
            raise ValueError("initializer must not be empty")
        else:
            self.a = a           
    
    #returns the coefficients of p with increasing i (x^i)
    def coeffs(self):
        if len(self) == 0:
            return [] 
        else:
            result = [] 
            for i in range(0,len(self)):
                result.append(a[i])
            return result  
            
    # calculates the polynomial height
    def height(self):
        return max(self.a)
        
    # calculates the p-norm = pow(sum(abs(coefficients^p)), 1/p)
    def p_norm(self, p):
        if not isinstance(p, int) or p < 1:
            raise ValueError("p must be an integer >= 1")
        else:
            print(self)
            sum = 0 
            for i in range(0, len(self)):
                sum += abs(self.a[i])**p 
            return pow(sum, 1/p)
            
                
    # returns the ith element of the polynomial a_i * x^i 
    # as result
    def factor(self, idx):    
        if idx not in range(0, len(self.a)):
            raise ValueError("index out of range")
        result = [] 
        if self.degree() == 0:
            return Polynomial([self.a[0]])
        for i in range(0, idx):
            result.append(0)
        result.append(self.a[idx])
        return Polynomial(result)
            
    # returns a polynomial that only consists of coeff*x^i
    def single_p(coeff, i):
        if i == 0:
            return Polynomial([coeff])
        else:
            if coeff == 0:
                return Polynomial([0])
            else:
                result = []
                for j in range(0, i): result.append(0)
                result.append(coeff)
                return Polynomial(result)
    
    # divide coefficients by common gcd (=> _gcd_mult())
    def normalize_gcd(self):
        g = Polynomial._gcdmult(self.a)
        if g == 1:
            return deepcopy(self)
        else:
            result = deepcopy(self)
            for i in range(0, len(result)):
                result[i] /= g
            return result
            
    # polynom will be divided by the coefficient of the highest 
    # ranked x^î which is self.degree()
    def normalize_coeff0(self):
        a = deepcopy(self)
        coeff0 = a.a[a.degree()] 
        if coeff0 == 1:
            return a
        else:
            res = []
            for i in range(0, len(a)):
                a.a[i] /= coeff0
            return a
        
    def linear_p(a0, a1):
        if a1 != 0 and a0 != 0:
            return Polynomial([a0,a1])
        
    # computes the value of the polynom at position x
    def compute(self, x):
        res = 0.0
        for i in range(0, len(self.a)):
            res += self.a[i] * pow(x,i)
        return res
        
    # degree of the highest ranked x^i in the polynom
    def degree(self): 
        return len(self)-1
        
    # returns a triple (number of results, result1, result2)
    # if no result exist (0, None, None) will be returned
    # if one result exists (1, result, result) will be returned
    # if two results exist (2, result1, result2) will be returned
    def solve_quadr_eq(self):
        if self.degree() != 2:
            raise ValueError("only quadratic polynomials are allowed")
        else:
            a = self[2]
            b = self[1]
            c = self[0]
            delta = b*b-4*a*c
            if delta < 0:
                return (0, None, None)
            elif delta == 0:
                result = -b / (2*a)
                return (1, result, result)
            else:
                sol1 = (-b + math.sqrt(delta)) / (2*a)
                sol2 = (-b - math.sqrt(delta)) / (2*a)
                return (2, sol1, sol2)
            
    # get single coefficient of polynom
    def __getitem__(self, arg):
        return self.a[arg]
        
    # set single coefficient of polynom
    def __setitem__(self, arg, val):
        self.a[arg] = val
        
    # derivation of a polynom     
    def derivation(self):
        res = []
        pos = 0
        for i in range(0, len(self.a)-1):
            if self.a[i+1] == 0:
                res.append(0)
            else:
                res.append((i+1) * self.a[i+1])
        return Polynomial(res)
        
    # Integration of polynomial 
    def integral(self):
        res = [0 for i in range(0, len(self.a) + 1)] 
        for i in range(0, self.degree() + 1):
            if self.a[i] == 0:
                continue
            else:
                res[i+1] = self.a[i] / (i+1)
        return Polynomial(res)
        
        
    # addition of polynoms
    def __add__(self, other):
        a = []
        if len(self.a) >= len(other.a):
            for i in range(0, len(other.a)):
                a.append(self.a[i] + other.a[i])
            for i in range(len(other.a), len(self.a)):
                a.append(self.a[i])
        else:
            for i in range(0, len(self.a)):
                a.append(self.a[i] + other.a[i])
            for i in range(len(self.a), len(other.a)):
                a.append(other.a[i])
        return Polynomial(a)
        
    # getting length of polynomial
    def __len__(self):
        return len(self.a)
        
    # helper method for __mul__
    def _max_index(self):
        last_i = 0
        for i in range(0, len(self)):
            if self.a[i] != 0:
                last_i = i
        return last_i
        
    # multiplication of polynoms
    def __mul__(self, other):
        m1 = self._max_index()
        m2 = other._max_index()
        arr = [0 for i in range(0, m1+m2+1)]
        for i in range(0, len(self)):
            for j in range(0, len(other)):
                arr[i+j] += self.a[i] * other.a[j]
        result = Polynomial(arr)
        return result
    
    # power of polynomials
    def __pow__(self, factor):
        if not isinstance(factor, int):
            raise ValueError("only integers >= 0 are allowd in pow()")
        elif (factor == 0):
            return Polynomial([1])
        elif (factor == 1):
            return self
        elif (factor == 2):
            return self * self
        else:
            result = self * pow(self, factor - 1)
            return result
            
    # division of polynoms
    # returns a tuple consisting of (quotient, remainder)
    def __truediv__(self, other):
        p1 = deepcopy(self)
        p2 = deepcopy(other)
        p1.a.reverse()
        p2.a.reverse()
        nominator = p1.a
        denominator = p2.a
        result = list(nominator) # Copy the dividend
        normalizer = denominator[0]
        for i in range(len(nominator)-(len(denominator)-1)):
            result[i] /= normalizer 
            coefficient = result[i]
            if coefficient != 0:
                for j in range(1, len(denominator)): 
                    result[i + j] += -denominator[j] * coefficient
        separator = -(len(denominator)-1)
        a1 = result[:separator]
        a1.reverse()
        a2 = result[separator:]
        a2.reverse()
        return Polynomial(a1), Polynomial(a2) 
            
    # negation of polynom, for example, x => -x
    def __neg__(self):
        neg = deepcopy(self)
        for i in range(0, len(neg.a)):
            neg.a[i] = -neg.a[i]
        return neg
        
    # identity
    def __pos__(self):
        return deepcopy(self)
        
    # subtraction of polynoms is delegated to negation and addition
    def __sub__(self,other):
        return self.__add__(-other)
        
    # check for equality
    def __eq__(self, other):
        if len(self.a) != len(other.a):
            return False
        else:
            for i in range(0, len(self.a)):
                if self.a[i] != other.a[i]:
                    return False
            return True 
            
            
    # check for inequality
    def __ne__(self, other):   
        return not self == other
            
    # string representation pf a polynom
    def __str__(self):
        res =""
        for i in range(0, len(self.a)):
            if i == 0:
                if self.a[0] != 0:
                    res += str(self.a[i])
            elif i == 1:
                if self.a[i] != 0:
                    if res == "": 
                        if self.a[i] == 1:
                            res += "x"
                        else:
                            res += str(self.a[i]) + "*x"
                    else:
                        if self.a[i] == 1:
                            res += " + x"
                        else:
                            res += " + " + str(self.a[i]) + "*x"
            else:
                if self.a[i] != 0:
                    if res == "": 
                        if self.a[i] == 1:
                            res += "x^" + str(i)
                        else:
                            res += str(self.a[i]) + "*x^" + str(i)
                    else:
                        if self.a[i] == 1:
                            res += " + x^" + str(i)
                        else:
                            res += " + " + str(self.a[i]) + "*x^" + str(i)
        return res 
        
    # calculation of roots of a polynomial 
    # using the Durand-Kerner method       
    # Arguments:
    # max_iter: how many iterations should the algorithm use 
    # epsilon:  what is the deviation of the imaginary part 
    #           w.r.t. 0
    #           if z.imag < epsilon,the complex number will  
    #           be transformed to a float (z.real)
    def roots(poly, max_iter = 50, epsilon = 1E-4):
        degree = poly.degree()
        p_old = [complex(0.4, 0.9) ** i for i in range(0, degree)]
        p_new = [0 for i in range(0, degree)]
        i = 0 
        while i < max_iter:
            for j in range(0, degree): # for all p,q,r,s ... 
                prod = 1 
                for k in range(0,degree):
                    if k == j: continue 
                    elif k < j:
                        prod *= p_old[j]-p_old[k]
                    else:
                        prod *= p_old[j]-p_new[k]
                p_new[j] = p_old[j] - poly.compute(p_old[j])/prod
            p_old = deepcopy(p_new)
            i += 1 
        for i in range(0,len(p_new)):
            if abs(p_new[i].imag) < epsilon:
                p_new[i] = float(p_new[i].real)
        return p_new 
        
    # given a sequence of roots this method calculates the 
    # corresponding polynomial    
    def poly_from_roots(*args):
        p0 = Polynomial([1])
        for arg in args:
            p0 = p0 * Polynomial([-arg,1])
        return p0
        

#################################################
################ class Rational #################
#################################################       


"""
class Rational implements rational numbers
for Python. It provides many useful operators,
methods to convert floats or ints to rational
numbers, and vice versa. It includes a 
method to obtain a rational representation
of e (Euler number). 

"""

def gcd(a, b):
    while b != 0:
        t = b
        b = a % b
        a = t
    return a
    
# convert number into a list of digits, for example 123 in [1,2,3]
def getDigits(n):
    list = []
    s = str(n)
    for i in range(0, len(s)):     
        list.append(int(s[i]))
    return list
    
# calc number of digits of number, for example, 4355 has 4 digits
def getLength(n):
    return len(getDigits(n))

#check if a number does not contain the digit 0
def checkForValidity(n):
    return not 0 in getDigits(n)
    
# helper function for convertig floats, periods, fractions 
# to rational numbers.
# for example denom(12345) == 99999, denom(2) == 9
def denom(period):
    sum = 0
    for i in range(0, getLength(period)): sum = sum * 10 + 9
    return sum

def fac(n):
    res = 1
    for i in range(2, n+1): res *= i
    return res

class Rational:
    def __init__(self, nom, denom):
        assert denom != 0, "denominator must be <> 0"
        gcd_nd = gcd(nom,denom)
        self.nom = int(nom / gcd_nd)
        self.denom = int(denom / gcd_nd)
        
    def nominator(self):
        return self.nom
        
    def denominator(self):
        return self.denom
        
    def __add__(self, r):
        x = self.nom * r.denom
        y = r.nom * self.denom
        n = x + y
        d = self.denom * r.denom
        return Rational(n, d)
        
    def __mul__(self, r):
        n = self.nom * r.nom
        d = self.denom * r.denom
        return Rational(n, d)
        
    def __truediv__(self, r):
        return self * r.reciprocal()
        
    def reciprocal(self):
        assert self.nom != 0; "Invertion with 0 nominator is not allowed"
        return Rational(self.denom, self.nom)
        
    def __neg__(self):
        return Rational(-self.nom, self.denom)
        
    def __pos__(self):
        return self 
        
    def __sub__(self,r):
        return self + -r
        
    def __eq__(self, r):
        return self.nom == r.nom and self.denom == r.denom
        
    def __ne__(self, r):
        return not self == r
        
    def __gt__(self, r):
        tmp = self - r 
        return tmp.nom > 0 and tmp.denom  > 0 
        
    def __ge__(self, r):
        return self > r or self == r
        
    def __lt__(self, r):
        return r > self
        
    def __le__(self, r):
        return self < r or self == r
        
    def __str__(self):
        if self.denom == 1:
            return str(int(self.nom))
        else:
            return str(int(self.nom)) + " / " + str(int(self.denom))
            
    def __repr__(self):
        return str(self.nom) + " / " + str(self.denom)
        
    def __pow__(self, exp):
        e = int(exp)
        return Rational(pow(self.nom, e), pow(self.denom, e))
        
    def __invert__(self):
        return self.reciprocal()
        
    # converts a rational number to an integer (heavy loss of precision)
    def __int__(self):
        return int(self.nom // self.denom)
        
    # converts a rational to float (mostly harmless, medium loss of precision)    
    def __float__(self):
        return 1.0 * self.nom / self.denom
        
    ################## class functions ##################
        
    def zero():
        return Rational(0, 1)
        
    def one():
        return Rational(1,1)
        
    def onehalf():
        return Rational(1,2)
        
    def onethird():
        return Rational(1,3)
        
        
    # calculates an approximation of Eulers number as rational
    # number. digits: number of Taylor nodes to be calculated
    def e(digits):
        assert digits > 0, "zero digits is not allowed"
        sum = Rational.one()
        for i in range(1,digits + 1): 
            sum += Rational(1,fac(i))
        return(sum)

    
    # convert a period like 3 (equivalent to 0.33333333) to a 
    # rational number. The argument leadingzeroes specifies
    # how many zeros there are between the decimal point and
    # the start of the period. for example, 0.0333333 has a 
    # period of 3 and a leadingzeros of 1
    def periodToRational(period, leadingzeros = 0):
        assert leadingzeros >= 0, "leadingzeros must be >= 0"
        assert checkForValidity(period), "period is not allowed to contain 0"
        return Rational(period, pow(10, leadingzeros)*denom(period))
        
    def intToRational(n):
        return Rational(n, 1)
        
    
    # creates an rational number that has the value 0.fraction
    # Number of leading zeros in leadingzeros: if for example, 
    # the number is 0.000fraction leadingzeros is 3
    def fractionToRational(n, leadingzeros = 0):
        if n == 0: 
            return Rational.zero()
        else:
            list = []
            s = str(n)
            for i in range(0, len(s)): list.append(int(s[i]))
            return Rational(n, pow(10, leadingzeros + len(list)))
        
    def floatToRational(x, digits=0):
        left = abs(int(f))
        right = abs(f - int(f))
        r_left = Rational.intToRational(left)
        zeros = 0
        tmp = right
        while (int(tmp) == 0):
            zeros += 1
            tmp = 10 * tmp
        zeros -= 1
        tmp = tmp / 10 * pow(10, digits)
        r_right = Rational.fractionToRational(int(tmp), leadingzeros = zeros)
        if x >= 0:
            return r_left + r_right
        else:
            return -r_left + r_right
            
            
    # for example, in the float 2.00125879879879 we would use 
    # number = 2, leadingzeros = 2, fraction = 125, period = 879
    def periodicFloatToRational(number, fraction, leadingzeros, period):
        r1 = Rational.intToRational(number)
        if fraction == 0:
            r2 = Rational.periodToRational(period,leadingzeros)
        else:
            r2 = Rational.periodToRational(period, leadingzeros+getLength(fraction))
        r3 = Rational.fractionToRational(fraction, leadingzeros)
        return r1+r2+r3
