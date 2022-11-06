
##############################################################
# This is an implementation of Matrix and Vector datatypes 
# in Python. 
# It is pretty useless given the fact that there are already
# much better solutions available such as pandas, numpy
##############################################################

import math 
from copy import deepcopy


################## class Matrix #################

class Matrix:
    
    # create size1 x size2 matrix with optional initial values
    def __init__(self, size1, size2, init_value = 0):
        if size1 <= 0 or size2 <= 0: raise ValueError("a matrix must have positive dimensions")
        self.m = [[] for r in range(0,size1)]
        self.dim1 = size1
        self.dim2 = size2
        for i in range(0, size1):
            for j in range(0, size2):
                self.m[i].append(init_value)
                
    # clone matrix
    def clone(self):
        return deepcopy(self)
                
    # transpose the matrix
    def T(self):
        m = Matrix(self.dim2, self.dim1)
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
        
            
    # set a position in matrix
    def change_item(self,i,j, val):
        if not i in range(0, self.dim1) or not j in range(0, self.dim2):
            raise ValueError("indices out of range")
        else:
            self.m[i][j] = val
        
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
            line = "|"
            for c in range(0, self.dim2):
                line += " " + str(self.m[r][c])
                if c != self.dim2-1:
                    line +=  "," 
                else:
                    line += " |\n"
            s += line
        return s
        
    # get a submatrix by leaving out all elements from row i and col j
    def minor(self, i, j):
        if not i in range(0, self.dim1) or not j in range(0, self.dim2):
            raise ValueError("out of range for indices")
        m = Matrix(self.dim1-1, self.dim2-1)
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
    def det(self):
        if self.dim1 != self.dim2:
            raise ValueError("Determinants can only be calculated for quadratic matrices")
        if self.dim1 == 1:
            return self.m[0][0]
        else: # developing around 0,0 
            det = 0
            for c in range(0, self.dim1):
                if c % 2 == 0:
                    factor =  1
                else:
                    factor = -1
                det += factor * self.m[0][c] * self.minor(0, c).det()
            return det 
    
    def cofactor(self, i, j):
        cof_ij = self.minor(i,j).det()
        if (i+j) % 2 == 0:
            return cof_ij
        else:
            return -cof_ij
            
    def cofactor_matrix(self):
        m = Matrix(self.dim1, self.dim2)
        for r in range(0, self.dim1):
            for c in range(0, self.dim2):
                m[r][c] = self.cofactor(r,c)
        return m
        
    def adjoint_matrix(self):
        return self.cofactor_matrix().T()
        
    def inverse_matrix(self):
        if self.det() == 0:
            raise ValueError("matrix with det == 0 has no inverse")
        else:
            return self.adjoint_matrix().mult_with_scalar(1 / self.det())
            
    def solve(self, vector):
        if self.det() == 0:
            raise ValueError("det == 0")
        if (vector.is_transposed()):
            raise ValueError("Vector must not be transposed")
        if not self.is_quadratic():
            raise ValueError("Matrix must be quadratic")
        if len(vector) != self.dim2:
            raise ValueError("dimensions of matrix and vector do not match")
        return self.inverse_matrix() * vector
            
        
    # get row vector 
    def row_vector(self, row):
        v = Vector(self.dim2, transposed = True)
        for c in range(0, self.dim2):
            v[c] = self.m[row][c]
        return v
            
    # get column vector
    def col_vector(self, col):
        v = Vector(self.dim1, transposed = False)
        for r in range(0, self.dim1):
            v[r] = self.m[r][col]
        return v
        
    # get Unit matrix for given size
    def identity(size):
        dim1 = size
        dim2 = size
        m = Matrix(size,size)
        for r in range(0, size):
            for c in range(0, size):
                if r == c:
                    m.m[r][c] = 1
                else:
                    m.m[r][c] = 0
        return m   
        
    # add two matrices with each other
    def __add__(self, other):
        if self.dim1 != other.dim1 and self.dim2 != other.dim2:
            raise ValueError("matrices must have similar sizes")
        else:
            m = Matrix(self.dim1, self.dim2)
            for r in range(0, self.dim1):
                for c in range (0, self.dim2):
                    m.m[r][c] = self.m[r][c] + other.m[r][c]
            return m
            
    # matrix multiplication
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
                m = Matrix(self.dim1, other.dim2)
                for r in range(0, self.dim1):
                    for c in range(0, other.dim2):
                        m.m[r][c] = self.row_vector(r) * other.col_vector(c)
                return m
        elif isinstance(other, Vector):
            if (not other.is_transposed() and self.dim1 != len(other)) or (other.is_transposed() and self.dim1 != 1):
                raise ValueError("incompatible dimensions of matrix and vector")
            else:
                if not other.is_transposed():
                    v = Vector(self.dim1, False)
                    for r in range(0, self.dim1):
                        value = 0
                        for k in range(0, self.dim2):
                            value += self.m[r][k] * other[k]
                        v[r] = value
                    return v
                else: # other.transposed and self.dim2 == 1
                    sum = 0
                    for k in range(0, len(other)):
                        sum += self.m[0][k] * other[k]
                    return sum
        else:
            raise ValueError("second argument must be matrix or vector")
            
            
    # subtracting one matrix from the other
    def __sub__(self, other):
        if self.dim1 != other.dim1 and self.dim2 != other.dim2:
            raise ValueError("matrices must have similar sizes")
        else:
            m = Matrix(self.dim1, self.dim2)
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
        
    # check whether matrix is symmetric, i.e., whether m == m.T
    def is_symmetric(self):
        if self.dim1 != self.dim2:
            raise ValueError("symmetry not defined for non-quadratic matrices")
        else:
            return self == self.T()
            
    # check for quadratic matrix
    def is_quadratic(self):
        return self.dim1 == self.dim2
            
    # calculate the standard norm
    def norm(self):
        n = 0
        for c in range(0, self.dim2):
            n = max(n, self.col_vector(c).norm())
        return n
            
    # create a matrix from a vector
    def from_vector(v):
        if v.is_transposed():
            m = Matrix(1, len(v))
            for i in range(0, len(v)): m.m[0][i] = v[i]
        else:
            m = Matrix(len(v),1)
            for i in range(0, len(v)): m.m[i][0] = v[i]
        return m
        
    # build absolute values of all matrix elements
    def __abs__(self):
        m = Matrix(self.dim1, self.dim2)
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
    def from_list(list):
        dim1 = len(list)
        dim2 = len(list[0])
        m = Matrix(dim1,dim2)
        m.m = list
        return m
        
    # return list of list of all matrix elements
    def to_list(self):
        return self.m
        
    # apply lambda to each element of matrix
    def apply(self, lambda_f):
        m = Matrix(self.dim1, self.dim2)
        for r in range(0, self.dim1):
            for c in range(0, self.dim2):
                m.m[r][c] = lambda_f(self.m[r][c])
        return m
        
    # like apply, but with lambda getting called with
    # row, col, value at (row,col) 
    def apply2(self, lambda_f):
        m = Matrix(self.dim1, self.dim2)
        for r in range(0, self.dim1):
            for c in range(0, self.dim2):
                m.m[r][c] = lambda_f(r, c, self.m[r][c])
        return m
        
            
                
                
 ################## class Vector #################
                
                                              
class Vector:
    
    _transposed = False
    
    # Initialize vector with size, initial values for
    # its elements and initial transposition-state
    def __init__(self, size, transposed = False, init_value = 0):
        self.v = [init_value for i in range(0,size)]
        self._transposed = transposed
        
    # clone vector
    def clone(self):
        return deepcopy(self)
        
    # check whether vector is transposed
    def is_transposed(self):
        return self._transposed
        
    # string representation of vector
    def __str__(self):
        if self._transposed:
            res = "<"
            for i in range(0, len(self.v)): 
                res += str(self.v[i])
                if i != len(self.v)-1: res += ","
            res += ">"
            return res
        else:
            res = "\n"
            for i in range(0, len(self.v)):
                res += "[" + str(self.v[i]) + "]\n"
            return res
    
    # vector transposition        
    def T(self):
        v = Vector(len(self), transposed = not self.is_transposed())
        for i in range(0, len(self)): 
            v[i] = self[i]
        return v
        
    # return length of vector
    def __len__(self):
        return len(self.v)
        
    # calculate absolute value of all elements
    def __abs__(self):
        res = Vector(len(self))
        for i in range(0, len(self)):
            res[i] = abs(self[i])
        return res
                   
    # access vector element, for example, x = v[6]
    # also allows slicing such as vec[0:] or v[0:2, 4:]
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
        
    # change vector element, for example: v[5] = 1
    def __setitem__(self, i, value):
        if i < 0 or i >= len(self.v):
            raise ValueError("index out of range")
        self.v[i] = value
        
    # multiplication of vectors 
    # vector multiplication with matrices is delegated
    # to class Matrix
    def __mul__(self, other):
        if isinstance(other, Matrix):
            m = Matrix.from_vector(self)
            return m * other
        if not isinstance(other, Vector):
            raise ValueError("other object must also be a vector")
        if len(other) != len(self):
            raise ValueError("incompatible lengths of vectors")
        else:
            if self._transposed:
                if not other._transposed:
                    return self.scalar_product(other)
                else:
                    v = Vector(len(self), self._transposed)
                    for i in range(0, len(self)):
                        v[i] = self[i] * other[i]
                    return v
            else: # not self.transposed  
                if other._transposed:
                    mat = Matrix(len(self), len(other))
                    for r in range(0, len(self)):
                        for c in range(0, len(other)):
                            mat.m[r][c] = self[r]*other[c]
                    return mat       
                else: 
                    v = Vector(len(self), self._transposed)
                    for i in range(0, len(self)):
                        v[i] = self[i] * other[i]
                    return v
    
    # add two vectors with each other
    def __add__(self, other):
        if len(other) != len(self):
            raise ValueError("incompatible lengths of vectors")
        elif self._transposed != other._transposed:
            raise ValueError("transposed and not transposed vectors cannot be added")
        else:
            res = Vector(len(self), self._transposed)
            for i in range(0, len(self)): res[i] = self[i] + other[i]
            return res
            
    # negative vector: all elements switch their sign
    def __neg__(self):
        res = Vector(len(self), self._transposed)
        for i in range(0, len(self)): res[i] = -self[i]
        return res
        
    # positive vector: nothing changes
    def __pos__(self):
        res = Vector(len(self), self._transposed)
        for i in range(0, len(self)): res[i] = +self[i]
        return res
        
    # build scalar product of two vectors
    def scalar_product(self, other):
        if len(self) != len(other):
            raise ValueError("incompatible lengths of vectors")
        else:
            res = 0
            for i in range(0,len(self)): res += self[i]*other[i]
            return res
            
    # subtract one vector from the other
    def __sub__(self, other):
        if len(other) != len(self):
            raise ValueError("incompatible lengths of vectors")
        elif self._transposed != other._transposed:
            raise ValueError("transposed and not transposed vectors cannot be subtracted from each other")
        else:
            res = Vector(len(self))
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
        res = 0.0
        for i in range(0,len(self)):
            res += self[i] * self[i]
        return math.sqrt(res)
        
    # get regular norm of vector 
    def norm(self):
        res = 0.0
        for i in range(0,len(self)):
            res += abs(self[i])
        return res
            
    # multiply all vector elements with a scalar
    def mult_with_scalar(self, scalar):
        res = Vector(len(self))
        for i in range(0, len(self)): res[i] = self[i] * scalar
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
    def unit_vector(size, i):
        v = Vector(size)
        for j in range(0,i): v[j] = 0
        v[i] = 1
        for j in range(i+1, size): v[j] = 0
        return v
        
    # retrieve all unit/base vectors for dimension = size
    def all_unit_vectors(size):
        vec_arr = []
        for i in range(0, size):
                vec_arr.append(Vector.unit_vector(size,i))
        return vec_arr
        
    # create a vector from a list
    def from_list(list, transposed = False):
        v = Vector(len(list), transposed)
        for i in range(0, len(v)): v[i] = list[i]
        return v
    
    # return list of all vector elements
    def to_list(self):
        return self.v
        
    # apply lambda to each element of vector
    def apply(self, lambda_f):
        v = Vector(len(self))
        for i in range(0, len(self)):
            v[i] = lambda_f(self[i])
        return v
        
    # same as apply, but with additional vector position passed to 
    # lambda
    def apply2(self, lambda_f):
        v = Vector(len(self))
        for i in range(0, len(self)):
            v[i] = lambda_f(i, self[i])
        return v

            
        
        

