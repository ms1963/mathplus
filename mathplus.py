
"""
mathplus.py is licensed under the
GNU General Public License v3.0

Permissions of this strong copyleft license are conditioned on making available complete source code of licensed works and modifications, which include larger works using a licensed work, under the same license. Copyright and license notices must be preserved. Contributors provide an express grant of patent rights.
"""

##############################################################
# This is an implementation of Matrix and Vector datatypes 
# in Python. In addition, classes for rational numbers and 
# Polynomials, and more are included.
# It may seem unnececessary given the fact that there are 
# already much better solutions available such as numpy,
# but offers a pure Python solution for those who do not
# need the full-blown functionality of numpy
##############################################################

from __future__ import print_function
from __future__ import division
import math 
from copy import deepcopy, copy
import random
from random import uniform, randrange, seed, shuffle
from functools import reduce
import operator
import numbers
import time
import matplotlib.pyplot as plt
import csv
import numpy as np # used to transfer array between 
                   # mathplus and numpy
from enum import Enum
from collections import Counter
from collections.abc import Sequence


#################################################
################## class Common #################
################################################# 
"""
as the name suggests, Common provides more general
functionality used by other modules.
"""
class Common:          
    def get_version():
        return "Version 0.1 of mathplus library distributed with GNU General Public License v3.0 in 2022"

    def get_author():
        return "Author: Michael Stal"
        
    # n!
    def fac(n):
        result = 1
        for i in range(2, n+1):
            result *= i
        return result
    
    # (n)  = n! // ((n-k)! * k!)
    # (k)
    def n_over_k(n,k):
        if k == 0:
            return 1
        elif k == 1:
            return n
        else:
            result = 1
            for i in range(n, n-k, -1):
                result *= i
            result = result // fac(k)
            return result
    
    # calculate multinomial coeff. n! / (n1! * n2! * ... * nk!)
    # with n1 + n2 + ... + nk = n        
    def multinomial_coeff(n, *args):
        sum = 0
        for arg in args:
            if arg < 0:
                raise ValueError("negative arguments are not allowed")
            sum += arg
        if (sum != n):
            return 0
        else:
            result = Common.fac(n)
            for arg in args:
                result /= Common.fac(arg)
            return result       
            
    # calculating the probabilty of an event: where every arg 
    # contains a tuple (ni,pi) with ni being the number the  
    # event occurs multiplied with its propability pi 
    # Result = (n! / (n1! * n2! * ... * nk!)) * p1^n1 * p2^n2 * ... * pk^nk
    def multinomial_probability(n, *args):
        sum1 = 0
        sum2 = 0
        for arg in args:
            if arg[0] < 0: # arg[0]: count of occurrences of event i
                raise ValueError("negative arguments n are not allowed")
            sum1 += arg[0]
            if arg[1] < 0:
                raise ValueError("negative arguments p are not allowed")
            sum2 += arg[1] # arg[1]: probability of event i
        if (sum1 != n) or (sum2 != 1):
            return 0
        else:
            result = Common.fac(n)
            for arg in args:
                result /= Common.fac(arg[0])
                result *= arg[1] ** arg[0] 
            return result
            
    # fibonacci series
    def fib(n):
        if n == 0:
            return 0
        elif n == 1:
            return 1
        else: return Common.fib(n-1) + Common.fib(n-2)
        
    # check whether n is prime
    def is_prime(n):
        if n <= 3:
            return n >= 1
        if  ((n % 2) == 0) or ((n % 3) == 0):
            return False
        i = 5
        limit = int(math.sqrt(n))
        while i <= limit:
            if  ((n % i) == 0) or  ((n % (i + 2)) == 0):
                return False
            i += 6
        return True
        
    # compute conjugate of complex numbers
    def conj(num):
        if isinstance(num, complex):
            return complex(num.real, -num.imag)
        else:
            return num
            
    # greatest common divisor
    def gcd(a, b):
        while b != 0:
            a,b = b,a%b
        return a
        
    # the kronecker delta is defined over fields, rings,  
    # groups. elem0 is the 0-element of the domain, 
    # elem1 is the 1-element.
    # x == y => delta(x,y) <- elem1
    # x != y => delta(x,y) <- elem0
    def kronecker_delta(x, y, elem0 = 0, elem1 = 1):
        if x == y:
            return elem1
        else:
            return elem0
        
    # calculate gcd for multiple numbers
    def gcdmult(args):
        if len(args) == 1:
            return args[0]
        elif len(args) == 2:
            return Common.gcd(args[0], args[1])
        elif len(args) == 0:
            return None
        else:
            g = Common.gcd(args[0], args[1])
            for i in range(2, len(args)):
                g = Common.gcd(g, args[i])
            return g
        
    # least common multiple
    def lcm(a, b):
        return abs(a*b)/gcd(a,b)       


    # factorization of integers into their prime factors.
    # the method returns a list of prime factors in ascending
    # order
    def factorize(n1):
        # helper method for factorization
        def _brent(N):
            if N % 2 == 0: 
                return 2
            y, c, m = randint(1, N-1), randint(1, N-1), randint(1, N-1)
            g, r, q = 1, 1, 1
            while g == 1:             
                x = y
                for i in range(r):
                    y = ( ( y * y ) % N + c ) %N
                k = 0
                while (k < r and g == 1):
                    ys = y
                    for i in range(min(m,r-k)):
                        y = ( ( y * y ) %N + c ) %N
                        q *= ( abs ( x - y ) ) %N
                    g = Common.gcd(q,N)
                    k = k + m
                r = r*2
            if g == N:
                while True:
                    ys = ( ( ys * ys ) % N + c ) %N
                    g = Common.gcd(abs(x - ys), N)
                    if g > 1: break
            return g # end of _brent

        if n1 == 0: 
            return []
        if n1 == 1: 
            return [1]
        n = n1
        b = []
        p = 0
        mx = 1000000
        while n % 2 == 0: 
            b.append(2)
            n //= 2
        while n % 3 == 0:  
            b.append(3) 
            n //= 3
        i = 5
        inc = 2
        while i <= mx:
           while n % i ==0: 
               b.append(i)
               n //= i
           i += inc
           inc = 6 - inc
        while n > mx:
            p1 = n
            while p1 != p:
                p = p1
                p1 = brent(p)
         
            b.append(p1)
            n //= p1 
        if n != 1:
            b.append(n)   
        return sorted(b)
    
    # defactorize expects an array of factors such as [2,3,5,7]
    # which results in 210
    def defactorize(factoriz):
        res = 1
        for i in factoriz:
            res *= i
        return res

    # x is the number, mu is the mean, sigma the standard deviation
    def z_score(x, mu, sigma):
        return (x - mu)/sigma
        
    # sigma is the standard deviation, mu the mean
    def variation_coeff(sigma, mu):
        return sigma/mu
            
    # gaussian distribution: mu is the median or expectation value of a data 
    # set, sigma its standard deviation
    def gaussian_distribution(x, mu, sigma):
        return (1 / (sigma * math.sqrt(2 * math.pi))) * math.exp(-0.5 * (((x-mu)/sigma) ** 2)) 

    # the different gaussian kernels for 1D, 2D, multidiemsional cases
    def gaussian_kernel_1D(x, sigma):
        return  (1 / (math.sqrt(2 * math.pi) * sigma)) * math.exp(- x**2/(2 * sigma ** 2)) 

    def gaussian_kernel_2D(x, y, sigma):
        return (1 / (2 * math.pi * sigma ** 2)) * math.exp(- (x**2 + y**2) / (2 * sigma ** 2))

    # dimension == len(xvect)
    def gaussian_kernel_multiD(xvect, sigma):
        return (1 / ((math.sqrt(2 * math.pi) * sigma)) ** len(xvect)) * math.exp(- (xvect.euclidean_norm() ** 2) / (2 * sigma **2))

    # calculate the covariance between two data series
    # zero result => dataset don't seem to have a relation
    # positive: if one data rises, the other one rises too 
    # negative: if one dataset rises, the other one falls
    def covariance(x_dataset, y_dataset):
        if len(x_dataset) != len(y_dataset):
            raise ValueError("both data sets must have the same number of data elements")
        else:
            sum = 0
            x_mean = Array.mean(x_dataset)
            y_mean = Array.mean(y_dataset)
            for i in range(len(x_dataset)):
                sum += (x_dataset[i] - x_mean) * (y_dataset[i] - y_mean)
            return sum / (len(x_dataset)-1)
            
            
    # calculate the correlation between two data series
    # 0 => no link between data, 1 => perfect positive 
    # correlation, -1 = perfect negative correlation
    def correlation(x_dataset, y_dataset):
        if len(x_dataset) != len(y_dataset):
            raise ValueError("both data sets must have the same number of data elements")
        else:
            sum1 = 0
            sum2 = 0
            sum3 = 0
            x_mean = Array.mean(x_dataset)
            y_mean = Array.mean(y_dataset)
            for i in range(len(x_dataset)):
                sum1 += (x_dataset[i] - x_mean) ** 2 
                sum2 += (x_dataset[i] - x_mean) * (y_dataset[i] - y_mean)
                sum3 += (y_dataset[i] - y_mean) ** 2
            return sum2 / math.sqrt(sum1 * sum3)

    # calculates expectation value for series of x-values.
    # If weights_array is used by caller each x-value will
    # be multiplied by the corresponding weight
    # If no weights_array is passed to the method, weight
    # will be 1/len(x_array)
    def expected_value(x_array, weights_array = None):
        if len(x_array) == 0:
            return 0
        if (weights_array != None) and (len(x_array) != len(weights_array)):
            raise ValueError("x_array and weights_array must have the same length") 
        if weights_array == None:
            weight = 1 / len(x_array)
            sum = 0
            for i in range(len(x_array)):
                sum += x_array[i] * weight
            return sum
        else:
            sum = 0
            for i in range(len(x_array)):
                sum += x_array[i] * weights_array[i]    
            return sum
            
    # Implementation of Fast Fourier Transform using Cooley-Tukey 
    # iterative in-place algorithm (radix-2 DIT). Number 
    # buffer must contain 2^n complex numbers.
    # Note: the algorithm operates in-place, i.e., the input buffer 
    # is overwritten
    def fft(buffer):
        def bitreverse(num, bits):
            n = num
            n_reversed = n
            ctr = bits - 1
            n = n >> 1
            while n > 0:
                n_reversed = (n_reversed << 1) | (n & 1)
                ctr -= 1
                n = n >> 1
            return ((n_reversed << ctr) & ((1 << bits) - 1))
            
        bits = int(math.log(len(buffer), 2))
        for j in range(1, len(buffer)):
            swap_pos = bitreverse(j, bits)
            if swap_pos <= j:
                continue
            (buffer[j], buffer[swap_pos]) = (buffer[swap_pos], buffer[j]) 

        N = 2
        while N <= len(buffer):
            for i in range(0, len(buffer), N):
                for k in range(0, N // 2):
                    even_idx = i + k
                    odd_idx = i + k + (N // 2)
                    even, odd = buffer[even_idx], buffer[odd_idx]
                    term = -2 * math.pi * k / float(N)
                    exp = complex(math.cos(term), math.sin(term)) * odd
                    buffer[even_idx], buffer[odd_idx] = even + exp, even - exp
            
            N = N << 1
            
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
                   
#################################################
################ class mparray  #################
#################################################

"""
mparray is based upon Python lists. It was created to
shield users of the mparray from its implementation, so that
the implementation may be changed (such as using C-arrays
underneath).
It implements "rectangular" arrays, but does not support
irregular shapes where a one-dimensional may contain numbers
and subarrays or subarrays may reveal different lengths.
Some methods are constrained to one- or two-dimensional
mparrays, for example, methods that may not be useful for other
dimensions. Consider transpose as an example.
"""
class mparray:
    # new mparray instances implement regular not rugged mparrays
    # example: a = mparray([[1,2],[3,4]], dtype = int)
    def __init__(self, array, dtype = float):
        if not isinstance(array, list):
            raise TypeError("mparray__init()__ expects a list as first parameter")
        if array != [] and not mparray._is_regular(array):
            raise ValueError("only 'rectangular' mparrays are supported")
        self.a = array
        self.dtype = dtype 
        self.ndim = len(self.shape)
        
    @property
    def count(self):
        shp = self.shape
        prod = 1
        for i in range(len(shp)):
            prod *= shp[i]
        return prod
        
    # creates a multidimensional array using a list
    def create_multidim_array(dims, lst, dtype = float):
        tmp = lst
        for i in range(dims-1): tmp = [tmp]
        return mparray(tmp, float)
            
    # creates a new mparray with shp as shape and dtype as type,
    # filled with init_value
    def filled_array(shp, init_value = 0, dtype = float):
        a = mparray._initializer(shp, init_value = init_value, dtype = dtype)
        return mparray(a, dtype)
        
    def ones(shp, dtype = float):
        return mparray.filled_array(shp, init_value = 1, dtype = dtype)
        
    def zeros(shp, dtype = float):
        return mparray.filled_array(shp, init_value = 0, dtype = dtype)
        
    # list passed to function is used to initialze new mparray object
    def from_list(array, dtype = float):
        if not mparray._is_regular(array):
            raise ValueError("only 'rectangular' mparrays are supported")
        shp = Array.shape(array)
        t = mparray(array, dtype)
        return t
        
    # rand() creates a mparray filled with random values in [0,1[ 
    # with the specified shape shp
    def rand(shp, seedval = 0):
        # determine number of elements
        prod = 1
        for i in range(len(shp)): prod *= shp[i]
        # create a flat array
        mpa = mparray.random_array([prod], fromvalue=0, tovalue = 1, dtype = float, seedval = seedval)
        # reshape the flat array in the required shp
        return mpa.reshape(shp)
        
    # extract a list from the mparray
    def to_list(self):
        return deepcopy(self.a)
        
    # extract a flat list from the mparray
    def to_flat_list(self):
        return self.flatten().to_list()
        
    # returns the actual size of the mparray
    def __len__(self):
        return len(self.flatten().to_list())
        
    def empty():
        return mparray.from_list([]) 
        
    def square_shaped(self):
        if len(self.shp) == 1:
            return False
        else:
            dim0 = self.shp[0]
            for i in range(1, len(self.shp)):
                if self.shp[i] != dim0:
                    return False
            return True
            
    # get a column of a 2-dimensional mparray
    def column(self, col):
        shp = self.shape
        if len(shp) != 2:
            raise ValueError("mparray.column() requires a 2-dimensonal mparray")
        if not col in range(0, shp[1]):
            raise ValueError("index (col) out of range")
        cvec = mparray.filled_array((shp[0],), dtype = self.dtype)
        for i in range(shp[0]):
            cvec[i] = self[i][col]  
        return cvec
        
    # get a row of a 2-dimensional mparray
    def row(self, row):
        shp = self.shape
        if len(shp) != 2:
            raise ValueError("mparray.row() requires a 2-dimensonal mparray")
        if not row in range(0, shp[0]):
            raise ValueError("index (row) out of range")
        return self[row]
        
            
    # create a new mparray using shp as shape
    def _initializer(shp, init_value = 0, dtype = float):
        if not isinstance(shp, tuple) and not isinstance(shp,list):
            raise ValueError("first argument must be a list or a tuple, not a " + str(type(shp)))
        if len(shp) == 1:
            a = []
            for i in range(shp[0]):
                if dtype == int:
                    a.append(int(init_value))
                else:
                    a.append(dtype(init_value))
            return a
        else: # len(shp) > 0
            a = []
            newshp = list(copy(shp))
            newshp.pop(0)
            newshp = tuple(newshp)
            for i in range(shp[0]):
                a.append(mparray._initializer(newshp, init_value, dtype))
            return a
            
    # helper function to initialize lists with random numbers
    def _random_initializer(dims, fromvalue, tovalue, seed_val = None, dtype = float):
        if seed_val != None:
            random.seed(seed_val)
        if len(list(dims)) == 1:
            a = []
            for i in range(dims[0]):
                if dtype == int:
                    a.append(int(random.randrange(fromvalue, tovalue)))
                else:
                    a.append(dtype(random.uniform(fromvalue, tovalue)))
            return a
        else: # len(dims) > 1
            a = []
            newshp = [copy(dims)]
            newshp.pop(0)
            newshp = tuple(newshp)
            for i in range(dims[0]):
                a.append(mparray._random_initializer(newshp, fromvalue, tovalue, None, dtype))
            return a
        
    # degree of mparray, i.e., its dimensions
    def degree(self):
        return len(self.shape)
        
    # this method shuffles all inner 1-dimensional arrays contained
    # in a mparray
    def shuffle(self):
        shp = self.shape
        if len(list(shp)) == 1:
            a = deepcopy(self.a)
            random.shuffle(a)
            return mparray(a, self.dtype)
        elif len(shp) == 2:
            a = []
            for i in range(shp[0]):
                elem = deepcopy(self.a[i])
                random.shuffle(elem)
                a.append(elem)
            return mparray(a, self.dtype)
        else:
            a = []
            for i in range(shp[0]):
                elem = mparray(self.a[i], self.dtype).shuffle()
                a.append(elem.to_list()) 
            return mparray(a, self.dtype)
            
        
    def _find(array, lambda_f):
        result = []
        shp = Array.shape(array)
        if len(shp) == 1:
            for i in range(shp[0]):
                if lambda_f(array[i]):
                    result.append(array[i])
            return result
        else:
            for i in range(shp[0]):
                result += mparray._find(array[i], lambda_f)
            return result
            
    def _find_with_path(array, path, lambda_f):
        result = []
        shp = Array.shape(array)
        if len(shp) == 1:
            for i in range(shp[0]):
                if lambda_f(array[i]):
                    result.append(path+[i])
            return result
        else:
            for i in range(shp[0]):
                result += mparray._find_with_path(array[i], path + [i], lambda_f)
            return result
            
    # search for all elements in mparray that meet a condition
    # defined by lambda_f
    def find(self, lambda_f):
        return mparray._find(self.a, lambda_f)
        
    def find_where(self, lambda_f):
        return mparray._find_with_path(self.a, [], lambda_f)
                    
    # apply a lambda on all mparray elements and create a
    # new mparray from result
    def apply(self, lambda_f):
        t = mparray.filled_array(self.shape, dtype = self.dtype)
        t.a = mparray._apply_op(self.a, lambda_f)
        return t
        
    def sign(self):
        def fun(n):
            if n == 0: return 0
            elif n > 0: return 1
            else: return -1
        return self.apply(fun)
            
    # helper function that gets two arrays as arguments
    # and is called for each pair. The result is then used
    # to create a new array with the same shape. This is useful
    # to implement operators such as __add__ for mparrays
    def _apply_on_multiDarrays(a1, a2, lambda_f):
        shp = Array.shape(a1)
        if len(shp) == 1:
            a = []
            for j in range(shp[0]):
                a.append(lambda_f(a1[j],a2[j]))
            return a
        else:
            a = []
            for j in range(shp[0]):
                a.append(mparray._apply_on_multiDarrays(a1[j],a2[j], lambda_f))
            return a
            
    # helper function for apply used to apply a lambda on each
    # element of a multidimensional list and create an square_shaped
    # new mparray from the results
    def _apply_op(array, lambda_f):
        shp = Array.shape(array)
        if len(shp) == 1:
            a = []
            for j in range(shp[0]):
                a.append(lambda_f(array[j]))
            return a
        else:
            a = []
            for j in range(0, shp[0]):
                a.append(mparray._apply_op(array[j], lambda_f))
            return a
                
    # get shape of mparray
    @property
    def shape(self):
        return Array.shape(self.a)
            
    def __str__(self):
        def helper(array):
            res = "["
            for i in range(len(array)):
                if not isinstance(array[i], list):
                    res += " " + str(array[i]) + " "
                else:
                    res += helper(array[i])
            res += "]"
            return res
        return helper(self.a)
        
    def __repr__(self):
        return str(self)
        
    # helper function to determine whether a list
    # is regular
    def _is_regular(array):
        if not isinstance(array, list):
            raise ValueError("_is_regular() only applicable to lists")
        if type(array[0]) == list:
            size = len(array[0])
            for i in range(1, len(array)):
                if type(array[i]) != list:
                    return False
                else:
                    if len(array[i]) != size:
                        return False
            for i in range(len(array)):
                if not mparray._is_regular(array[i]):
                    return False
            return True
        else:
            for i in range(1, len(array)):
                if isinstance(array[i], list):
                    return False
            return True
            
    # subtraction of two equal-shaped mparrays. Pairwise addition
    # with results used to create a new mparray
    def __sub__(self, other):
        if isinstance(other, mparray):
            if self.shape != other.shape:
                raise ValueError("shapes of operands do not match")
            t = mparray.filled_array(self.shape, init_value = self.dtype(0), dtype=self.dtype)
            t.a = mparray._apply_on_multiDarrays(self.a, other.a, lambda x,y: x-y)
            return t
        elif isinstance(other, float) or isinstance(other, int):
            t = mparray.filled_array(self.shape, init_value = self.dtype(0), dtype = self.dtype)
            t.a = mparray._apply_op(self.a, lambda x: x - self.dtype(other))
            return t
        else:
            raise TypeError("both operands must be mparrays")
        
    # addition of mparrays: similar implementation like that in __sub__
    def __add__(self, other):
        if isinstance(other, mparray):
            if self.shape != other.shape:
                raise ValueError("shapes of operands do not match")
            t = mparray.filled_array(self.shape, init_value = 0,dtype=self.dtype)
            t.a = mparray._apply_on_multiDarrays(self.a, other.a, lambda x,y: x+y)
            return t
        elif isinstance(other, float) or isinstance(other, int):
            t = mparray.filled_array(self.shape, init_value = 0, dtype = self.dtype)
            t.a = mparray._apply_op(self.a, lambda x: x + self.dtype(other))
            return t
        else:
            raise TypeError("operands must be mparrays or a mparray and a number")
        
    # scalar multiplication and multiplication between mparrays is supported
    def __mul__(self, other):
        return mparray.multiply(self, other)
        
    # pair-wise multiplication
    def multiply(arr1, arr2):
        if isinstance(arr1, mparray) and not isinstance(arr2, mparray):
            return deepcopy(arr1).apply(lambda x: x * arr2)
        elif not isinstance(arr1, mparray) and isinstance(arr2, mparray):
            return deepcopy(arr2).apply(lambda x: x * arr1)
        elif not isinstance(arr1, mparray) and not isinstance(arr2, mparray):
            return arr1 * arr2
        # both operands are mparrays:
        if arr1.shape != arr2.shape:
            raise ValueError("multiply only defined for arrays with same shape")
        shp = arr1.shape
        result = mparray.filled_array(shp, dtype = arr1.dtype)
        if arr1.degree() == 1:
            for i in range(shp1[0]):
                result[i] = arr1[i] * arr2[i]
            return result
        elif arr1.degree() == 2:
            for i in range(arr1.shape[0]):
                for j in range(arr1.shape[1]):
                    result[i][j] = arr1[i][j] * arr2[i][j]
            return result
        else:
            for i in range(arr1.shape[0]):
                result[i] = mparray.mul_pairwise(arr1[i], arr2[i])
            return result
        
        
    # negation of mparray:
    def __neg__(self):
        return self.apply(lambda x: -x)
        
    # return mparray where all elems of self are powered to exponent
    def __pow__(self, exponent):
        return self.apply(lambda x: x ** exponent)
        
    # expects arrays with equal shapes
    def _compare_arrays(a1, shp1, a2, shp2):
        if shp1 != shp2:
            return False
        elif len(shp1) == 1:
            for i in range(shp1[0]):
                if a1[i] != a2[i]:
                    return False
            return True
        else:
            newshp = list(copy(shp1))
            newshp.pop(0)
            newshp = tuple(newshp)
            for i in range(shp1[0]):
                if mparray._compare_arrays(a1[i], newshp, a2[i], newshp):
                    continue
                else: return False
            return True
            
    # allclose checks for two array whether their elements are equal
    # within a tolerance. It returns whether all elements were 
    # considered equal (True) ao at least one element was considered
    # not equal
    def allclose(a1, a2, tol = 1E-10):
        shp1 = a1.shape
        shp2 = a2.shape
        if shp1 != shp2:
            raise ValueError("can not compare mparrays with different shapes")
        else:
            if len(shp1) == 1:
                for i in range(shp1[0]):
                    print(a1)
                    print(a2)
                    if abs(a1[i] - a2[i]) <=  tol:
                        continue
                    else:
                        return False
            else:
                for i in range(shp1[0]):
                    if mparray.allclose(a1[i], a2[i], tol):
                        continue
                    else:
                        return False
            return True
            
            
    # isclose checks two arrays whether their elements are equal
    # within a tolerance. It returns a mparray with bools specifying
    # where the elements were considered equal and where they were 
    # considered not equal
    def isclose(a1, a2, tol = 1E-10):
        shp1 = a1.shape
        shp2 = a2.shape
        if shp1 != shp2:
            raise ValueError("can not compare mparrays with different shapes")
        else:
            result = []
            if len(shp1) == 1:
                for i in range(shp1[0]):
                    if abs(a1[i] - a2[i]) <=  tol:
                        result.append(True)
                    else:
                        result.append(False)
            else:
                for i in range(shp1[0]):
                    result.append(mparray.isclose(a1[i], a2[i], tol))       
            return result
            
            
        
    def __eq__(self, other):
        if not (isinstance(other, mparray) or isinstance(other, list)):
            raise TypeError("left operand must be mparray, right an mparray or a list")
        if isinstance(other, mparray):
            operand2 = other
        else:
            operand2 = mparray(other, self.dtype)
        if self.shape != operand2.shape:
            raise ValueError("cannot compare objects with different shapes")
        return mparray._compare_arrays(self.a, self.shape, operand2.a, operand2.shape)

    def _ne__(self, other):
        return not self == other
        
    # create a random array of given shape and size
    def random_array(shp, fromvalue, tovalue, seedval = None, dtype = float):
        a = mparray._random_initializer(shp, fromvalue, tovalue, seedval, dtype)
        t = mparray(a, dtype)
        return t
        
    # recursive helper method for mparray flattening
    def _flattener(array):
        a = []
        shp = Array.shape(array)
        if len(shp) == 1:
            for i in range(shp[0]): a.append(array[i])
            return a
        else:
            for i in range(shp[0]):
                sub_array = mparray._flattener(array[i])
                a += sub_array
            return a
    
    # convert mparray object to flat mparray
    def flatten(self):
        return mparray.from_list(mparray._flattener(self.a), dtype = self.dtype)
        
    # recursive helper function for mparray reshaping
    def _filler(array, idx, shp):
        if len(shp) == 1:
            tmp = []
            for i in range(shp[0]):
                tmp.append(array[idx])
                idx += 1
            return (tmp, idx)
        else:
            new_idx = idx
            tmp = []
            new_shape = list(copy(shp))
            new_shape.pop(0)
            new_shape = tuple(new_shape)
            for i in range(shp[0]):
                sub_arr, new_idx = mparray._filler(array, new_idx, new_shape)
                tmp.append(sub_arr)
            return (tmp, new_idx)
            
    def reshape(self, new_shape):
        # Flatten the whole mparray to a mpone-dimenional array
        array = self.flatten()
        # calculate size required by new shape
        prod = 1
        for num in new_shape: prod *= num
        # if sizes do not match raise ValueError
        if prod != len(array.a):
            raise ValueError("cannot reshape a + " + str(self.shape) + " array into a " + str(new_shape) + " array")
        else:
            # call _filler with flattened array, index 0, and desired shape
            new_a, _ = mparray._filler(array.a, 0, new_shape)
            # use list to create mparray
            return mparray.from_list(new_a, dtype = self.dtype)
    
    # method adds an additional matrix on front of mparray,
    # copies elements of self and adds them to the expanded
    # mparray. This, the degree of the new mparray is one plus
    # degree of old mparray.
    # When using newarray[0] this sub array whill have
    # exactly the shape and elements of self
    def expand(self):
        tmp=[]
        tmp.append(deepcopy(self.a))
        newarray = mparray.from_list(tmp, self.dtype)
        return newarray
        
    def compress(self):
        if len(self.shape) > 1 and self.shape[0] == 1:
            tmp = self.a[0]
            return mparray.from_list(tmp, self.dtype)
        
    # checks whether addressed element is a scalar and returns that scalar
    # Otherwise, it returns an mparray object
    def __getitem__(self, arg):
        tmp = self.a.__getitem__(arg)
        if not isinstance(tmp, list):
            return tmp
        else:
            return mparray.from_list(tmp)
        
    # delegates to corresponding access pattern for lists
    def __setitem__(self, arg, cont):
        self.a.__setitem__(arg, cont)
        
    # iterator implementation. The iterator walks through
    # all elements of mparray. For this purpose, the flattened
    # mparray is used
    def __iter__(self):
        self.idx = 0
        self.elements = (self.flatten()).a
        return self
        
    def __next__(self):
        if self.idx < len(self.elements):
            result = self.elements[self.idx]
            self.idx += 1
            return result
        else:
            raise StopIteration
            
    # clip cuts values so that they fit in the specified interval
    # _clip_helper support clip
    def _clip_helper(arr, a_min, a_max, in_situ = False):
        shp = arr.shape
        result = []
        if in_situ:
            if len(shp) == 1:
                for i in range(shp[0]):
                    if arr[i] < a_min:
                        arr[i] = a_min
                    elif arr[i] > a_max:
                        arr[i] = a_max
            else:
                for i in range(shp[0]):
                    mparray._clip_helper(arr[i], a_min, a_max, in_situ)
        else:
            if len(shp) == 1:
                for i in range(shp[0]):
                    if arr[i] < a_min:
                        result.append(a_min)
                    elif arr[i] > a_max:
                        result.append(a_max)
                    else:
                        result.append(arr[i])
                return result
            else:
                for i in range(shp[0]):
                    result.append(mparray._clip_helper(arr[i], a_min, a_max, in_situ))
                return result
                
    def clip(arr, a_min, a_max, in_situ = False):
        if in_situ:
            mparray._clip_helper(arr, a_min, a_max, in_situ)
        else:
            return mparray(mparray._clip_helper(arr, a_min, a_max, in_situ), dtype = arr.dtype)
        
            
    # check whether a condition holds for at least one element
    # of the mparray
    def any(self, cond):
        tmp = self.flatten()
        for  elem in tmp:
            if cond(elem):
                return True
        return False
        
    # check whether a condition holds for all elements
    # of the mparray
    def all(self, cond):
        tmp = self.flatten()
        for  elem in tmp:
            if not cond(elem):
                return False
        return True
        
    # creates a list with all entries from start to stop
    # using step as increment. The values created are in
    # [start, stop[
    def arange(start = 0, stop = 1, step = 1, dtype = float):
        result = []
        act_value = dtype(start)
        while act_value < stop:
            result.append(act_value)
            act_value += step
        return mparray(result, dtype)
        
    # creates a linear distribution from start point startp to endpoint endp
    # consisting of size elements with the specified datatype dtype, If
    # with_endp is set to True the distribution will contain the endpoint.
    def lin_distribution(startp, endp, size, with_endp = False, dtype = float):
        array = []
        if size == 1:
            sz = 1
        else:
            sz = size - 1 if with_endp else size
        
        if dtype == int:
            tmp = float((endp-startp) / sz)
            if sz > abs(endp-startp) or tmp - math.floor(tmp) > 0:
                raise ValueError("cannot create an equidistant distibution of integers in this interval")
        incr =(endp-startp)/sz
    
        for i in range(0, size):
            array.append(dtype(startp + i * incr))
        return mparray.from_list(array, dtype)
        
    # creates a logarithmic distribution of size elements starting
    # with base ** startp and ending with base ** endp (if with_endp
    # is set to True)
    def log_distribution(startp, endp, size, base = 10.0, with_endp = True, dtype = float):
        array = []
        if size == 1:
            sz = 1
        else:
            sz = size - 1 if with_endp else size
        
        incr = float((endp-startp)/sz)
    
        for i in range(0, size):
            array.append(dtype(base ** (startp + i * incr)))
        return mparray.from_list(array, dtype)
        
    # calculates the mean of mparray elements
    def mean(self):
        sum = 0
        t = self.flatten()
        for elem in t:
            sum += elem
        return sum / len(t.a)
        
    # calculates the standard deviation of array elemments
    def std_dev(self):
        sum = 0
        mu = self.mean()
        t = self.flatten()
        for elem in t:
            sum += (elem - mu) ** 2
        res = math.sqrt(sum / (len(t.a)-1))
        return res
        
    # calculates the variance of mparray elements
    def variance(self):
        return self.std_dev() ** 2
        

    # calculate the covariance between two data series
    # zero result => dataset don't seem to have a relation
    # positive: if one data rises, the other one rises too 
    # negative: if one dataset rises, the other one falls
    def covariance(x_dataset, y_dataset):
        shp1 = x_dataset.shape
        shp2 = y_dataset.shape
        if shp1 != shp2 or len(shp1) != 1:
            raise ValueError("both data sets must be 1-dimensional and have the same number of data elements")
        else:
            sum = 0
            x_mean = x_dataset.mean()
            y_mean = y_dataset.mean()
            for i in range(len(x_dataset)):
                sum += (x_dataset[i] - x_mean) * (y_dataset[i] - y_mean)
            return sum / (len(x_dataset)-1)
           
           
    # this function does delegate to reduce()
    # the lmbda is applied to all elements of
    # mparray with init_val being the aggregator
    def reduce_general(self, lmbda, init_val = None):
        array = self.flatten().a
        if init_val == None:
            return reduce(lmbda, array)
        else:
            return reduce(lmbda, array, init_val)
        
    # calculate the sum of all elements in a mparray
    # using init_val as the base value
    def sum(self, init_val = None):
        if init_val == None:
            return self.reduce_general(operator.add)
        else:
            return self.reduce_general(operator.add, init_val)
        
    # calculate the product of all elements in a mparray
    # using init_val as the base value
    def mul(self, init_val = None):
        if init_val == None:
            return self.reduce_general(operator.mul)
        else:
            return self.reduce_general(operator.mul, init_val)
            
    # transpose() is defined for all 2D mparrays. It returns a new
    # mparray with: new_array[r][c] = array[c][r] for all
    # valid (row,column)-combinations
    def transpose(self):
        assert self.degree() == 2, "transpose only defined for 2d-arrays"
        result = mparray.array_transpose(self.a)
        return mparray(result, self.dtype)
    
    # returns the transposed array
    @property
    def T(self):
        return self.transpose()
        
    # returns the total number of elements in the mparray
    @property
    def size(self):
        prod = 1
        shp = self.shape
        for i in range(len(shp)):
            prod *= shp[i]
        return prod
    
        
    def dot(arg1, arg2):
        if isinstance(arg1, mparray) and not isinstance(arg2, mparray):
            return arg1 * arg2 # multiplication with scalar
        elif not isinstance(arg1, mparray) and isinstance(arg2, mparray):
            return arg2 * arg1 # multiplication with scalar
        elif not isinstance(arg1, mparray) and not isinstance(arg2, mparray):
            return arg1*arg2 # simple scalar multiplication
        else: # isinstance(arg1,mparray) and isinstance(arg2.mparray)
            shp1 = arg1.shape
            shp2 = arg2.shape
            # inner product
            if arg1.degree() == 1 and arg2.degree() == 1:
                result = mparray.filled_array(shp1, dtype=arg1.dtype)
                sum = 0
                for i in range(shp1[0]):
                    sum += arg1[i] * arg2[i]
                return sum
            elif arg1.degree() == 2 and arg2.degree() == 1 and shp1[1] == shp2[0]: # matrix * vector 
                n = shp1[0]
                m = shp1[1]
                k = shp2[0]
                result = mparray.filled_array([n], dtype=arg1.dtype)
                for i in range(n):
                    sum = 0
                    for j in range(k):
                        sum += arg1[i][j]*arg2[j]
                    result[i] = sum
                return result
            elif arg1.degree() == 2 and arg2.degree() == 2:
                return arg1 @ arg2
            elif arg1.degree() > 2 and arg2.degree() > 2:
                result = mparray.filled_array(shp1, dtype=arg1.dtype)
                for i in range(shp1[0]):
                    result[i] = mparray.dot(arg1[i], arg2[i])
                return mparray([result])
            else:
                raise TypeError("Incompatible operands for mparray.dot()")
        
    def __matmul__(self , other):
        if not isinstance(other, mparray):
            return self.apply(lambda x: x * other)
        shp1 = self.shape
        shp2 = other.shape
        if len(shp1) == 2 and len(shp2) == 2:
            if shp1[1] != shp2[0]:
                raise ValueError("the 2-dimensional mparrays have incompatible shapes for multiplication")
            res = mparray.filled_array([shp1[0], shp2[1]], dtype=self.dtype)
            for r in range(shp1[0]):
                for c in range(shp2[1]):
                    sum = 0
                    for j in range(shp1[1]):
                        sum += self[r][j] * other[j][c]
                    res[r][c] = sum
            return res
        elif len(shp1) == 2 and len(shp2) == 1:
            if shp1[1] == shp2[0]:
                res = mparray.filled_array([shp1[0], 1], dtype=self.dtype)
                for i in range(shp1[0]):
                    sum = 0
                    for j in range(shp1[1]):
                        sum += self[i, j] * other[j]
                    res[i] = sum
                return res
            else:
                raise ValueError("the 2- and -dimensional mparrays have incompatible shapes for multiplication")
            
        
    # transpose() transposes nxm-arrays
    def array_transpose(array):
        shp = Array.shape(array)
        n = shp[0]
        m = shp[1]
        result = [[0 for i in range(n)] for j in range(m)]
        for r in range(m):
            for c in range(n):
                result[r][c] = array[c][r]
        return result
        
    # extract a subarray from mparray
    def subarray(self, top, bottom, left, right):
        shp = self.shape
        dim1 = shp[0]
        dim2 = shp[1]
        if not(left <= right and top <= bottom and right <= dim2 and bottom <= dim1):
            raise ValueError("index out of range")
            
        if left == right and top == bottom:
            return self[top][left]
        else: 
            result = []
            for r in range(bottom - top + 1):
                row = []
                for c in range(right - left + 1):
                    row.append(self[top + r][left + c])
                result.append(row)
            return mparray(result, dtype=self.dtype)
        
    # split splits a mparray in different sub-arrays. It is only defined
    # for 1-D and 2-D mparrays.
    # arg is an integer when the split should happen into same-size-pieces
    # or as a tupel if the split should happen at specified positions.
    # For 2-D arrays axis defines along which axis the split is going to
    # happen
    def split(self, arg, axis = 0):
        shp = self.shape
    
        def split_1D(array, arg):
            n = len(array)
            if isinstance(arg, int):
                if arg == 0 or n % arg != 0:
                    raise ValueError("equal split of array with size = " + str(n) + " impossible with arg = " + str(arg))
                else:
                    result = []
                    i = 0
                    while i < n:
                        arr = []
                        for j in range(n // arg):
                            arr.append(array[i + j])
                        i += n // arg
                        result.append(arr)
                return result
            elif isinstance(arg, tuple) or isinstance(arg, list):
                last_pos = 0
                split_indices = [0]
                for pos in arg:
                    if pos < 0 or pos >= n:
                        raise ValueError("attempt to split mparray of size = " + str(n) + " at nonexistent position " + str(pos))
                    split_indices.append(pos)
                split_indices.append(len(array))
                split_indices = list(set(split_indices))
                split_indices.sort()
                result = []
                for i in range(len(split_indices)-1):
                    tmp = []
                    for j in range(split_indices[i], split_indices[i+1]):
                        tmp.append(array[j])
                    result.append(tmp)
                return result
    
        def split_2D(array, arg, axis = 0):
            shp = Array.shape(array)
            n = shp[0]
            m = shp[1]
            if axis == 0:
                if isinstance(arg, int):
                    if arg == 0 or m % arg != 0:
                        raise ValueError("equal split of mparray with size = " + str(m) + " impossible with arg = " + str(arg))
                    else:
                        result = []
                        tmp = []
                        for i in range(n):
                            tmp.append(split_1D(array[i], arg))
                        for j in range(len(tmp[0])):
                            single = []
                            for i in range(n):
                                single.append(tmp[i][j])
                            result.append(single)
                        return result
                elif isinstance(arg, tuple) or isinstance(arg, list):
                    result = []
                    tmp = []
                    for i in range(n):
                        tmp.append(split_1D(array[i], arg))
                    for j in range(len(tmp[0])):
                        single=[]
                        for i in range(n):
                            single.append(tmp[i][j])
                        result.append(single)
                    return result
            else: # axis==1
                if isinstance(arg, int):
                    if arg == 0 or n % arg != 0:
                        raise ValueError("equal split of array with size = " + str(n) + " impossible with arg = " + str(arg))
                result = []
                array_tmp = mparray.array_transpose(array)
                tmp = split_2D(array_tmp, arg, axis = 0)
            
                for k in range(len(tmp)):
                    result.append(mparray.array_transpose(tmp[k]))
                return result
                
        array = self.a
        if self.degree() == 1:
            return split_1D(array, arg)
        elif self.degree() == 2:
            return split_2D(array, arg, axis)
        else:
            raise ValueError("split only defined for one- and two-dimensional mparrays")

    # block allows to combine different mparrays to a larger
    # mparray
    def block(lst, axis = 0):
        shp = Array.shape(lst)
        if len(shp) == 1:
            if shp[0] == 0:
                return mparray([])
            else:
                result = lst[0]
                for i in range(1, shp[0]):
                    result = mparray.concat(result, lst[i], axis = 0)
                return result
        elif len(shp) == 2:
            rows = []
            for i in range(shp[0]):
                rows.append(mparray.block(lst[i]))
            result = rows[0]
            for j in range(1, len(rows)):
                result = mparray.concat(result, rows[j], axis = 1)
            return result       
        else:
            raise ValueError("mparray.block() onls supports 1-dimensional and 2-dimensional arrays")
        
    # concat combines two mparrays along axis 0 or axis 1
    def concat(mp1, mp2, axis = 0):
        result = Array.concat(mp1.to_list(), mp2.to_list(), axis)
        return mparray(result, mp1.dtype)
                
    # vstack is based on concat. Two arrays are stacked horizontally
    def vstack(self):
        return self.concat(other, axis = 0)
        
     # hstack is based on concat. Two arrays are stacked vertically
    def hstack(self):
        return self.concat(other, axis = 1)
        
    # diff takes in each row a[r,c]-a[r,c-1] for c in 1 .. len(row)
    # if the original mparray has dimension dim1 x dim2, then the result
    # will have dimension dim1 x (dim2 - 1)
    def diff(self):
        assert self.degree() == 1 or self.degree() == 2, "diff only defined for 1d and 2d mparrays"
        shp = self.shape
        if self.degree() == 2:
            result = []
            for r in range(shp[0]):
                row = []
                for c in range(1, shp[1]):
                    row.append(self[r][c] - self[r][c-1])
                result.append(row)
            return mparray.from_list(result, self.dtype)
        elif self.degree() == 1:
            result = []
            for i in range(1, shp[0]):
                result.append(self[i]-self[i-1])
            return result
            
    # An 2D mparray can be rotated 90° to the left or right.
    # if left = True  => left  rotation
    # if left = False => right rotation
    def rotate(self, left = True):
        assert self.degree() == 2, "can only rotate 2d mparrays"
        shp = self.shape
        n = shp[0]
        m = shp[1]
        res = []
        if left:
            for c in range(m):
                row = []
                for r in range(n):
                    row.append(self[r][m-c-1])
                res.append(row)
            return mparray.from_list(res, self.dtype)
        else: # right!
            for c in range(m):
                row = []
                for r in range(n):
                    row.append(self[n-r-1][c])
                res.append(row)
            return mparray.from_list(res, self.dtype)
        
    # remove duplicates from a one-dimensional array
    def remove_duplicates(arr):
        return list(set(arr))
        
    # the following helper functions are wrappers
    # that use mparray.apply. They are implemented
    # for the users convenience.
    
    # sin for mparrays based on apply()
    def sin(self):
        res = self.apply(math.sin)
        return res
        
    # cos for mparrays based on apply()
    def cos(self):
        res = self.apply(math.cos)
        return res
        
    # tan for mparrays based on apply()
    def tan(self):
        res = self.apply(math.tan)
        return res
                
    # tanh for mparrays based on apply()
    def tanh(self):
        res = self.apply(math.tanh)
        return res           
                
    # exp for mparrays based on apply()
    # apply_1D
    def exp(array):
        res = self.apply(math.exp)
        return res
        
    # log for mparrays based on apply()
    def log(self, base):
        res = self.apply(lambda x: math.log(x,base))
        return res
        
    # pow for mparrays based on apply()
    def pow(self, exponent):
        res = self.apply(lambda x: pow(x,exponent))
        return res
        
    # calculate the mean-normalized form of the
    # input mparray:
    def mean_normalization(self):
        array = self.flatten().to_list()
        maximum = max(array)
        mean    = self.mean()
        result  = [(array[i] - mean) / maximum for i in range(0, len(array))]
        return mparray(result, self.dtype)
        
    def euclidean_norm(self):
        sum = 0
        array = self.flatten()
        for i in range(array.shape[0]):
            sum += array[i] ** 2
        return math.sqrt(sum)
        
    # calculate minima of mparray
    def argmin(self, axis = None):
        if self.degree() > 2:
            raise ValueError("argmin only defined for 1-D or 2-D")
            
        if axis == None or self.degree() == 1:
            return min(self.a)
        elif axis == 0:
            result = []
            for i in range(0, self.shape[0]):
                result.append(min(self.a[i]))
            return mparray(result, self.dtype)
        elif axis == 1:
            result = []
            for c in range(0, self.shape[1]):
                tmp = []
                for r in range(0, self.shape[0]):
                    tmp.append(self.a[r][c])
                result.append(min(tmp))
            return mparray(result, self.dtype)
                
    # calculate maxima of mparray
    def argmax(self, axis = None):
        if self.degree() > 2:
            raise ValueError("argmax only defined for 1-D or 2-D")
        if axis == None or self.degree() == 1:
            return max(self.a)
        elif axis == 0:
            result = []
            for i in range(0, self.shape[0]):
                result.append(max(self[i]))
            return mparray(result, self.dtype)
        elif axis == 1:
            result = []
            for c in range(0, self.shape[1]):
                tmp = []
                for r in range(0, self.shape[0]):
                    tmp.append(self.a[r][c])
                result.append(max(tmp))
            return mparray(result, self.dtype)
            
    # delete elements from an array with indices of elements
    # to delete given in indices
    def delete(array, indices):
        new_array = []
        for i in range(0, len(array)):
            if not i in indices:
                new_array.append(array[i])
        return new_array
        
    # erase values from array so that that none of
    # these numbers is left in the mparray
    def erase(array, values):
        new_array = deepcopy(array)
        for val in values:
            while val in new_array:
                new_array.remove(val)
        return new_array
        
    # This method sorts the passed array in-situ
    # It return an array of indices that shows
    # which element of the unsorted array was
    # moved to which index due to sorting
    # for example,
    #     array = [4,1,3,2,1,3,0]
    #     will end up in [0, 1, 1, 2, 3, 3, 4]
    #     after sort()
    # The returned index array looks like:
    #     [6, 1, 4, 3, 2, 5, 0]
    # Due to sorting the element formerly located
    # at index 6 is now at index 0 after sorting
    # and the lement formerly located at position 0
    # is at position 6 after sort()
    # if in_situ = False, the sort will be
    # conducted on a copy of a so that a remains
    # unchanged       
    def sort(array, in_situ = True):
        def quicksort(array, indices):
            smaller = []
            equal   = []
            larger  = []
            idx_smaller = []
            idx_equal   = []
            idx_larger  = []
        
            if len(array) > 1:
                pivot = array[0]
                for i in range(len(array)):
                    if array[i] < pivot:
                        smaller.append(array[i])
                        idx_smaller.append(indices[i])
                    elif array[i] == pivot:
                        equal.append(array[i])
                        idx_equal.append(indices[i])
                    elif array[i] > pivot:
                        larger.append(array[i])
                        idx_larger.append(indices[i])
                sma, sidx = quicksort(smaller, idx_smaller)
                equ, eidx = equal, idx_equal
                lar, lidx = quicksort(larger, idx_larger)
                return (sma + equ + lar, sidx + eidx + lidx)
            elif len(array) == 1: 
                return array, [indices[0]]
            else:
                return [],[]
                
        sortedarr, indices = quicksort(array, [i for i in range(len(array))])
        if in_situ:
            for i in range(len(array)):
                array[i] = sortedarr[i]
            return indices
        else:
            return sortedarr, indices
            
    # calculates expectation value for series of x-values.
    # If weights_array is used by caller each x-value will
    # be multiplied by the corresponding weight
    # If no weights_array is passed to the method, weight
    # will be 1/len(x_array)
    def expected_value(x_array, weights_array = None):
        if len(x_array) == 0:
            return 0
        if (weights_array != None) and (len(x_array) != len(weights_array)):
            raise ValueError("x_array and weights_array must have the same length")
        if weights_array == None:
            weight = 1 / len(x_array)
            sum = 0
            for i in range(0, len(x_array)):
                sum += x_array[i] * weight
            return sum
        else:
            sum = 0
            for i in range(len(x_array)):
                sum += x_array[i] * weights_array[i]
            return sum
            
        
    # calculate covariance matrix    
    def covariance_matrix(dataset, rows = True):
        shp = dataset.shape
        if len(shp) != 2:
            raise ValueError("covariance_matrix only defined for 2-dimensional datasets")
        if rows:
            result = Matrix(shp[0],shp[0], dtype = float)
            for i in range(shp[0]):
                for j in range(shp[0]):
                    if i == j: 
                        result[i,j] = mparray.variance(dataset[i])
                    else:
                        result[i,j] = mparray.covariance(dataset[i], dataset[j])
            return result
        else:
            cov = mparray.covariance_matrix(dataset.T)
            return cov.T
    
        
    # calculate correlation matrix    
    def correlation_matrix(dataset, rows = True):
        shp = dataset.shape
        if len(shp) != 2:
            raise ValueError("correlation_matrix only defined for 2-dimensional datasets")
        if rows:
            result = Matrix(shp[0],shp[0], dtype = float)
            for i in range(shp[0]):
                for j in range(shp[0]):
                    if i == j: 
                        result[i,j] = 1
                    else:
                        result[i,j] = mparray.covariance(dataset[i], dataset[j]) / (dataset[i].variance() * dataset[j].variance())
            return result
        else:
            cor = mparray.covariance_matrix(dataset.T)
            return cor.T
            
    # meshgrid allows to combine two 1D mparrays x, y to a coordinate
    # system spanned by x and y. It returns 2 2D arrays
    def meshgrid(mpx, mpy):
        shp1 = mpx.shape
        shp2 = mpy.shape
        if len(shp1) != 1 or len(shp2) != 1:
            raise ValueError("meshgrid expects 1D arrays as input")
        dim1 = shp1[0]
        dim2 = shp2[0]
        xarray = []
        for r in range(dim2):
            row = []
            for  c in range(dim1):
                row.append(mpx[c])
            xarray.append(row)
        x = mparray(xarray, mpx.dtype)
        yarray = []
        for r in range(dim2):
            row = []
            for c in range(dim1):
                row.append(mpy[r])
            yarray.append(row)
        y = mparray(yarray, mpx.dtype)
        return (x,y)
        
    # creates a matrix from a 2-dimensional mparray        
    def asmatrix(self):
        if self.ndim != 2:
            raise ValueError("Cannot convert an array with  " + str(ndim) + " dimensions to a matrix")
        else:
            return Matrix.from_list(self.a, dtype = self.dtype)
            
    # creates a vector from a 1-dimensional mparray
    def asvector(self):
        if self.ndim != 1:
            raise ValueError("Cannot convert an array with  " + str(ndim) + " dimensions to a vector")
        else:
            return Vector.from_list(self.a, dtype = self.dtype)
            
            
        
#################################################
################## class Array ##################
################################################# 
"""
Array contains functionality for dealing with 
Python arrays (aka. lists).

"""

class Array:

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
        
    # count elements of list/array
    def count(array):
        shp = Array.shape(array)
        prod = 1
        for i in range(len(shp)):
            prod *= shp[i]
        return prod 
        
    # this function does delegate to reduce() 
    # the lmbda is applied to all elements of 
    # array with init_val being the aggregator
    def reduce_general(lmbda, array, init_val = None):
        if init_val == None:
            return reduce(lmbda, array)
        else:
            return reduce(lmbda, array, init_val)
        
    # calculate the sum of all elements in an array 
    # using init_val as the base value
    def sum(array, init_val = None):
        if init_val == None:
            return reduce(operator.add, array)
        else:
            return reduce(operator.add, array, init_val)
        
    # calculate the product of all elements in an array 
    # using init_val as the base value
    def mul(array, init_val = None):
        if init_val == None:
            return reduce(operator.mul, array)
        else:
            return reduce(operator.mul, array, init_val)
          
    # methods to create array with different dimensions filled
    # with init_value
    def create_1Darray(count, init_value = 0):
        return [init_value for i in range(0, count)]
        
    def create_2Darray(rowcount, colcount, init_value = 0):
        return [[init_value for i in range(0, colcount)] for j in range(0, rowcount)]

    # transpose() is defined for all 2Darrays. It returns a new
    # array with: new_array[r][c] = array[c][r] for all
    # valid (row,column)-combinations
    def transpose(array):
        n,m = Array.shape(array)
        result = Array.create_2Darray(m,n)
        for r in range(m):
            for c in range(n):
                result[r][c] = array[c][r]
        return result
        
    def create_3Darray(dim1count, dim2count, dim3count, init_value = 0):
        return [[[init_value for i in range(0, dim3count)] for j in range(0, dim2count)] for k in range(dim1count)]

    # split_1D splits a one-dimensional array in arg pieces if 
    # arg is an integer, or at the specified positions if arg 
    # is a list or tupel
    def split_1D(array, arg):
        n = len(array)
        if isinstance(arg, int):
            if arg == 0 or n % arg != 0:
                raise ValueError("equal split of array with size = " + str(n) + " impossible with arg = " + str(arg))
            else:
                result = []
                i = 0
                while i < n:
                    arr = []
                    for j in range(n // arg):
                        arr.append(array[i + j])
                    i += n // arg
                    result.append(arr)
            return result
        elif isinstance(arg, tuple) or isinstance(arg, list):
            last_pos = 0
            split_indices = [0]
            for pos in arg:
                if pos < 0 or pos >= n:
                    raise ValueError("attempt to split array of size = " + str(n) + " at nonexistent position " + str(pos))
                split_indices.append(pos)
            split_indices.append(len(array))
            split_indices = list(set(split_indices))
            split_indices.sort()
            result = []
            for i in range(len(split_indices)-1):
                tmp = []
                for j in range(split_indices[i], split_indices[i+1]):
                    tmp.append(array[j])
                result.append(tmp)
            return result
    
    # split_2D splits a two-dimensional array in arg pieces if 
    # arg is an integer, or at the specified positions if arg 
    # is a list or tupel. The split can be conducted along 
    # axis = 0 or axis = 1
    def split_2D(array, arg, axis = 0):
        n,m = Array.shape(array)
        if axis == 0:
            if isinstance(arg, int):
                if arg == 0 or m % arg != 0:
                    raise ValueError("equal split of array with size = " + str(m) + " impossible with arg = " + str(arg))
                else:
                    result = []
                    tmp = []
                    for i in range(n):
                        tmp.append(Array.split_1D(array[i], arg))
                    for j in range(len(tmp[0])):
                        single = []
                        for i in range(n):
                            single.append(tmp[i][j])
                        result.append(single)
                    return result
                        
            elif isinstance(arg, tuple) or isinstance(arg, list):
                result = []
                tmp = []
                for i in range(n):
                    tmp.append(Array.split_1D(array[i], arg))
                for j in range(len(tmp[0])):
                    single=[]
                    for i in range(n):
                        single.append(tmp[i][j])
                    result.append(single)
                return result
        else: # axis==1  
            if isinstance(arg, int):
                if arg == 0 or n % arg != 0:
                    raise ValueError("equal split of array with size = " + str(n) + " impossible with arg = " + str(arg))
            result = []
            array_tmp = Array.transpose(array)
            tmp = Array.split_2D(array_tmp, arg, axis = 0)
            
            for k in range(len(tmp)):
                result.append(Array.transpose(tmp[k]))
            return result
            
    # extract a subarray from array
    def subarray(array, top, bottom, left, right):
        shp = Array.shape(array)
        dim1 = shp[0]
        dim2 = shp[1]
        if not(left <= right and top <= bottom and right <= dim2 and bottom <= dim1):
            raise ValueError("index out of range")
            
        if left == right and top == bottom:
            return array[top][left]
        else: 
            result = []
            for r in range(bottom - top + 1):
                row = []
                for c in range(right - left + 1):
                    row.append(array[top + r][left + c])
                result.append(row)
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
                
    # returns the covariance matrix
    def covariance_matrix(dataset, rows = True):
        marray = mparray(dataset)
        return mparray.covariance_matrix(marray, rows)
        
    # returns the correlation_matrix
    def correlation_matrix(dataset, rows = True):
        marray = mparray(dataset)
        return mparray.correlation_matrix(marray, rows)
    
    # method to analyze the shape of a list or other sequence
    def shape_analyzer(a):
        def all(a, lambda_f):
            for i in range(len(a)):
                if not lambda_f(a[i]): return False
            return True
            
        if isinstance(a[0], list):
            a0len = len(a[0])
            if not all(a, lambda x: a0len == len(x)):
                raise ValueError("irregular sequence")
            else:
                return (len(a),) + Array.shape_analyzer(a[0])
        else:
            return (len(a),)
        
    # just a wrapper for shape_analyzer
    def shape(a):
        return Array.shape_analyzer(a)
        
    
    # diff_2d takes in each row a[r,c]-a[r,c-1] for c in 1 .. len(row)
    # if the original array has dimension dim1 x dim2, then the result
    # will have dimension dim1 x (dim2 - 1)
    def diff_2D(arr):
        dim1, dim2 = Array.shape(arr)
        result = []
        for r in range(dim1):
            row = []
            for c in range(1, dim2):
                row.append(arr[r][c] - arr[r][c-1])
            result.append(row)
        return result
        
    # same as diff_2D but for a one-dimensional array. The 
    # result has the length: len(arr) - 1
    def diff_1D(arr):
        result = []
        for i in range(1, len(arr)):
            result.append(arr[i]-arr[i-1])
        return result
        
    # sorting mparray. in_situ = True => array is sorted in place.
    # otherwise, the sorted array is returned leaving the original
    # array unchanged. The return value consists of two arrays,  
    # the first one containing the sorted array, the second one 
    # containing the information where the original items have
    # moved to due to sorting. As the name suggests the second
    # return value contains the indices of the original array 
    # elements, i.e. 0..len(self)-1
    def sort(self, in_situ = True):
        array, indices =  mparray.sort(self.a, in_situ = in_situ)
        if not in_situ:
            return mparray(array), mparray(indices)
    
    # mul_2d multiplies each element of arr with num  
    def mul_2D(num, arr):
        dim1, dim2 = Array.shape(arr)
        result = []
        for r in range(dim1):
            row = []
            for c in range(0, dim2):
                row.append(num * arr[r][c])
            result.append(row)
        return result
        
    # same as mul_2D, but for a one-dimensional array
    def mul_1D(num, arr):
        result = []
        for i in range(0, len(arr)):
            result.append(num * arr[i])
        return result
        
    # remove duplicates from a one-dimensional array 
    def remove_duplicates(arr):
        return list(set(arr))
        
    # This method sorts the passed array in-situ
    # It return an array of indices that shows
    # which element of the unsorted array was 
    # moved to which index due to sorting
    # for example,       
    #     array = [4,1,3,2,1,3,0]
    #     will end up in [0, 1, 1, 2, 3, 3, 4]
    #     after sort()
    # The returned index array looks like:
    #     [6, 1, 4, 3, 2, 5, 0]
    # Due to sorting the element formerly located
    # at index 6 is now at index 0 after sorting 
    # and the lement formerly located at position 0 
    # is at position 6 after sort()
    # if in_situ = False, the sort will be 
    # conducted on a copy of a so that a remains
    # unchanged
    def sort(array, in_situ = True):
        def quicksort(array, indices):
            smaller = []
            equal   = []
            larger  = []
            idx_smaller = []
            idx_equal   = []
            idx_larger  = []
        
            if len(array) > 1:
                pivot = array[0]
                for i in range(len(array)):
                    if array[i] < pivot:
                        smaller.append(array[i])
                        idx_smaller.append(indices[i])
                    elif array[i] == pivot:
                        equal.append(array[i])
                        idx_equal.append(indices[i])
                    elif array[i] > pivot:
                        larger.append(array[i])
                        idx_larger.append(indices[i])
                sma, sidx = quicksort(smaller, idx_smaller)
                equ, eidx = equal, idx_equal
                lar, lidx = quicksort(larger, idx_larger)
                return (sma + equ + lar, sidx + eidx + lidx)
            elif len(array) == 1: 
                return array, [indices[0]]
            else:
                return [],[]
                
        sortedarr, indices = quicksort(array, [i for i in range(len(array))])
        if in_situ:
            for i in range(len(array)):
                array[i] = sortedarr[i]
            return indices
        else:
            return sortedarr, indices
        
    # concatenate two rectangular arrays on axis 0 or 1 
    def concat(arr1, arr2, axis = 0):
        shp1 = Array.shape(arr1)
        shp2 = Array.shape(arr2)
        if len(shp1) > 2 or len(shp2) > 2:
            raise ValueError("concat not defined for arrays with dimension > 2")
        if axis == 1:
            if len(shp1) == 2 and len(shp2) == 2 and shp1[1] == shp2[1]:
                result = []
                for r in range(0, shp1[0]): 
                    result.append(arr1[r])
                for r in range(0, shp2[0]):
                    result.append(arr2[r])
                return result
            elif len(shp1) == 1 and len(shp2) == 2 and len(arr1) == shp2[1]:
                result == []
                result.append(arr1)
                for r in range(0, shp2[0]):
                    result.append(arr2[r])
                return result
            elif len(shp1) == 2 and len(shp2) == 1 and shp1[1] == len(arr2):
                result = []
                for r in range(0, shp1[0]): 
                    result.append(arr1[r])
                result.append(arr2)
                return result
            elif len(shp1) == 1 and len(shp2) == 1 and len(arr1) == len(arr2):
                result = []
                result.append(arr1)
                result.append(arr2)
                return [arr1, arr2]
            else: # dimensions don't fit
                raise ValueError("cannot concatenate array with different number of columns on axis 1")
        else: # axis == 0 
            if len(shp1) == 1 and len(shp2) == 1:
                return arr1 + arr2
            elif len(shp1) == 2 and len(shp2) == 2 and shp1[0] == shp2[0]:
                result = []
                for r in range(shp1[0]):
                    result.append(arr1[r]+arr2[r])    
                return result
            else: 
                raise ValueError("cannot concatenate array with different number of rows on axis 0")
            
    # vstack is based on concat. Two arrays are stacked horizontally
    def vstack(arr1, arr2):
        return Array.concat(arr1, arr2, axis = 0)
        
     # hstack is based on concat. Two arrays are stacked vertically
    def hstack(self):
        return Array.concat(arr1, arr2, axis = 1)
                           
    # summing up all array elements on axis 0 or 1
    def sum_2D(arr, axis = 0):
        shp = Array.shape(arr)
        result = []
        if axis == 0:
            for c in range(0, shp[1]):
                sum = 0
                for r in range(0, shp[0]):
                    sum += arr[r][c]   
                result.append(sum)
        else: # axis != 0 
            for r in range(0, shp[0]):
                sum = 0
                for c in range (0, shp[1]):
                    sum += arr[r][c]
                result.append(sum)
        return result
        
    # summing up all array elements
    def sum_1D(arr):
        if len(arr) == 0:
            return 0
        else:
            sum = 0
            for elem in arr:
                sum += elem
            return sum
        
    # calculates the mean of array elements
    def mean(arr):
        sum = 0
        for elem in arr:
            sum += elem
        return sum / len(arr)
        
    # calculates the standard deviation of array elemments
    def std_dev(arr):
        sum = 0
        mu = Array.mean(arr)
        for i in range(0, len(arr)):
            sum += (arr[i] - mu) ** 2
        res = math.sqrt(sum / (len(arr)-1))
        return res 
        
    # calculates the variance of array elements
    def variance(arr):
        return Array.std_dev(arr) ** 2
        
    # calculates the median of array elements
    def median(arr):
        a = deepcopy(arr)
        a.sort()
        len_a = len(a)
        if len(a) % 2 == 1:
            return a[len_a // 2]
        else:
            return (a[(len_a - 1)//2]+a[(len_a+1)//2])/2 
            
    # checks whether condition cond (lambda or function type)
    # holds for all elements of an array
    def all(array, cond):
        for i in range(0, len(array)):
            if not cond(array[i]):
                return False
        return True
    
    # checks whether condition cond (lambda or function type)
    # holds for at least one element of an array
    def any(array, cond):
        for i in range(0, len(array)):
            if cond(array[i]):
                return True
        return False
        
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
        
    # creates a linear distribution from start point startp to endpoint endp
    # consisting of size elements with the specified datatype dtype, If 
    # with_endp is set to True the distribution will contain the endpoint.
    def lin_distribution(startp, endp, size, with_endp = False, dtype = float):
        array = []
        if size == 1:
            sz = 1
        else:
            sz = size - 1 if with_endp else size
        
        if dtype == int:
            tmp = float((endp-startp) / sz)
            if sz > abs(endp-startp) or tmp - math.floor(tmp) > 0:
                raise ValueError("cannot create an equidistant distibution of integers in this interval")
        incr =(endp-startp)/sz
    
        for i in range(0, size):
            array.append(dtype(startp + i * incr))
        return array
        
    # creates a logarithmic distribution of size elements starting 
    # with base ** startp and ending with base ** endp (if with_endp 
    # is set to True)
    def log_distribution(startp, endp, size, base = 10.0, with_endp = True, dtype = float):
        array = []
        if size == 1:
            sz = 1
        else:
            sz = size - 1 if with_endp else size
        
        incr =(endp-startp)/sz
    
        for i in range(0, size):
            array.append(dtype(base ** (startp + i * incr)))
        return array
        
    # calculate the mean-normalized form of the 
    # input array: 
    def mean_normalization(array):
        maximum = max(array)
        mean    = Array.mean(array)
        result  = [(array[i] - mean) / maximum for i in range(0, len(array))]
        return result
        
    def euclidean_norm(array):
        sum = 0
        for i in range(len(array)):
            sum += array[i] ** 2
        return math.sqrt(sum)
            
    # calculate minima of array        
    def argmin(array, axis = None):
        if axis == None or Array.shape(array)[0] == 1:
            return min(array)
        elif axis == 0:
            result = []
            for i in range(0, Array.shape(array)[0]):
                result.append(min(array[i]))
            return result
        elif axis == 1:
            result = []
            for c in range(0, Array.shape(array)[1]):
                tmp = []
                for r in range(0, Array.shape(array)[0]):
                    tmp.append(array[r][c])
                result.append(min(tmp))
            return result
                
    # calculate maxima of array
    def argmax(array, axis = None):
        if axis == None or Array.shape(array)[0] == 1:
            return max(array)
        elif axis == 0:
            result = []
            for i in range(0, Array.shape(array)[0]):
                result.append(max(array[i]))
            return result
        elif axis == 1:
            result = []
            for c in range(0, Array.shape(array)[1]):
                tmp = []
                for r in range(0, Array.shape(array)[0]):
                    tmp.append(array[r][c])
                result.append(max(tmp))
            return result
            
    # the predicate function/lambda is applied to all 
    # elements in the array. All elements satisfying the 
    # predicate are appended to the result list 
    # returns result list
    def find(lambda_f, array):
        result = [] 
        dim1, dim2 = Array.shape(array)
        for i in range(0, dim1):
            for j in range(0, dim2):
                if lambda_f(array[i][j]):
                    result.append(array[i][j])
        return result
    
    # same as find, but returns as a list all of all indices
    # where the condition lambda_f holds            
    def find_where(lambda_f, array):
        result = [] 
        dim1, dim2 = Array.shape(array)
        for i in range(0, dim1):
            for j in range(0, dim2):
                if lambda_f(array[i][j]):
                    result.append((i,j))
        return result
        
    # apply_1D applies a function to all elements 
    # of a 1D array
    def apply_1D(lambda_f, array):
        result = []
        for i in range(len(array)):
            result.append(lambda_f(array[i]))
        return result
            
    # apply_2D applies a function to all elements
    # of a 2D array
    def apply_2D(lambda_f, array):
        d1,d2 = Array.shape(array)
        result = []
        for i in range(d1):
            row = []
            for j in range(d2):
                row.append(lambda_f(array[i][j]))
            result.append(row)
        return result
    
    # An 2D array can be rotated 90° to the left or right.
    # if left = True  => left  rotation
    # if left = False => right rotation           
    def rotate(array, left = True):
        n,m = Array.shape(array)
        res = []
        if left:
            for c in range(m):
                row = []
                for r in range(n):
                    row.append(array[r][m-c-1])
                res.append(row)
            return res
        else: 
            for c in range(m):
                row = []
                for r in range(n):
                    row.append(array[n-r-1][c])
                res.append(row)
            return res
        
    # the following helper functions are wrappers
    # that use Array.apply_1D. They are implemented 
    # for the users convenience.
    
    # sin for one dimensional arrays based on 
    # apply_1D
    def sin(array):
        res = Array.apply_1D(math.sin, array)
        return res
        
    # cos for one dimensional arrays based on 
    # apply_1D
    def cos(array):
        res = Array.apply_1D(math.cos, array)
        return res
        
    # tan for one dimensional arrays based on 
    # apply_1D
    def tan(array):
        res = Array.apply_1D(math.tan, array)
        return res
                

    # exp for one dimensional arrays based on 
    # apply_1D
    def exp(array):
        res = Array.apply_1D(math.exp, array)
        return res
        
    # log for one dimensional arrays based on 
    # apply_1D
    def log(array, base):
        res = Array.apply_1D(lambda x: math.log(x,base), array)
        return res
        
    # pow for one dimensional arrays based on 
    # apply_1D
    def pow(array, exponent):
        res = Array.apply_1D(lambda x: pow(x,exponent), array)
        return res
        
    # get a two dimensional array filled with random numbers
    def random_2D(shp, fromvalue, tovalue, dtype, seedval = None):
        if seedval != None:
            seed(seedval)
        rows = shp[0]
        cols = shp[1]
        array = [[dtype(0) for i in range(cols)] for j in range(rows)]
        for r in range(0, rows):
            for c in range(0, cols):
                if dtype == int:
                    array[r][c] = int(randrange(fromvalue, tovalue))
                else:
                    array[r][c] = dtype(uniform(fromvalue, tovalue))
        return array
        
    # get a one-dimensional array filled with random numbers
    def random_1D(length, fromvalue, tovalue, dtype, seedval = None):
        if seedval != None:
            seed(seedval)
        array = [dtype(0) for i in range(length)]
        if dtype == int:
            for i in range(length):
                array[i] = int(randrange(fromvalue, tovalue))
        else:
            for i in range(length):
                array[i] = dtype(uniform(fromvalue, tovalue))
        return array
        
    def asmatrix(lst):
        shape = Array.shape(lst)
        if len(shape) != 2:
            raise ValueError("cannot convert a list/array with " + str(len(shape)) + " dimensions to a matrix")
        else:
            dtype = type(lst[0][0])
            return Matrix.from_list(lst, dtype=dtype)
            
    def asvector(lst):
        shape = Array.shape(lst)
        if len(shape) != 1:
            raise ValueError("cannot convert a list/array with " + str(len(shape)) + " dimensions to a vector")
        else:
            dtype = type(lst[0])
            return Vector.from_list(lst, dtype=dtype)
            
        
#################################################
################## class Matrix #################
#################################################


class Matrix:
    
    fstring = '{:4}'
    left_sep = ""
    right_sep = ""
    
    # create size1 x size2 matrix with optional initial values
    # dtype specifies the desired data type of matrix elements
    def __init__(self, dim1, dim2, dtype = float, init_value = 0):
        if dim1 <= 0 or dim2 <= 0: raise ValueError("a matrix must have positive dimensions")
        self.m = [[dtype(init_value) for c in range(dim2)] for r in range(dim1)]
        self.dim1 = dim1
        self.dim2 = dim2
        self.dtype = dtype
        
    # get the shape of the matrix, i.e. its numbers of rows and columns
    @property
    def shape(self):
        return (self.dim1, self.dim2)
        
        
    # returns the total number of elements in matrix
    @property
    def count(self):
        return self.dim1 * self.dim2
        
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
    @property
    def T(self):
        return Matrix.from_list(Array.transpose(self.m), dtype = self.dtype)
        
    @property
    def H(self):
        if self.dtype != complex: return self.T
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
            slclen =   Array.slice_length(arg, [0 for i in range(0,self.dim1)])
            dim1, dim2  = Array.shape(val)
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
                    length_required = Array.slice_length(s2, self.m[s1])
                    dim1, dim2 = Array.shape(val)
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
                    length_required = Array.slice_length(s1,self.column_vector(0))
                    if not isinstance(val, list):
                        raise TypeError("list expected as argument")
                    dim1, dim2 = Array.shape(val)
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
                    length1_required = Array.slice_length(s1, self.column_vector(0).v)
                    length2_required = Array.slice_length(s2, self.row_vector(0).v)
                    if (length1_required, length2_required) != Array.shape(val):
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
            dim1, dim2 = Array.shape(val)
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
        
    # extract a submatrix from a matrix given in self
    def submatrix(self, top, bottom, left, right):
        if not(left <= right and top <= bottom and right <= self.dim2 and bottom <= self.dim1):
            raise ValueError("index out of range")
            
        if left == right and top == bottom:
            return self.m[top][left]
        else: 
            result = []
            for r in range(bottom - top + 1):
                row = []
                for c in range(right - left + 1):
                    row.append(self.m[top + r][left + c])
                result.append(row)
            return Matrix.from_list(result, dtype = self.dtype)
                
    # contains operator for searching elements of the matrix
    def __contains__(self, key):
        for r in range(self.dim1):
            if key in self.m[r]: return True
        return False
        
    # recursive calculation of the determinant using sub-matrices
    # returns a ValueError is matrix is not quadratic
    def det(self):
        if self.dim1 != self.dim2:
            raise ValueError("Determinants can only be calculated for quadratic matrices")
        if self.dim1 == 1:
            return self.m[0][0]
        else: # developing around 0,0 (i.e., row 0 --> )
            det = self.dtype(0)
            for c in range(0, self.dim1):
                if c % 2 == 0:
                    factor =  self.dtype(1)
                else:
                    factor = self.dtype(-1)
                det += factor * self.m[0][c] * self.minor(0, c).det()
            return det 
    
    
    # sets all elements below the specified diag to 0 
    # diag = 0 is the main diagonal        
    def upper_triangle(self, diag = 0):
        dim1, dim2 = self.shape
        res = self.clone()
        max_axis = min(dim1,dim2)
        for r in range(0, dim1):
            for c in range(0, dim2):
                if (c - diag) > r: 
                    res[r,c] = 0
        return res
        
    # sets all elements above the specified diag to 0 
    # diag = 0 is the main diagonal        
    def lower_triangle(self, diag = 0):
        dim1, dim2 = self.shape
        res = self.clone()
        max_axis = min(dim1,dim2)
        for r in range(0, dim1):
            for c in range(0, dim2):
                if (r - diag) > c: 
                    res[r,c] = 0
        return res
        
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
        return self.cofactor_matrix().T
        
    # creates the inverse matrix iff det != 0. Raises a  
    # ValueError if that is not the case
    def inverse_matrix(self):
        if self.det() == 0:
            raise ValueError("matrix with det == 0 has no inverse")
        else:
            return self.adjoint_matrix().scalar_product(self.dtype(1) / self.det())
            
    # inverse diagonal creates the inverse of a diagonal matrix
    def inverse_diagonal(self):
        dim1, dim2 = self.shape
        if (dim1 != dim2):
            raise ValueError("inverse_diagonal() only defined for square matrices")
        if not self.is_diagonal():
            raise ValueError("inverse_diagonal() only defined for diagonal matrices")
        diag = [] 
        for i in range(dim1):
            if self[i,i] == 0:
                raise ValueError("inverse_diagonal() only defined for diagonal matrices where all diagonal values are not zero")
            diag.append(1 / self[i,i])
        return self.diagonal_matrix(dtype = self.dtype)
        
    # creates a diagonal-matrix with diagonal-elements populated with 
    # inverse values of the given diagonal matrix self
    def inverse_pseudo_diagonal(self):
        dim1, dim2 = self.shape
        lst = []
        for i in range(min(dim1, dim2)):
            if self[i, i] != 0:
                lst.append(1 / self[i,i])
            else:
                lst.append(0)
        return Matrix.pseudo_diagonal_matrix(self.shape, lst, self.dtype)
        
            
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
        if not self.is_orthonormal():
            return self.inverse_matrix() * vector
        else:
            return self.T * vector
            
    # requires a tridiagonal matrix a vector with len(vector)==matrix.dim1
    # returns the solution x of the equation matrix@x = vector
    def thomas_algorithm(self, d):
        if not self.is_tridiagonal():
            raise ValueError("thomas algorithm works with tridiagonal matrices only")
        n, _ = self.shape
        if len(d) != n:
            raise ValueError("vector must have same length as self.dim1")
        if self.det() == 0:
            raise ValueError("matrix must be invertible")
        
        c_ = [0 for i in range(n)]
        d_ = [0 for i in range(n)]
        x  = [0 for i in range(n)]
        a  = [0] + self.diagonal(-1)
        b  = self.diagonal(0)
        c  = self.diagonal(+1) + [0]
        
        # forward propagation
        c_[0] = c[0]/b[0]
        d_[0] = d[0]/b[0]
        for i in range(1,n-1):
            c_[i] = c[i] / (b[i] - c_[i-1]*a[i])
            d_[i] = (d[i]-d_[i-1] * a[i]) / (b[i]-c_[i-1]*a[i])
        d_[n-1] = (d[n-1]-d_[n-2] * a[n-1]) / (b[n-1]-c_[n-2]*a[n-1])
                    
        # backward propagation
        x[n-1] = d_[n-1]
        for i in range(n-2, -1, -1):
            x[i] = d_[i]-c_[i]*x[i+1]
            
        return Vector.from_list(x)
                    
    # The Jacobi method is an approximation approach to solve 
    # a linear equation system. As termination criterion this 
    # implementation uses the change of the result vector between 
    # different iterations. If the tolerance is achieved the 
    # method returns the result. If after max_iter iterations
    # the required tolerance is not achieved then None is returned.
    def jacobi_method(self, b, tolerance = 1E-20, max_iter = 100):
        _max_iter = max_iter
        _tolerance = 1E-20
        n,_ = self.shape
        if not self.is_square():
            raise ValueError("only square matrices can be used")
        if len(b) != n:
            raise ValueError("dimensions of b and self are not equal")
        if self.det == 0:
            raise ValueError("algorithm only works for matrices with det <> 0")
        
        x_old = Vector.random_vector(n, -2, +2, dtype = self.dtype) # initialze
        iter = 0
        while iter < _max_iter:
            x = b.clone()
            for i in range(n):
                for j in range(n):
                    if j != i:
                        x[i] = x[i] - self[i,j] * x_old[j]
                x[i] = x[i] / self[i,i]
            if (x_old - x).euclidean_norm() <= tolerance:
                return x
            x_old = x
        return None
        
    # NOTE: svd() is the only function taken from numpy as long as there is no 
    # implementation in mathplus. It returns U, s, VT
    def svd(self, full_matrices = True):
        U,s,VT = np.linalg.svd(self.m)
        return Matrix.from_list(U), s.tolist(), Matrix.from_list(VT)

        
    # computes the pseudo inverse of a matrix using the Moore-Penrose method
    def pseudo_inverse(self):
        U, s, VT = self.svd()
        V_ = VT.T
        U_ = U.H # since U, V are unitary matrices
        shp1 = V_.shape
        shp2 = U_.shape
        shp = (shp1[1], shp2[0])
        S_ = Matrix.pseudo_diagonal_matrix(shp, s, self.dtype).inverse_pseudo_diagonal()
        return V_ @ (S_ @ U_)
        
    # get row vector for row
    def row_vector(self, row):
        v = Vector(self.dim2, dtype = self.dtype, transposed = True)
        v.v = self.m[row]
        return v
        
    # get all row vectors at once
    def all_row_vectors(self):
        shape = self.shape
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
        shape = self.shape
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
    # diag == 0 => main diagonal
    # diag = -1 => diagonal one step below the main diagonal
    # diag =  1 => diagonal one step above the main diagonal
    # ... 
    # diag must exist, otherwise a ValueError is thrown
    def diagonal(self, diag = 0):
        dim1, dim2 = self.shape
        max_diag = min(dim1, dim2)-1
        if (abs(diag) > max_diag):
            raise ValueError("diag must be in [" + str(-max_diag) + "," + str(max_diag) + "]")
        list = []
        for r in range(0, dim1):
            for c in range(0,dim2):
                if c - diag == r:
                    list.append(self[r,c])
        return list
        
    # fills the diagonal of a square matrix with a value 
    # value is the value to be filled into the diagonal
    # diag:
    # diag == 0 => main diagonal
    # diag = -1 => diagonal one step below the main diagonal
    # diag =  1 => diagonal one step above the main diagonal
    # ... 
    # diag must exist, otherwise a Valuuerror is thrown
    # If in_situ = True => the value will be filled directly
    # in the matrix (self)
    # otherwise         => a copy of the matrix self will be 
    # used for this purpose
    def fill_diagonal(self, value, diag = 0, in_situ = False):
        if in_situ:
            m = self
        else:
            m = self.clone()
            
        dim1, dim2 = self.shape
        max_diag = min(dim1, dim2)-1
        if (abs(diag) > max_diag):
            raise ValueError("diag must be in [" + str(-max_diag) + "," + str(max_diag) + "]")
        for r in range(0, dim1):
            for c in range(0,dim2):
                if c - diag == r:
                    m[r,c] = value
        return m
        
    # checks whether matrix is in tridiagonal form
    def is_tridiagonal(self):
        if not self.is_square():
            raise ValueError("is_tridiagonal only defined for square matrices")
        dim1, dim2 = self.shape
        for d in range(-dim1+1, dim1):
            n = Array.euclidean_norm(self.diagonal(d))
            if not d in range(-1,2) and n != 0:
                return False
            else: 
                continue
        return True
        
    # checks whether matrix is diagonal
    def is_diagonal(self):
        dim1, dim2 = self.shape
        m = min(dim1,dim2)
        for i in range(m):
            for j in range(m):
                if i == j:
                    continue
                else:
                    if self[i,j] != 0:
                        return False
        for i in range(m, dim1):
            for j in range(m, dim2):
                if self[i, j] != 0:
                    return False
        return True
    
    # add two matrices with each other
    #  if their sizes are not the same, a ValueError is raised
    def __add__(self, other):
        if self.dim1 != other.dim1 or self.dim2 != other.dim2:
            raise ValueError("matrices must have similar sizes")
        else:
            m = Matrix(self.dim1, self.dim2, dtype = self.dtype)
            for r in range(0, self.dim1):
                for c in range (0, self.dim2):
                    m.m[r][c] = self.m[r][c] + other.m[r][c]
            return m
            
    # this is the += operator that changes self in contrast to __add__
    def __iadd__(self, other):
        if self.dim1 != other.dim1 or self.dim2 != other.dim2:
            raise ValueError("matrices must have similar sizes")
        else:
            for r in range(0, self.dim1):
                for c in range (0, self.dim2):
                    self.m[r][c] += other.m[r][c]
        
            
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
        elif isinstance(other, numbers.Number):
            return self.scalar_product(other)
        else:
            raise TypeError("second argument must be matrix or vector")
            
    def __matmul__(self, other):
        return self * other
        
    def multiply(self, other):
        if not isinstance(other, Matrix):
            m = deepcopy(self)
            for r in range(self.dim1):
                for c in range(self.dim2):
                    m.m[r][c] *=  other
            return m
        if self.dim1 != other.dim1 or self.dim2 != other.dim2:
            raise ValueError("matrices must have equal shape for pairwise multiplication")
        m = deepcopy(self)
        for r in range(self.dim1):
            for c in range(self.dim2):
                m.m[r][c] *= other.m[r][c]
        return m
            
    # subtracting one matrix from the other. Raises ValueError if sizes are
    # not equal
    def __sub__(self, other):
        if self.dim1 != other.dim1 or self.dim2 != other.dim2:
            raise ValueError("matrices must have similar sizes")
        else:
            m = Matrix(self.dim1, self.dim2, dtype = self.dtype)
            for r in range(0, self.dim1):
                for c in range (0, self.dim2):
                    m.m[r][c] = self.m[r][c] - other.m[r][c]
            return m
            
    # implementation of -= operator. Here, self will be changed
    # in contrast to __sub()__
    def __isub__(self, other):
        if self.dim1 != other.dim1 or self.dim2 != other.dim2:
            raise ValueError("matrices must have similar sizes")
        else:
            for r in range(0, self.dim1):
                for c in range (0, self.dim2):
                    self.m[r][c] -= other.m[r][c]
            
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
            return self == self.T
            
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
            
    def is_unitary(self):
        if not self.is_square():
            raise ValueError("only square matrices permitted")
        if self.det() == 0:
            return False
        else:
            return self.H == self.inverse_matrix()
            
    def is_orthonormal(self):
        if not self.is_square():
            raise ValueError("orthomality is only defined for square matrices")
        return self.T @ self == Matrix.identity(self.dim1)
            
    # calculate the standard norm for column vectors
    def norm(self):
        n = self.dtype(0)
        for c in range(0, self.dim2):
            n = max(n, self.column_vector(c).norm())
        return n
        
    def euclidean_norm(self):
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
        
    # calculate exp(t*Matrix)
    def exp(self, t = 1):
        max_expansion = 20
        shape = self.shape
        if shape[0] != shape[1]:
            raise ValueError("exp only defined for square matrices")
        if t == 0:
            return Matrix.identity(shape[0])
        else:
            res = Matrix.identity(shape[0]) + self.scalar_product(t)
            for i in range(2, max_expansion + 1):
                res += (self ** i).scalar_product(t**i/Common.fac(i))
            return res
            
    # calculate sin of matrix    
    def sin(self):
        max_expansion = 20
        shape = self.shape
        if shape[0] != shape[1]:
            raise ValueError("sin only defined for square matrices")
        res = Matrix.identity(shape[0])
        for i in range(2, max_expansion + 1):
            res +=  (self ** (2*i+1)).scalar_product((-1 ** i)/Common.fac(2*i))
        return res
        
    # calculate cosine of matrix 
    def cos(self):
        max_expansion = 20
        shape = self.shape
        if shape[0] != shape[1]:
            raise ValueError("cos only defined for square matrices")
        res = self.clone()
        for i in range(1, max_expansion + 1):
            res +=  (self ** (2*i)).scalar_product((-1 ** i)/Common.fac(2*i))
        return res
    
                
    # multiply all matrix elements with n
    def mult_n_times(self, n):
        if not self.is_square():
            raise ValueError("can only multiply a square matrix with itself")
        if n == 0:
            return Matrix.identity(self.dim1, dtype = self.dtype)
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
        try:
            dim2 = len(_list[0])
        except TypeError:
            m = Matrix(1,1,dtype=dtype)
            m[0][0] = _list[0]
            return m
        
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
        
    # return an mparray
    def to_mparray(self):
        return mparray(self.a, dtype=self.dtype)
        
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
            
    # if a pattern as an array is passed to from_pattern
    # then rows are assembled by concatenating the pattern
    # d2 x to build a row which is then repeated d1 times 
    # to build the matrix returned to the caller.
    # for example, Matrix.from_pattern(2,3,[1,2]) leads to 
    #     [[1,2,1,2,1,2],  
    #      [1,2,1,2,1,2]]
    #
    def from_pattern(d1, d2, pattern):
        if len(pattern) == 0:
            raise ValueError("pattern must not be empty")
        row = []
        for i in range(d2): row += pattern
        mat = [row for j in range(d1)]
        return Matrix.from_list(mat)
        
    def reshape(self, shape, dtype = float):
        if shape == None: 
            raise ValueError("shape must not be None")
        elif shape[0] == self.dim1 and shape[1] == self.dim2:
            return self.clone()
        elif self.dim1 * self.dim2 != shape[0] * shape[1]:
            raise ValueError("shape does not correspond to dim1*dim2")
        else:
            list = self.to_flat_list()
            return Matrix.from_flat_list(list, shape, dtype = self.dtype)
            
    # apply applies lambda to each element of matrix
    def apply(self, lambda_f):
        m = Matrix(self.dim1, self.dim2, dtype = self.dtype)
        for r in range(0, self.dim1):
            for c in range(0, self.dim2):
                m.m[r][c] = lambda_f(self.m[r][c])
        return m
        
    # apply lambda_f to a single column of the matrix. returns new matrix 
    # with changed column
    def apply_column(self, column, lambda_f):
        m = Matrix(self.dim1, self.dim2, dtype = self.dtype)
        if column > self.dim2:
            raise ValueError("column " + str(column) + " does not exist")
        for r in range(0, self.dim1):
            m.m[r][column] = lambda_f(self.m[r][column])
        return m
        
    # apply lambda_f to a single row of the matrix. returns new matrix 
    # with changed row
    def apply_row(self, row, lambda_f):
        m = Matrix(self.dim1, self.dim2, dtype = self.dtype)
        if row > self.dim1:
            raise ValueError("row " + str(row) + " does not exist")
        for c in range(0, self.dim2):
            m.m[row][c] = lambda_f(self.m[row][c])
        return m
        
    # like apply, but with lambda getting called with
    # row, col, value at (row,col) 
    def apply2(self, lambda_f):
        m = Matrix(self.dim1, self.dim2, dtype = self.dtype)
        for r in range(0, self.dim1):
            for c in range(0, self.dim2):
                m.m[r][c] = lambda_f(r, c, self.m[r][c])
        return m
        
    # find returns  all matrix elements that satisfy the 
    # predicate lambda_f as a list
    def find(self, lambda_f):
        return Array.find(lambda_f, self.m)
        
    # same as find, but does not return the elements 
    # but the index-locations where the condition lambda_f
    # holds as a list
    def find_where(self, lambda_f):
        return Array.find_where(lambda_f, self.m)
        
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
                shape = vec_array[0].shape
                m = Matrix(len(vec_array), shape[0])
                i = 0
                for vec in vec_array:
                    v_shape = vec.shape
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
                shape = vec_array[0].shape
                m = Matrix(shape[0], len(vec_array))
                i = 0
                for col in range(0, len(vec_array)):
                    vec = vec_array[col]
                    v_shape = vec.shape
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
        shape = self.shape
        e = [None for i in range(0, shape[1])]
        u = [None for i in range(0, shape[1])]
        if not shape[1] >= shape[0]:
            raise ValueError("number of columns must be >= number of rows")
        u[0] = a[0]
        e[0] = u[0].scalar_product(1 / u[0].euclidean_norm())
        u[1] = a[1] - e[0] * (a[1].T*e[0])
        e[1] = u[1].scalar_product(1 / u[1].euclidean_norm())
        for k in range(2, shape[1]):
            u[k] = a[k]
            for i in range(0,k):
                u[k] = u[k]-e[i] * (a[k].T * e[i])
            e[k] = u[k].scalar_product(1/u[k].euclidean_norm())
        Q = Matrix.from_column_vectors(e)
        R = Matrix(shape[1], shape[0], dtype = self.dtype)
        for i in range(0, shape[1]):
            for j in range(0, shape[0]):
                if i > j: R.m[i][j] = 0
                else:
                    R.m[i][j] = a[j].T * e[i]
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
            diff = abs(A_new - A_orig).euclidean_norm()
            i += 1
        return (A_new.diagonal(), A_new.all_column_vectors())
        
    # LU decomposition of square matrices using Gaussian decomposition
    def lu_decomposition(self):
        m, n = self.shape
        if not self.is_square():
            raise ValueError("LU decomposition only defined for square matrices")
    
        l = self.clone()
        u = Matrix(n, n, dtype = self.dtype, init_value=self.dtype(0))
    
        # compute l, u in-place in l
        for k in range(0, n-1):
            if l[k,k] == 0:
                break
            
            for i in range(k+1,n):
                l[i,k] = l[i,k]/l[k,k]
            
            for j in range(k+1, n):
                for i in range(k+1, n):
                    l[i,j] -= l[i,k] * l[k,j]
                
        # separate result in l and u            
        for r in range(0,n):
            for c in range(0,n):
                if r > c:
                    u[r,c] = 0
                elif r == c:
                    u[r,c] = l[r,c]
                    l[r,c] = 1
                else:
                    u[r,c] = l[r,c]
                    l[r,c] = 0
        return l, u    
        
        
    # the Cholesky decomposition tries to find a matrix L 
    # for the symmetric / hermitian and positive matrix A 
    # so that A = L@L.T 
    def cholesky_decomposition(self):
        a = self
        if not a.is_square():
            raise ValueError("Cholesky decomposition only defined for square matrices")
        if not a.is_hermitian():
            raise ValueError("matrix must be symmetric or hermitian")
        n = a.dim1
        l = Matrix(n, n, dtype = a.dtype)
        for i in range(n):
            for j in range(i+1):
                sum = 0
                for k in range(j):
                    sum += l[i,k] * Common.conj(l[j,k])
                if (i == j):
                    if sum > a[i,i]:
                        raise ValueError("matrix is not positive definite")
                    else:
                        l[i,j] = math.sqrt(a[i,i] - sum)
                else:
                    l[i,j] = (1.0 / l[j,j] * (a[i,j] - sum))
        return l
        

    # computing the characteristic polynomial of a matrix
    # using the Faddeev–LeVerrier algorithm
    # M0 = 0                    ,   cn = 1 
    # M1 = I                    ,   cn-1 = -tr(A)
    # ....
    # Mm = Sum cn-m+k * A^k-1   , cn-m = -1/m * Sum cn-m+k * tr(A^k)
    #     k = 1,..,m                          k = 1, .., m
    # results are returned as a coefficient list [a0,a1,a2, ... , an]
    # which implies the polynomial is defined as
    #  p(x) = a0 + a1 * x + a2 * x**2 + an * x**n
    # 
    def char_poly(self):
        if not self.is_square():
            raise ValueError("characteristic polynomials do only exist for square matrices")
        n = self.dim1
        if n == 1:
            return -self.tr()
        coeff_list = [0 for i in range(0, n+1)]
        coeff_list[n]   = 1 # cn
        coeff_list[n-1] = -self.tr()
        for m in range(2,n+1):
            sum = 0
            for k in range(1, m+1):
                sum += (self ** k).tr() * coeff_list[n-m+k] 
            coeff_list[n-m] = -1/m * sum
        return coeff_list
        
    # creates a diagonal matrix with the list 
    # elements populating the diagonal
    def diagonal_matrix(lst, dtype = float):
        m = Matrix(len(lst), len(lst), dtype)
        for i in range(0, len(lst)):
            m[i,i] = lst[i]
        return m
        
    # creates a m x n-matrix with diagonal populated using lst 
    # shp: the desired size of the matrix
    # lst: the list of elemens the matrix diagonal is populated with
    # dtype: the type of the matrix elements
    # returns: the diagonal matrix
    def pseudo_diagonal_matrix(shp, lst, dtype = float):
        dim1, dim2 = shp
        n = len(lst)
        if n > min(dim1, dim2):
            raise ValueError("passed list has more elements than the diagonal in pseudo_diagonal_matrix()")
        m = Matrix(dim1, dim2, dtype, init_value = 0)
        for i in range(n):
            m[i,i] = lst[i]
        return m
        
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
        
    def rotation3D_x(angle_x, dtype=float):
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
        
    def rotation3D_y(angle_y, dtype=float):
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
        
    def rotation3D_z(angle_z, dtype = float):
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
        rot_x = Matrix.rotation3D_z(angle_x, dtype)
        rot_y = Matrix.rotation3D_y(angle_y, dtype)
        rot_z = Matrix.rotation3D_z(angle_z, dtype)
        return rot_z @ rot_y @ rot_x
    
    # remove a row from self and return result as a new matrix 
    def remove_row(self, i):
        (dim1, dim2) = self.shape
        if not i in range(0, dim1):
            raise ValueError("row does not exist")
        else:
            dim1 -= 1
            row_vectors = self.all_row_vectors()
            row_vectors.remove(row_vectors[i])
            return Matrix.from_row_vectors(row_vectors)
            
    # remove a column from self and return result as a new matrix
    def remove_column(self, j):
        (dim1, dim2) = self.shape
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
        dim1, dim2 = self.shape
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
        dim1, dim2 = self.shape
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
        dim1, dim2 = self.shape
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
        dim1, dim2 = self.shape
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
            
    # Implementation of standard operators __iter__ and 
    # __next__ for matrices. 
    # Note: this iterator traverses all row vectors.
    # To traverse each row vector, use the Vector implementation
    # of __iter__, __next__.
    
    def __iter__(self):
        self.row = 0
        return self
        
    def __next__(self):
        d1, _ = self.shape
        if self.row < d1:
            result = self.row_vector(self.row)
            self.row += 1
            return result
        else:
            raise StopIteration
            
                  
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
        return Array.sum(self.m, axis)
        
    # concatenation of the rows or columns of matrix other
    # to matrix self depending on axis
    def concatenate(self, other, axis = 0):
        array =  Array.concatenate(self.m, other.m, axis)
        m = Matrix.from_list(array, dtype = self.dtype)
        return m
        
    # this method stacks the rows or columns of a m x n-matrix
    # on top of each other which results in a 1 x m*n row vector 
    # (axis == 0) or in a m*n x 1 column vector
    def vectorize(self, axis = 0):
        array = []
        if axis == 0:
            for i in range(0, self.dim1):
                array += self.row_vector(i).v
            return Vector.from_list(array, dtype = self.dtype, transposed = True)
        else: # axis != 0 
            for i in range(0, self.dim2):
                array += self.column_vector(i).v
            return Vector.from_list(array, dtype = self.dtype, transposed = False)
        
    # 2-dim array required as input 
    def covariance_matrix(array, rows):
        if isinstance(array, list):
            return mparray.covariance_matrix(mparray(array), rows)
        elif isinstance(array, mparray):
            return mparray.covariance_matrix(array)
        else:
            raise ValueError("either list or mparray object expected as argument in Matrix.covariance_matrix()")
        
   # 2-dim array required as input 
    def correlation_matrix(array, rows):
        if isinstance(array, list):
            return mparray.correlation_matrix(mparray(array), rows)
        elif isinstance(array, mparray):
            return mparray.correlation_matrix(array)
        else:
            raise ValueError("either list or mparray object expected as argument in Matrix.correlation_matrix()")
            
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
    @property
    def shape(self):
        if self.is_transposed():
            return (len(self), True)
        else:
            return (len(self), False)
    
    # returns number of elements in vector (same as len(vector))        
    @property    
    def count(self):
        return len(self)
        
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
        return str(self)
    
    # vector transposition        
    @property
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
        elif isinstance(other, numbers.Number):
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
                       
    # pairwise multiplication of vectors
    def multiply(self, other):
        if not isinstance(other, Vector):
            v = deepcopy(self)
            for i in len(v): v[i] *= other
        if len(self.v) != len(other.v):
            raise ValueError("vectors must have same length for pairwise multiplication")
        m = deepcopy(self)
        for r in range(len(self.v)):
            m.v[r] *= other.v[r]
        return m
    
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
      
     # this is the += operator that overwrites self. If you don't
     # want that behavior, use a = a + b instead
    def __iadd__(self, other):
        if len(other) != len(self):
            raise ValueError("incompatible lengths of vectors")
        elif self._transposed != other._transposed:
            raise ValueError("transposed and not transposed vectors cannot be added")
        else:
            for i in range(0, len(self)): self[i] += other[i]
            
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
        
    # contains operator for searching elements of the vector
    def __contains__(self, key):
        return key in self.v
        
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
            res = Vector(len(self), dtype = self.dtype, transposed = self._transposed)
            for i in range(0, len(self)): res[i] = self[i] - other[i]
            return res
            
    # implementation of -= operator. This operator overwrites self.
    # If you don't want that behavior, use a = a - b instead
    def __isub__(self, other):
        if len(other) != len(self):
            raise ValueError("incompatible lengths of vectors")
        elif self._transposed != other._transposed:
            raise ValueError("transposed and not transposed vectors cannot be subtracted from each other")
        else:
            for i in range(0, len(self)): self[i] -= other[i]
            
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
        
    # return dot x product 
    def dot(self, other):
        if not isinstance(other, Vector):
            raise TypeError("wrong type of second argument")
        elif len(self) != len(other):
            raise ValueError("lengths of self and other are different")
        else:
            sum = self.dtype(0)
            for i in range(0, len(self)):
                sum += self[i] * other[i]
            return sum
            
    # get euclidean norm of vector
    def euclidean_norm(self):
        return math.sqrt(self.dot(self))
        
    def l2_norm(self, other):
        return (self-other).euclidean_norm()
   
    def l1_norm(self, other):
        delta = self - other
        sum = 0
        for i in range(len(delta)):
            sum += abs(self.v[i] - other.v[i])
        return sum
        
    def lp_norm(self, other, p):
        sum = 0
        for i in range(len(self)):
            sum += abs(self[i] - other[i]) ** p
        return sum ** (1 / p)
                   
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
        
    def add_multiple_of_vector(self, other, factor):
        if len(self) != len(other):
            raise ValueError("dimensions of vectors differ")
        elif self.is_transposed() != other.is_transposed():
            raise ValueError("vectors must both be transposed or not transposed")
        else:
            vector = deepcopy(self)
            for i in range(0, len(self)):
                vector[i] += other[i] * factor
            return vector
                
    # check whether one vector is orthogonal to the other
    def is_orthogonal_to(self, other):
        if len(other) != len(self):
            raise ValueError("incompatible lengths of vectors")
        if self.is_transposed() or other.is_transposed():
            raise ValueError("vectors must not be transposed")
        else:
            return self.cross_product(other) == 0
    
    # proj_u(v) := <u, u>/<v, v> * u 
    # <u, v> is the dot product/inner product
    def proj(u, v):
        return u.scalar_product(u.dot(v)/u.dot(u))
        
    # orthonormalization of vectors following the 
    # Gram-Schmidt Process.
    # Input v  -> a list of k vectors
    # returns a tupel (u, e) where u denotes a  
    # system of k orthogonal vectors and e is the 
    # set of k corresponding normalized vectors
    def orthonormalize_vectors(v):
        k = len(v)
        if k == 0:
            return [] 
        
        u = [None for i in range(0, k)]
        e = [None for i in range(0, k)]
        if k >= 1:
            u[0] = v[0]
            e[0] = u[0].scalar_product(1 / u[0].euclidean_norm())
        
        for i in range(1, k):
            u[i] = v[i]
            for j in range(0,i):
                u[i] = u[i] - Vector.proj(u[j], v[i])
            e[i] = u[i].scalar_product(1 / u[i].euclidean_norm())
        return (u, e)
    
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
        if len(list) == 0:
            raise ValueError("initialization list must not be empty")
        v = Vector(len(list), transposed = transposed)
        v.dtype = dtype
        for i in range(0, len(v)): v[i] = dtype(list[i])
        return v


    # get a vector filled with random numbers
    def random_vector(length, fromvalue, tovalue, dtype, transposed = False, seedval = None):
        if seedval != None:
            seed(seedval)
        v = Vector(length, transposed, dtype)
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
        
    # return the vector transformed to an mparray
    def to_mparray(self):
        return mparray(self.v, dtype=self.dtype)
        
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
            
    def __iter__(self):
        self.idx = 0
        return self
        
    def __next__(self):
        if self.idx < len(self):
            result = self[self.idx]
            self.idx += 1
            return result
        else:
            raise StopIteration
    
    def mean(self):
        return Array.mean(self.v)
        
    # apply applies lambda to each element of vector
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
        
    # find returns  all vector elements that satisfy the 
    # predicate lambda_f as a list
    def find(self, lambda_f):
        res = []
        for i in range(0, len(self)):
            if lambda_f(self.v[i]):
                res.append(self.v[i])
        return res
        
    # identical to find, but returns not elements
    # found but their indices as a list
    def find_where(self, lambda_f):
        res = []
        for i in range(0, len(self)):
            if lambda_f(self.v[i]):
                res.append(i)
        return res
        
#################################################
################## class Tensor #################
################################################# 
# This is a highly experimental Tensor implementation.
# with very limited functionality.
# It will grow in the future.
class Tensor:
    def __init__(self, arr, dtype = float):
        if isinstance(arr , list):
            self.mpa = mparray(arr)
            self.dtype = type(arr[0])
        elif isinstance(arr, mparray):
            self.mpa = arr
            self.dtype = arr.dtype
        else:
            raise ValueError("tensor __init__ requires a list or an mparray as initializer")
        
    @property
    def shape(self):
        return self.mpa.shape
        
    @property
    def ndim(self):
        return len(self.mpa.shape)
        
    @property
    def count(self):
        shp = self.shape
        prod = 1
        for i in range(len(shp)):
            prod *= shp[i]
        return prod
        
    def clone(self):
        return deepcopy(self)
        
    def __str__(self):
        return "tensor" + str(self.mpa)
        
    def filled_tensor(shp, init_value = 0, dtype = float):
        mpa = mparray.filled_array(shp, init_value, dtype)
        return Tensor(mpa)
        
    # apply a function on everey element of the tensor
    # if in_situ == True, all modifications will be made
    # to the tensor itself. Otherwise a new tensor is 
    # created
    def apply(self, lambda_f, in_situ = False):
        if in_situ:
            Tensor._apply_helper(self.mpa.a, lambda_f, in_situ)
        else:
            return Tensor(mparray(Tensor._apply_helper(self.mpa.a, lambda_f, in_situ)))
                    
    def _apply_helper(lst, lambda_f,  in_situ):
        shp = Array.shape(lst)
        if in_situ: 
            if len(shp) == 1:
                for i in range(shp[0]):
                    lst[i] = lambda_f(lst[i])
            else:
                for i in range(shp[0]):
                    Tensor._apply_helper(lst[i], lambda_f, in_situ)
        else:
            if len(shp) == 1:
                lst_new = []
                for i in range(shp[0]):
                    lst_new.append(lambda_f(lst[i]))
                return lst_new
            else:
                lst_new = []
                for i in range(shp[0]):
                    lst_new.append(Tensor._apply_helper(lst[i], lambda_f, in_situ))
                return lst_new
            
    def _apply(mpa, mpa1, mpa2, lambda_f):
        shp = mpa.shape
        if len(shp) == 1:
            for i in range(shp[0]):
                mpa[i] = lambda_f(mpa1[i],mpa2[i])
        else:
            for i in range(shp[0]):
                Tensor._apply(mpa[i], mpa1[i], mpa2[i], lambda_f)
                
    def _apply_single(mpa, mpa1, lambda_f):
        shp = mpa.shape
        if len(shp) == 1:
            for i in range(shp[0]):
                mpa[i] = lambda_f(mpa1[i])
        else:
            for i in range(shp[0]):
                Tensor._apply_single(mpa[i], mpa1[i], lambda_f)
                
    def _apply_const(mpa, lambda_f):
        shp = mpa.shape
        if len(shp) == 1:
            for i in range(shp[0]):
                mpa[i] = lambda_f(mpa[i])
        else:
            for i in range(shp[0]):
                Tensor._apply_const(mpa[i], lambda_f)
        
    def __add__(self, other):
        if self.shape != other.shape:
            raise ValueError("only tensors with same shape can be added")
        mpa = mparray.filled_array(self.shape, init_value = 0, dtype = self.mpa.dtype)
        Tensor._apply(mpa, self.mpa, other.mpa, lambda x,y: x+y)
        return Tensor(mpa)
        
    def __sub__(self, other):
        if self.shape != other.shape:
            raise ValueError("only tensors with same shape can be subtracted")
        mpa = mparray.filled_array(self.shape, init_value = 0, dtype = self.mpa.dtype)
        Tensor._apply(mpa, self.mpa, other.mpa, lambda x,y: x-y)
        return Tensor(mpa)
        
    def __mul__(self, other):
        if isinstance(other, int) or isinstance(other, float) or isinstance(other, complex):
            mpa = deepcopy(self.mpa)
            Tensor._apply_const(mpa, lambda x: x * other)
            return Tensor(mpa)
        elif isinstance(other, Tensor):
            if self.shape != other.shape:
                raise ValueError("only tensors with same shape can be multiplied")
            mpa = mparray.filled_array(self.shape, init_value = 0, dtype = self.mpa.dtype)
            Tensor._apply(mpa, self.mpa, other.mpa, lambda x,y: x*y)
            return Tensor(mpa)
        elif isinstance(other, list):
            return self * Tensor(mparray(other))
        elif isinstance(other, mparray):
            return self * Tensor(other)
        else:
            raise ValueError("cannot multiply a tensor with a " + str(type(other)))
        
    def __truediv__(self, other):
        if self.shape != other.shape:
            raise ValueError("only tensors with same shape can be divided")
        mpa = mparray.filled_array(self.shape, init_value = 0, dtype = self.mpa.dtype)
        Tensor._apply(mpa, self.mpa, other.mpa, lambda x,y: x/y)
        return Tensor(mpa)
        
    def mult(t1,t2): 
        t1a = t1.flatten()
        t2b = t2.flatten()
        shp1a = t1a.shape
        shp2b = t2b.shape
        newshp= shp1a + shp2b
        result = []
        for i in range(shp1a[0]):
            for j in range(shp2b[0]):
                result.append(t1a.mpa[i] * t2b.mpa[j])
        return Tensor(mparray(result, t1.dtype).reshape(newshp))
        
    def __matmul__(self, other):
        if isinstance(other, float) or isinstance(other, complex) or isinstance(other, int):
            mpa = deepcopy(self.mpa)
            Tensor._apply_const(lambda x: x * other, mpa)
            return Tensor(mpa)
        elif isinstance(other, Tensor):
            return Tensor.mult(self, other)
        elif instance(other, list):
            return self * Tensor(mparray(lst))
        elif isinstance(other, mparray):
            return self * Tensor(other)
        else:
            raise ValueError("tensor product not compatible with type " + str(type(other)))

    def mult(t1, t2):
        return Tensor(kmult_helper(t1.mpa.a, t2.mpa.a), dtype = t1.dtype)
        
        
    def kmult_helper(l1, l2):
        def apply(l, lambda_f):
            shp = Array.shape(l)
            if len(shp) == 1:
                result = []
                for i in range(shp[0]):
                    result.append(lambda_f(l[i]))
                return result
            else:
                result = []
                for i in range(shp[0]):
                    result.append(apply(l[i], lambda_f))
                return result
        if (isinstance(l1,int) or isinstance(l1,float) or isinstance(l1, complex)) and (isinstance(l2, int) or isinstance(l2, float) or isinstance(l2, complex)):
            return l1 * l2 
        elif isinstance(l1, int) or isinstance(l1,float) or isinstance(l1, complex):
            return apply(l2, lambda x: x * l1)
        elif isinstance(l2, int) or isinstance(l2, float) or isinstance(l2, complex):
            return apply(l1, lambda x: x * l2)
        
        shp1 = Array.shape(l1)
        shp2 = Array.shape(l2)
        ndim1 = len(shp1)
        ndim2 = len(shp2)
        if ndim1 == 1 and ndim2 == 1:
            result = []
            for i in range(shp1[0]):
                row = []
                for j in range(shp2[0]):
                    row.append(l1[i] * l2[j])
                result.append(row)
            return result
        elif ndim1 == 1 and ndim2 == 2:
            result = []
            for i in range(shp1[0]):            
                result.append(kmult_helper(l1[i], l2))
            return result
        elif ndim1 == 2 and ndim2 == 1:
            result = [] 
            for i in range(shp1[0]):
                row = []
                for j in range(shp1[1]):
                    row.append(kmult_helper(l1[i][j],l2))
                result.append(row)               
            return result
        elif ndim1 == 2 and ndim2 == 2:
            result = Array.create_2Darray(shp1[0], shp1[1])
            for i in range(shp1[0]):
                for j in range(shp1[1]):
                    result[i][j] = kmult_helper(l1[i][j], l2)
            return result
        elif ndim1 >= 1:
            result = []
            for i in range(shp1[0]):
                result.append(kmult_helper(l1[i], l2))
            return result

    def _eq_helper(mpa1, mpa2):
        if len(mpa1.shape) == 1:
            for i in range(mpa1.shape[0]):
                if mpa1[i] != mpa2[i]:
                    return False
            return True
        else:
            for i in range(mpa1.shape[0]):
                if Tensor._eq_helper(mpa1[i], mpa2[i]):
                    continue
                else: 
                    return False
            return True
    
    # checks two tensors for equality        
    def __eq__(self, other):
        shp1 = self.shape
        shp2 = other.shape
        if shp1 != shp2:
            return False
        else:
            return Tensor._eq_helper(self.mpa, other.mpa)
                        
    def __ne__(self, other):
        return not self == other
        
    def reshape(self, shp):
        t = deepcopy(self)
        t.mpa = t.mpa.reshape(shp)
        return t
        
    def flatten(self):
        t = deepcopy(self)
        t.mpa = self.mpa.flatten()
        return t
        
    def random_tensor(shp, fromvalue=0, tovalue=1, seed_value = 0, dtype=float):
        mpa = mparray.random_array(shp, fromvalue, tovalue, seed_value, dtype)
        return Tensor(mpa)
        
    # checks whether addressed element is a scalar and returns that scalar
    # Otherwise, it returns an mparray object
    def __getitem__(self, arg):
        tmp = self.mpa.__getitem__(arg)
        if not isinstance(tmp, list):
            return tmp
        else:
            return mparray.from_list(tmp)
        
    # delegates to corresponding access pattern for moparrays
    def __setitem__(self, arg, cont):
        self.mpa.__setitem__(arg, cont)
        
    # returns the Tensor as a list 
    def to_list(self):
        return deepcopy(self.mpa.to_list())
        
    def to_mparray(self):
        return deepcopy(self.mpa)
        
    # turns a 2-dimensional tensor into a matrix
    def asmatrix(self):
        if self.ndim != 2:
            raise ValueError("cannot convert a Tensor with " + str(self.ndim) + " dimensions to a matrix")
        else:
            lst = deepcopy(self.mpa.to_list())
            return Matrix.from_list(lst, dtype = self.dtype)
            
    # turns a 1-dimensional tensor to a vector
    def asvector(self):
        if self.ndim != 1:
            raise ValueError("cannot convert a Tensor with " + str(self.ndim) + " dimensions to a vector")
        else:
            lst = deepcopy(self.mpa.to_list())
            return Vector.from_list(lst, dtype = self.dtype)
        
#################################################
############## class FunctionMatrix #############
################################################# 

# The elements of an FunctionMatrix are functions instead of numbers
# If it is applied to another number-based matrix (with the same shape), 
# it will take the corresponding element of the regular matrix and 
# apply the contained function to it 

class FunctionMatrix:
    # identity function
    def id(x):
        return x
        
    # null function
    def null_func(x):
        return 0
        
    # the constructor is called with the dimension of the matrix as 
    # well as an optional init_value which per default is the lambda
    # FunctionMatrix.null
    def __init__(self, dim1, dim2, init_value = null_func):
        self.m = [[init_value for i in range(0, dim2)] for j in range(0, dim1)]
        self.dim1 = dim1
        self.dim2 = dim2
        
    # deep copy of matrix is returned
    def clone(self):
        return deepcopy(self)
    
    # return own shape
    @property
    def shape(self):
        return(self.dim1,self.dim2)
        
    # retrieve number of rows
    def dim1(self):
        return self.dim1
        
    # retrieve number of columns
    def dim2(self):
        return self.dim2
        
    # check for equal dimensions
    def is_square(self):
        return self.dim1 == self.dim2
        
    # the identity matrix consists of
    # FunctionMatrix.id in the diagonal,
    # and FunctionMatrix.null in all
    # remaining places.
    def identity(size):
        o = FunctionMatrix(size, size)
        for r in range(0, size):
            for c in range(0, size):
                if r == c:
                    o[r,c] = FunctionMatrix.id
                else:
                    o[r,c] = FunctionMatrix.null
        return o
     
    # null matrix has all elements set to
    # FunctionMatrix.null
    def null_matrix(dim1, dim2):
        o = FunctionMatrix(dim1, dim2)
        for r in range(0, dim1):
            for c in range(0, dim2):
                o[r,c] = FunctionMatrix.null_func
        return o
        
    # transpose of FunctionMatrix works as
    # expected
    @property
    def T(self):
        o = FunctionMatrix(self.dim2, self.dim1)
        for r in range(0, self.dim1):
            for c in range(0, self.dim2):
                o.m[r][c] = self.m[c][r]
        return o
    # FunctionMatrix fm allows to access
    # individual functions: f = fm[r,c]   
    def __getitem__(self, arg):
        if isinstance(arg, tuple):
            if len(arg) > 2: 
                raise ValueError("maximum of 2 slices allowed")
            else:
                s1 = arg[0]
                s2 = arg[1]
                (x,y) = (isinstance(s1 ,int), isinstance(s2, int))
                if (x,y) == (True, True): # we got instance[i,j]
                    return self.m[s1][s2]
                else:
                    raise ValueError("FunctionMatrix does not support slicing")       
        else:
            raise TypeError("only accesses like matrix[i,j] are allowed")


    # FunctionMatrices fm allow to set 
    # individual elements using fm[r,c]=val
    def __setitem__(self, arg, val):
        if isinstance(arg, tuple):
            if len(arg) > 2: 
                raise ValueError("maximum of 2 slices allowed")
            else:
                s1 = arg[0]
                s2 = arg[1]
                (x,y) = (isinstance(s1 ,int), isinstance(s2, int))
                if (x,y) == (True, True): # we got instance[i,j]
                    self.m[s1][s2] = val
                else: 
                    raise ValueError("FunctionMatrix does not support slicing")
        else:
            raise TypeError("only accesses like matrix[i,j] are allowed")
        
    # multiplication of a FunctionMatrix with a 
    # regular vector is provided by multiplying
    # the row vectors of the FunctionMatrix with
    # the vector. Each vector element if the
    # FunctionMatrix is applied to the corres-
    # ponding element of the regular vector.
    # The results are then added to become
    # elements of the result vector.
    # The same thing is done multiple times
    # when a FunctionMatrix is multiplied with
    # a regular matrix.
    # Multiplying a FunctionMatrix with another
    # FunctionMatrix works different. Multi-
    # plying functions is implemented as
    # function conposition. Summing up elements
    # is implemented by a lambda that combines
    # the function compositions with a sum-up-
    # lanbda
    def __matmul__(self, other):
        if isinstance(other, Matrix):
            o = Matrix(self.dim1, other.dim2, dtype = other.dtype)
            for r in range(0, self.dim1):
                for c in range(0, other.dim2):
                    sum = 0
                    for i in range(0, self.dim2):
                        sum += self.m[r][i](other.m[i][c])
                    o.m[r][c] = sum
            return o
        elif isinstance(other, Vector):
            if self.dim2 != len(other):
                raise ValueError("matrix.dim2 != len(other)")
            v = Vector(self.dim1, dtype = other.dtype)
            for r in range(0, self.dim1):
                sum = 0
                for i in range(0, self.dim2):
                    sum+= self.m[r][i](other[i])
                v.v[r] = sum
            return v
        elif isinstance(other, FunctionMatrix):
            o = FunctionMatrix(self.dim1, other.dim2)
            for r in range(0, self.dim1):
                for c in range(0, self.dim2):
                    lambdas = [None for i in range(0, self.dim2)]
                    for i in range(0, self.dim2):
                        if i == 0:
                            lambdas[0] = lambda x: self.m[r][i](other.m[i][c](x))
                        else:
                            lambdas[i] = lambda x: lambdas[i-1](x) + self.m[r][i](other.m[i][c](x))
                o[r,c] = lambdas[self.dim2-1]
            return o
        else: 
            raise TypeError("can only multiply a FunctionMatrix with a Matrix, another FunctionMatrix, or a Vector")
                    
    def __mul__(self, other):
        return self.__matmul__(other)
        
    # summing two FunctionMatrix objects is 
    # implemented by combining the functions
    # in both matrices using a sum-up lambda:
    # f1, f2  results in 
    # lambda x: f1(x)+f2(x)
    def __add__(self, other):
        if not isinstance(other, FunctionMatrix):
            raise TypeError("can only add a FunctionMatrix to an FunctionMatrix")
        if self.shape != other.shape:
            raise ValueError("both matrices must have the same shape")
        o = FunctionMatrix(self.dim1, self.dim2)
        for r in range(0, self.dim1):
            for c in range(0, self.dim2):
                f1 = self.m[r][c]
                f2 = other.m[r][c]
                o.m[r][c] = lambda x: f1(x) + f2(x)
        return o
        
    def __str__(self):
        res = "\n[\n"
        for r in range(0, self.dim1):
            line = " [ "
            for c in range(0, self.dim2):
                line += str(self[r,c]) + "\t"
            line += " ] "
            res += line + "\n"
        res += "]\n"
        return res 
            
    # apply expects a equally shaped regular
    # matrix. The functions in the Function-
    # Matrix are element-wise applied to the
    # corresponding elements in the regular
    # matrix.
    def apply(self, n):
        if self.shape != n.shape:
            raise ValueError("n must have the same dimensions like FunctionMatrix")
        result = Matrix(self.dim1, self.dim2)
        for r in range(0, self.dim1):
            for c in range(0, self.dim2):
                result[r,c] = self[r,c](n[r,c])
        return result
        
    # initialize a FunctionMatrix using
    # a list. The list shape is mapped to the
    # Matrix shape.
    def from_list(array):
        shape = Array.shape(array)
        o = FunctionMatrix(shape[0], shape[1])
        for r in range (0, shape[0]):
            for c in range(0, shape[1]):
                o.m[r][c] = array[r][c]
        return o
        
#################################################
################ class Polynomial ###############
#################################################       

class Polynomial:
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
    
    # returns the coefficients of p with increasing i (x^i)
    def coeffs(self):
        if len(self) == 0:
            return [] 
        else:
            result = [] 
            for i in range(0,len(self)):
                result.append(a[i])
            return result  
            
    # returns the intercept a0 of the polynomial
    def intercept(self):
        if len(self) == 0:
            raise ValueError("polynomial with length > 0 expected")
        else:
            return self[0]
            
    # calculates the polynomial height
    def height(self):
        return max(self.a)
        
    # calculates the p-norm = pow(sum(abs(coefficients^p)), 1/p)
    def p_norm(self, p):
        if not isinstance(p, int) or p < 1:
            raise ValueError("p must be an integer >= 1")
        else:
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
        g = Common.gcdmult(self.a)
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
    def derivative(self):
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
        
    # multiplication of polynomials
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
        
    # returns the result of a polynomial division (/-operator)
    # without the reminder
    def __floordiv__(self, other):
        result = self / other
        return result[0]
        
    # returns the remainder of a polynomial division (/-operator)
    def __mod__(self, other):
        result = self / other
        return result[1]
               
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
        
    # method fit: calculates a polynomial that fits the training
    # date passed in x_vector, y_vector.
    # fit() id currently implemented without vectorization which
    #    might change in the future.    
    # x_vector and y_vector contain the training data, i.e. 
    #    polynomial_to_be_searched.compute(x_vector[i]) = y_vector[i]
    #    m := length of x_vector and y_vector
    # degree: degree of polynomial_to_be_searched
    # learningrate: learning rate that learning of fit() should use 
    # epochs: number of iterations through all training data 
    #    returns a Polynomial which can be used to predict results
    #    of other argument values
                
    def fit(x_vector, y_vector, n, learningrate = 0.0001, epochs = 1000):

        if len(x_vector) != len(y_vector):
            raise ValueError ("x_vector and y_vector must have same length")
        if x_vector.is_transposed() or y_vector.is_transposed():
            raise ValueError("x_vector and y_vector must not be transposed")
        # theta has n+1 elements, x_vector and y_vector m elements
        theta = Vector(n+1, dtype = x_vector.dtype, init_value=0)
        n_plus_1 = len(theta)
        m        = len(x_vector)
        theta = Vector.from_list([0 for i in range(0,n_plus_1)])
        pow_matrix = Matrix(m, n_plus_1, dtype = x_vector.dtype)
        
        
        for r in range(0, m):
            for c in range(0, n_plus_1):
                pow_matrix[r][c] = x_vector[r] ** c
                
        for epoch in range(0, epochs):
            
            diff = (pow_matrix * theta - y_vector)
            delta = (pow_matrix.T * diff).scalar_product(learningrate/m)
            theta = theta - delta 
            
        return Polynomial(theta.v)
            
            
            
            
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
        gcd_nd = Common.gcd(nom,denom)
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

#################################################
############## class Interpolation ##############
#################################################   

class Interpolation:
    # This class implements natural cubic splines.
    # Parameters:
    #     xarray: array with  x-coordinates of the 
    #             points. 
    #             Prereq.:      x0 < x2 < ... < xn 
    #     yarray: array  with y-coordinates of the
    #             points,  i.e. Pi = (xi,yi)
    #             Prereq.: len(xarray)==len(yarray)
    #     To initialize the interpolation, calling
    #     the constructor is sufficient.
    #     After that, interpolation values  can be 
    #     computed using method interpolate(x0)
    # 
    class CubicSplines:
        # constructor which does the heavy lifting
        # of the cubic spline interpolation
        def __init__(self, xarray, yarray):
            lam_0       = 0
            lam_n_min_1 = 0
            b_0         = 0
            b_n_min_1   = 0
            mu_0        = 1
            mu_n_min_1  = 1
            self.n = len(xarray)
            self.a = [0 for i in range(self.n-1)]
            self.b = [0 for i in range(self.n-1)]
            self.c = [0 for i in range(self.n-1)]
            self.d = [0 for i in range(self.n-1)]
            self.x = deepcopy(xarray)
            self.y = deepcopy(yarray)
        
            if len(self.x) != len(self.y):
                raise ValueError("x and y vector must have equal length")
            
            increasing_x = True
            for i in range(len(self.x)-1):
                if self.x[i] < self.x[i+1]:
                    continue
                else:
                    increasing_x = False
                    break
            if not increasing_x:
                raise ValueError("violation of precondition x[0] < x[1] < x[2] ... ")
            self.h  = [self.x[i+1]-self.x[i] for i in range(self.n-1)]
            self.b = [0 for i in range(self.n)]
            self.b[0]   = b_0
            self.b[self.n-1] = b_n_min_1
            for i in range(1, self.n-1):
                self.b[i] = (self.y[i+1]-self.y[i])/self.h[i] - (self.y[i]-self.y[i-1])/self.h[i-1]
            mat = Matrix(self.n, self.n)
            mat[0,0] = mu_0
            mat[0,1] = lam_0
            mat[self.n-1,self.n-2] = lam_n_min_1
            mat[self.n-1,self.n-1] = mu_n_min_1
            for i in range(1, self.n-1):
                mat[i,i]     = (self.h[i-1]+self.h[i])/3
                mat[i-1,i]   = self.h[i-1]/6
                mat[i,i+1]   = self.h[i]/6
            self.M = mat.thomas_algorithm(Vector.from_list(self.b))
            for i in range(self.n-1):
                self.c[i] = (self.y[i+1] - self.y[i])/self.h[i] - (self.h[i]*(self.M[i+1]-self.M[i]))/6
                self.d[i] = self.y[i]-(self.h[i]**2 * self.M[i])/6
                
        # helper method responsible to search for a x0 the right 
        # polynomial Si(x0) to be used. <<called by interpolate().
        def _search_interval(self, x0):
            if x0 <= self.x[0]:
                return 0
            elif x0 >= self.x[self.n-1]:
                return self.n-2
            else:
                for i in range(0,self.n-1):
                    if x0 >= self.x[i] and x0 <=self.x[i+1]:
                        return i
        
        # computes the function value at x0. If x0 < xarray[0], 
        # S0 will be used. If x0 > xarray[n-1], Sn-1 will be 
        # used. In all other cases, interpolate() uses the 
        # right Si where i is determinded by calling 
        # _search_interval                  
        def interpolate(self, x0):
            i = self._search_interval(x0)
            res =  (self.x[i+1]-x0)**3 * self.M[i]/self.h[i]
            res += (x0-self.x[i])**2 * self.M[i+1]/self.h[i]          
            res = res / 6
            res += self.c[i]*(x0-self.x[i])+self.d[i]
            return res            
 
    # Interpolation1D takes a x-and a y-array of same length and
    # connects all points (xi, yi) using lines. The points 
    # approximate a function f(x).
    # All heavy lifting is done in the constructor.
    # By calling the method interpolate() with argument x0
    # the caller receives an interpolation value for f(x0).
    # The arrays x and y must contain at least 2 elements. 
    # x-values must be in ascending order. Note: For values 
    # outside the interval [ x[0],x[n] ] an interpolation
    # value is returned which deviates largely from the 
    # interpolated function.
    # If the aforementioned  preconditions are not fullfilled,   
    # ValueErrors are raised        
    class Interpolation1D:
        def __init__(self, xarr, yarr):
            if len(xarr) != len(yarr):
                raise ValueError("x and y array must have equal size")
            self.n = len(xarr)
            if self.n <= 1:
                raise ValueError("arrays must contain more than one point")
            for i in range(self.n-1):
                if xarr[i] >= xarr[i+1]:
                    raise ValueError("x values must ascend") 
            self.x = xarr
            self.y = yarr
            self.b=[]
            self.a=[]
            for i in range(self.n - 1):
                self.a.append((self.y[i+1]-self.y[i]) / (self.x[i+1]-self.x[i]))
                self.b.append(self.y[i+1]-self.a[i]*self.x[i+1])
                
        def search_interval(self, x0):
            if x0 <= self.x[0]:
                return 0
            elif x0 >= self.x[self.n-1]:
                return self.n-2
            else:
                for i in range(self.n-2):
                    if x0 <= self.x[i+1]: return i
            
        def interpolate(self, x0):
            i = self.search_interval(x0)
            return self.a[i] * x0 + self.b[i]
       
#################################################
################ class Regression ###############
#################################################   
        
class Regression:
  
    def compute_multinomial(coeffs, values):
        if len(coeffs) != 1 + len(values):
            raise ValueError("len(coefficients) + 1 and len(x-values) must be the same")
        res = coeffs[0]
        for i in range(0, len(coeffs)-1):
            res += coeffs[i+1] * values[i]
        return res
                
    # x_matrix and y_vector contain the training data,  
    # a0 + a1*x[0] + a2 * x[2] + .. + an * x[n] = y_vector[i]
    #    m := length of x_vector and y_vector
    # learningrate: learning rate that learning of fit() should use 
    # epochs: number of iterations through all training data 
    #   
    # returns the cofficients a0, a1, .... ,an
    #       
    # These coefficients can be used in combination with the 
    # function compute_multinomial to predict results for  
    # other value vectors
    def multivariate_fit(x_matrix, y_vector, learningrate, epochs):
        if x_matrix.dim1 != len(y_vector):
            raise ValueError ("x_matrix.dim1 and len(y_vector) must be the same")
        if y_vector.is_transposed():
            raise ValueError("y_vector must not be transposed")
        n_plus_1 = x_matrix.dim2 + 1
        m        = x_matrix.dim1
        # theta has n+1 elements
        theta = Vector(n_plus_1, dtype = x_matrix.dtype, init_value=x_matrix.dtype(0))
        ones = Matrix(m,1,dtype=x_matrix.dtype, init_value = 1)
        ext_matrix = Matrix(m, n_plus_1, dtype=x_matrix.dtype)
        
        for r in range(0, m):
            for c in range(1, n_plus_1):
                if c == 0:
                    ext_matrix[r][c] = ext_matrix.dtype(1)
                else:
                    ext_matrix[r][c] = x_matrix[r][c-1]
                
        for epoch in range(0, epochs):
            diff = (ext_matrix * theta - y_vector)
            delta = (ext_matrix.T * diff).scalar_product(learningrate/m)
            theta = theta - delta
        theta_1_n = theta.v 
        return theta_1_n

#################################################
############## class Classification #############
################################################# 
class Classification:
     
    """
    KNearestNeighbor:
    datapoints[] is a two dimensional array with number of rows
    specifying the number of neighbors and number of columns
    specifying the coordinates of each row.
    labels[] must have the same number of elements, the datapoints[]
    array has rows. Thus, the vector datapoints[i] has the label
    labels[i]
    norm is the norm used such as the euclidean norm
        2 is euclidean/L2 norm, 1 is Manhattan/L1 norm, 2 is Lp norm
    """
    class KNearestNeighbor:
        def most_frequent(self, array):
            data = Counter(array)
            return data.most_common(1)[0][0]

        def __init__(self, datapoints, labels, norm):
            self.datapoints = datapoints
            self.labels = labels
            self.norm = norm
            if norm < 0 or norm > 2:
                raise ValueError("unsupported norm")
            self.shp = Array.shape(datapoints)
            if self.shp[0] != len(labels):
                raise ValueError("need a label for each datapoint") 
        
        # x is a Vector. 
        def k_nearest_neighbor(self, x, k):
            if len(x) != self.shp[1]:
                raise ValueError("x must have the same fields as each of the datapoints")
            distances = []
            for i in range(self.shp[0]):
                if self.norm == 2:
                    distance = x.l2_norm(Vector.from_list(self.datapoints[i]))
                elif self.norm == 1:
                    distance = x.l1_norm(Vector.from_list(self.datapoints[i]))
                elif self.norm == 0:
                    distance = x.lp_norm(Vector.from_list(self.datapoints[i]), 3)
                distances.append(distance)
            sorted_arr, indices = Array.sort(distances, in_situ = False)
            neighbor_labels = []
            for i in range(k): 
                neighbor_labels.append(self.labels[indices[i]])
            return self.most_frequent(neighbor_labels)



    
#################################################
#################### class ANN ##################
#################################################  
########### Implementation  of a simple Artificial Neural Network ######### 

            


# Base class
class ANN:
    class Layer:
        def __init__(self):
            self.input = None
            self.output = None

        # responsible to calculate the layer output
        # using the layer input
        def forward_propagation(self, input):
            pass

        # on basis of output error calculate input error for
        # previous layer
        def backward_propagation(self, output_error, learning_rate):
            pass
        

    # Concrete Layer Classes
    class FullyConnectedLayer(Layer):
        # input_size = number of input neurons
        # output_size = number of output neurons
        def __init__(self, in_neurons, out_neurons):
            self.weights = mparray.rand((in_neurons, out_neurons), seedval=time.gmtime()) - 0.5
            self.bias = mparray.rand((1, out_neurons),seedval=time.gmtime()) - 0.5
        
        # receice signals on channels, assign a weight to them. Returns  
        # corresponding output
        def forward_propagation(self, input_data):
            self.input = input_data
            self.output = mparray.dot(self.input, self.weights) + self.bias
            return self.output

        # computes the input_error from the output_error
        def backward_propagation(self, output_error, learning_rate):
            input_error = mparray.dot(output_error, self.weights.T)
            weights_error = mparray.dot(self.input.T, output_error)
            # dBias = output_error

            # update parameters
            self.weights = self.weights - weights_error * learning_rate
            self.bias = self.bias - output_error * learning_rate
            return input_error
        
    # Activation Layer
    class ActivationLayer(Layer):
        def __init__(self, activation, activation_prime):
            self.activation = activation
            self.activation_prime = activation_prime

        # returns the activated input
        def forward_propagation(self, input_data):
            self.input = input_data
            self.output = self.activation(self.input)
            return self.output

        # propagate error back to input layer
        def backward_propagation(self, output_error, learning_rate):
            return self.activation_prime(self.input).multiply(output_error)
        
    # activation functions and derivatives
    def tanh(x):
        return x.tanh()

    def tanh_prime(x):
        return -x.tanh()**2+1
    
    def reLU(x):
        return x.apply(lambda x: max(0,x))
    
    def reLU_prime(x):
        return x.apply(lambda x: 0 if x < 0 else 1)
    
    def sigmoid(x):
        return x.apply(lambda x: 1.0 / (1 + np.exp(-x)))
      
    def sigmoid_prime(x):
      return sigmoid(x) * (-sigmoid(x) + 1)
    
    # loss function and its derivative
    def mse(y_true, y_pred):
        return ((y_true-y_pred) ** 2).mean()

    def mse_prime(y_true, y_pred):
        return (y_pred - y_true) * (2 / y_true.count)
    
    class Network:
        def __init__(self):
            self.layers = []
            self.loss = None
            self.loss_prime = None

        # add layer to network
        def add(self, layer):
            self.layers.append(layer)

        # set loss to use
        def use(self, loss, loss_prime):
            self.loss = loss
            self.loss_prime = loss_prime

        # predict output for given input
        def predict(self, input_data):
            # sample dimension first
            samples = input_data.shape[0]
            result = []

            # run network over all samples
            for i in range(samples):
                # forward propagation
                output = input_data[i]
                for layer in self.layers:
                    output = layer.forward_propagation(output)
                result.append(output)
            return result

        # train the network
        def fit(self, x_train, y_train, epochs, learning_rate, autostop = False):
            # sample dimension first
            samples = x_train.shape[0]
            prev_err = 1
            # training loop
            for i in range(epochs):
                err = 0
                for j in range(samples):
                    # forward propagation
                    output = x_train[j]
                    for layer in self.layers:
                        output = layer.forward_propagation(output)

                    # compute loss (for display purpose only)
                    err += self.loss(y_train[j], output)
                
                    # backward propagation
                    error = self.loss_prime(y_train[j], output)
                
                    for layer in reversed(self.layers):
                        error = layer.backward_propagation(error, learning_rate)
            
                # calculate average error on all samples
                err /= samples
                if err > prev_err: 
                    print("training aborted cause the error started to rise again")
                    break
                prev_err = err
                print('epoch %d/%d   error=%f' % (i+1, epochs, err))
    

    
#################################################
################ class Clustering ###############
#################################################  
class Clustering:
    # k_means expects data points in argument points, 
    # a cluster number k, 
    # and a tolerance
    # The algorithm clusters all data points    
    # with respect to centroids until the 
    # distance between old and new centroids 
    # falls under the tolerance

    def k_means(points, k, tolerance = 1E-5):
        # measures the distance of a point from
        # all centroids and return these 
        # distances as an array
        def measure(point, _centroid):
            _distance = []
            for i in range(0, len(_centroid)):
                delta = _centroid[i] - point
                _distance.append(delta.euclidean_norm())
            return _distance
    
        # if less points than clusters, an exception is raised
        if len(points) < k:
            raise ValueError("number of points smaller than k")
        elif len(points) == k: # if number of points equals clusters,
            return points      # each point defines its own cluster
        
        # make a copy of data points
        _points = deepcopy(points)
        # set sequence of centroids to empty list
        _centroid = []
        # shuffle points
        shuffle(_points)
        # take the first k points randomly as centroids
        for i in range(0,k): 
            _centroid.append(_points[i])
    
        in_equilibrium = False
        while not in_equilibrium:
            # remember old centroids
            old_centroid = deepcopy(_centroid)
            # initialize list of clusters
            _clusters = [[] for j in range(0, k)]
            # iterate through all points
            for _point in _points:
                # measure the point's distance from all centroids
                _distances = measure(_point, _centroid)
                # check to which centroid the point is closest
                for i in range(0,k):
                    # and enter the point to the cluster belonging
                    # to the closest centroid
                    if _distances[i] == min(_distances):
                        _clusters[i].append(_point)
                        # if we found closest centroid we can skip the for loop
                        break
            _vector_size = len(_points[0])
    
            # walk through all clusters
            for i in range(0,k):
                cluster_len = len(_clusters[i])
                _new_centroid = Vector(_vector_size, init_value = 0)
                # and determine a new centroid per cluster 
                # as the mean of all coordinates in that cluster 
                # Note: while the initial randomly selected centroids
                # are real data points, the new ones are virtual
                # data points
                for point in _clusters[i]:
                    _new_centroid = _new_centroid + point.scalar_product(1 / cluster_len)
                _centroid[i] = _new_centroid.clone()
            
            # calculate the difference between new and old centroids
            change = 0            
            for i in range(0,k):
                change += (old_centroid[i] - _centroid[i]).euclidean_norm()
            # if difference is smaller than tolerance, terminate algorithm
            if change < tolerance:
                in_equilibrium = True
        # and return result which are all the clusters found
        return _clusters 
        
    # k-means mini batch clustering is sensitive to the hyper
    # parameters. Its intended use is for very large datasets.
    # It expects:
    # points:         the data points to be clustered
    # k:              the number of clusters 
    # batchsize:      the batchsize to be used (must be smaller
    #                 than the number of data points)
    # iterations:     number that determines how often the batches
    #                 should be processed
    # An internal hyperparameter in the code i the number how often
    # the dataset should be randomly shuffled.
    def k_means_mini_batch(points, k, batchsize, iterations = 200):
        # measures the distance of a point from
        # all centroids and return these 
        # distances as an array
        def measure(point, _centroid):
            _distance = []
            for i in range(0, k):
                delta = _centroid[i] - point
                _distance.append(delta.euclidean_norm())
            minimum_distance = min(_distance)
            for i in range(0, len(_centroid)):
                if _distance[i] == minimum_distance:
                    return i
    
        
        # if less points than clusters, an exception is raised
        if len(points) < k:
            raise ValueError("number of points smaller than k")
        elif len(points) == k: # if number of points equals clusters,
            return points      # each point defines its own cluster
        elif batchsize > len(points):
            raise ValueError("batchsize must be smaller than dataset")
        
        # make a copy of data points
        _points = deepcopy(points)
        # set sequence of centroids to empty list
        _centroid = []
        # shuffle points
        for i in range(1,10) : shuffle(_points)
        # take the first k points randomly as centroids
        for i in range(0,k): 
            _centroid.append(_points[i])
    
        _cluster_learning_rate = [0 for i in range(0, k)]
        _cluster_counters = [0 for i in range(0,k)]

        iter = 1 
        # initialize list of clusters
        _clusters = [[] for j in range(0, k)]
        
        while iter < iterations:

            shuffle(_points)
            # batch[i] is the ith point in the batch 
            # cluster[i] is the cluster batch[i] was assigned to
            batch = []
            cluster = [0 for i in range(0,batchsize)]
            # randomly choose batchsize data points from _points
            # cluster for these points is initialized with 0
            for i in range(0, batchsize):
                batch.append(_points[i])
                cluster[i] = measure(batch[i], _centroid)
                
            # iterate through points in batch again   
            for i in range(0, batchsize):
                for j in range(0, k):
                    if batch[i] in _clusters[j]: _clusters[j].remove(batch[i])
                _clusters[cluster[i]].append(batch[i])
                _cluster_counters[cluster[i]] += 1 
                _cluster_learning_rate[cluster[i]] = 1/_cluster_counters[cluster[i]]
                _centroid[cluster[i]] =  _centroid[cluster[i]] + (batch[i] - _centroid[cluster[i]]).scalar_product(_cluster_learning_rate[cluster[i]])
            iter+= 1
        # and return result which are all the clusters found
        return _clusters

        
        
#################################################
############## class Newton method ##############
#################################################
# Implementation of the Newton method 
# f: a lambda or function
# fderivative (optional) is the first derivative
# of f
# eps: the tolerance with which the Newton method
# should work, i.e., when convergence is assumed
# In the Taylor series f(x+h) = f(x) + h f'(x) + h**2/2!*f''(x) ... 
# We stop after the first derivative: f(x+h) = f(x) + h f'(x)
# Assuming f(x+h) should be zero, we get 0 = f(x) + h f'(x)
# h = xnew - x => 0 = f(x) + (xnew - x)f'(x) =>
# xnew = x - f(x)/f'(x). Note: If f'(x) is close _clusterto zero, 
# the approximation will fail.
# Example usage:
# res = Newton(lambda x: x**2, lambda x: 2*x)
# print(res.compute(1,2))
# find a solution for: x**2 = 2 and start with x=1


class Newton:
    def __init__(self, f, fderivative = None, eps = 01E-10):
        self.f = f
        self.y0 = 0
        self.eps = eps # tolerance
        self.max_iter = 100 # maximum number of iterations
        if fderivative != None:
            self.fder = fderivative
        else:
            self.fder = None

    # calls the function f
    def fun(self,x):
        return self.f(x) - self.y0
        

    # calculates the derivative of f at x. If the user 
    # specified a derivative of f, the this function is 
    # used. Otherwise, an approximation is done.
    def derivation(self, x):
        try:
            approx = (self.fun(x+self.eps) - self.fun(x)) / self.eps
        except OverflowError as oe:
            print("overflow")
        except ZeroDivisionError as ze:
            print("division by zero")
        return approx
    
    # the work horse of the Newton method. y0 is the result the
    # function should have. if, for example.
    # Example:f is y = x**2 and y0 == 2, then compute tries to
    # solve the equation x**2 = y0 which is sqrt(y0). For this 
    # case, the call would look like: 
    #     n = Newton(lambda x: x**2, lambda x: 2*x)
    #     res = n.compute(1,2)
    # the x specified as argument is the initial value 
    # compute() should start with
    
    def compute(self, x, y0):
        self.y0 = y0 
        iter = 0
        while iter < self.max_iter:
            fun_r = self.fun(x)
            if self.fder != None:
                fun_d = self.fder(x)
            else:
                fun_d = self.derivation(x)
            x_old = x 
            try:
                x = x - fun_r / fun_d
            except OverflowError as oe:
                print("overflow error")
                return None
            except ZeroDivisionError as ze:
                print("division by zero")
                return None
            if abs(x-x_old) < self.eps: break
            iter += 1
        return x


#################################################
##############  class Measurement  ##############
#################################################

class Measurement:
    def circle_perimeter(r):
        return 2 * r * math.pi
        
    def circle_area(r):
        return math.pi * r**2
        
    # alpha is the angle of the segment
    def circle_segment_area(alpha, r):
        return Measurement.circle_surface_area(r) * angle/(2*math.pi)
        
    def sphere_area(r):
        return 4 * r**2 * math.pi
        
    def sphere_volume(r):
        return 4/3 * r**3 * math.pi
        
    def ellipse_surface_area(ra, rb):
        return ra * rb * math.pi
        
    def ellipse_perimeter(ra, rb):
        val1 = math.pi* (ra+rb)
        val2 = math.pi * (1.5*(ra+rb)-math.sqrt(ra*rb))
        return (val1 + val2)/2
            
    def ellipse_eccentricity(ra, rb):
        if ra > rb:
            return math.sqrt(1-(rb/ra)**2)
        else:
            return math.sqrt(1-(ra/rb)**2)
            
    def ellipse_foci(ra, rb):
        if ra >= rb:
            xcoord = math.sqrt(ra**2 - rb**2)
        else:
            xcoord = math.sqrt(rb**2 - ra**2)
        return((-coord,0),(+coord,0))
        
    def cylinder_volume(r, h):
        return r**2 * h * math.pi
        
    def cylinder_area(r, h):
        return 2 * Measurement.circle_surface_area(r) + h * 2 * math.pi * r
        
    def cone_volume(r, h):
        return 1/3 * math.pi * r**2 * h
        
    def cone_area(r, h):
        l = math.sqrt(r**2 + h**2)
        return math.pi * r * l + math.pi * r**2
        
    # if all 3 lines are given:
    def triangle_area_sides(l1, l2, l3):
        s = 0.5*(l1 + l2 + l3)
        return math.sqrt(s * (s-l1) * (s-l2) * (s-l3))
        
    # if two lines and the angle between them is given:
    def triangle_area_sides_angle(l1, l2, angle_between_l1_and_l2):
        return 0.5 * l1 * l2 * math.sin(angle_between_l1_and_l2)
        
    # if a base line and a perpendicular height is given: 
    def triangle_area_base_height(base, height):
        return 0.5 * base * height
        
    # b is the length of the parallelogram and h the height perpendicular
    # to b
    def parallelogram_area(b, h):
        return b * h
        
    # a, b are the parallel lines, h the height between them
    def trapezoid_area(a, b, h):
        return 0.5 * (a + b) * h
        
    def rectangle_area(a, b):
        return a * b
        
    def square_volume(a,b,c):
        return a * b * c
        
    def square_area(a, b, c):
        return 2 * (a * b + b * c + a * c)
        
    def pyramid_volume(h, base_area):
        return 1/3 * base_area * h
        
        
#################################################
################  class Transfer  ###############
#################################################

# Functionality to transfer mparrays, Vectors, Matrices 
# between mathplus and other libraries, primarily numpy
        
class Transfer:
    def numpy_to_array(nparray):
        return nparray.tolist()
        
    def array_to_numpy(array):
        return np.array(array)
        
    def matrix_to_numpy(m):
        return np.asmatrix(Transfer.array_to_numpy(m.m))
        
    def vector_to_numpy(v):
        if not v.is_transposed():
            return Transfer.array_to_numpy(v.v)
        else:
            list = [] 
            for i in range(0, len(v)):
                list.append([v[i]])
            return Transfer.array_to_numpy(list)
        
    def numpy_to_matrix(nparray):
        npshp = nparray.shape
        if len(npshp) > 2:
            raise ValueError("mathplus only support one or two dimensions")
        return Matrix.from_list(nparray.tolist())
        
    def numpy_to_vector(nparray):
        npshp = nparray.shape
        if len(npshp) > 2:
            raise ValueError("mathplus only supports one or two dimensions")
        elif len(npshp) == 2:
            if npshp[0] != 1 and npshp[1] != 1:
                raise ValueError("cannot convert two-dimensional numpy-array to a vector")
            elif npshp[0] == 1:
                return Vector_from_list(nparray[0])
            else: # npshp[1] == 1 => use transposed vector 
                list = []
                for i in range(0, npshp[0]):
                    list.append(nparray[i, 0])
                return Vector.from_list(list, transposed = True)
        else: # len(npshp) == 1 
            return Vector.from_list(nparray.tolist())
            
    def numpy_to_mparray(npa):
        return mparray(npa.tolist())
        
    def mparray_to_numpy(mpa):
        shp = tuple(mpa.shape)
        npa = np.array(mpa.flatten().to_list())
        return np.reshape(npa, shp)
            
            
    # the draw function expects a mparray x with the arguments
    # and thre function to be applied as a lambda. This function 
    # is drawn using matplotlib
    def draw_function_2D(xmp, lambda_f, color = 'r', label = ""): 
        ymp = xmp.apply(lambda_f)
        x = Transfer.mparray_to_numpy(xmp)
        y = Transfer.mparray_to_numpy(ymp)
        # setting the axes at the centre
        fig = plt.figure()
        ax = fig.add_subplot(1, 1, 1)
        ax.spines['left'].set_position('center')
        ax.spines['bottom'].set_position('zero')
        ax.spines['right'].set_color('none')
        ax.spines['top'].set_color('none')
        ax.xaxis.set_ticks_position('bottom')
        ax.yaxis.set_ticks_position('left')
        # plot the function
        plt.plot(x, y, color = color, label = label)
        plt.legend(loc='upper left')
        plt.show()
        
    # draw_functions_2D is used to draw multiple functions
    # at once. It expects an input list as an argument that
    # contain entries of the form (lambda, color, label)
    def draw_functions_2D(xmp, input):
        if len(input) == 0:
            raise ValueError("received an empty input vector")
        x = Transfer.mparray_to_numpy(xmp)
        y_array = [] 
        col_array = []
        lbl_array = []
        for entry in input:
            lambda_f, color, label = entry
            y_array.append(Transfer.mparray_to_numpy(xmp.apply(lambda_f)))
            col_array.append(color)
            lbl_array.append(label)
        # setting the axes at the centre
        fig = plt.figure()
        ax = fig.add_subplot(1, 1, 1)
        ax.spines['left'].set_position('center')
        ax.spines['bottom'].set_position('zero')
        ax.spines['right'].set_color('none')
        ax.spines['top'].set_color('none')
        ax.xaxis.set_ticks_position('bottom')
        ax.yaxis.set_ticks_position('left')
        
        for i in range(len(y_array)):
            plt.plot(x, y_array[i], color = col_array[i], label = lbl_array[i])
        plt.legend(loc='upper left')
        plt.show()
        
    def draw_function_3D(tmp, lambda_f1, lambda_f2, title ="mathplus 3D function"):
        fig = plt.figure(figsize = (8,8))
        ax = plt.axes(projection='3d')
        ax.grid()
        
        xmp = tmp.apply(lambda_f1)
        ymp = tmp.apply(lambda_f2)
        t = Transfer.mparray_to_numpy(tmp)
        x = Transfer.mparray_to_numpy(xmp)
        y = Transfer.mparray_to_numpy(ymp)

        ax.plot3D(x, y, t)
        ax.set_title(title)

        # Set axes label
        ax.set_xlabel('x', labelpad=20)
        ax.set_ylabel('y', labelpad=20)
        ax.set_zlabel('t', labelpad=20)

        plt.show()
        
    def draw_surface(xmp, ymp, lambda_x, lambda_y):
        fig = plt.figure(figsize = (12,10))
        ax = plt.axes(projection='3d')

        xmp = mparray.arange(-5, 5.1, 0.2)    
        ymp = mparray.arange(-5, 5.1, 0.2)

        Xmp, Ymp = mparray.meshgrid(xmp, ymp)
        Zmp = Xmp.apply(lambda_x) * Ymp.apply(lambda_y)

        X = Transfer.mparray_to_numpy(Xmp)
        Y = Transfer.mparray_to_numpy(Ymp)
        Z = Transfer.mparray_to_numpy(Zmp)

        surf = ax.plot_surface(X, Y, Z, cmap = plt.cm.cividis)

        # Set axes label
        ax.set_xlabel('x', labelpad=20)
        ax.set_ylabel('y', labelpad=20)
        ax.set_zlabel('z', labelpad=20)

        fig.colorbar(surf, shrink=0.5, aspect=8)
        plt.show()



            
    # the following read/write-methods writea matrix or array to a
    # file or read a matrix or array from a file. The format is "csv".
    # Note: mparrays are stored as flattened arrays to the CSV file.
    # The header will store the dtype of the array, as well as its 
    # shape. When retrieving the mparray back from file, a flattened
    # mparray will be created which will then be reshaped into the 
    # original shape with the original type using the file's header 
    # information.
    # The headers of vectors will store the type as well as the 
    # transposition-state which will then be used to restore the 
    # vector. For a matrix all rows will be stored plus the type
    # of the matrix. 

                
    def create_tensor_header(a):
        header = ["#"]
        header.append("tensor")
        if a.dtype == int:
            header.append(0)
        elif a.dtype == float:
            header.append(1)
        elif a.dtype == complex:
            header.append(2)
        shp = a.shape
        header.append(len(shp))
        for i in range(len(shp)):
            header.append(shp[i])
        return header
        
    def create_vector_header(v):
        header = ["#"]
        header.append("vector")
        if v.dtype == int:
            header.append(0)
        elif v.dtype == float:
            header.append(1)
        elif v.dtype == complex:
            header.append(2)
        if v.is_transposed():
            header.append(1)
        else:
            header.append(0)
        return header
        
    def create_mparray_header(a):
        header = ["#"]
        header.append("mparray")
        if a.dtype == int:
            header.append(0)
        elif a.dtype == float:
            header.append(1)
        elif a.dtype == complex:
            header.append(2)
        shp = a.shape
        header.append(len(shp))
        for i in range(len(shp)):
            header.append(shp[i])
        return header
            
    def create_matrix_header(m):
        header = ["#"]
        header.append("matrix")
        if m.dtype == int:
            header.append(0)
        elif m.dtype == float:
            header.append(1)
        elif m.dtype == complex:
            header.append(2)
        else: header.append(42)
        return header
              
    def map_tensor_header(row):
        dtype = None
        shp = [] 
        if row[1] != "tensor":
            raise ValueError("file does not contain a tensor but a " + str(row[1]))
        if int(row[2]) == 0:
            dtype = int
        elif int(row[2]) == 1:
            dtype = float
        elif int(row[2]) == 2:
            dtype = complex
        len_shp = int(row[3])
        for i in range(len_shp):
            shp.append(int(row[i+4]))
        return (dtype, shp)
        
    def map_mparray_header(row):
        dtype = None
        shp = [] 
        if row[1] != "mparray":
            raise ValueError("file does not contain a mparray but a " + str(row[1]))
        if int(row[2]) == 0:
            dtype = int
        elif int(row[2]) == 1:
            dtype = float
        elif int(row[2]) == 2:
            dtype = complex
        len_shp = int(row[3])
        for i in range(len_shp):
            shp.append(int(row[i+4]))
        return (dtype, shp)
        
    def map_vector_header(row):
        trans = None
        dtype = None
        if row[1] != "vector":
            raise ValueError("file does not contain a vector but a " + str(row[1]))
        if int(row[2]) == 0:
            dtype = int
        elif int(row[2]) == 1:
            dtype = float
        elif row[2] == 2:
            dtype = complex
        if int(row[3]) == 0:
            trans = False
        elif int(row[3]) == 1:
            trans = True          
        return (dtype, trans)
    
    def map_matrix_header(row):
        dtype = None
        if row[1] != "matrix":
            raise ValueError("file does not contain a matrix but a " + str(row[1]))
        if int(row[2]) == 0:
            dtype = int
        elif int(row[2]) == 1:
            dtype = float
        elif row[2] == 2:
            dtype = complex
        return dtype
            
    def readMParrayFromCSV(filename, verbose = False):
        csv.register_dialect('excel', delimiter=',', quoting=csv.QUOTE_NONE)
        counter = 0
        array = []
        shp   = None
        dtype = None
        
        with open(filename, newline='') as f:
            try: 
                reader = csv.reader(f, delimiter=",", quoting=csv.QUOTE_NONE)
                if verbose: print("... reading file " + filename + " ...")
                if verbose: print()
                counter = 0
                for row in reader:
                    if row[0] == "#": 
                        (dtype, shp) = Transfer.map_mparray_header(row)
                        continue
                    if counter > 1:
                        raise ValueError("more than one row in file: this cannot be a flattened mparray")
                    array_row = []
                    for num in row:
                        array.append(dtype(num))
            except csv.Error as e:
                print('file {}, line {}: {}'.format(filename, reader.line_num, e))
        a = mparray(array, dtype)
        return a.reshape(shp)

    def writeMParrayToCSV(a, filename, verbose = False):
        with open(filename, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile, delimiter=',',quoting=csv.QUOTE_NONE)
            lst = a.flatten().to_list()
            if verbose: print("... writing file " + filename + "...")
            if verbose: print()
            writer.writerow(Transfer.create_mparray_header(a))
            writer.writerow(lst)                
               
    def readVectorFromCSV(filename, verbose = False):
        csv.register_dialect('excel', delimiter=',', quoting=csv.QUOTE_NONE)
        counter = 0
        array = []
        trans = None
        dtype = None
        
        with open(filename, newline='') as f:
            try: 
                reader = csv.reader(f, delimiter=",", quoting=csv.QUOTE_NONE)
                if verbose: print("... reading file " + filename + " ...")
                if verbose: print()
                counter = 0
                for row in reader:
                    if row[0] == "#": 
                        dtype, trans = Transfer.map_vector_header(row)
                        continue
                    if counter > 1:
                        raise ValueError("more than one row in file: this cannot be a vector")
                    array_row = []
                    for num in row:
                        array.append(dtype(num))
            except csv.Error as e:
                print('file {}, line {}: {}'.format(filename, reader.line_num, e))
        v = Vector.from_list(array, dtype, trans)
        return v
        
    def writeVectorToCSV(v, filename, verbose = False):
        with open(filename, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile, delimiter=',',quoting=csv.QUOTE_NONE)
            
            if verbose: print("... writing file " + filename + "...")
            if verbose: print()
            writer.writerow(Transfer.create_vector_header(v))
            writer.writerow(v.v)                

    def readMatrixFromCSV(filename, verbose = False):
        csv.register_dialect('excel', delimiter=',', quoting=csv.QUOTE_NONE)
        array = []
        dtype = None
        with open(filename, newline='') as f:
            try: 
                reader = csv.reader(f, delimiter=",", quoting=csv.QUOTE_NONE)
                if verbose: print("... reading file " + filename + " ...")
                if verbose: print()
                first_row = True
                for row in reader:
                    if first_row:
                        dtype = Transfer.map_matrix_header(row)
                        first_row = False
                        continue
                    array_row = []
                    for num in row:
                        array_row.append(dtype(num))
                    array.append(array_row)
            except csv.Error as e:
                print('file {}, line {}: {}'.format(filename, reader.line_num, e))
        return Matrix.from_list(array, dtype)
        
    def writeMatrixToCSV(m, filename, verbose = False):
        with open(filename, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile, delimiter=',',quoting=csv.QUOTE_NONE)
            if verbose: print("... writing file " + filename + "...")
            if verbose: print()
            r, c = m.shape
            writer.writerow(Transfer.create_matrix_header(m))
            for i in range(r):
                row = m.m[i]
                writer.writerow(row)
                
    def readTensorFromCSV(filename, verbose = False):
        csv.register_dialect('excel', delimiter=',', quoting=csv.QUOTE_NONE)
        counter = 0
        array = []
        shp   = None
        dtype = None
        
        with open(filename, newline='') as f:
            try: 
                reader = csv.reader(f, delimiter=",", quoting=csv.QUOTE_NONE)
                if verbose: print("... reading file " + filename + " ...")
                if verbose: print()
                counter = 0
                for row in reader:
                    if row[0] == "#": 
                        (dtype, shp) = Transfer.map_tensor_header(row)
                        continue
                    if counter > 1:
                        raise ValueError("more than one row in file: this cannot be a flattened tensor")
                    array_row = []
                    for num in row:
                        array.append(dtype(num))
            except csv.Error as e:
                print('file {}, line {}: {}'.format(filename, reader.line_num, e))
        t = Tensor(mparray(array, dtype).reshape(shp))
        return t

    def writeTensorToCSV(a, filename, verbose = False):
        with open(filename, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile, delimiter=',',quoting=csv.QUOTE_NONE)
            lst = a.flatten().to_list()
            if verbose: print("... writing file " + filename + "...")
            if verbose: print()
            writer.writerow(Transfer.create_tensor_header(a))
            writer.writerow(lst)                