
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
from copy import deepcopy
from random import uniform, randrange, seed, shuffle
from functools import reduce
import operator
import numpy as np # used to transfer array between 
                   # mathplus and numpy

#################################################
################## class Common #################
################################################# 
"""
as the name suggests, Common provides more general
functionality used by other modules.
"""
class Common:          
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
            
    # n!
    def fac(n):
        result = 1
        for i in range(2, n+1):
            result *= i
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
                if g > 1:  
                    break
        return g

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
            

    # factorization of integers into their prime factors.
    # the method returns a list of prime factors in ascending
    # order
    def factorize(n1):
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
                p1 = Common._brent(p)
         
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
        return  (1 / (sqrt(2 * math.pi) * sigma)) * math.exp(- x**2/(2 * sigma ** 2)) 

    def gaussian_kernel_2D(x, y, sigma):
        return (1 / (2 * math.pi * sigma ** 2)) * math.exp(- (x**2 + y**2) / (2 * sigma ** 2))

    # dimension == len(xvect)
    def gaussian_kernel_multiD(xvect, sigma):
        return (1 / ((sqrt(2 * math.pi) * sigma)) ** len(xvect)) * math.exp(- (xvect.euclidean_norm() ** 2) / (2 * sigma **2))

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
            for i in len(x_dataset):
                sum += (x_dataset[i] - x_mean) * (y_dataset[i] - y_mean)
            return sum/(len(x_dataset) - 1)
            
            
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
            for i in len(x_dataset):
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
            for i in range(0, len(x_array)):
                sum += x_array[i] * weight
            return sum
        else:
            sum = 0
            for i in range(0, len(x_array)):
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
                    raise ValueError("attempt to split array of size at nonexistent position " + str(pos))
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
    def sort(a, in_situ = True):
        def partition(a, indices, left, right):
            pivot = a[right]
            i = left
            j = right - 1
            
            while i < j:
                while i < j and a[i] <= pivot: i += 1
                while j > i and a[j] >  pivot: j -= 1
                if a[i] > pivot:
                    a[i],a[right] = a[right],a[i]
                    indices[i], indices[right] = indices[right], indices[i]
                else:
                    i = right
            return i
            
        def quicksort(a, indices, left, right):
            if left < right:
                idx = partition(a, indices, left, right)
                quicksort(a, indices, left, idx - 1)
                quicksort(a, indices, idx + 1, right)        
        if not in_situ:
            a = deepcopy(a)
        indices = [i for i in range(len(a))]
        quicksort(a, indices, 0, len(a)-1)
        return indices
        
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
        shp1 = Array.shape(arr1)
        shp2 = Array.shape(arr2)
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
        res = math.sqrt(sum / len(arr))
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
        return sum / len(array)
            
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
    def argmax(array, axis = 1):
        if axis == None or CommArrayon.shape(array)[0] == 1:
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
    def filter(lambda_f, array):
        result = [] 
        dim1, dim2 = Array.shape(array)
        for i in range(0, dim1):
            for j in range(0, dim2):
                if lambda_f(array[i][j]):
                    result.append(array[i][j])
        return result
    
    # same as filter, but returns as a list all indices
    # where the condtion lambda_f holds            
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
        
    # returns the total number of elements in matrix
    def size(self):
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
    
    
    # sets all elements below the specified diag to 0 
    # diag = 0 is the main axis        
    def upper_triangle(self, diag = 0):
        dim1, dim2 = self.shape()
        res = self.clone()
        max_axis = min(dim1,dim2)
        for r in range(0, dim1):
            for c in range(0, dim2):
                if (c - diag) > r: 
                    res[r,c] = 0
        return res
        
    # sets all elements above the specified diag to 0 
    # diag = 0 is the main axis        
    def lower_triangle(self, diag = 0):
        dim1, dim2 = self.shape()
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
        if not self.is_orthonormal():
            return self.inverse_matrix() * vector
        else:
            return self.T() * vector
            
    # requires a tridiagonal matrix a vector with len(vector)==matrix.dim1
    # returns the solution x of the equation matrix@x = vector
    def thomas_algorithm(self, d):
        if not self.is_tridiagonal():
            raise ValueError("thomas algorithm works with tridiagonal matrices only")
        n, _ = self.shape()
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
    # implimentation uses the change of the result vector between 
    # different iterations. If the tolerance is achieved the 
    # method returns the result. If after max_iter iterations
    # the required tolerance is not achieved then None is returned.
    def jacobi_method(self, b, tolerance = 1E-20, max_iter = 100):
        _max_iter = max_iter
        _tolerance = 1E-20
        n,_ = self.shape()
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
    # diag == 0 => main diagonal
    # diag = -1 => diagonal one step below the main diagonal
    # diag =  1 => diagonal one step above the main diagonal
    # ... 
    # diag must exist, otherwise a ValueError is thrown
    def diagonal(self, diag = 0):
        dim1, dim2 = self.shape()
        max_diag = min(dim1, dim2)-1
        if (abs(diag) > max_diag):
            raise ValueError("diag must be in [" + str(-max_diag) + "," + str(max_diag) + "]")
        list = []
        for r in range(0, dim1):
            for c in range(0,dim2):
                if c - diag == r:
                    list.append(self[r,c])
        return list
        
    # checks whether matrix is in tridiagonal form
    def is_tridiagonal(self):
        if not self.is_square():
            raise ValueError("is_tridiagonal only defined for square matrices")
        dim1, dim2 = self.shape()
        for d in range(-dim1+1, dim1):
            n = Array.euclidean_norm(self.diagonal(d))
            if not d in range(-1,2) and n != 0:
                return False
            else: 
                continue
        return True
        
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
            
    def is_unitary(self):
        if not self.is_square():
            raise ValueError("only square matrices permitted")
        if self.det() == 0:
            return False
        else:
            return self.H() == self.inverse_matrix()
            
    def is_orthonormal(self):
        if not self.is_square():
            raise ValueError("orthomality is only defined for square matrices")
        return self.T() @ self == Matrix.identity(self.dim1)
            
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
        
    # calculate exp(t*Matrix)
    def exp(self, t = 1):
        max_expansion = 20
        shape = self.shape()
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
        shape = self.shape()
        if shape[0] != shape[1]:
            raise ValueError("sin only defined for square matrices")
        res = Matrix.identity(shape[0])
        for i in range(2, max_expansion + 1):
            res +=  (self ** (2*i+1)).scalar_product((-1 ** i)/Common.fac(2*i))
        return res
        
    # calculate cosine of matrix 
    def cos(self):
        max_expansion = 20
        shape = self.shape()
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
        
    # apply lambda_f to a single column of the matrix. returns new matrix 
    # with changed column
    def map_column(self, column, lambda_f):
        m = Matrix(self.dim1, self.dim2, dtype = self.dtype)
        if column > self.dim2:
            raise ValueError("column " + str(column) + " does not exist")
        for r in range(0, self.dim1):
            m.m[r][column] = lambda_f(self.m[r][column])
        return m
        
    # apply lambda_f to a single row of the matrix. returns new matrix 
    # with changed row
    def map_row(self, row, lambda_f):
        m = Matrix(self.dim1, self.dim2, dtype = self.dtype)
        if row > self.dim1:
            raise ValueError("row " + str(row) + " does not exist")
        for c in range(0, self.dim2):
            m.m[row][c] = lambda_f(self.m[row][c])
        return m
        
    # like map, but with lambda getting called with
    # row, col, value at (row,col) 
    def map2(self, lambda_f):
        m = Matrix(self.dim1, self.dim2, dtype = self.dtype)
        for r in range(0, self.dim1):
            for c in range(0, self.dim2):
                m.m[r][c] = lambda_f(r, c, self.m[r][c])
        return m
        
    # filter returns  all matrix elements that satisfy the 
    # predicate lambda_f as a list
    def filter(self, lambda_f):
        return Array.filter(lambda_f, self.m)
        
    # same as filter, but does not return the elements 
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
        
    # LU decomposition of square matrices using Gaussian decomposition
    def lu_decomposition(self):
        m, n = self.shape()
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
    # results are returned as a coefficient list with c0 to cn 
    # going from left to right 
    # It can be directly used in class Polynomial to create a polynomial 
    # such as in:
    #     coeffs = char_poly(M)
    #     p = Polynomial(coeffs)

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
    def diagonal_matrix(list, dtype = float):
        m = Matrix(len(list), len(list), dtype)
        for i in range(0, len(list)):
            m[i][i] = list[i]
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
    
    # returns number of elements in vector (same as len(vector))        
    def size(self):
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
            res = Vector(len(self), dtype = self.dtype, transposed = self._transposed)
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
        v = Vector(len(list), transposed = transposed)
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
        return Array.mean(self.v)
        
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
        
    # filter returns  all vector elements that satisfy the 
    # predicate lambda_f as a list
    def filter(self, lambda_f):
        res = []
        for i in range(0, len(self)):
            if lambda_f(self.v[i]):
                res.append(self.v[i])
        return res
        
    # identical to filter, but returns not elements
    # found but there indices as a list
    def find_where(self, lambda_f):
        res = []
        for i in range(0, len(self)):
            if lambda_f(self.v[i]):
                res.append(i)
        return res
        
        
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
        if self.shape() != other.shape():
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
        if self.shape() != n.shape():
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
            delta = (pow_matrix.T() * diff).scalar_product(learningrate/m)
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
    class cubic_splines:
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
            delta = (ext_matrix.T() * diff).scalar_product(learningrate/m)
            theta = theta - delta
        theta_1_n = theta.v 
        return theta_1_n


    
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
                    _new_centroid += point.scalar_product(1 / cluster_len)
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

# Functionality to transfer arrays, Vectors, Matrices 
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
            
