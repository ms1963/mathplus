

"""
mathplusdemo.py is licensed under the
GNU General Public License v3.0

Permissions of this strong copyleft license are conditioned on making available complete source code of licensed works and modifications, which include larger works using a licensed work, under the same license. Copyright and license notices must be preserved. Contributors provide an express grant of patent rights.
"""

from mathplus import *


import math

# just a show case to illustrate usage of Matrix and Vector classes
# in matrix

print("This demo illustrates a subset of the functionality provided")
print()

Matrix.set_separators("|", "|")
Vector.set_separators("[", "]")
print("Separators for Matrix set to " + Matrix.left_sep + " and " + Matrix.right_sep)
print("Separators for Vector set to " + Vector.left_sep + " and " + Vector.right_sep)
print()

m1 =  Matrix(4,4)
print("m1 = " + str(m1))
m2 = Matrix.identity(4)
print("m2 = " + str(m2))
print("det(m2) = " + str(m2.det()))
print("m1 @ m2 + m2 = " + str(m1 @ m2 + m2))
print("For matrix multiplication you may eiher use '*' or '@'")

v1  = Vector.unit_vector(4,3)
print("v1 = " + str(v1))

print("||v1|| = " + str(v1.norm()))
print()

v2 = Vector.unit_vector(4,2)
print("v2 = " + str(v2))

print("Getting all unit vectors for size == 3: ")
for v in Vector.all_unit_vectors(3):
    print(str(v))

print("Are v1 and v2 orthogonal? " + str(v1.is_orthogonal_to(v2)))
print()

print("v2.T() = " + str(v2.T()))
print()

m3 = Matrix.from_list([[1,4,1,3], [0,2,-1,6], [1,0,1,2], [4, 0, 1 ,2]])
print("m3 = " + str(m3))

print("m3 * v1 = " + str(m3 * v1))

print("m3 = " + str(m3))

print("det(m3) = " + str(m3.det()))
print()

print("||m3|| = " + str(m3.norm()))
print()

print("abs(m3) = " + str(abs(m3)))

print("m3 transposed with m3.T() + " + str(m3.T()))

for i in range(0, m3.dim1):
    print("row vector [" + str(i) + "] = " + str(m3.row_vector(i)))
print()

for i in range(0, m3.dim2):
    print("col vector [" + str(i) + "] = " + str(m3.column_vector(i)))
    
print("Multiplication of m3 with unit matrix  = " + str(m2 * m3))

print("Multiplying m3 with itself for 3 times = " + str(m3.mult_n_times(3)))

print("Is m3 == m1? " + str(m1 == m3))
print("Is m1 == m1? " + str(m1 == m1))
print()
print()

print("Creating a Matrix from v2.T(): " + str(Matrix.from_vector(v2.T())))

print("Multiplying v2 with v2.T: ", str(v2 * v2.T()))
print()

print("Multiplying v2.T with v2: ", str(v2.T() * v2))
print()

v4 = Vector.from_list([1,1,1,1])
print("v4 = " + str(v4))
print("Multiplication of v4.T() with m3", str(v4.T() * m3))

print("Multiplying unit matrix (size == 5) with 2: ", str(Matrix.identity(5).scalar_product(2)))

m4 = Matrix.from_list([[1,2,3], [3, 2, 1]])
m5 = Matrix.from_list([[1,1], [1,1], [1,1]])
print("m4 = " + str(m4))
print("m5 = " + str(m5))
print("m5 @ m4 = " + str(m5 @ m4))

print("m4 @ m5 = " + str(m4 @ m5))

print("Use map() to apply the lambda x -> x^2 to each element of m3: " + str(m3.map(lambda x: x * x)))

print("You may also use m3.map(math.sin) for vectors or matrices")
print(m3.map(math.sin))
print()

print("Use map() to apply the lambda x -> x + 1 to each element of v4: " + str(v4.map(lambda x: x + 1)))

print()
print("Calculating the diagonal of matrix m5 @ m4")
list = (m5 @ m4).diagonal()
print(list)
print()

m6 = m4 @ m5
print("Creating matrix m6 = m4 @ m5: " + str(m6))
print()
print("pow(m6, 2) = ")
print(m6.mult_n_times(2))

print("m7a = m6.clone(); m7b = m6")
m7a = m6.clone()
m7b = m6
print("m7a = " + str(m7a))
print("m7b = " + str(m7b))

m6.change_item(0, 0, 42)
print("Changing element 0,0 of matrix m6 to 42 => " + str(m6))

print("m7a = " + str(m7a))
print("m7b = " + str(m7b))

v = Vector.from_list([i for i in range(0,10)], transposed = True)
print("v = " + str(v))
print("Slice v[0:3, 5:8] = " + str(v[0:3, 5:8]))
print()
print("Matrix Slicing")
print("m = [[i*j+1 for i in range(0,5)] for j in range(0,5)]")
m = Matrix.from_list([[i*j+1 for i in range(0,10)] for j in range(0,10)])
print(m)
print()
print("m[2:4] = ")
print(m[2:4]) # get rows - slice
print()
print("m[2] = ")
print(m[2]) # get single row - index
print()
print("m[3,3] = ")
print(m[3,3]) # get element
print()
print("m[3][0:2] = ")
print(m[3, 0:2]) # get specific elements from row 3
print()
print("m[0:2, 3] = ")
print(m[0:2, 3]) # get col 3 from specific rows
print()
print("m[0:2,0:3] = ")
print(m[0:2,0:3]) # get matrix with row 0 and 1 and columns 0,1,2 per row
m7 = Matrix.from_list([[5,9,2],[1,8,5],[3,6,4]])
print("m7 = " + str(m7))
print("det(m7) = " +str(m7.det()))
print()
print("setting Matrix and Vector format string to {0:.4f}")
Matrix.set_format('{:8.4f}')
Vector.set_format('{:8.4f}')
print("Cofactor matrix of m7 = " + str(m7.cofactor_matrix()))
print("Inverse(m7) = " + str(m7.inverse_matrix()))
print("m7 * inverse(m7)" + str(m7 * m7.inverse_matrix()))
print("Solving equation m7 * x = [1][2][0]")
print(m7.solve_equation(Vector.from_list([1, 2, 0])))
print("Euclidean norm of m7 = " + str(m7.euclidean_norm()))
print()
Matrix.set_format('{:.8f}')
Vector.set_format('{:.8f}')
print("m8 = Matrix.from_flat_list(list = [1,2,3,4,5,6,7,8,9,10,11,12], shape=(3,4))")
m8 = Matrix.from_flat_list(list = [1,2,3,4,5,6,7,8,9,10,11,12], shape=(3,4))
print(m8)
print("m9 = Matrix.from_flat_list(list = [1,2,3,4,5,6,7,8,9,10,11,12], shape=(4,3)")
m9 = Matrix.from_flat_list( list = [1,2,3,4,5,6,7,8,9,10,11,12], shape=(4,3))
print(m9)
print("m8.reshape(shape = (4,3))")
print(m8.to_flat_list())
m8 = m8.reshape((4,3))
print(m8)
print("m8 == m9 ? " + str(m8==m9))
print()
print("v5 = Vector(6, dtype = int, init_value = 1, transposed = True)")
v5 = Vector(6, dtype = int, init_value = 3, transposed = True)
print(v5)
print("v6 = Vector(6, dtype = int, init_value = 2, transposed = False)")
v6 = Vector(6, dtype = int, init_value = 2, transposed = False)
print(v6)
print("v5 * v6 =  ")
print(v5*v6)

print("Swap operations using m9")
print(m9)
print("swapping rows 0 and 2" + str(m9.swap_rows(0,2)))
print("swapping columns 1 and 2" + str(m9.swap_columns(1,2)))
m10 = Matrix.from_list([[0,1,2], [1,2,1], [0,3,6]])
print("m10 = " +str(m10))
print("m10.shape() = " + str(m10.shape()))
print("m10 Echolon form is ", end="")
print(m10.echolon())
print("m10 Reduced Echolon is ", end="")
print(m10.reduced_echolon())
print("Solving m7 * x [1,2,0].T with reduced echolon matrix")
m7_enhanced  = Matrix.from_list([[5,9,2,1],[1,8,5,2],[3,6,4,0]])
print("m7_enhanced = " + str(m7_enhanced))
print("Solution in column 4 of reduced echolon matrix" + str(m7_enhanced.reduced_echolon()))
print("The rank of a matrix can be computed by creating its row echolon form")
print("and then counting its rows that are not 0 which is the rank.")
print("The rank of m10 is "+str(m10.rank()))
print()
print("The trace of a square matrix is defined as the sum of all elements on its diagonal. m10.tr() = " + str(m10.tr()))
print()

v7 = Vector.from_list([i for i in range(0,10)])
print("v7 = " + str(v7))
print("v7[0:2, 5:7] = [-1,-2,-3,-4]")
v7[0:2, 5:7] = [-1.0,-2.0,-3.0,-4.0]
print("result is " + str(v7))
print("v7[9] = 99")
v7[9] = 99
print(v7)



print()
print("QR factorization of m = ")
m = Matrix.from_list([[1,1,0],[1,0,1],[0,1,1]])
print(m)
(Q,R) = m.qr_decomposition()
print("Q = ")
print(Q)
print("R = ")
print(R)



print("For matrix : ")
m = Matrix.from_list([[5,4],[1,2]])
print(m)
(eigenvalues, eigenvectors) = m.eigen(epsilon = 1E-16, i_max = 2000)
print("the eigenvalues are " + str(eigenvalues))
print("and the eigenvectors are :")
for ev in eigenvectors: print(ev)

print()
print("We can also compute the characteristic polynomial of m")
print("Its coefficients in ascending order from x^0 to x^2 are: " + str(m.char_poly()))
print()

print("LU decomposition of a matrix is supported by Matrix.lu_decomposition")
m = Matrix.from_list([[2, 1, 8],[-1, 3, 5], [1, 0, -1]])
print("Matrix m = " + str(m))
l,u = m.lu_decomposition()
print("If we call lu_decomposition(m) , we get l = " + str(l))
print("and u = " + str(u))
print()

print("Cholesky decomposition of a symmetric/hermitian positive definite square matrix")
print("Matrix m is: ")
print()
m = Matrix.from_list([[4, 12, -16], [12, 37, -43], [-16, -43, 98]])
print(m)
print()
print("Result L returned from m.cholesky_decomposition(m) is ")
print()
l = m.cholesky_decomposition()
print(l)
print()
print("Let us multiply L with L.T :")
print()
print(l@l.T())
print()

print("Create diagonal matrix from [1,2,3]")
m = Matrix.diagonal_matrix([1,2,3], dtype = int)
print(m)


print("Using Array.arange(start = 0, stop = 5, step = 0.5, dtype = float) ")
print(Array.arange(start = 0, stop = 5, step = 0.5, dtype = int))
print()


print("Creating a vector filled with random numbers")
print (Vector.random_vector(10, 1, 5, int))
print()

print("Creating a matrix filled with random numbers")
print(Matrix.random_matrix((3,3), fromvalue = 1, tovalue = 10, dtype = int, seedval = 42))


print("mc is a matrix with complex numbers:")
mc = Matrix.from_list([[complex(1,2), complex(3,5), complex(0,1)]], dtype = complex)
print(mc)
print("md is a matrix with complex numbers as well: ")
md = Matrix.from_list([[complex(4,2)],[complex(1,0)], [complex(1,1)] ], dtype = complex)
print(md)
print("mc @ md = " + str(mc @ md))
print("md @ mc = " + str(md @ mc))

print("Hermitian conjugate of mc = " + str(mc.H()))


md = Matrix.from_list([   
        [complex(1,0), complex(3,5), complex(1,1)],
        [complex(3,-5), complex(2,0), complex(4,2)], 
        [complex(1,-1),complex(4,-2), complex(1,0)]
    ], dtype = complex)
print("md = " + str(md))
print("Is md a Hermitian matrix ? " + str(md.is_hermitian()))
print()

print("Obtaining information about vector v7 using repr(v7) = ")
print(repr(v7))
print()
print("Obtaining information about matrix md using repr(md) = ")
print(repr(md))
print()


print("Example of a general rotation with angle_x = math.pi/2, angle_y = math.pi/2, angle_z = math(pi)")
m = Matrix.rotation3D(math.pi/2, math.pi/2, math.pi)
print(m)

print()
print("Normalization of vector v =")
v = Vector.from_list([2,4,5])
print(v)
v = v.normalize()
print("leads to")
print(v)
print("Euclidean norm = ", end = "")
print(v.euclidean_norm())
print()

print("Random matrix m")
m = Matrix.random_matrix((5,5), fromvalue = -1, tovalue = 1, dtype = float)
print(m)
print("After removing rows 3 & 4 as well as columns 3 & 4")
m = m.remove_rows([3,4])
m = m.remove_columns([3,4])
print(m)

v = Vector.random_vector(6, dtype = float, fromvalue = -2, tovalue = 2)
print("Generating random vector = \n" + str(v))
print("Removing values at vector[1,3, and 5] " + str(v.remove(indices=[1,3,5])))
print()
array = [[1 for i in range(0,5)] for row in range(0,5)]
m = Matrix.from_list(array, dtype = int)
print("Matrix m = " + str(m))
v = Vector.from_list([1 for i in range(0,5)], dtype = int)
print("Vector v is " + str(v))
print("Adding vector v to all columns of m" + str(m.add_vector_to_all_columns(v)))
print("And now adding v.T() to a subset of rows of m", str(m.add_vector_to_rows({0, 2, 4}, v.T())))
print("The three base vectors of a base relative to the standard base:")
b1_new = Vector.from_list([0, -1,  2])
b2_new = Vector.from_list([1,  2,  0])
b3_new = Vector.from_list([1,  1,  1])
print(b1_new)
print(b2_new)
print(b3_new)
print("form the transformation matrix t")
T = Matrix.from_column_vectors([b1_new, b2_new, b3_new])
print(T)
print("The vector v_old ")
v_old = Vector.from_list([1,1,1])
print(str(v_old))
print("gets mapped to t @ v")
print(T @ v_old)

print("To get the other direction we need the inverse of T:")
T_inv = T.inverse_matrix()
print(T_inv)
print("using the following vector:")
v_new = Vector.from_list([2,2,3])
print(v_new)
print("and multiplying it with T_inv yields: ")
print(T_inv @ v_new)


# Demo usage of polynomials:
print()
print("Polynomials")
q1 = Polynomial([-8,-4,5,7])
q2 = Polynomial([-1,1])
print("q1 = " + str(q1))
print("q2 = " + str(q2))
res = (q1/q2)
print("q1 / q2 = ")
print("Quotient = " + str(res[0]))
print("Remainder = " + str(res[1]))
p = Polynomial([2,1]) * Polynomial([1,1]) * Polynomial([-4,1])
print("p = " + str(p))
print("What are the roots of p? ")
print(p.roots(100, 1E-5))
p = Polynomial([2,-7,5])
print("Polynom p = " + str(p))
print("Solving the quadratic equation leads to " + str(Polynomial([2,-7,5]).solve_quadr_eq()))
print("Polynomial q is " + str(Polynomial.linear_p(-1,1)))
print("Deviding p by q ")
res = Polynomial([2,-7,5]) / Polynomial.linear_p(-1,1)
print("yields " + str(res[0].normalize_coeff0()))
print("Derivative of p = " + str(p.derivative()))
print("Integral of p = "  + str(p.integral()))
print("pow(p,2) = " + str(pow(p,2)))
print("pow(p,3) = " + str(pow(p,3)))
print()
# Demo usage of Rational
print("Rational numbers")
r1 = Rational(1,2)
r2 = Rational(1,3)
print("Rational number r1 = " + str(r1))
print("Rational number r2 = " + str(r2))
print("r1 + r2 = " + str(r1+r2))
print("r1 * r2 = " + str(r1*r2))
print()
print("e as a Rational is ", Rational.e(20))
print()
print("What is the rational number that leads to 4711.000123666666..?")
print(Rational.periodicFloatToRational(number = 4711, fraction=123, leadingzeros = 3, period = 6))
print("float division yields ")
print(float(Rational.periodicFloatToRational(number = 4711, fraction=123, leadingzeros = 3, period = 6)))
print()
# Example to illustrate a 3D projection
print("Let us carry out a central projection with homogeneous corrdinates")
v = Vector.from_list([2,3,4,1])
print("Viewpoint is " + str(v))
print("The plane p is defined by ax+by+az=0 => n.T = (a,b,c,0).T")
n = Vector.from_list([1,4,7,0])
print(n)
M = v * n.T()
print("The projection matrix is given by v * n.T " + str(M))
p = Vector.from_list([-1,1,0,0])
print("Vector p is a point in 3D space " + str(p))
print("It is projected onto p' on the plane by M @ p " + str(M @ p))
print("which is in normal coordinates: ")
p_coord = Vector.from_list((M@p)[0:3])
print(p_coord)
print("By the way, det(M) is " + str(M.det()))
print("but obviously a projection from 3D space to 2D space cannot be invertible.")

print()
print("Printing matrices and vectors with print_matrix respectively print_vector")
print()
Array.print_matrix(M)
print()
Array.print_vector(M.row_vector(0))
print()
Array.print_vector(M.column_vector(0))
print()

m = Matrix.from_list([[1,2,3], [4,5,6], [7,8,9]], dtype = float)
print("Vectorization of matrix m = " + str(m))
print("using axis 0 => " + str(m.vectorize(axis = 0)))
print("using axis 1 => " + str(m.vectorize(axis = 1)))
print()


print("Polynomial regression")
x = Vector.from_list([1,2,3,4,5])
y = Vector.from_list([1,4,9,16,25])
print("Training data x = " + str(x))
print("Ground truth given by y = " + str(y))
print("Calling Polynomial.fit(x, y, 2, 0.0001, 10000)")
p = Polynomial.fit(x, y, 2, 0.0001, 10000)
print("We got back p = " + str(p))
print("Now, let us compare the ground truth with the results of the returned polynomial")
for i in range(0,len(x)):
    print("Estimation = " + str(p.compute(x[i])))
    print("Ground truth y = " + str(y[i]))
print()
    
print("Multivariate regression")
x = Matrix.from_list([[1,3], [2,4], [3,5],[4,8], [5,10]])
y = Vector.from_list([2,4,7,8,12])
print("Training data: x = " + str(x))
print("Training data: y = " + str(y))

coeffs = Regression.multivariate_fit(x,y,0.00001, 100000)
print("Coefficients determined by regression: " + str(coeffs))
for i in range(0, len(y)):
    print("Estimation = " + str(Regression.compute_multinomial(coeffs, x.m[i])))
    print("Ground truth y = " + str(y[i]))
    
print()
print("k-means clustering algorithm")
points = []
    
print("creating 60 random data points for k-means clustering")
for i in range(0,60):
    v = Vector.random_vector(3, fromvalue = -100, tovalue = 100, dtype = float, transposed=False)
    points.append(v.clone())
    
print("k-means clustering is called with random data points, cluster size = 3")
clusters = Clustering.k_means(points, 3)
for i in range(0, 3):
    print()
    print("CLUSTER + " + str(i))
    print("with size :" + str(len(clusters[i])))
    print("=================================================")
    
print()
print("k-means mini batch clustering algorithm")
points = []
    
print("creating 100 random data points for k-means clustering")
for i in range(0,100):
    v = Vector.random_vector(3, fromvalue = -100, tovalue = 100, dtype = float, transposed=False)
    points.append(v.clone())
    
print("k-means clustering is called with random data points, cluster size = 5, batchsize = 10")
clusters = Clustering.k_means_mini_batch(points, 5, 10, 100)
for i in range(0, 5):
    print()
    print("CLUSTER + " + str(i))
    print("with size :" + str(len(clusters[i])))
    print("=================================================")


print()
print("Using the Newton method to find the solution of X**2 = 2")
print("which means find the root of: x**2 - 2 = 0  (sqrt(2))")
res = Newton(lambda x: x**2, lambda x: 2*x)
print("The initial value for x is one")
print("Result is : " + str(res.compute(1,2)))
print()
print("Factorization of numbers by Common.factorize(), e.g. 210:")
f = Common.factorize(210)
print(f)
print("which can be defactorized by Common.defactorize()")
print(Common.defactorize(f))
n = 7853 * 6947 * 5701 * 2749 * 661 * 5 * 3 * 2
print("Another example is multiplying the prime numbers 7853 * 6947 * 5701 * 2749 * 661 * 5 * 3 * 2 = " + str(n))
f = Common.factorize(n)
print("Let us factorize it: " + str(f))
print("And now back again : " + str(Common.defactorize(f)))
print()
print("FunctionMatrices contain functions/lambdas instead of numbers")
m = FunctionMatrix(2,2)

m[0,0] = FunctionMatrix.null_func
m[0,1] = lambda x: -math.sin(x)
m[1,0] = math.cos
m[1,1] = FunctionMatrix.null_func
print("An example is matrix m " + str(m))
n = Matrix.from_list([[1,1],[1,1]])
print("A FunctionMatrix can be muliplied with a regular Matrix n " + str(n))
print("m @ n leads to vector multiplications where  functional elements of the FunctionMatrix are applied to the numbers of the regular matrix")
print("The results will be added with each other to set the corresponding element in the resulting regular matrix" + str(m @ n))
print()
print("An application of a FunctionMatrix to a same-shape regular matrix results in applying the individual function elements of the FunctionMatrix to the corresponding elements of the regular matrix. Let us apply m to n " + str(m.apply(n)))
print()
print("The multiplication of one FunctionMatrix with another FunctionMatrix is a little bit more complex. Multiplication of individual elements results in function composition, while the summing up of elements results in introducing lambdas that combine the composed functions using addition. E.g. m @ m = " + str(m @ m))
print()
print("Demo of Fast Fourier Transform in mathplus")
buffer = [complex(1), complex(1), complex(1), complex(1), complex(0), complex(0), complex(0), complex(0)]
print("The input buffer to fft() is as follows: ")
print(buffer)
print("FFT returns back the following result:")
Common.fft(buffer)
print(buffer)
print()
print("The Transfer category allows to exchange data between mathplus and numpy")
print()
m = Matrix.from_list([[11,12,13,14,15],[21,22,23,24,25],[31,32,33,34,35],[41,42,43,44,45],[51,52,53,54,55]])          
print("Let m be the following matrix " + str(m))
n = Transfer.matrix_to_numpy(m)
print("Transfering it to numpy creates the numpy matrix n = " + str(n))
m = Transfer.numpy_to_matrix(n)
print()
print("Transfering it back to mathplus ends up in m " + str(m))
print()
v= Vector.from_list([1,2,3,4], transposed=True)
print("The same holds for Vectors. Let v be transposed vector: \n" + str(v))
nv = Transfer.vector_to_numpy(v)
print()
print("After transfering it to numpy we get \n" + str(nv))
print("Transferring it back to mathplus results in : " + str(Transfer.numpy_to_vector(nv)))
print()
print("Interpolation with cubic natural splines")
print(" Given are 8 points with the x-coordinates: ")
print()
x = [0,1,2,3,4,5,6,7]
print(x)
print("and the y-coordinates: ")
y = [3,-1,0,2,1,4,0,-1]
print()
print(y)
print("Spline interpolation is initialized by calling cs = Interpolation.CubicSplines(x,y)")
cs = Interpolation.CubicSplines(x,y)
x = -1
while x <= 8:
    print("  x = " + str(x) + " Interpolation delivers : " + str(cs.interpolate(x)))
    x += 0.5
print()
print("Interpolation with connected line segments")
x = Array.lin_distribution(-3,3,13,float)
y = Array.apply_1D(math.sin, x)
print("x values are " + str(x))
print("y values are " + str(y))
print("The y-values are the sin of the corresponding x-values.")
print("Hence, we approximate math.sin.")
ip = Interpolation.Interpolation1D(x,y)
x0 = -3
while (x0 <= 3):
    res = ip.interpolate(x0)
    print("At x0 = " + str(x0) + " the interpolation value is " + str(res))
    x0 += 0.5


