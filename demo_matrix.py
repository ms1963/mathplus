from matrix import Vector, Matrix, Common
import math

# just a show case to illustrate usage of Matrix and Vector classes
# in matrix


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
print("m1 * m2 + m2 = " + str(m1 * m2 + m2))

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

print("Multiplying unit matrix (size == 5) with 2: ", str(Matrix.identity(5).mult_with_scalar(2)))

m4 = Matrix.from_list([[1,2,3], [3, 2, 1]])
m5 = Matrix.from_list([[1,1], [1,1], [1,1]])
print("m4 = " + str(m4))
print("m5 = " + str(m5))
print("m5 * m4 = " + str(m5 * m4))

print("m4 * m5 = " + str(m4 * m5))

print("Use apply() to apply the lambda x -> x^2 to each element of m3: " + str(m3.apply(lambda x: x * x)))

print("You may also use m3.apply(math.sin) for vectors or matrices")
print(m3.apply(math.sin))
print()

print("Use apply() to apply the lambda x -> x + 1 to each element of v4: " + str(v4.apply(lambda x: x + 1)))

print()
print("Calculating the diagonal of matrix m5 * m4")
list = (m5*m4).diagonal()
print(list)
print()

m6 = m4 * m5
print("Creating matrix m6 = m4 * m5: " + str(m6))

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
Matrix.set_format('{:.4f}')
Vector.set_format('{:.4f}')
print("Cofactor matrix of m7 = " + str(m7.cofactor_matrix()))
print("Inverse(m7) = " + str(m7.inverse_matrix()))
print("m7 * inverse(m7)" + str(m7 * m7.inverse_matrix()))
print("Solving equation m7 * x = [1][2][0]")
print(m7.solve_equation(Vector.from_list([1, 2, 0])))
print("Frobenius norm of m7 = " + str(m7.frobenius_norm()))
print()
Matrix.set_format('{:4}')
Vector.set_format('{:4}')
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

print("Eigenvalues of m ")
print(m.eigenvalues())
print()

print("For matrix : ")
m = Matrix.from_list([[5,4],[1,2]])
print(m)
print("the eigenvalues are " + str(m.eigenvalues()))


print("Create diagonal matrix from [1,2,3]")
m = Matrix.diagonal_matrix([1,2,3], dtype = int)
print(m)


print("Using Matrix.arange(start = 0, stop = 5, step = 0.5, dtype = float) ")
print(Matrix.arange(start = 0, stop = 5, step = 0.5, dtype = int))
print()


print("Creating a vector filled with random numbers")
print (Vector.random_vector(10, 1, 5, int))
print()

print("Creating a matrix filled with random numbers")
print(Matrix.random_matrix((3,3), fromvalue = 1, tovalue = 10, dtype = int, seedval = 42))


print(Common.delete([i for i in range(0, 10)], [1,3,5,7]))























