from matrix import Vector, Matrix

# just a show case to illustrate usage of Matrix and Vector classes
# in matrix

m1 =  Matrix(4,4)
print("m1 = " + str(m1))
m2 = Matrix.Unit(4)
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
    print("col vector [" + str(i) + "] = " + str(m3.col_vector(i)))
    
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

print("Multiplying unit matrix (size == 5) with 2: ", str(Matrix.Unit(5).mult_with_scalar(2)))

m4 = Matrix.from_list([[1,2,3], [3, 2, 1]])
m5 = Matrix.from_list([[1,1], [1,1], [1,1]])
print("m4 = " + str(m4))
print("m5 = " + str(m5))
print("m5 * m4 = " + str(m5 * m4))

print("m4 * m5 = " + str(m4 * m5))

print("Use apply() to apply the lambda x -> x^2 to each element of m3: " + str(m3.apply(lambda x: x * x)))


print("Use apply() to apply the lambda x -> x + 1 to each element of v4: " + str(v4.apply(lambda x: x + 1)))



m6 = m4 * m5
print("Cloning a matrix m6 = m3.clone(): " + str(m3.clone()))

print("m7a = m6.clone(); m7b = m6")
m7a = m6.clone()
m7b = m6
print("m7a = " + str(m7a))
print("m7b = " + str(m7b))

m6.set_item(0, 0, 42)
print("Changing element 0,0 of matrix m6 to 42 => " + str(m6))

print("m7a = " + str(m7a))
print("m7b = " + str(m7b))

