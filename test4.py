from matrix import Vector, Matrix

# checks the shape of an array 
def array_shape(a):
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
        
# determines how many items are covered in a slice        
def slice_dimension(slc, array):
    return len(array[slc])
    
    

"""
v1 = Vector.from_list([1,2,3,4])

v2 = Vector.from_list([1,1,1,1])

m1 = Matrix.from_list([[1,1,1,1],[1,1,1,1], [1,1,1,1], [1,1,1,1]])

m1[2][2:4] = [7,8]
print(m1)


e = Matrix.identity(4)


print(e*m1)
print(m1*e)
print(e @ e.inverse_matrix() * m1)

m1 = Matrix.from_list([[0,1,0,0],[1,0,0,0], [0,0,0,1], [0,0,1,0]])
print(m1)
print(m1.inverse_matrix())

print(v1.T()  * v1)

m1[1]= [0,1,2,3]
print(m1)
m1[1,2] = 42
print(m1[0:2, 1:3])
print(m1)
print(m1[:,:])
print(m1[1])
print(m1[0,0:2])
print(v1)

print(v1[:])
"""

array = [1,2,3,4,5]
slc = slice(None,None,None)
print(array[slc])

print(array_shape([[1,2,3]]))
print(slice_dimension(slc,[1,2,3,4]))