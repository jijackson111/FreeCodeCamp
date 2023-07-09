import numpy as np

# 1. Create an array of size 10, filled with zeros
a_0 = np.zeros(10)
print('1.', a_0)

# 2. Create array with values from 10 to 49
a_19 = np.arange(10, 50)
print('\n2.', a_19)

# 3. Create a matrix of 2*2 integers that is filled with ones
m20 = np.ones([2, 2], dtype=np.int64)
print('\n3.', m20)

# 4. Create a matrix of 4*4, filled with fives
m45 = np.ones([4, 4], dtype=np.int64) * 5
print('\n4.', m45)

# 5. Create a new numpy array with the same shape and type as X, but that is filled with sevens
X = np.arange(4, dtype=np.int64)
xx = np.ones_like(X) * 7
print('\n5.', xx)

# 6. Create an array, filled with 3 random integer values between 1 and 10
ra = np.random.randint(10, size=3)
print('\n6.', ra)

# 7. Create a 3*3*3 matrix, filled with random float values
rm = np.random.randn(3, 3, 3)
print('\n7.', rm)

# 8. Convert the given list into an array
XX = [1, 2, 3]
xar = np.array(XX)
print('\n8.', xar, type(xar))

# 9. Make a copy of the given array, and store it on the variable 'Y'
XA = np.array([5, 2, 3], dtype=np.int64)
Y = np.copy(XA)
print('\n9.', Y)

# 10. Create an array with the odd numbers between 1 and 10
ao10 = np.arange(1, 11, 2)
print('\n10.', ao10)

# 11. Create an array with numbers from 1 to 10, in descending order
a10 = np.arange(1, 11)[::-1]
print('\n11.', a10)

# 12. Create a 3*3 matrix, filled with values from 0 to 8
m38 = np.arange(9).reshape(3, 3)
print('\n12.', m38)

# 13. Show the memory size of the given matrix
Z = np.zeros((10,10))
print("\n13.", "%d bytes" % (Z.size * Z.itemsize))