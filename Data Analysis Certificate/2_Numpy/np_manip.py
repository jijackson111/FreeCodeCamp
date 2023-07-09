import numpy as np

# 1. Convert the given integer array to float
x1 = [-5, -3, 0, 10, 40]
xf = np.array(x1, np.float32)
print('\n1.', xf)

# 2. Reverse the given array
x2 = x1
xr = x2[::-1]
print('\n2.', xr)

# 3. Sort/order the given array
x3 = [0, 10, -5, 40, -3]
xs = x3.sort()
print('\n3.', xs)

# 4. Set the fifth element equal to 1
x4 = np.zeros(10)
x4[4] = 1
print('\n4.', x4)

# 5. Swap the 50 in the given array to a 40
x5 = np.array([10, 20, 30, 50])
x5[-1] = 40
print('\n5.', x5)

# Given Matrix
mx = np.array([
    [1,   2,  3,  4],
    [5,   6,  7,  8],
    [9,  10, 11, 12],
    [13, 14, 15, 16]
])

# 6. Change the last row of the matrix to contain all 1s
mx[-1] = [1, 1, 1, 1]
print('\n6.', mx)

# 7. Change the last item on the last row to a 0
mx[-1, -1] = 0
print('\n7.', mx)

# 8. Add 5 to every element 
mx += 5
print('\n8.', mx)