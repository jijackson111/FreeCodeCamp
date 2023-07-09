import numpy as np

# Given array
x = np.array([-1, 2, 0, -4, 5, 6, 0, 0, -9, 10])

# 1. Get the negative elements
m1 = x <= 0
print('\n1.', x[m1])

# 2. Get numbers higher than 5
m2 = x > 5
print('\n2.', x[m2])

# 3. Get numbers higher than the mean of the elements
m3 = x > x.mean()
print('\n3.', x[m3])

# 4. Get numbers equal to 2 or 10
m4 = (x == 2) | (x == 10)
print('\n4.', x[m4])

# 5. Return true if no elements are zero
print('\n5.', x.all())

# 6. Return true if any of its elements are zero
print('\n6.', x.any())

# 7. Show sum of elements
xs = x.sum()
print('\n7.', xs)

# 8. Show max value of elements
x_max = x.max()
print('\n8.', x_max)

# Given matrix
m = np.array([
    [1,   2,  3,  4],
    [5,   6,  7,  8],
    [9,  10, 11, 12],
    [13, 14, 15, 16]
])

# 9. Show sum of columns
msc = m.sum(axis=0)
print('\n9.', msc)

# 10. Show mean value of rows
mmr = m.mean(axis=1)
print('\n10.', mmr)

