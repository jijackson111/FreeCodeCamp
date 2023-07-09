import numpy as np

# Given Array
x = np.array(['A', 'B', 'C', 'D', 'E'])

# 1. Show first element
x1 = x[0]
print('\n1.', x1)

# 2. Show last element
xl = x[-1]
print('\n2.', xl)

# 3. Show first three elements
x3 = x[0:3]
print('\n3.', x3)

# 4. Show all middle elements
xm = x[1:-1]
print('\n4.', xm)

# 5. Show elements in reverse position
xr = x[::-1]
print('\n5.', xr)

# 6. Show all elements at an odd position
xo = x[::2]
print('\n6.', xo)

# Given Matrix
y = np.array([
    [1,   2,  3,  4],
    [5,   6,  7,  8],
    [9,  10, 11, 12],
    [13, 14, 15, 16]
])

# 7. Show first row elements
y1 = y[0]
print('\n7.', y1)

# 8. Show last row elements
yl = y[-1]
print('\n8.', yl)

# 9. Show first element on first row
y11 = y[0, 0]
print('\n9.', y11)

# 10. Show last element on last row 
yll = y[-1, -1]
print('\n10.', yll)

# 11. Show middle row elements
ym = y[1:-1, 1:-1]
print('\n11/', ym)

# 12. Show first two elements on first two rows
y12 = y[0:2, 0:2]
print('\n12.', y12)

# 13. Show last two elements on last two rows
yl2 = y[2:, 2:]
print('\n13.', yl2)