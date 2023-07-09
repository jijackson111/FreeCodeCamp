import numpy as np
import pandas as pd

# Given Pandas Series
x = pd.Series(['A','B','C'])

# 1. Name the series 'My Letters'
x.name = 'My Letters'
print('\n1.', x.name)

# 2. Show the values in the series
print('\n2.', x.values)

# 3. Assign index names to the given series
index_names = ['first', 'second', 'third']
x.index = index_names
print('\n3.', x)

# 4. Show the first element
print('\n4.', x[0])

# 5. Show the last element
print('\n5.', x[-1])

# Given Pandas Series 2
y = pd.Series(['A','B','C','D','E'],
              index=['first','second','third','forth','fifth'])

# 6. Show all middle elements
ym = y[1:-1]
print('\n6.', ym)

# 7. Show the elements in reverse position
yr = y[::-1]
print('\n7.', yr)

# 8. Show the first and last elements
print('\n8.', y[0], y[-1])

# Given Pandas Series 3
z = pd.Series([1,2,3,4,5],
              index=['A','B','C','D','E'])

# 9. Convert the integers in the series to floats
zf = pd.Series(z, dtype=np.float16)
print('\n9.', zf)

# 10. Order the series
z = z.sort_values()
print('\n10.', z)

# 11. Set the fifth element equal to 10
z[4] = 10
print('\n11.', z[4])

# 12. Change all middle elements to 0
z[1:-1] = 0
print('\n12.', z)

# 13. Add 5 to every element
z = z + 5
print('\n13.', z)

# Given Pandas Series 4
s = pd.Series([-1, 2, 0, -4, 5, 6, 0, 0, -9, 10])

# 14. Create a mask showing negative elements
m1 = s < 0
print('\n14.', s[m1])

# 15. Create a mask to get numbers higher than 5
m2 = s > 5
print('\n15.', s[m2])

# 16. Create a mask to get elements higher than the mean of the series
m3 = s > s.mean()
print('\n16.', s[m3])

# 17. Use a mask to get numbers equal to 2 or 10
m4 = (s == 2) | (s == 10)
print('\n17.', s[m4])

# 18. Return True if no elements are zero
print('\n18.', s.all())

# 19. Return True if any elements are zero
print('\n19.', s.any())

# 20. Show sum of elements
ss = s.sum()
print('\n20.', ss)

# 21. Show minimum value of elements
smin = s.min()
print('\n21.', smin)