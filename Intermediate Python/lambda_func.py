# lambda arguments: expression
add10 = lambda x: x + 10
print('Through lambda:', add10(5))

# Iterative function
def add10_func(x):
    return x + 10
add10_f = add10_func(5)
print('Through function:', add10_f)

# Multiple function lambda
mult = lambda x,y: x*y
print("5 and 10:", mult(5, 10))

# Sort list with lambda
points2D = [(1, 2), (15, 1), (5, -1), (10, 4)]
points2D_sorted = sorted(points2D, key=lambda x: x[1])
print('\nUnsorted list:', points2D)
print('Sorted list by y-value:', points2D_sorted)

# Map function
a = [1, 2, 3, 4, 5]
b = map(lambda x: x*2, a)
print('\nMultiplied by 2:', list(b))

# List comprehension
c = [x*2 for x in a]
print('Comprehension:', c)

# Filter function
d = filter(lambda x: x%2==0, a)
print('Filter function:', list(d))

