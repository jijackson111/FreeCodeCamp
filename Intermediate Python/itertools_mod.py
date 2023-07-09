from itertools import *

# Product function
a = [1, 2]
b = [3, 4]
prod = product(a, b)
print('\nList of product elements:\n', list(prod))

# permutations function
c = [1, 2, 3]
perm = permutations(c, 2)
print('\nList of permutations from input:\n', list(perm))

# combinations function
d = [1, 2, 3]
comb = combinations(d, 2)
print('\nList of combinations from input:\n', list(comb))

# accumulate
e = [1, 2, 3, 4]
acc = accumulate(e)
print('\nList of accumulations from input:\n', list(acc))

# groupby
def smaller_than_3(x):
    return x < 3
f = [1, 2, 3, 4]
group_obj = groupby(f, key=smaller_than_3)
print('\nIterating through a dataset using a given key:')
for key, value in group_obj:
    print(key, list(value))