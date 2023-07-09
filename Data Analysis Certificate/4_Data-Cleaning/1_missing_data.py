import numpy as np

# 1. Falsy values
print('1.')
falsy_values = (0, False, None, '', [], {})
print(any(falsy_values))

# 2. np.nan (behaves as a virus)
print('\n2.')
print(np.nan)
print(3 + np.nan)
lt = np.array([1, 2, 3, np.nan])
print(lt.mean())

# 3. np.inf (also behaves as a virus)
print('\n3.')
print(np.inf)
print(3 + np.inf)
lti = np.array([1, 2, 3, np.inf])
print(lti.mean())

# 4. Checking for nan or inf
print('\n4.')
a = np.array([1, 2, 3, np.nan, np.inf, 4])
print(np.isnan(a))
print(np.isinf(a))
print(np.isfinite(a))

# 5. Filtering out nan and inf
print('\n5.')
print(a[~np.isnan(a)])
afin = a[np.isfinite(a)]
print(afin)
print(afin.sum())