import numpy as np
import pandas as pd

# 1. Pandas utility functions (nan in pandas no longer acts as a virus, and is just ignored)
print('1.')
print(pd.isnull(None))
print(pd.isnull(3))
print(pd.notnull(None))
print(pd.notnull(3))

# 2. Pandas utility functions with series and dataframes
print('\n2.')
print(pd.isnull(pd.Series([1, np.nan, 7])))
print('\n', pd.isnull(pd.DataFrame({
    'Column A': [1, np.nan, 7],
    'Column B': [np.nan, 2, 3],
    'Column C': [np.nan, 2, np.nan]
})))

# 3. Filtering missing data
x = pd.Series([1, 2, 3, np.nan, np.nan, 4])
print('\n3.')
print(pd.notnull(x))
print('Number of missing:', pd.isnull(x).sum())
xn = x[pd.notnull(x)]
print(xn)

# 4. Dropping null values
df = pd.DataFrame({
    'Column A': [1, np.nan, 30, np.nan],
    'Column B': [2, 8, 31, np.nan],
    'Column C': [np.nan, 9, 32, 100],
    'Column D': [5, 8, 34, 110],
})
print('\n4.')
print(df.dropna(axis=1))
print(df.dropna(how='all'))
print(df.dropna(thresh=3))

# 5. Fill null values
print('\n5.')
print(df.fillna(method='ffill', axis=0))
