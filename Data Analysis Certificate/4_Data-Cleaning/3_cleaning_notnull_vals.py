import numpy as np
import pandas as pd

# Given Data Frame
df = pd.DataFrame({
    'Sex': ['M', 'F', 'F', 'D', '?'],
    'Age': [29, 30, 24, 290, 25],
})
print(df)

# 1. Finding unique values
print('\n1.')
print(df['Sex'].unique())
print(df['Sex'].value_counts())
over_100 = df[df['Age'] > 100]
print(over_100)
df.loc[df['Age'] > 100, 'Age'] = df.loc[df['Age'] > 100, 'Age'] / 10
print(df)

# Given Data Frame 2
ambassadors = pd.Series([
    'France',
    'United Kingdom',
    'United Kingdom',
    'Italy',
    'Germany',
    'Germany',
    'Germany',
], index=[
    'Gérard Araud',
    'Kim Darroch',
    'Peter Westmacott',
    'Armando Varricchio',
    'Peter Wittig',
    'Peter Ammon',
    'Klaus Scharioth '
])
    
# 2. Duplicates
print('\n2.')
print(ambassadors.duplicated())
print('\n', ambassadors.duplicated(keep='last'))
print('\n', ambassadors.drop_duplicates())