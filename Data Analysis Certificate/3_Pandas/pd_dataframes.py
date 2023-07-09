import numpy as np
import pandas as pd

# Given Data
marvel_data = [
    ['Spider-Man', 'male', 1962],
    ['Captain America', 'male', 1941],
    ['Wolverine', 'male', 1974],
    ['Iron Man', 'male', 1963],
    ['Thor', 'male', 1963],
    ['Thing', 'male', 1961],
    ['Mister Fantastic', 'male', 1961],
    ['Hulk', 'male', 1962],
    ['Beast', 'male', 1963],
    ['Invisible Woman', 'female', 1961],
    ['Storm', 'female', 1975],
    ['Namor', 'male', 1939],
    ['Hawkeye', 'male', 1964],
    ['Daredevil', 'male', 1964],
    ['Doctor Strange', 'male', 1963],
    ['Hank Pym', 'male', 1962],
    ['Scarlet Witch', 'female', 1964],
    ['Wasp', 'female', 1963],
    ['Black Widow', 'female', 1964],
    ['Vision', 'male', 1968]
]

# 1. Create a dataframe
df = pd.DataFrame(data=marvel_data)

# 2. Add column names to the df
df.columns= ['Name', 'Sex', 'Year']

# 3. Add index names (use the character name as index)
df.index = df['Name']

# 4. Drop the name column as it is now the index
del df['Name']

# 5. Drop the rows for Namor and Hank Pym
df = df.drop(['Namor', 'Hank Pym'], axis=0)

# Print dataframe each step
print(df)

# 6. Show first five elements
print('\n6.', df.iloc[:5])

# 7. Show last five elements
print('\n7.', df.iloc[-5:])

# 8. Show the sex of the first five elements
print('\n8.', df.iloc[:5].Sex)

# 9. Show first and last elements
print('\n9.', df.iloc[[0, -1],])

# 10. Change the year of Vision to 1964
df.loc['Vision', 'Year'] = 1964

# 11. Add a new column called 'Since'
df['Since'] = 2018 - df['Year']

# Print dataframe
print('\n', df)

# 12. Use a mask to show all female characters
m1 = df['Sex'] == 'female'
print('\n12.', df[m1])

# 13. Get male characters with first year past 1970
m2 = (df['Sex'] == 'male') & (df['Year'] > 1970)
print('\n13.', df[m2])

# 14. Show the basic statistics of the dataframe
print('\n14.', df.describe())

# 15. Show the mean value of first year
fym = df['Year'].mean()
print('\n15.', fym)

# 16. Show maximum of year since introduction
imax = df['Since'].max()
print('\n16.', imax)

# 17. Reset index names
df = df.reset_index()
print('\n17.', df)

# 18. Plot values of Year
print(df.Year.plot())
