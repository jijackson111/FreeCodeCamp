import numpy as np
import pandas as pd

# Import data
df = pd.read_csv('adult.data.csv')

# Race counts
race = df['race']
race_val = race.value_counts()
print('Race counts:\n', race_val)

# Average age of men
men = df[df['sex'] == 'Male']
age_avg = men['age'].mean()
print('\nAverage age of men:', age_avg)

# Percentage of people with Bachelors
bach = df[df['education'] == 'Bachelors']
perc = (len(bach) / len(df)) * 100
print('\nPercentage with Bachelors:', perc)

# What percentage of people with advanced education make >50k
adv_list = ['Bachelors', 'Masters', 'Doctorate']
adv = df[df['education'] == adv_list]
print(adv)