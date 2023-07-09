import sqlite3
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Import Data
conn = sqlite3.connect('data/sakila.db')
df = pd.read_sql('''
    SELECT
        rental.rental_id, rental.rental_date, rental.return_date,
        customer.last_name AS customer_lastname,
        store.store_id,
        city.city AS rental_store_city,
        film.title AS film_title, film.rental_duration AS film_rental_duration,
        film.rental_rate AS film_rental_rate, film.replacement_cost AS film_replacement_cost,
        film.rating AS film_rating
    FROM rental
    INNER JOIN customer ON rental.customer_id == customer.customer_id
    INNER JOIN inventory ON rental.inventory_id == inventory.inventory_id
    INNER JOIN store ON inventory.store_id == store.store_id
    INNER JOIN address ON store.address_id == address.address_id
    INNER JOIN city ON address.city_id == city.city_id
    INNER JOIN film ON inventory.film_id == film.film_id
    ;
''', conn, index_col='rental_id', parse_dates=['rental_date', 'return_date'])
print("\n List of columns:\n", df.columns)

# Mean of Film Rental Duration
rent_dur = df['film_rental_duration']
print('\nAverage Duration of Film Rental:', rent_dur.mean())

# Bar Plot with All Durations
rent_durc = rent_dur.value_counts()
bp_dur = rent_durc.plot(kind='bar')

# Pie Plot for Rental Rates
rent_rate = df['film_rental_rate']
rr_vals = rent_rate.value_counts()
pp_dur = rr_vals.plot(kind='pie', figsize=(6,6))

# Density plot of replacement costs, with a red line on the mean and a green line on the median
rep_co = df['film_replacement_cost']
sns.displot(rep_co)
plt.xlabel('Cost')
plt.ylabel('Frequency')
plt.title('Film Replacement Costs')
plt.axvline(x=rep_co.mean(), color='red')