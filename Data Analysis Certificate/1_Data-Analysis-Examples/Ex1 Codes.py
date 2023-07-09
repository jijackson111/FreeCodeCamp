import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from csv import reader

# Import and Preview Data
file = 'C:/Users/JI_mu/OneDrive/Documents/FCC/Pandas/data/sales_data.csv'
data = pd.read_csv(file)


# Mean Customer Age 
ages = data['Customer_Age']
age_mean = ages.mean()
print('\nMean Customer Age:', age_mean)


# Mean Order Quantity
oq = data['Order_Quantity']
oq_mean = oq.mean()
print('Mean Order Quantity:', oq_mean)


# Number of Sales per Annum
y13 = (data.loc[data['Year'] == 2013])
y14 = (data.loc[data['Year'] == 2014])
y15 = (data.loc[data['Year'] == 2015])
y16 = (data.loc[data['Year'] == 2016])
spa_data = [[2013, len(y13)], [2014, len(y14)], [2015, len(y15)], [2016, len(y16)]]
spa_df = pd.DataFrame(spa_data, columns = ['Year', 'Sales'])
print('\n', spa_df)


# Country with Highest Quantity of Sales
nat_data = data['Country']
nat_md = nat_data.mode()
nat_val = nat_md.values
for val in nat_val:
    print('\nCountry with Highest Number of Sales:', val)
  
# List of All Products Sold
data_prod = data['Product']
prod_list = data_prod.unique()
print('\nA List Containing Every Product Sold:')
print(prod_list)

#   Scatter Plot for Unit Cost and Unit Price
spd = {'x': data['Unit_Cost'], 'y': data['Unit_Price']}
spdf = pd.DataFrame(spd)
plt.scatter(spdf.x, spdf.y)
plt.xlabel('Unit Cost')
plt.ylabel('Unit Price')
plt.title('Cost vs Price of Units')

#   Box Plot for Profit per Country
nat_list = nat_data.unique()
data_list = []
for nat in nat_list:
     n = (data.loc[data['Country'] == nat])
     p = n['Profit'].sum()
     data_list.append(p)
data_dict = {nat_list[0]: data_list[0], nat_list[1]: data_list[1], nat_list[2]: data_list[2], 
             nat_list[3]: data_list[3], nat_list[4]: data_list[4], nat_list[5]: data_list[5]}
ddf = pd.DataFrame.from_dict(data_dict, orient='index')
fig, ax = plt.subplots()
ax.boxplot(ddf)
plt.title('Profits per Country')
plt.show()

# Function to Convert Month Names to Numbers for Next Problem
def month_name_to_number(x):
    if x == 'January':
        return 1
    elif x == 'February':
        return 2
    elif x == 'March':
        return 3
    elif x == 'April':
        return 4
    elif x == 'May':
        return 5
    elif x == 'June':
        return 6
    elif x == 'July':
        return 7
    elif x == 'August':
        return 8
    elif x == 'September':
        return 9
    elif x == 'October':
        return 10
    elif x == 'November':
        return 11
    elif x == 'December':
        return 12
    else:
        return 'error'

# Create a Column Labelled 'Calculated_Date' with the Format 'YYYY-MM-DD'
date_list = []
with open(file, 'r') as read_obj:
    csv_reader = reader(read_obj)
    for row in csv_reader:
        d = row[1]
        mm = row[2]
        m = month_name_to_number(mm)
        str_m = str(m)
        y = row[3]
        if len(d) == 1:
            ds = '0{}'.format(d)
        else:
            ds = d
        if len(str_m) == 1:
            ms = '0{}'.format(m)
        else:
            ms = m
        calc_d = '{}/{}/{}'.format(y, ms, ds)
        date_list.append(calc_d)
date_list.remove('Year/error/Day')
data['Calculated_Date'] = date_list
print('\n', data.head())        

# Parse Calculated_Date into a Datetime Object
data['Calculated_Date'] = pd.to_datetime(data['Calculated_Date'])
print('\n', data.head())

# How Many Bike Rack Orders were Made from Canada
cnd = (data.loc[data['Country'] == 'Canada'])
bkr = (cnd.loc[cnd['Sub_Category'] == 'Bike Racks'])
print('\nThe Number of Bike Rack Orders from Canada is:', len(bkr))

#    Create a Visual Model of Sales per Category
cg = data['Product_Category']
cg_list = list(cg.unique())
cg_gr = data.groupby(data['Product_Category'])
acc = cg_gr.get_group('Accessories')
clo = cg_gr.get_group('Clothing')
bik = cg_gr.get_group('Bikes')
acc_num = len(acc)
clo_num = len(clo)
bik_num = len(bik)
bar_data = pd.DataFrame({'Sales': [acc_num, clo_num, bik_num]}, 
                        index = [cg_list[0], cg_list[1], cg_list[2]])
bar_data.plot(kind='bar', title='Sales per Category')

# Number of Orders in May 2016
yr16 = data.loc[data['Year'] == 2016]
mn5 = yr16.loc[yr16['Month'] == 'May']
print('\nTotal Orders in May 2016:', len(mn5))