import pandas as pd
import numpy as np
import LinearModels as LM
import matplotlib.pyplot as plt
import NonLinearModels_post as nlm
from matplotlib.ticker import FormatStrFormatter

pd.options.display.float_format = '{:.4f}'.format
path = "/Users/Lasse/Documents/Bachelor/Seb/MotorRegisterData-main/"

# CHOICE DATA
choice_data_subset = pd.read_csv(path + 'clogit__data_subset.csv', delimiter=',', encoding = 'unicode_escape')
choice_data_subset = choice_data_subset.rename(columns = {'Key' : 'key'})

# BILBASEN SCRAPE
bilbasen_data = pd.read_csv(path + 'bilbasen_scrape.csv', delimiter=';', encoding = 'unicode_escape')
bilbasen_data.rename(columns = {'duns': 'make model year'}, inplace = True)
bilbasen_data['key'] = bilbasen_data['make model year'] + '-' + bilbasen_data['drivkraft']
bilbasen_subset = bilbasen_data[['kmL', 'key']]
print("\nBilbasen subset:\n", bilbasen_subset)

# COMBINING THE TWO DATASETS
combined_data = pd.merge(choice_data_subset, bilbasen_subset, on = "key")
print("\ncombined_data before drop duplicates:\n", combined_data)
combined_data.drop_duplicates(subset = ['key'], keep = 'last', inplace = True, ignore_index = True)
combined_data['kmL'] = combined_data['kmL'].str.replace('km/l', '')
combined_data['kmL'] = combined_data['kmL'].str.replace('km', '')
combined_data['kmL'] = combined_data['kmL'].str.replace("\(NEDC\)", '')
combined_data['kmL'] = combined_data['kmL'].str.replace(',', '.')

for i in combined_data.index:
    share_check = combined_data['MarketShare'].loc[i]
    fuel_check = combined_data['Fuel'].loc[i]
    weight_check = combined_data['Weight (kg)'].loc[i]
    engine_check = combined_data['Engine effect (kW)'].loc[i]
    kmL_check = combined_data['kmL'].loc[i]
    price_check = combined_data['Prices (2015-DKK)'].loc[i]
    if share_check == '-' or fuel_check == '-' or weight_check == '-' or engine_check == '-' or kmL_check == '-' or price_check == '-':
        combined_data = combined_data.drop(labels = i, axis = 0)

combined_data['kmL'] = pd.to_numeric(combined_data['kmL'], errors = 'coerce')
combined_data['kmL'] = combined_data['kmL'].astype(float)
combined_data = combined_data.reset_index(drop = True)
combined_data.rename(columns = {'Prices (2015-DKK)': 'Nypris'}, inplace = True)
combined_data.rename(columns = {'Weight (kg)': 'Weight'}, inplace = True)
print("\ncombined_data after drop duplicates:\n", combined_data)
combined_data.to_csv(path + 'finales.csv')

#


# MAKING FUEL EFFICIENCY PLOT
fe_data = combined_data[['Year', 'kmL', 'Fuel']].set_index(['Fuel'])
print("\nFE data for plotting:\n", fe_data)

fe_el_data = fe_data.loc['El']
fe_el_means = fe_el_data.groupby('Year')['kmL'].mean()
print("\nFE EL means:\n", fe_el_means)

fe_benzin_data = fe_data.loc['Benzin']
fe_benzin_means = fe_benzin_data.groupby('Year')['kmL'].mean()
print("\nFE Benzin means :\n", fe_benzin_means)

fe_diesel_data = fe_data.loc['Diesel']
fe_diesel_means = fe_diesel_data.groupby('Year')['kmL'].mean()
print("\nFE Diesel means :\n", fe_diesel_means)

fig, ax1 = plt.subplots()

ax1.set_xlabel('Years')
ax1.set_ylabel('EV: km pr. charging', color = 'black') 
plot_1 = ax1.plot(list(fe_el_data['Year'].drop_duplicates()), fe_el_means, color = 'magenta', label = 'EV') 
ax1.tick_params(axis ='y', labelcolor = 'black') 

ax2 = ax1.twinx()
ax2.set_ylabel('Petrol/Diesel: km/l', color = 'black') 
plot_2 = ax2.plot(list(fe_benzin_data['Year'].drop_duplicates()), fe_benzin_means, color = 'blue', label = 'Petrol') 
ax2.tick_params(axis ='y', labelcolor = 'black')

plot_3 = plt.plot(list(fe_diesel_data['Year'].drop_duplicates()), fe_diesel_means, color = 'cyan', label = 'Diesel') 

plt.title("Range") 

lns = plot_1 + plot_2 + plot_3
labels = [l.get_label() for l in lns]
plt.legend(lns, labels, loc = 0)


plt.show()

# Engineeffect plot
e_data = choice_data[['Year', 'Engine effect (kW)', 'Fuel']].set_index(['Fuel'])
print("\nFE data for plotting:\n", e_data)

e_el_data = e_data.loc['El']
e_el_means = e_el_data.groupby('Year')['Engine effect (kW)'].mean()
print("\nFE EL means:\n", e_el_means)

e_benzin_data = e_data.loc['Benzin']
e_benzin_means = e_benzin_data.groupby('Year')['Engine effect (kW)'].mean()
print("\nFE Benzin means :\n", e_benzin_means)

e_diesel_data = e_data.loc['Diesel']
e_diesel_means = e_diesel_data.groupby('Year')['Engine effect (kW)'].mean()
print("\nFE Diesel means :\n", e_diesel_means)

fig, ax1 = plt.subplots()

ax1.set_xlabel('Years')
ax1.set_ylabel('EV', color = 'black') 
plot_1 = ax1.plot(list(e_el_data['Year'].drop_duplicates()), e_el_means, color = 'magenta', label = 'EV') 
ax1.tick_params(axis ='y', labelcolor = 'black') 

ax2 = ax1.twinx()
ax2.set_ylabel('Petrol/Diesel', color = 'black') 
plot_2 = ax2.plot(list(e_benzin_data['Year'].drop_duplicates()), e_benzin_means, color = 'blue', label = 'Petrol') 
ax2.tick_params(axis ='y', labelcolor = 'black')

plot_3 = plt.plot(list(e_diesel_data['Year'].drop_duplicates()), e_diesel_means, color = 'cyan', label = 'Diesel') 

plt.title("Engine effect") 

lns = plot_1 + plot_2 + plot_3
labels = [l.get_label() for l in lns]
plt.legend(lns, labels, loc = 0)

plt.show()


ee_data = choice_data[['Year', 'Engine effect (kW)', 'Fuel']].set_index(['Fuel'])
print("\nSize data for plotting:\n", ee_data)

ee_el_data = ee_data.loc['El']
ee_el_means = ee_el_data.groupby('Year')['Engine effect (kW)'].mean()
print("\nSize EL means:\n", ee_el_means)

ee_benzin_data = ee_data.loc['Benzin']
ee_benzin_means = ee_benzin_data.groupby('Year')['Engine effect (kW)'].mean()
print("\nSize Benzin means :\n", ee_benzin_means)

ee_diesel_data = ee_data.loc['Diesel']
ee_diesel_means = ee_diesel_data.groupby('Year')['Engine effect (kW)'].mean()
print("\nSize Diesel means :\n", ee_diesel_means)

plt.plot(list(ee_el_data['Year'].drop_duplicates()), ee_el_means, color = 'm', label = 'EL')
plt.plot(list(ee_benzin_data['Year'].drop_duplicates()), ee_benzin_means, color = 'b', label = 'Benzin')
plt.plot(list(ee_diesel_data['Year'].drop_duplicates()), ee_diesel_means, color = 'c', label = 'Diesel')

plt.xlabel("Years")
plt.ylabel("Engine effect (kW)")
plt.title("Engine Effect")
  
plt.legend()
plt.show()

# Weight in kg plot
weight_data = choice_data[['Year', 'Weight (kg)', 'Fuel']].set_index(['Fuel'])
print("\nSize data for plotting:\n", weight_data)

weight_el_data = weight_data.loc['El']
weight_el_means = weight_el_data.groupby('Year')['Weight (kg)'].mean()
print("\nSize EL means:\n", weight_el_means)

weight_benzin_data = weight_data.loc['Benzin']
weight_benzin_means = weight_benzin_data.groupby('Year')['Weight (kg)'].mean()
print("\nSize Benzin means :\n", weight_benzin_means)

weight_diesel_data = weight_data.loc['Diesel']
weight_diesel_means = weight_diesel_data.groupby('Year')['Weight (kg)'].mean()
print("\nSize Diesel means :\n", weight_diesel_means)

plt.plot(list(weight_el_data['Year'].drop_duplicates()), weight_el_means, color = 'm', label = 'EL')
plt.plot(list(weight_benzin_data['Year'].drop_duplicates()), weight_benzin_means, color = 'b', label = 'Benzin')
plt.plot(list(weight_diesel_data['Year'].drop_duplicates()), weight_diesel_means, color = 'c', label = 'Diesel')

plt.xlabel("Years")
plt.ylabel("Weight in kilogram (kg)")
plt.title("Weight")
  
plt.legend()
plt.show()


# MAKING SIZE (m3) PLOT
size_data = choice_data[['Year', 'Size (m3)', 'Fuel']].set_index(['Fuel'])
print("\nSize data for plotting:\n", size_data)

size_el_data = size_data.loc['El']
size_el_means = size_el_data.groupby('Year')['Size (m3)'].mean()
print("\nSize EL means:\n", size_el_means)

size_benzin_data = size_data.loc['Benzin']
size_benzin_means = size_benzin_data.groupby('Year')['Size (m3)'].mean()
print("\nSize Benzin means :\n", size_benzin_means)

size_diesel_data = size_data.loc['Diesel']
size_diesel_means = size_diesel_data.groupby('Year')['Size (m3)'].mean()
print("\nSize Diesel means :\n", size_diesel_means)

plt.plot(list(size_el_data['Year'].drop_duplicates()), size_el_means, color = 'm', label = 'EL')
plt.plot(list(size_benzin_data['Year'].drop_duplicates()), size_benzin_means, color = 'b', label = 'Benzin')
plt.plot(list(size_diesel_data['Year'].drop_duplicates()), size_diesel_means, color = 'c', label = 'Diesel')

plt.xlabel("Years")
plt.ylabel("Cubic metres (m3)")
plt.title("Size")
  
plt.legend()
plt.show()

# MARKET SHARES
ms_data = choice_data[['Year', 'No. of registrations', 'Fuel']].set_index(['Fuel'])
print("\nSize data for plotting:\n", ms_data)

ms_el_data = ms_data.loc['El']
ms_el_means = ms_el_data['No. of registrations'] / ms_el_data.groupby('Year')['No. of registrations'].transform('sum')
print("\nSize EL means:\n", ms_el_means)

ms_benzin_data = ms_data.loc['Benzin']
ms_benzin_means = ms_benzin_data['No. of registrations'] / ms_benzin_data.groupby('Year')['No. of registrations'].transform('sum')
print("\nSize Benzin means :\n", ms_benzin_means)

ms_diesel_data = ms_data.loc['Diesel']
ms_diesel_means = ms_diesel_data['No. of registrations'] / ms_diesel_data.groupby('Year')['No. of registrations'].transform('sum')
print("\nSize Benzin means :\n", ms_diesel_means)

plt.plot(list(ms_el_data['Year'].drop_duplicates()), ms_el_means, color = 'm', label = 'El Small')
plt.plot(list(ms_benzin_data['Year'].drop_duplicates()), ms_benzin_means, color = 'b', label = 'El Large')
plt.plot(list(ms_diesel_data['Year'].drop_duplicates()), ms_diesel_means, color = 'c', label = 'Diesel')


plt.xlabel("Years")
plt.ylabel("Market Shares")
plt.title("Market Shares: EV size")
  
plt.legend()
plt.show()

# PRICE
price_data = choice_data[['Year', 'Prices (2015-DKK)', 'Fuel-size segment']].set_index(['Fuel-size segment'])
print("\nSize data for plotting:\n", price_data)

price_el_data = price_data.loc['ElSmall']
price_el_means = price_el_data.groupby('Year')['Prices (2015-DKK)'].mean()
print("\nSize EL means:\n", price_el_means)

price_el_data1 = price_data.loc['ElLarge']
price_el_means1 = price_el_data1.groupby('Year')['Prices (2015-DKK)'].mean()
print("\nSize EL means:\n", price_el_means1)

price_benzin_data = price_data.loc['BenzinSmall']
price_benzin_means = price_benzin_data.groupby('Year')['Prices (2015-DKK)'].mean()
print("\nSize Benzin means :\n", price_benzin_means)

price_benzin_data1 = price_data.loc['BenzinLarge']
price_benzin_means1 = price_benzin_data1.groupby('Year')['Prices (2015-DKK)'].mean()
print("\nSize Benzin means :\n", price_benzin_means1)

price_diesel_data = price_data.loc['DieselSmall']
price_diesel_means = price_diesel_data.groupby('Year')['Prices (2015-DKK)'].mean()
print("\nSize Diesel means :\n", price_diesel_means)

price_diesel_data1 = price_data.loc['DieselLarge']
price_diesel_means1 = price_diesel_data1.groupby('Year')['Prices (2015-DKK)'].mean()
print("\nSize Diesel means :\n", price_diesel_means1)

fig, ax = plt.subplots()

ax.yaxis.set_major_formatter(FormatStrFormatter('%.0f'))

plt.plot(list(price_el_data['Year'].drop_duplicates()), price_el_means, color = 'm', label = 'ELSmall')
plt.plot(list(price_el_data1['Year'].drop_duplicates()), price_el_means1, color = 'b', label = 'ElLarge')
plt.plot(list(price_diesel_data['Year'].drop_duplicates()), price_diesel_means, color = 'c', label = 'DieselSmall')
plt.plot(list(price_diesel_data1['Year'].drop_duplicates()), price_diesel_means1, color = 'r', label = 'DieselLarge')
plt.plot(list(price_benzin_data['Year'].drop_duplicates()), price_benzin_means, color = 'g', label = 'BenzinSmall')
plt.plot(list(price_benzin_data1['Year'].drop_duplicates()), price_benzin_means1, color = 'k', label = 'BenzinLarge')

plt.xlabel("Years")
plt.ylabel("Price")
plt.title("Prices for all segments")
  
plt.legend()
plt.show()

