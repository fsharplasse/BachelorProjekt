import pandas as pd
import numpy as np
import LinearModels as LM
import NonLinearModels as NLM
import matplotlib.pyplot as plt


pd.options.display.float_format = '{:.4f}'.format
path = "/Users/nicholaihjelme/Documents/GitHub/Bachelorprojekt/"

# CHOICE DATA
choice_data = pd.read_csv(path + 'data/choice_data.csv', delimiter=';', encoding = 'unicode_escape')
#choice_data = choice_data.rename(columns = {'Make-model-year-fuel' : 'key'})
choice_data['key'] = choice_data['Model'].str.replace('jaguar ', '-')
choice_data['key'] = choice_data['Make'] + '-' + choice_data['Model'] + '-' + choice_data['Year'].astype(str) + '-' + choice_data['Fuel']
choice_data['key'] = choice_data['key'].str.replace(' ', '-')
choice_data_subset = choice_data[['Year', 'Shares', 'Fuel', 'Weight (kg)', 'Engine effect (kW)', 'Prices (2015-DKK)', 'key']]
print("Choice_data subset:\n", choice_data_subset)

# BILBASEN SCRAPE
bilbasen_data = pd.read_csv(path + 'data/bilbasen_scrape.csv', delimiter=';', encoding = 'unicode_escape')
bilbasen_data.rename(columns = {'Unnamed: 6': 'make model year'}, inplace = True)
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
    share_check = combined_data['Shares'].loc[i]
    fuel_check = combined_data['Fuel'].loc[i]
    weight_check = combined_data['Weight (kg)'].loc[i]
    engine_check = combined_data['Engine effect (kW)'].loc[i]
    kmL_check = combined_data['kmL'].loc[i]
    price_check = combined_data['Prices (2015-DKK)'].loc[i]
    if share_check == '-' or fuel_check == '-' or weight_check == '-' or engine_check == '-' or kmL_check == '-' or price_check == '-':
        combined_data = combined_data.drop(labels = i, axis = 0)

combined_data['kmL'] = pd.to_numeric(combined_data['kmL'], errors = 'coerce')
combined_data['kmL'] = combined_data['kmL'].astype(float)
fuel_dummy = pd.get_dummies(combined_data['Fuel'])
combined_data = combined_data.join(fuel_dummy).astype(int)
combined_data = combined_data.reset_index(drop = True)
print("\ncombined_data after drop duplicates:\n", combined_data)
combined_data.to_csv(path + 'final_data.csv')

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
ax1.set_ylabel('km - EL', color = 'black') 
plot_1 = ax1.plot(list(fe_el_data['Year'].drop_duplicates()), fe_el_means, color = 'green', label = 'EL') 
ax1.tick_params(axis ='y', labelcolor = 'black') 

ax2 = ax1.twinx()
ax2.set_ylabel('km/l - Benzin/Diesel', color = 'black') 
plot_2 = ax2.plot(list(fe_benzin_data['Year'].drop_duplicates()), fe_benzin_means, color = 'red', label = 'Benzin') 
ax2.tick_params(axis ='y', labelcolor = 'black')

plot_3 = plt.plot(list(fe_diesel_data['Year'].drop_duplicates()), fe_diesel_means, color = 'blue', label = 'Diesel') 

lns = plot_1 + plot_2 + plot_3
labels = [l.get_label() for l in lns]
plt.legend(lns, labels, loc = 0)

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

plt.plot(list(size_el_data['Year'].drop_duplicates()), size_el_means, color = 'g', label = 'EL')
plt.plot(list(size_benzin_data['Year'].drop_duplicates()), size_benzin_means, color = 'r', label = 'Benzin')
plt.plot(list(size_diesel_data['Year'].drop_duplicates()), size_diesel_means, color = 'b', label = 'Diesel')

plt.xlabel("Years")
plt.ylabel("m3")
plt.title("Size")
  
plt.legend()
plt.show()

# MAKING LINEAR REGRESSION
combined_data = combined_data.values

y = combined_data[:, 1].astype(float)
y = pd.DataFrame({'Shares' : y})
print("\ny:\n", y)

x1 = combined_data[:, 2].astype(str)
x2 = np.ones(x1.shape).astype(float)
x3 = combined_data[:, 3].astype(float)
x4 = combined_data[:, 4].astype(float)
x5 = combined_data[:, 7].astype(float)
x6 = combined_data[:, 5].astype(float)

x = pd.DataFrame({'Constant' : x2, 'Fuel' : x1, 'Weight' : x3, 'Engine effect' : x4, 'Full efficiency (km/l)' : x5, 'Price' : x6})
x = pd.get_dummies(data = x, drop_first = True)
print("\nX:\n", x)

y_label = list(y.columns)[0]
print("\ny_label:", y_label)

x_labels = list(x.columns)
print("x_labels:", x_labels)

y = np.array(y)
#print("y:\n", y)

x = np.array(x)
#print("X:\n", x)

pols = LM.estimate(y, x, robust_se = True)

print("\n")
LM.print_table((y_label, x_labels), pols, title = "Linear regression", floatfmt = '.7f')


print("\n")