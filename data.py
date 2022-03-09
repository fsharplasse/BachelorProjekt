import pandas as pd
import numpy as np
## import LinearModels as LM

path = "/Users/Lasse/Documents/Bachelor/Seb/MotorRegisterData-main/"

choice_data = pd.read_csv(path + 'choice_data.csv', delimiter=';', encoding = 'unicode_escape')
choice_data = choice_data.rename(columns = {'Make-model-year-fuel' : 'key'})
#choice_data["Full name"] = choice_data["Make"] + ' ' + choice_data["Model"] + ' ' + choice_data["Year"].astype(str) + ' ' + choice_data['Fuel']
choice_data_subset = choice_data[['Weight (kg)', 'Engine effect (kW)', 'No. of registrations', 'Shares', 'Prices (2015-DKK)', 'key']]
print("Choice_data subset:\n", choice_data_subset)


bilbasen_data = pd.read_csv(path + 'bilbasen_scrape.csv', delimiter=';', encoding = 'unicode_escape')
bilbasen_data.rename(columns = {'duns': 'make model year'}, inplace = True)
bilbasen_data['key'] = bilbasen_data['make model year'] + '-' + bilbasen_data['drivkraft']
#bilbasen_data['key2'] = bilbasen_data['Unnamed: 2'] + '-' + bilbasen_data['Unnamed: 3'] + '-' + bilbasen_data['aargang'].astype(str) + '-' + bilbasen_data['drivkraft']
bilbasen_subset = bilbasen_data[['horsepower_hk', 'nypris_kr', 'drivkraft', '0-100kmt_sek', 'aargang', 'tophastighed_kmt', 'kmL', 'forbrug (WLTP)', 'Tank', 'key']]
print("\nBilbasen subset:\n", bilbasen_subset)


combined_data = pd.merge(choice_data_subset, bilbasen_subset, on = "key")
print("\nBefore drop duplicates:\n", combined_data)
combined_data.drop_duplicates(subset = ['key'], keep = 'last', inplace = True, ignore_index = True)
for i in combined_data.index:
    pris_check = combined_data['nypris_kr'].loc[i]
    tid_check = combined_data['0-100kmt_sek'].loc[i]
    hastighed_check = combined_data['tophastighed_kmt'].loc[i]
    horsepower_check = combined_data['horsepower_hk'].loc[i]
    if pris_check == '-' or tid_check == '-' or hastighed_check == '-' or horsepower_check == '-':
        #print("Deleting:", combined_data['key'].loc[i])
        combined_data = combined_data.drop(labels = i, axis = 0)
combined_data = combined_data.reset_index(drop = True)
print("\nAfter drop duplicates:\n", combined_data)
combined_data.to_excel(path + 'final_data.xlsx')