#!/usr/bin/python
# William Dew

import sys
import pickle
sys.path.append("../tools/")

import pandas as pd
import numpy as np

from tester import dump_classifier_and_data

payment_columns = ['salary',
                   'bonus',
                   'long_term_incentive',
                   'deferred_income',
                   'deferral_payments',
                   'loan_advances',
                   'other',
                   'expenses',
                   'director_fees',
                   'total_payments']

stock_columns = ['exercised_stock_options',
                 'restricted_stock',
                 'restricted_stock_deferred',
                 'total_stock_value']

email_columns = ['to_messages',
                 'from_messages',
                 'from_poi_to_this_person',
                 'from_this_person_to_poi',
                 'shared_receipt_with_poi']             
              
features_list = ['poi'] + payment_columns + stock_columns + email_columns

### Load the dictionary containing the dataset
with open("final_project_dataset.pkl", "r") as data_file:
    data_dict = pickle.load(data_file)
    
# transfer dictionary to dataframe for easier data manipulation
df = pd.DataFrame.from_dict(data_dict, orient='index')
# replace all 'NaN' with numpy 'nan'
df = df.replace('NaN', np.nan)
# reorder dataframe columns to match features_list
df = df[features_list]

# Fill in the NaN payment and stock values with zero 
df[payment_columns] = df[payment_columns].fillna(0)
df[stock_columns] = df[stock_columns].fillna(0)

# Create a poi dataframe and nonpoi dataframe
df_poi = df[df["poi"]]
df_nonpoi = df[df['poi'] == False]
# Fill in the NaN email values with column mean in the poi dataframe
df_poi[email_columns] = df_poi[email_columns].fillna(df_poi[email_columns].mean())
df_nonpoi[email_columns] = df_nonpoi[email_columns].fillna(df_nonpoi[email_columns].mean())
# update the df with new poi dataframe and nonpoi dataframe
df = df_poi.append(df_nonpoi)

# Correct 2 error values in order taken from the official financial PDF
fixed_bel_rob = [0, 0, 0, -102500, 0, 0, 0, 3285, 102500, 3285, 0, 44093, -44093, 0]
fixed_bha_san = [0, 0, 0, 0, 0, 0, 0, 137864, 0, 137864, 15456290, 2604490, -2604490, 15456290]
# Putting the fixed values into the correct rows
df.loc["BELFER ROBERT", 1:15] = fixed_bel_rob
df.loc["BHATNAGAR SANJAY", 1:15] = fixed_bha_san

# Remove outliers
df.drop(axis=0, labels=['TOTAL', 'THE TRAVEL AGENCY IN THE PARK','FREVERT MARK A', 'BAXTER JOHN C', 'LAVORATO JOHN J', 'KEAN STEVEN J'], inplace=True)

# Create new feature(s)
df["from_poi_ratio"] = df["from_poi_to_this_person"] / df["from_messages"]
df["to_poi_ratio"] = df["from_this_person_to_poi"] / df["to_messages"]
df["shared_poi_ratio"] = df["shared_receipt_with_poi"] / df["to_messages"]
df["salary_ratio"] = df["salary"] / df["total_payments"]

features_list.append('to_poi_ratio')
features_list.append('from_poi_ratio')
features_list.append('shared_poi_ratio')
features_list.append('salary_ratio')

df.fillna(value=0, inplace=True)
df = df.replace('inf', 0)

# Scale the dataset and send it back to a dictionary
from sklearn.preprocessing import scale
scaled_df = df.copy()
scaled_df.iloc[:,1:] = scale(scaled_df.iloc[:,1:])
my_dataset = scaled_df.to_dict(orient='index')

from sklearn.pipeline import Pipeline
from sklearn.svm import SVC

# create my clf from SVC classifier
clf = Pipeline([('classify', SVC(C=100, kernel='linear'))])

# dump my clf, modified dataset, and modified features_list
dump_classifier_and_data(clf, my_dataset, features_list)