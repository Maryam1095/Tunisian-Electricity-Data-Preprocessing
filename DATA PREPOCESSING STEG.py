# -*- coding: utf-8 -*-
"""
Created on Mon Jan  1 15:28:27 2024

@author: USER
"""
#DATA PREPROCESSING 

import pandas as pd
from sklearn.impute import SimpleImputer
data1_pandas = pd.read_csv("STEG_BILLING_HISTORY.csv", low_memory=False)

# Display information about the DataFrame
data1_pandas.info()

# Display the first few rows of the DataFrame
#data1_pandas_head = data1_pandas.head()

# Display the last few rows of the DataFrame
#data1_pandas_tail = data1_pandas.tail()

# Display the ten first lines and store in 'client_0_bills'
client_0_bills =data1_pandas.head(10)
print(client_0_bills)

# Check the data type of 'client_0_bills'
print(type(client_0_bills))

# Display general information about the dataset
print(data1_pandas.info())

# Get the number of rows and columns
num_rows, num_columns = data1_pandas.shape
print(f"Number of rows: {num_rows}, Number of columns: {num_columns}")

# Identify categorical features
categorical_features = data1_pandas.select_dtypes(include=['object']).columns
num_categorical_features = len(categorical_features)
print(f"Number of categorical features: {num_categorical_features}")

# Check for missing values
missing_values = data1_pandas.isnull().sum()


#Counting missing Values 
missing_values_total_count = data1_pandas.isnull().sum().sum()

#To replace the missing values, I chose imputing with median values because there is a relatively small number of missing values in the very large dataset using the median is a reasonable and efficient approach.
columns_with_missing = data1_pandas.columns[data1_pandas.isnull().any()]  # Identify columns with missing values

# Filter the DataFrame to include only columns with missing values
data_to_impute = data1_pandas[columns_with_missing]

# Initialize SimpleImputer with strategy set to "median"
imputer = SimpleImputer(strategy="median")

# Fit and transform the imputer on columns with missing values
data_imputed = pd.DataFrame(imputer.fit_transform(data_to_impute), columns=data_to_impute.columns)

# Update the missing values in the original DataFrame 'data1_pandas'
data1_pandas[columns_with_missing] = data_imputed

# Descriptive analysis
numeric_features = data1_pandas.select_dtypes(include=['float64', 'int64'])
data1_pandas_descriptive_statistics = numeric_features.describe()

# Select bills records for 'train_Client_0' using DataFrame filtering (Method 1)
bills_train_Client_0_1 = data1_pandas[data1_pandas['client_id'] == 'train_Client_0']


# Transform 'counter_type' feature to a numeric variable using encoding
# For instance, you can use LabelEncoder from scikit-learn
#from sklearn.preprocessing import LabelEncoder

#label_encoder = LabelEncoder()
#data1_pandas['counter_type_encoded'] = label_encoder.fit_transform(data1_pandas['counter_type'])

# Delete 'counter_statue' feature
#data1_pandas.drop('counter_statue', axis=1, inplace=True)




