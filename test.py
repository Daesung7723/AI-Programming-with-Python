# ------------------------------------------------------------------
# 1. Setup Kaggle API and Download Data
# ------------------------------------------------------------------
# Install the Kaggle library
!pip install kaggle

# Setup for uploading the kaggle.json file
from google.colab import files
print("Please upload your Kaggle API token (kaggle.json) file.")
files.upload() # This code will open a file selection window.

# Create Kaggle directory and set permissions
!mkdir -p ~/.kaggle
!cp kaggle.json ~/.kaggle/
!chmod 600 ~/.kaggle/kaggle.json

# Download the Kaggle dataset (Car Dekho dataset)
print("\nStarting dataset download...")
!kaggle datasets download -d nehalbirla/vehicle-dataset-from-cardekho
!unzip vehicle-dataset-from-cardekho.zip # Unzip the file
print("Dataset download and extraction complete.")
print("-" * 50)


# ------------------------------------------------------------------
# 2. Load Data and Check Basic Information
# ------------------------------------------------------------------
import pandas as pd
import numpy as np

# Read the CSV file into a pandas DataFrame
df = pd.read_csv('CAR DETAILS FROM CAR DEKHO.csv')

print("--- Checking Basic Data Information ---")
print("First 5 rows of the data:")
print(df.head())
print("\nData summary information:")
df.info()
print("-" * 50)


# ------------------------------------------------------------------
# Goal 1: Check and Handle Missing Values (This dataset has no missing values)
# ------------------------------------------------------------------
print("--- Checking for Missing Values ---")
print(df.isna().sum())
print("There are no missing values to handle in this dataset.")
print("-" * 50)


# ------------------------------------------------------------------
# Goal 2: Create 'Brand' Column and Convert 'year' Column Type
# ------------------------------------------------------------------
print("--- Data Cleaning and Type Conversion ---")
# Extract the first word from the 'name' column to create a 'Brand' column
df['Brand'] = df['name'].str.split(' ').str[0]

# Convert the data type of the 'year' column to integer
df['year'] = df['year'].astype(int)

print("Data after creating 'Brand' column and converting 'year' type:")
print(df.head())
print("\nData types after conversion:")
print(df.dtypes)
print("-" * 50)


# ------------------------------------------------------------------
# Goal 3: Calculate Average Price by 'Brand'
# ------------------------------------------------------------------
print("--- Calculating Average Selling Price by Brand ---")
# The selling_price is in Indian Rupees (INR).
# Convert to integer for cleaner display (no decimals).
avg_price_by_brand = df.groupby('Brand')['selling_price'].mean().astype(int).sort_values(ascending=False)
print(avg_price_by_brand)
print("-" * 50)


# ------------------------------------------------------------------
# Goal 4: Filter Cars with Mileage Less Than 10,000 km
# ------------------------------------------------------------------
print("--- Filtering cars with km_driven < 10,000 ---")
# The mileage in the original data is low, so we filter by 10,000 km instead of 100,000 km.
low_mileage_cars = df[df['km_driven'] < 10000]
print(low_mileage_cars)
print("-" * 50)

print("🎉 Kaggle Data Analysis Mini-Project Complete! 🎉")
