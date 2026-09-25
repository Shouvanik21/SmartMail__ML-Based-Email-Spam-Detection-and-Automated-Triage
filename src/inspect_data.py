#this line loads the pandas library as pd which is used to work with tabular data
import pandas as pd
import os

# Get the SmartMail project folder
project_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Build the dataset path
file_path = os.path.join(project_dir, "data", "SMSSpamCollection")

#load dataset
#we are telling pandas how to read this dataset |its a tab seperated thats why \t means separate the fields wherever there is a tab | we also gave two column names label and messages
df = pd.read_csv(file_path, sep="\t", header=None, names=["label", "message"])

#-------------------------------------------
# 1. BASIC INFORMATION
#-------------------------------------------

#this tells us the no of rows and columns in the  dataset
print("Dataset Shape:")
print(df.shape)

#shows the first five datas
print("\nFirst 5 rows:")
print(df.head())

#shows the column names and their datatypes
print("\nColumn names:")
print(df.columns)

#-------------------------------------------
# 2. CLASS DISTRIBUTION
#-------------------------------------------

#shows how many ham and spam messages we have each| tells us how many examples belongs to each class
print("\nClass Distribution:")
print(df["label"].value_counts())

#shows how many ham and spam messages we have each in percentage
print("\nClass Distribution in percent:")
print(df["label"].value_counts(normalize=True)*100)

#-------------------------------------------
# 3. MISSING VALUES
#-------------------------------------------

#checks whether something is missing
print("\nMissing Values:")
print(df.isnull().sum())

#-------------------------------------------
# 4. DUPLICATE VALUES
#-------------------------------------------

#checks whether the same complete row appears multiple times
print("\nDuplicate Rows:")
print(df.duplicated().sum())

#-------------------------------------------
# 5. MESSAGE LENGTH
#-------------------------------------------

#create a message length column "HELLO" --> 5
df["message_length"]=df["message"].str.len()

print("\nAverage Message Length:")
print(df["message_length"].mean())

print("\nShortest Message Length:")
print(df["message_length"].min())

print("\nLongest Message Length:")
print(df["message_length"].max())
