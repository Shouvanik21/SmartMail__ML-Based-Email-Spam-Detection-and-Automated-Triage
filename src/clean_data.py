import pandas as pd
import os
import matplotlib.pyplot as plt

# SmartMail project folder
project_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Dataset path
file_path = os.path.join(project_dir, "data", "SMSSpamCollection")

# load dataset
df = pd.read_csv(file_path, sep="\t", header=None, names=["label", "message"])

print("Original dataset shape/Before cleaning dataset shape:", df.shape)

# --------------------------------------------------
# 1. CLEANING THE DATASET
# --------------------------------------------------

# remove the missing values from the dataset
df = df.dropna(subset=["label", "message"])

# remove the duplicate values from the dataset
df = df.drop_duplicates()

# keep only the valid values from the dataset something other than ham/spam will be discarded
df = df[df["label"].isin(["ham", "spam"])]

# reset the row numbers in the dataset
df = df.reset_index(drop=True)

print("\nAfter cleaning dataset shape:", df.shape)

# --------------------------------------------------
# 2. SAVE CLEANED DATASET
# --------------------------------------------------

# this is thepath where our cleaned dataset will be stored
cleaned_data_path = os.path.join(project_dir, "data", "cleaned_SMSSpamCollection.csv")

# takes our cleaned DATAFRAME and converts it to csv and stores at the desired location
df.to_csv(cleaned_data_path, index=False)

print("\nCleaned dataset saved to:", cleaned_data_path)

# --------------------------------------------------
# 3. CLASS DISTRIBUTION
# --------------------------------------------------

# shows how many ham and spam messages we have each
class_counts = df["label"].value_counts()

print("\nClass Distribution:", class_counts)

# --------------------------------------------------
# 4. VISUALIZATION
# --------------------------------------------------

plt.figure(figsize=(7, 5))

plt.bar(class_counts.index, class_counts.values)

plt.title("Spam vs Ham Messages")
plt.xlabel("Message Type")
plt.ylabel("Number of Messages")

plt.tight_layout()

# Save graph in this location
plot_path = os.path.join(project_dir, "plots", "class_distribution.png")

plt.savefig(plot_path)
print("\nGraph saved to:", plot_path)

plt.show()
