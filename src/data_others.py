# Code to unify every instance of a cathegory that is less common than 2% into a new class called "others"

import pandas as pd

# Load the data
df = pd.read_csv('T1/Dados/car_data_original.csv')

# List of categorical features to check
categorical_features = ["class", "drive", "fuel_type", "make", "model", "transmission"]

# Threshold for considering a category as rare
threshold = 10

# Replace less common categories with 'others'
for feature in categorical_features:
    # Count occurrences of each category
    category_counts = df[feature].value_counts()
    
    # Identify categories that occur less than the threshold
    rare_categories = category_counts[category_counts < threshold].index
    
    # Replace rare categories with 'others'
    df[feature] = df[feature].apply(lambda x: 'others' if x in rare_categories else x)

# Save the modified DataFrame to a new CSV file
df.to_csv('car_data_others.csv', index=False)

print("Processing complete. The new file has been saved as 'car_data_others.csv'.")