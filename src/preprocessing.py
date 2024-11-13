import pandas as pd

def remove_instances_with_nan(df):
    return df.dropna()

def filter_categories(df, categorical_features, threshold=10):
# Replace less common categories with 'others'
    for feature in categorical_features:
        # Count occurrences of each category
        category_counts = df[feature].value_counts()
        
        # Identify categories that occur less than the threshold
        rare_categories = category_counts[category_counts < threshold].index
        
        # Replace rare categories with 'others'
        df[feature] = df[feature].apply(lambda x: 'others' if x in rare_categories else x)

    return df