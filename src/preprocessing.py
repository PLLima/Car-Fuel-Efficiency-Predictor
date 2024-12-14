import numpy as np

def drop_atributes(df, atributes_to_drop):
    return df.drop(columns=atributes_to_drop)

def remove_instances_with_nan(df):
    return df.dropna()

def remove_outliers(df, atributes):
    for atribute in atributes:
        q1 = np.percentile(df[atribute], 25)
        q3 = np.percentile(df[atribute], 75)
        iqr = q3 - q1
        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr
        df = df[(df[atribute] >= lower_bound) & (df[atribute] <= upper_bound)]
    return df

def filter_categories(df, categorical_features, threshold=0.1):
# Replace less common categories with 'others'
    total_instances = len(df)
    for feature in categorical_features:
        # Count occurrences of each category
        category_counts = df[feature].value_counts()
        
        # Identify categories that occur proportionally less than the threshold
        rare_categories = category_counts[(category_counts/total_instances) < threshold].index
        
        # Replace rare categories with 'others'
        df[feature] = df[feature].apply(lambda x: 'others' if x in rare_categories else x)
    return df