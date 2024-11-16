import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.neural_network import MLPRegressor
from sklearn.svm import SVR

from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import RobustScaler
from sklearn.preprocessing import PolynomialFeatures

from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split

# Extra iterations to calculate average and increase reproducibility
n_repeats = 1
k = 9 # K for Knn
Ksplit = 10
plot_flag = False  

# Electric cars dont have some values, which causes errors in some models
# Removed from datasets
data_cases = [
    "no_electric_cars"
#    "grouped_categories",
]

methods = [
    "knn",
    "random_forest",
    "linear_regression",
    "neural_networks",
    "svm",
]

for data_case in data_cases:
    test_size = 0.1
    df = pd.read_csv(f'src/data/car_data_{data_case}.csv', delimiter=',')
    # Variable chosen to be predicted : combination_mpg
    # Dropped city_mpg and highway_mpg -> combination_mpg is a combination of the two
    # Predictor might realize pattern
    df_filtered = df.drop(columns=["city_mpg", "highway_mpg"])

    if plot_flag:
        plt.figure(figsize=(10, 6))
        sns.histplot(df_filtered['combination_mpg'], kde=True,color='skyblue')
        plt.xlabel('Fuel Consumption (Miles per Gallon)')
        plt.ylabel('Number of Vehicles')
        plt.title('Distribution of Combination MPG')
        plt.savefig('src/plots/combination_dist.png')

        # Changed the car classes plot to a pie chart
        #plt.figure(figsize=(10, 6))
        #sns.countplot(data=df_filtered, x='class', order=df['class'].value_counts().index)
        #plt.title('Distribution of Car Classes')
        #plt.xticks(rotation=45)
        #plt.savefig('src/plots/classes_dist.png')
        
        plt.figure(figsize=(10, 6))
        car_class_counts = df_filtered['class'].value_counts()
        plt.pie(car_class_counts, labels=car_class_counts.index, 
                autopct='%1.1f%%', startangle=140, colors=sns.color_palette('pastel'))
        plt.title('Distribution of Car Classes')
        plt.savefig(f'src/plots/classes_{data_case}_dist.png')

        numeric_df = df_filtered.select_dtypes(include=[np.number])
        plt.figure(figsize=(10, 8))
        sns.heatmap(numeric_df.corr(), annot=True, cmap='coolwarm', fmt='.2f')
        plt.title('Correlation Heatmap of Numeric Features')
        plt.savefig('src/plots/num_heatmap.png')

    # Choosing the variables to be analyzed (X) and predicted (y)
    X = df_filtered.drop(columns=["combination_mpg"])
    y = df_filtered["combination_mpg"]

    ##################################################################################################
    ###### Forcing at least one instance of every category to appear in the fossil_fuel dataset ######
    ##################################################################################################
    # As in no_electric_cars, there isnt any strategy for dealing with non represented instances
    # during the testing phase, we force every instance to appear at least once in the training case
    if data_case == "no_electric_cars":
        train_indices = set()
        categorical_features = X.select_dtypes(include=['object', 'bool']).columns
        for feature in categorical_features:
            unique_values = X[feature].unique()
            for value in unique_values:
                index = X[X[feature] == value].index[0]
                train_indices.add(index)

        train_indices = list(train_indices)
        X_train_mandatory = X.loc[train_indices]
        y_train_mandatory = y.loc[train_indices]

        X = X.drop(train_indices)
        y = y.drop(train_indices)
        
        test_size = test_size * (len(X) + len(X_train_mandatory)) / len(X)

    for method in methods:
        mae_total = 0
        mse_total = 0
        all_predictions = []  # List to store predictions for each split
        for split_random_state in range(0, n_repeats):
            #X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=split_random_state)
            kf = KFold(n_splits=Ksplit, shuffle=True, random_state=split_random_state)
            for train_index, test_index in kf.split(X):
                X_train, X_test = X.iloc[train_index], X.iloc[test_index]
                y_train, y_test = y.iloc[train_index], y.iloc[test_index]
                strat_size = len(y_test) / len(y_train)
                X_train, X_val, y_train, y_val  = train_test_split(X_train, y_train, test_size=strat_size, random_state=split_random_state)
                # If no_electric_cars, join the forced test_case with the random test_case
                if data_case == "no_electric_cars":
                    X_train = pd.concat([X_train_mandatory, X_train])
                    y_train = pd.concat([y_train_mandatory, y_train])

                # Preprocessing is sensitive to type
                # Separate analyzed features into numerical and categorical
                # As to apply preprocessing only to valid features
                numerical_features = X_train.select_dtypes(include=['int64', 'float64']).columns
                categorical_features = X_train.select_dtypes(include=['object', 'bool']).columns

                # ColumnTransformer applies preprocessing patterns e.g. StandardScaler() and
                # OneHotEncoder() to groups, e.g. numerical_features and categorical_features
                # Preprocessor will be applied to every dataset
                preprocessor = ColumnTransformer([
                    #("num", StandardScaler(), numerical_features),
                    #("cat", OneHotEncoder(), categorical_features)
                    #("num", PolynomialFeatures(degree=2, include_bias=False), numerical_features),
                    ("num", StandardScaler(), numerical_features),
                    ("cat", OneHotEncoder(), categorical_features)
                ])
                # Select model and configuration for use in the pipeline
                match method:
                    case "knn":
                        model = KNeighborsClassifier(n_neighbors=k)
                    case "random_forest":
                        model = RandomForestRegressor(max_depth=6, random_state=split_random_state)
                    case "linear_regression":
                        model = LinearRegression()
                    case "neural_networks":
                        model = MLPRegressor(random_state=split_random_state, max_iter=2500)
                    case "svm":
                        model = SVR(C=1.0, epsilon=0.2)
                # Pipeline applies the preprocessed dataset to the model for fitting
                pipe = Pipeline([
                    ("preprocessor", preprocessor),
                    ("regressor", model)
                ])

                # Fitting the pipeline
                pipe.fit(X_train, y_train)

                # Prediction and evaluation
                y_pred = pipe.predict(X_test)
                # all_predictions.append(y_pred)

                if plot_flag:
                    plt.figure(figsize=(10, 6))
                    plt.scatter(y_test, y_pred, color='blue', alpha=0.6, label='Average Predicted vs Actual')
                    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', lw=2, label='Ideal Fit Line')
                    plt.xlabel('Actual Combination MPG')
                    plt.ylabel('Average Predicted Combination MPG')
                    plt.title(f'{method.upper()} Model: Average Predicted vs Actual Combination MPG')
                    plt.legend()
                    plt.grid(True)
                    plt.savefig(f'src/plots/{method}_{data_case}_actual_vs_predicted.png')

                mae = mean_absolute_error(y_test, y_pred)
                mse = mean_squared_error(y_test, y_pred)
                mae_total += mae
                mse_total += mse

            #print("Mean Absolute Error (MAE) for iteration {} of {} using the {} method:".format(split_random_state+1, data_case, method), mae)
            #print("Mean Squared Error (MSE) for iteration {} of {} using the {} method:".format(split_random_state+1, data_case, method), mse)
            #print("\n")
        
        #all_predictions = np.array(all_predictions)
        #print("ALL")
        #print(all_predictions)
        #average_predictions = np.mean(all_predictions, axis=0)
        #print("Avg")
        #print(average_predictions)

        mae_med = mae_total/n_repeats
        mse_med = mse_total/n_repeats
        print("Average MAE for case {} using the {} method: {}".format(data_case, method, mae_med))
        print("Average MSE for case {} using the {} method: {}".format(data_case, method, mse_med))
        print("\n")
