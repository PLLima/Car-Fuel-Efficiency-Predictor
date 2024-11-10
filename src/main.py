import pandas as pd
import matplotlib.pyplot as plt

from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.neural_network import MLPRegressor
from sklearn.svm import SVR

from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split

n_repeats = 5 # For split_tests (Linear_Regression)
k = 9 # For Knn

# Original not actually original -> have to test which methods work with NaN
# Electric cars dont have some values, which causes errors in some models
# Removed from original dataset
data_cases = [
    "original",
    "others"
]

methods = [
    "knn",
    "random_forest",
    "linear_regression",
    "neural_networks",
    "svm"
]

for data_case in data_cases:
    mae_total = 0
    mse_total = 0
    df = pd.read_csv(f'src/data/car_data_{data_case}.csv', delimiter=',')
    df_filtered = df.drop(columns=["city_mpg", "highway_mpg"])
    # Choosing the variables to be analyzed (X) and predicted (y)
    # Variable chosen to be predicted : combination_mpg
    # Dropped city_mpg and highway_mpg -> combination_mpg is a combination of the two
    # Dropped model -> models tend to have the same consumption
    # Predictor might realize pattern
    X = df_filtered.drop(columns=["combination_mpg"])
    y = df_filtered["combination_mpg"]
    numerical_features = X.select_dtypes(include=['int64', 'float64']).columns
    categorical_features = X.select_dtypes(include=['object', 'bool']).columns

    ###############################################################################################
    ###### Forcing at least one instance of every category to appear in the original dataset ######
    ###############################################################################################
    if data_case == "original":
        train_indices = set()
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

    # Choose a common preprocessing for every model for a specific dataset
    preprocessor = ColumnTransformer([
        ("num", StandardScaler(), numerical_features),
        ("cat", OneHotEncoder(), categorical_features)
    ])

    for method in methods:
        for split_random_state in range(0, n_repeats):
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=split_random_state)
            if data_case == "original":
                X_train = pd.concat([X_train_mandatory, X_train])
                y_train = pd.concat([y_train_mandatory, y_train])

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

            pipe = Pipeline([
                ("preprocessor", preprocessor),
                ("regressor", model)
            ])

            # Treinamento do modelo
            pipe.fit(X_train, y_train)

            # Previsão e avaliação
            y_pred = pipe.predict(X_test)
            mae = mean_absolute_error(y_test, y_pred)
            mse = mean_squared_error(y_test, y_pred)
            mae_total += mae
            mse_total += mse

            #print("Mean Absolute Error (MAE) for iteration {} of {} using the {} method:".format(split_random_state+1, data_case, method), mae)
            #print("Mean Squared Error (MSE) for iteration {} of {} using the {} method:".format(split_random_state+1, data_case, method), mse)
            #print("\n")
        
        mae_med = mae_total/n_repeats
        mse_med = mse_total/n_repeats
        print("Average MAE for case {} using the {} method: {}".format(data_case, method, mae_med))
        print("Average MSE for case {} using the {} method: {}".format(data_case, method, mse_med))
        print("\n")
