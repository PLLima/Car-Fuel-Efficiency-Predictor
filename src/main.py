import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor

from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

n_repeats = 1 # For split_tests (Linear_Regression)
k_max = 11 # For Knn
mae_total = 0
mse_total = 0

# Original not actually original -> have to test which methods work with NaN
# Electric cars dont have some values, which causes errors in some models
# Removed from original dataset
data_cases = [
    "original",
    "others"
]

methods = [
    #"knn",
    #"random_forest",
    "linear_regression"
    #"neural_networks",
    #"svm"
]

for data_case in data_cases:
    df = pd.read_csv(f'T1/Dados/car_data_{data_case}.csv', delimiter=',')
    df_filtered = df.drop(columns=["model", "city_mpg", "highway_mpg"])
    # Choosing the variables to be analyzed (X) and predicted (y)
    # Variable chosen to be predicted : combination_mpg
    # Dropped city_mpg and highway_mpg -> combination_mpg is a combination of the two
    # Dropped model -> models tend to have the same consumption
    # Predictor might realize pattern
    df_encoded = pd.get_dummies(df_filtered)
    X = df_encoded.drop(columns=["combination_mpg"])
    y = df_encoded["combination_mpg"]

    ###############################################################################################
    ###### Forcing at least one instance of every category to appear in the original dataset ######
    ###############################################################################################
    #if data_case == "original":
    #    train_indices = set()
    #    for feature in categorical_features:
    #        unique_values = X[feature].unique()
    #        for value in unique_values:
    #            index = X[X[feature] == value].index[0]
    #            train_indices.add(index)

    #    train_indices = list(train_indices)
    #    X_train_mandatory = X.loc[train_indices]
    #    y_train_mandatory = y.loc[train_indices]

    #    X = X.drop(train_indices)
    #    y = y.drop(train_indices)


    for split_random_state in range(0, n_repeats):
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=split_random_state)
        scaler = StandardScaler()
        # Fit the train data and transform it into normal distribution
        X_train = scaler.fit_transform(X_train)
        # Transform the test data into normal distribution
        X_test = scaler.transform(X_test)
        #if data_case == "original":

            # Combine the mandatory and the rest of the training data
        #    X_train = pd.concat([X_train_mandatory, X_train])
        #    y_train = pd.concat([y_train_mandatory, y_train])

        for method in methods:
            match method:
                case "knn":
                    for k in range(1,2, k_max):
                        model = KNeighborsClassifier(n_neighbors=k)
                case "random_forest":
                    model = RandomForestRegressor(max_depth=4, min_samples_split=5, n_estimators=500,
                      oob_score=True, random_state=split_random_state, warm_start=True)
                case "linear_regression":
                    #preprocessor = ColumnTransformer([
                    #    ("num", "passthrough", numeric_features),
                    #    ("cat", OneHotEncoder(), categorical_features)
                    #])
                    model = LinearRegression()
                    #    ("preprocessor", preprocessor),
                    #    ("regressor", LinearRegression())
                    #])

                case "neural_networks":
                    model = MLPRegressor(random_state=split_random_state)
                    

        # Treinamento do modelo
        model.fit(X_train, y_train)

        # Previsão e avaliação
        y_pred = model.predict(X_test)
        mae = mean_absolute_error(y_test, y_pred)
        mse = mean_squared_error(y_test, y_pred)
        mae_total += mae
        mse_total += mse

        print("Mean Absolute Error (MAE) for iteration {} of {}:".format(split_random_state+1, data_case), mae)
        print("Mean Squared Error (MSE) for iteration {} of {}:".format(split_random_state+1, data_case), mse)
        print("\n")
    mae_med = mae_total/n_repeats
    mse_med = mse_total/n_repeats
    print("Average MAE for case {}: {}".format(data_case, mae_med))
    print("Average MSE for case {}: {}".format(data_case,mse_med))
    print("\n")
