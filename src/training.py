import pandas as pd

from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.neural_network import MLPRegressor
from sklearn.svm import SVR

from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.model_selection import KFold

import plots

def train(n_repeats, kfolds, k, plot_flag, X_train, y_train, X_train_mandatory, y_train_mandatory, method):
    mae_total = 0
    mse_total = 0
    all_predictions = []  # List to store predictions for each split
    for split_random_state in range(0, n_repeats):
        # Separate validation data and the remaining instances
        #X_train, X_val, y_train, y_val  = train_test_split(X, y, test_size=val_size)
        # Split the remaining instances in kfolds parts
        kf = KFold(n_splits=kfolds, shuffle=True)
        for train_index, test_index in kf.split(X_train):
            X_train, X_val = X_train.iloc[train_index], X_train.iloc[test_index]
            y_train, y_val = y_train.iloc[train_index], y_train.iloc[test_index]
            # If no_electric_cars, join the forced test_case with the random test_case
            # Concatanate the mandatory and training sets
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
                    model = RandomForestRegressor(max_depth=6)
                case "linear_regression":
                    model = LinearRegression()
                case "neural_networks":
                    model = MLPRegressor(max_iter=2500)
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
            y_pred = pipe.predict(X_val)
            # all_predictions.append(y_pred)

            if plot_flag:
                plots.plot(method, y_val, y_pred)

            mae = mean_absolute_error(y_val, y_pred)
            mse = mean_squared_error(y_val, y_pred)
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
    print("Average MAE using the {} method: {}".format(method, mae_med))
    print("Average MSE using the {} method: {}".format( method, mse_med))
    print("\n")