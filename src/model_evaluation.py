import numpy as np
import pandas as pd

from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.neural_network import MLPRegressor
from sklearn.svm import SVR

from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import GridSearchCV

from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error

def generate_train_validation_sets(train_indexes, validation_indexes, X_train, X_mandatory, y_train, y_mandatory):
    X_train, X_val = X_train.iloc[train_indexes], X_train.iloc[validation_indexes]
    y_train, y_val = y_train.iloc[train_indexes], y_train.iloc[validation_indexes]
    # If no_electric_cars, join the forced test_case with the random test_case
    # Concatanate the mandatory and training sets
    X_train = pd.concat([X_mandatory, X_train])
    y_train = pd.concat([y_mandatory, y_train])
    return X_train, X_val, y_train, y_val

def evaluate_model(method, X_train, X_val, y_train, y_val, random_state):
    # Preprocessing is sensitive to type
    # Separate analyzed features into numerical and categorical
    # As to apply preprocessing only to valid features
    numerical_features = X_train.select_dtypes(include=['int64', 'float64']).columns
    categorical_features = X_train.select_dtypes(include=['object', 'bool', 'string']).columns

    # ColumnTransformer applies preprocessing patterns e.g. MinMaxScaler() and
    # OneHotEncoder() to groups, e.g. numerical_features and categorical_features
    # Preprocessor will be applied to every dataset
    preprocessor = ColumnTransformer([
        ("num", MinMaxScaler(), numerical_features),
        ("cat", OneHotEncoder(handle_unknown='infrequent_if_exist'), categorical_features)
    ])
    # Select model and configuration for use in the pipeline
    match method:
        case "knn":
            model = KNeighborsRegressor()
        case "random_forest":
            model = RandomForestRegressor(random_state=random_state)
        case "linear_regression":
            model = LinearRegression()
        case "neural_networks":
            model = MLPRegressor(max_iter=2500, random_state=random_state)
        case "svm":
            model = SVR()
    # Pipeline applies the preprocessed dataset to the model for fitting
    pipe = Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("regressor", model)
    ])

    # Fitting the pipeline
    pipe.fit(X_train, y_train)

    # Prediction and evaluation
    model_prediction = pipe.predict(X_val)

    return mean_squared_error(y_val, model_prediction)

def optimize_hyperparameters(method, X_train, y_train, kfolds, random_state):
    # Preprocessing is sensitive to type
    # Separate analyzed features into numerical and categorical
    # As to apply preprocessing only to valid features
    numerical_features = X_train.select_dtypes(include=['int64', 'float64']).columns
    categorical_features = X_train.select_dtypes(include=['object', 'bool', 'string']).columns

    # ColumnTransformer applies preprocessing patterns e.g. MinMaxScaler() and
    # OneHotEncoder() to groups, e.g. numerical_features and categorical_features
    # Preprocessor will be applied to every dataset
    preprocessor = ColumnTransformer([
        ("num", MinMaxScaler(), numerical_features),
        ("cat", OneHotEncoder(handle_unknown='infrequent_if_exist'), categorical_features)
    ])
    # Select model and configuration for use in the pipeline
    match method:
        case "random_forest":
            hyperparameters = {
                "regressor__n_estimators": np.arange(300, 500, 25),
                "regressor__ccp_alpha": np.arange(0, 0.4, 0.025),
            }
            model = RandomForestRegressor(random_state=random_state)
        case "neural_networks":
            hyperparameters = {
                "regressor__solver": ['lbfgs', 'adam'],
                "regressor__hidden_layer_sizes": [(25,), (50,), (100,), (25, 50, 25), (50, 50, 25), (50, 100),
                                                  (50, 50), (100, 100), (50, 100, 50), (100, 100, 100)],
            }
            model = MLPRegressor(max_iter=2500, random_state=random_state)
    
    # Pipeline applies the preprocessed dataset to the model for fitting
    pipe = Pipeline([
        ("preprocessor", preprocessor),
        ("regressor", model)
    ])

    # Parameters optimization
    clf = GridSearchCV(pipe, hyperparameters, scoring='neg_mean_squared_error', cv=kfolds, verbose=1)

    # Fitting the pipeline
    clf.fit(X_train, y_train)

    return clf.best_params_, clf.best_score_

def final_training(X_opt, y_opt, model):

    numerical_features = X_opt.select_dtypes(include=['int64', 'float64']).columns
    categorical_features = X_opt.select_dtypes(include=['object', 'bool', 'string']).columns

    preprocessor = ColumnTransformer([
        ("num", MinMaxScaler(), numerical_features),
        ("cat", OneHotEncoder(handle_unknown='infrequent_if_exist'), categorical_features)
    ])

    modelPipe = Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("regressor", model)
    ])

    # Fitting the pipeline
    modelPipe.fit(X_opt, y_opt)

    return modelPipe