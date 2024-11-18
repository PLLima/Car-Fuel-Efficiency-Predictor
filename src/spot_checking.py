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

from sklearn.metrics import mean_squared_error
from sklearn.model_selection import KFold

import plots

def train(kfolds, X_train, y_train, X_train_mandatory, y_train_mandatory, method):
    metric_total = 0
    model_metrics = []  # List to store metrics for each split

    # Separate validation data and the remaining instances
    #X_train, X_val, y_train, y_val  = train_test_split(X, y, test_size=val_size)
    # Split the remaining instances in kfolds parts
    kf = KFold(n_splits=kfolds, shuffle=True, random_state=42)
    for train_index, validation_index in kf.split(X_train):
        X_train, X_val = X_train.iloc[train_index], X_train.iloc[validation_index]
        y_train, y_val = y_train.iloc[train_index], y_train.iloc[validation_index]
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
            ("num", StandardScaler(), numerical_features),
            ("cat", OneHotEncoder(), categorical_features)
        ])
        # Select model and configuration for use in the pipeline
        match method:
            case "knn":
                model = KNeighborsClassifier()
            case "random_forest":
                model = RandomForestRegressor()
            case "linear_regression":
                model = LinearRegression()
            case "neural_networks":
                model = MLPRegressor(max_iter=2500)
            case "svm":
                model = SVR()
        # Pipeline applies the preprocessed dataset to the model for fitting
        pipe = Pipeline([
            ("preprocessor", preprocessor),
            ("regressor", model)
        ])

        # Fitting the pipeline
        pipe.fit(X_train, y_train)

        # Prediction and evaluation
        y_pred = pipe.predict(X_val)

        metric = mean_squared_error(y_val, y_pred)
        model_metrics.append(metric)
        metric_total += metric

    return model_metrics, y_pred