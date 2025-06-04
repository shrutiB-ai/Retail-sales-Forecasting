from sklearn.linear_model import LinearRegression
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error,mean_squared_error,r2_score
import numpy as np

def train_linear(x_train, y_train):
    model = LinearRegression()
    model.fit(x_train, y_train)
    return model

def train_xgboost(x_train, y_train):
    model = XGBRegressor(objective = 'reg:squarederror', random_state=42)
    model.fit(x_train, y_train)
    return model

def evaluate_model(model , x_test, y_test):
    predictions = model.predict(x_test)
    mae = mean_absolute_error(y_test, predictions)
    rmse = np.sqrt(mean_squared_error(y_test,predictions))
    r2 = r2_score(y_test, predictions)
    return mae,rmse,r2