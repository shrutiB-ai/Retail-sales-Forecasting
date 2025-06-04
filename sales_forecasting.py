import random
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import yaml

from src.preprocessing import load_data, preprocess_data
from src.model import train_linear, train_xgboost, evaluate_model
from src.visualization import plot_predictions
from sklearn.model_selection import train_test_split

with open('config.yaml','r') as f:
	config = yaml.safe_load(f)

data_path = config['data']['path']
test_size = config['train_test_split']['test_size']
random_state = config['train_test_split']['random_state']
plot_path = config['output']['plot_path']

sns.set(style = 'whitegrid')

# Load Data
try:
	df = load_data('../data/train.csv')
except FileNotFoundError:
	print('Data not Found. Using Dummy data instead')
	df= pd.DataFrame({
		'Store':[1,2,3,4,5,6,7,8,9,10],
		'Dept':[1,1,1,1,2,2,2,3,3,3],
		'Date':pd.date_range(start ='2022-01-01',periods=10,freq ='W'),
		'Weekly_sales':[20000,30000,25000,22000,21000,19000,23000,27000,29000,21000],
		'IsHoliday':[False,True,False,False,False,False,False,False,False,False]
	})

# Preprocess
df_clean = preprocess_data(df)
X= df_clean.drop_(columns=['Weekly_sales'])
Y= df_clean['Weekly_sales']

x_train,x_test,y_train,y_test = train_test_split(
	X , Y, test_size=0.2, random_state=42)

#Train Linear Regression
lin_model = train_linear(x_train,y_train)
mae_lin,rmse_lin,r2_lin = evaluate_model(lin_model,x_test,y_test)

print("Linear Regression Results")
print(f"MAE:{mae_lin:.2f} | RMSE : {rmse_lin:.2f} | R2 score {r2_lin:.2f}")

# train XGBoost

xgb_model = train_xgboost(x_train,y_train)
mae_xgb,rmse_xgb,r2_xgb = evaluate_model(xgb_model,x_test,y_test)

print("XGBoost Results")
print(f"MAE:{mae_xgb:.2f} | RMSE : {rmse_xgb:.2f} | R2 score {r2_xgb:.2f}")


plot_predictions(y_test,y_pred_lin,y_pred_xgb,save_path=plot_path)