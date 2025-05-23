# SGD-Regressor-for-Multivariate-Linear-Regression

## AIM:
To write a program to predict the price of the house and number of occupants in the house with SGD regressor.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
#### 1. Load California housing data, select features and targets, and split into training and testing sets.
#### 2. Scale both X (features) and Y (targets) using StandardScaler.
#### 3. Use SGDRegressor wrapped in MultiOutputRegressor to train on the scaled training data.
#### 4. Predict on test data, inverse transform the results, and calculate the mean squared error.


## Program:

```
import pandas as pd
import numpy as np 
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.multioutput import MultiOutputRegressor
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import SGDRegressor
dataset = fetch_california_housing()
df = pd.DataFrame(dataset.data,columns=dataset.feature_names)
df['HousingPrice'] = dataset.target
df.head()
df.info()
df.isnull().sum()
df.describe()
import seaborn as sns
correlation_matrix = df.corr()
_=sns.heatmap(correlation_matrix, annot=True,  fmt=".2f")
X = df.drop(columns=['AveOccup','HousingPrice'])
X
X.info()
y = df[['AveOccup','HousingPrice']]
y
y.info()
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2,random_state=42)
print('Shape of X_train : ',X_train.shape)
print('Shape of X_test : ',X_test.shape)
print('Shape of y_train : ',y_train.shape)
print('Shape of y_test : ',y_test.shape)
scaler_X = StandardScaler()
scaler_y = StandardScaler()
scaler_X
X_train = scaler_X.fit_transform(X_train)
X_test = scaler_X.transform(X_test)
y_train = scaler_y.fit_transform(y_train)
y_test = scaler_y.transform(y_test)
print(X_train)
print(y_train)
print(X_test)
print(y_test)
sgd = SGDRegressor(max_iter=1500,tol=1e-3)
multi_output_sgd = MultiOutputRegressor(sgd)
multi_output_sgd
multi_output_sgd.fit(X_train,y_train)
multi_output_sgd
y_pred = multi_output_sgd.predict(X_test)
y_pred
y_pred = scaler_y.inverse_transform(y_pred)
y_test = scaler_y.inverse_transform(y_test)
print('y_pred : \n', y_pred)
print('y_test : \n',y_test)
mse = mean_squared_error(y_test,y_pred)
print('Mean Squared Error : ',mse)
import matplotlib.pyplot as plt
_=plt.scatter(y_test[0],y_pred[0])

```

## Output:
![image](https://github.com/user-attachments/assets/4764de0b-c504-4c8f-a262-aaf74119c0fd)
![image](https://github.com/user-attachments/assets/9fc72129-1efb-472d-b63d-63fd315689c7)

![image](https://github.com/user-attachments/assets/fed8337d-fa34-4510-a4ce-325c9750c519)
![image](https://github.com/user-attachments/assets/7a969447-2e96-41d0-ad35-16dbbfd5af73)


## Result:
Thus the program to implement the multivariate linear regression model for predicting the price of the house and number of occupants in the house with SGD regressor is written and verified using python programming.
