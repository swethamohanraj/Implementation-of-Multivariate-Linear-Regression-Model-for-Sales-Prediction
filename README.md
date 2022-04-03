Implementation-of-Multivariate-Linear-Regression-Model-for-Sales-Prediction
AIM:
To write a program to implement the multivariate linear regression model for sales prediction.

EQUIPMENTS REQUIRED:
Hardware – PCs
Anaconda – Python 3.7 Installation / Moodle-Code Runner
ALGORITHM:
Import the required standard libraries.
Upload the dataset
Declare X and Y variable with respect to the dataset.
Predict the values.
Find the required line that fits the data using Mean Square Error(MSE).
Print the Mean Square Error(MSE) and R square error.
End the program.
PROGRAM:

Program to implement the multivariate linear regression model for sales prediction.
Developed by:K.M.SWETHA
RegisterNumber: 212221240055

import pandas as pd
import matplotlib.pyplot as plt
df = pd.read_csv("Advertising.csv")
df.head()
df.describe()
df.isnull().sum()
df.shape
X = df[["TV", "Radio", "Newspaper"]]
X
Y = df["Sales"]
Y
from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test = train_test_split(X,Y,test_size = 0.2,random_state = 101)
from sklearn.linear_model import LinearRegression
l = LinearRegression()
l.fit(X_train,Y_train)
Y_pred = l.predict(X_test)
X_test
print("Regression slope: ",l.coef_[0])
print("Regression Intercept: ",l.intercept_)
Y_pred
from sklearn import metrics
MSE = metrics.mean_squared_error(Y_test,Y_pred)
print("MSE is {}".format(MSE))
r2 = metrics.r2_score(Y_test,Y_pred)
print("R squared error is {}".format(r2))
l.predict([[150.3,240.5,234.5]])

OUTPUT:
![image](https://user-images.githubusercontent.com/94228215/161429522-eafe1178-47c3-4090-bcec-79009f2d2f8f.png)



## Result:
Thus the program to implement the multivariate linear regression model for sales prediction is written and verified using python programming.
