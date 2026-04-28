# Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored

## AIM:
To write a program to predict the marks scored by a student using the simple linear regression model.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
Step 1: Start
Step 2: Import Required Libraries
Import libraries such as:
NumPy
Pandas
Matplotlib
Scikit-learn
Step 3: Load the Dataset
Read the dataset (e.g., CSV file) containing:
Independent variable (e.g., hours studied → X)
Dependent variable (marks scored → Y)
Step 4: Separate Input and Output Variables
Assign:
X = independent variable (hours)
Y = dependent variable (marks)
Step 5: Split the Dataset
Divide data into:
Training set (e.g., 80%)
Testing set (e.g., 20%)
Step 6: Create the Linear Regression Model
Initialize the regression model.
Step 7: Train the Model
Fit the model using training data:
Model learns relationship between X and Y
Step 8: Predict the Results
Use the trained model to predict marks for test data:
Y_pred = model.predict(X_test)
Step 9: Evaluate the Model
Calculate performance metrics:
Mean Squared Error (MSE)
R² Score
Step 10: Visualize the Results (Optional)
Plot:
Actual data points
Regression line
Step 11: Predict for New Input
Input new value (e.g., hours studied)
Use model to predict marks
Step 12: Display Output
Print predicted marks
Step 13: End
Key Idea Behind the Model

The regression line follows:

Y=mX+b
m → slope
b → intercept 

## Program:
```
/*
Program to implement the simple linear regression model for predicting the marks scored.
Developed by: Ramsanjay C
RegisterNumber:  212224220077
*/
```
```
import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error,mean_squared_error
df=pd.read_csv('student_scores.csv')
df.head()
df.tail()
X=df.iloc[:,:-1].values
print(*X)
Y=df.iloc[:,1].values
print(*Y)
from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=1/3,random_state=0)
from sklearn.linear_model import LinearRegression
regressor=LinearRegression()
regressor.fit(X_train,Y_train)
Y_pred=regressor.predict(X_test)
Y_pred
print(*Y_pred)
Y_test
print(*Y_test)
plt.scatter(X_train,Y_train,color="orange")
plt.plot(X_train,regressor.predict(X_train),color="red")
plt.title("Hours vs Scores(Training Set)")
plt.xlabel("Hours")
plt.ylabel("Scores")
plt.show()
plt.scatter(X_test, Y_test, color="blue")
plt.plot(X_test, regressor.predict(X_test), color="green")
plt.title('Testing set (Hours vs Scores)')
plt.xlabel("Hours")
plt.ylabel("Scores")
plt.show()

mae = mean_absolute_error(Y_test, Y_pred)
mse = mean_squared_error(Y_test, Y_pred)
rmse = np.sqrt(mse)
print("Mean Absolute Error:", mae)
print("Mean Squared Error:", mse)
print("Root Mean Squared Error:", rmse)
```
## Output:

<img width="970" height="1002" alt="Screenshot (6)" src="https://github.com/user-attachments/assets/35c6fb65-4493-4430-bdc2-3307d2263bb6" />

## Result:
Thus the program to implement the simple linear regression model for predicting the marks scored is written and verified using python programming.
