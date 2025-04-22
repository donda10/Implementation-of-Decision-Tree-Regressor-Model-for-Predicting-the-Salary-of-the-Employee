# Implementation-of-Decision-Tree-Regressor-Model-for-Predicting-the-Salary-of-the-Employee

## AIM:
To write a program to implement the Decision Tree Regressor Model for Predicting the Salary of the Employee.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Import and preprocess the dataset: handle missing values and encode categorical data.
2. Split the dataset into training and testing sets.
3. Train a Decision Tree Regressor model using the training data.
4. Evaluate the model with predictions and compute the R² score for accuracy.

## Program:
```
/*
Program to implement the Decision Tree Regressor Model for Predicting the Salary of the Employee.
Developed by: Simon Malachi S
RegisterNumber: 212224040318 
*/
import pandas as pd
data=pd.read_csv("Salary.csv")
data.head()
data.info()
data.isnull().sum()
from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
data["Position"]=le.fit_transform(data["Position"])
data.head()
x=data[["Position","Level"]]
x.head()
y=data["Salary"]
y.head()
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=2)
from sklearn.tree import DecisionTreeRegressor
dt=DecisionTreeRegressor()
dt.fit(x_train,y_train)
y_pred=dt.predict(x_test)
print("Y_Prediction: ",y_pred)
from sklearn.metrics import r2_score
r2=r2_score(y_test,y_pred)
print("R2_Score:",r2)
dt.predict([[5,6]])
```

## Output:

![Screenshot 2025-04-23 000606](https://github.com/user-attachments/assets/6fe053af-9087-46f0-8b47-d125a1bbe508)
![Screenshot 2025-04-23 000602](https://github.com/user-attachments/assets/fd1d5f65-ed06-4358-b58e-688d6aa37c0d)


## Result:
Thus the program to implement the Decision Tree Regressor Model for Predicting the Salary of the Employee is written and verified using python programming.
