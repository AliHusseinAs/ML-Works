import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

dataset = pd.read_csv("salary_simple_linear_dataset.csv")
x = dataset.iloc[:, :-1].values # years of experience or independent vars the data we give to the model
y = dataset.iloc[:, -1].values # salaries or dependent vars that we want to predict 
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)
regressor = LinearRegression()
regressor.fit(x_train, y_train)
y_pred = regressor.predict([[5]])
# plt.scatter(x_test, y_test, color="red")
# plt.plot(x_train, regressor.predict(x_train), color="blue")
# plt.title("salar v experience (Training set)")
# plt.xlabel("years of experience")
# plt.ylabel("annual salary")
# plt.show()
print(y_pred)



