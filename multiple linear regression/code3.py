# the provided dataset is for 50 startups, and an ML model is required to select which startup makes the 
# most sense to invest in basde on profitability { dependet variable } based on factors stated in 
# dataset { independent variables }


import pandas as pd

dataset = pd.read_csv("dataset_for_multiple_regression.csv")
X_train = dataset.iloc[:, :-1]
Y_train = dataset.iloc[:, -1]
