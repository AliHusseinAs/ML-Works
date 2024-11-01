#first lecture Section 2 

# import numpy as np
# import pandas as pd

# dataset = pd.read_csv("data.csv") 
# # feature var X 
# X = dataset.iloc[:, :-1]
# #iloc[row:column] 
# # in this code you can get all columns but the last one iloc[:, :-1]
# # dependent variable vector Y
# Y = dataset.iloc[:, -1] # will get the last column
# print(X)
# print("dependent vars")
# print(Y)

#TAKING CARE OF MISSING DATA

# from sklearn.impute import SimpleImputer
# import pandas as pd
# import numpy as np

# dataset = pd.read_csv("data.csv")
# X = dataset.iloc[:, :-1].values
# Y = dataset.iloc[:, -1].values
# Imputer = SimpleImputer(missing_values=np.nan, strategy='mean')
# Imputer.fit(X[:, 1:3])
# X[:, 1:3] = Imputer.transform(X[:, 1:3])
# print(X)


#//////////////////////////
# THIS PART IS IMPORTANT
#//////////////////////////


# ENCODING COLUMN
# THE CODE BELOW ENCODED THE COUNTRY COLUMN INTO BINARY NUMS THIS METHOD IS BEST 
# TO TRAIN THE MACHINE
# import pandas as pd
# import numpy as np
# from sklearn.compose import ColumnTransformer
# from sklearn.preprocessing import OneHotEncoder

# dataset = pd.read_csv("data.csv")
# X = dataset.iloc[:, :-1].values
# Y = dataset.iloc[:, -1].values

# ct = ColumnTransformer(transformers=[('encoder', OneHotEncoder(), [0])],
#                        remainder='passthrough')
# X = np.array(ct.fit_transform(X))


# Encoding for the dependent vars

# import pandas as pd
# import numpy as np
# from sklearn.preprocessing import LabelEncoder


# dataset = pd.read_csv("data.csv")
# X = dataset.iloc[:, :-1].values
# Y = dataset.iloc[:, -1].values
# lable = LabelEncoder()
# Y = lable.fit_transform(Y)
# print(Y)


# splitting the dataset into training and testing data

from sklearn.model_selection import train_test_split
import pandas as pd
from sklearn.impute import SimpleImputer
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import LabelEncoder
# feature scaling
from sklearn.preprocessing import StandardScaler

dataset = pd.read_csv("data.csv")
X = dataset.iloc[:, :-1].values
Y = dataset.iloc[:, -1].values
Imputer = SimpleImputer(missing_values=np.nan, strategy='mean')
Imputer.fit(X[:, 1:3])
X[:, 1:3] = Imputer.transform(X[:, 1:3])
ct = ColumnTransformer(transformers=[('encoder', OneHotEncoder(), [0])],
                       remainder='passthrough')
X = np.array(ct.fit_transform(X))
lable = LabelEncoder()
Y = lable.fit_transform(Y)
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.2,
                                                     random_state=1 )

sc = StandardScaler()
X_train[:, 3:] = sc.fit_transform(X_train[:, 3:]) # the [:, 3:] is to cound all columns but the dummy variables we added 
#since feature scaling will be applied to train data it must be applied to test data too
X_test[:, 3:] = sc.fit_transform(X_test[:, 3:])





