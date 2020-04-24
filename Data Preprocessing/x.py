#importing necessary datasets
import pandas as pd
import numpy as np

#reading the datasets
dataset = pd.read_csv("Data.csv")
X = dataset.iloc[ :, : -1].values
y = dataset.iloc[ :,3]

#importing the imputer class
from sklearn.impute import SimpleImputer 

#replacing missing values with mean
X_mean = np.array(X)
imputer = SimpleImputer(missing_values=np.nan, strategy='mean')
X_mean[:,1:] = imputer.fit_transform(X_mean[:,1:])

#replacing missing values with medianm
X_median = np.array(X)
imp_median = SimpleImputer(missing_values=np.nan, strategy = "median")
X_median[:,1:] = imp_median.fit_transform(X_median[:, 1:])


#replacing missing values with mode
X_mode = np.array(X)
imp_mode = SimpleImputer(missing_values=np.nan, strategy = "most_frequent")
X_mode[:,1:] = imp_mode.fit_transform(X_mode[:,1:])


#replacing missing values with 0
X_cons = np.array(X)
imp_mode = SimpleImputer(missing_values=np.nan, strategy = "constant", fill_value=0)
X_cons[:,1:] = imp_mode.fit_transform(X_cons[:,1:])




#encoding Categorical Data
X = np.array(X_mean)
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer
labelencoder_x = LabelEncoder()
X[:, 0] = labelencoder_x.fit_transform(X[:, 0])

labelencoder_y = LabelEncoder()
y = labelencoder_y.fit_transform(y)

ct = ColumnTransformer([("encoder",OneHotEncoder(),[0])],remainder='passthrough')
X = ct.fit_transform(X)

#Removing The First Column to avoid Dummy Variable Trap
X = X[:,1:]


#Splitting The Data Into Train and Test Datasets
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 0)


# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)