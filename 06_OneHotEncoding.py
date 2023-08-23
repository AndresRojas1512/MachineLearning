import pandas as pd
from sklearn import linear_model
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
import numpy as np

# df = pd.read_csv('homepricesT.csv')
# dummies = pd.get_dummies(df['town'])
# merged = pd.concat([df, dummies], axis='columns')
# final = merged.drop(['town', 'west windsor'], axis='columns')
# Price is our dependt variable. The other data are features.
# x = final.drop('price', axis='columns')
# y = df['price']
# model.fit(x, y)
# prediction = model.predict([[3400, 0, 0]])
df = pd.read_csv('homepricesT.csv')
le = LabelEncoder()
df['town'] = le.fit_transform(df['town'])
X = df[['town', 'area']].values
y = df['price']
ohe = OneHotEncoder(categories='auto', drop='first')
X_ohe = ohe.fit_transform(X).tolist()