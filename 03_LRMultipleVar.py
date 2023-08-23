import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import linear_model
import math

df = pd.read_csv('homepricesC.csv')
median_bedrooms = math.floor(df['bedrooms'].median())
df['bedrooms'] = df['bedrooms'].fillna(median_bedrooms)
reg = linear_model.LinearRegression()
reg.fit(df[['area', 'bedrooms', 'age']], df['price'])
prediction = reg.predict([[3000, 3, 2]])
print(prediction)