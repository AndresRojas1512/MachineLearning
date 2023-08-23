import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import linear_model

# plt.xlabel('area(ft^2)')
# plt.ylabel('price($)')
# plt.scatter(df.area, df.price, color='red', marker='+')
# # plt.show()

# predicted_price = reg.predict(np.array([[3300]]))
# print(predicted_price)
# print(reg.coef_)
# print(reg.intercept_)

df = pd.read_csv("homeprices.csv")
reg = linear_model.LinearRegression()
reg.fit(df[['area']], df['price'])
prediction = reg.predict(df[['area']])
plt.scatter(df['area'], df['price'], color='red', marker='+')
plt.plot(df['area'], prediction, color='blue')
plt.show()

# Fillin with new data once the model is trained
da = pd.read_csv('areas.csv')
predicted_prices = reg.predict(da[['area']])

if 'prices' not in da.columns:
    da['prices'] = predicted_prices
    da.to_csv('areas.csv', index=False)