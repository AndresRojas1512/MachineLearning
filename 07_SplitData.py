import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
df = pd.read_csv('carprices.csv')
# Price(Mile)
plt.scatter(df['Mileage'], df['Sell Price($)'])
plt.xlabel('Miles')
plt.ylabel('Price')
# Price(Age)
plt.scatter(df['Age(yrs)'], df['Sell Price($)'])
plt.xlabel('Age')
plt.ylabel('Price')

X = df[['Mileage', 'Age(yrs)']]
y = df['Sell Price($)']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=10)
model = LinearRegression()
model.fit(X_train, y_train)
prediction = model.predict(X_test)
# print(X_test)
# for i in range(len(prediction)):
#     print(prediction[i])

print(model.score(X_test, y_test))
