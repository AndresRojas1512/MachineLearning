import pandas as pd
from sklearn.datasets import load_iris
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
#SVM takes the largest margin
iris = load_iris()
# General DataFrame
df = pd.DataFrame(iris.data, columns=iris.feature_names)
df['target'] = iris.target
df['flowerName'] = df['target'].apply(lambda x : iris.target_names[x])
# Filtered DataFrames
df0 = df[df['target'] == 0]
df1 = df[df['target'] == 1]
df2 = df[df['target'] == 2]
# Plot
plt.xlabel('Sepal lenght')
plt.ylabel('Sepal width')
plt.scatter(df0['petal length (cm)'], df0['petal width (cm)'], color='green', marker='+')
plt.scatter(df1['petal length (cm)'], df1['petal width (cm)'], color='blue', marker='+')
# Data prep
X = df.drop(df[['flowerName', 'target']], axis='columns')
y = df['target']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
# Model
model = SVC()
model.fit(X_train, y_train)
predictions = model.predict(X_test)
# Printing predictions and checking
print(X_test)
for i in range(len(predictions)):
    print(predictions[i])

# setosa = 0
# versicolor = 1
# verginica = 2