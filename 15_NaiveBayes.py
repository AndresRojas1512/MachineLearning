import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB

df = pd.read_csv('titanic.csv')
df.drop(['Name', 'PassengerId', 'SibSp', 'Parch', 'Ticket', 'Cabin', 'Embarked'], axis='columns', inplace=True)
target = df['Survived']
features = df.drop('Survived', axis='columns')
dummies = pd.get_dummies(features['Sex'])
features = pd.concat([features, dummies], axis='columns')
features.drop('Sex', axis='columns', inplace=True)
features.columns[features.isna().any()]
features['Age'] = features['Age'].fillna(features['Age'].mean())
X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2)
model = GaussianNB()
model.fit(X_train, y_train)
score = model.score(X_test, y_test)
predictions = model.predict(X_test[:10])
probability = model.predict_proba(X_test[:10])
print(predictions)