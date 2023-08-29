import pandas as pd
from sklearn.datasets import load_digits
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
# Ensemle : multiple algorithms

digits = load_digits()
direc = dir(digits)
df = pd.DataFrame(digits.data)
df['target'] = digits.target
X = df.drop(['target'], axis='columns')
y = df['target']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Model
model = RandomForestClassifier()
model.fit(X_train,y_train)
