from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.model_selection import cross_val_score
import numpy as np
import pandas as pd

# Cross Validation: Compare model
digits = load_digits()
direc = dir(digits)
df = pd.DataFrame(digits.data)
df['target'] = digits.target
X = df.drop(['target'], axis='columns')
y = df['target']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

# Models
logisticModel = LogisticRegression()
logisticModel.fit(X_train, y_train)
# print(logisticModel.score(X_test, y_test))

svcModel = SVC()
svcModel.fit(X_train, y_train)
# print(svcModel.score(X_test, y_test))

rfcModel = RandomForestClassifier()
rfcModel.fit(X_train, y_train)

# kf = KFold(n_splits=3)
# for train_intex, test_index in kf.split([1, 2, 3, 4, 5, 6, 7, 8, 9]):
#     print(train_intex, test_index)

def get_score(model, X_train, X_test, y_train, y_test):
    model.fir(X_train, y_train)
    return model.score(X_test, y_test)

folds = StratifiedGroupKFold(n_splits=3)
scoresLogReg= []
scoresSVM = []
scoresRF = []
for train_index, test_index in folds.split(digits.data):
    X_train, X_test, y_train, y_test = digits.data[train_index], digits.data[test_index], digits.target[train_index], digits.target[test_index]
    scoresLogReg.append(get_score(LogisticRegression(), X_train, X_test, y_train, y_test))
    scoresSVM.append(get_score(SVC(), X_train, X_test, y_train, y_test))
    scoresRF.append(get_score(RandomForestClassifier(), X_train, X_test, y_train, y_test))

print(cross_val_score(LogisticRegression(), digits.data, digits.target))