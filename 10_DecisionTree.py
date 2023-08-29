import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn import tree

df = pd.read_csv('salaries.csv')
features = df[['company', 'job', 'degree']]
target = df['salary_more_then_100k']
encoderCompany = LabelEncoder()
encoderJob = LabelEncoder()
encoderDegree = LabelEncoder()
features['companyEncoded'] = encoderCompany.fit_transform(features['company'])
features['jobEncoded'] = encoderJob.fit_transform(features['job'])
features['degreeEncoded'] = encoderDegree.fit_transform(features['degree'])
encodedFeaturesDf = features.drop(['company', 'job', 'degree'], axis='columns')
model = tree.DecisionTreeClassifier()
model.fit(encodedFeaturesDf, target)

prediction = model.predict([[2, 0, 1]])
print(prediction)