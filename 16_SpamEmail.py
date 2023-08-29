import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline

df = pd.read_csv('spam.csv')
df['spam'] = df['Category'].apply(lambda x: 1 if x == 'spam' else 0)
df.drop('Category', axis='columns', inplace=True)
X = df['Message']
y = df['spam']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25)

CVobject = CountVectorizer()
X_train_Matrix = CVobject.fit_transform(X_train.values) # Create the matrix
# Create the model
model = MultinomialNB()
model.fit(X_train_Matrix, y_train)
emails = [
    'Hey Mohan, can we get together to watch the footbal game tomorrow',
    'Buy the New OFFER!'
]
emails_Matrix = CVobject.transform(emails) # Create the matrix fpr test
prediction = model.predict(emails_Matrix)
X_test_Matrix = CVobject.fit_transform(X_test)
# score = model.score(X_test_Matrix, y_test)

# Creating sklearn pipeline
clf = Pipeline([
    ('vectorizer', CountVectorizer()), # Convert the text
    ('nb', MultinomialNB()) #Creating the model
])

clf.fit(X_train, y_train)
score = clf.score(X_test, y_test)
# print(type(score))
print(clf.predict(y_test))