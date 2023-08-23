from sklearn import linear_model
import joblib
import pandas as pd
import pickle

df = pd.read_csv('homeprices.csv')
model = linear_model.LinearRegression()
model.fit(df[['area']], df['price'])

# with open('model_pickle', 'wb') as f:
#     pickle.dump(model, f)

# with open('model_pickle', 'rb') as f:
#     mp = pickle.load(f)

joblib.dump(model, 'model_joblib')
mj = joblib.load('model_joblib')
print(mj.predict([[5000]]))