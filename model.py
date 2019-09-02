import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pickle

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor

np.random.seed(42)

train = pd.read_csv('Admission_Predict_Ver1.1.csv')

# convert to float type
train = train.astype(float)

# train-test 
x_train = train.drop(['Serial No.', 'Chance of Admit '], axis=1)
y_train = train['Chance of Admit ']

# train
rf = RandomForestRegressor(n_estimators=100, n_jobs=-1)
rf = rf.fit(x_train, y_train)

# save the model
pickle.dump(rf, open('model.pkl', 'wb'))

# load the model
model = pickle.load(open('model.pkl', 'rb'))