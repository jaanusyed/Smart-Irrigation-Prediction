from importlib.resources import path
import pandas as pd
import numpy as np
from sklearn import metrics
from sklearn.metrics import classification_report
from sklearn import tree
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import pickle


df = pd.read_csv('datasets.csv')

Components = ['CropType', 'CropDays',
              'SoilMoisture', 'temperature', 'Humidity']
Features = df[Components]
Target = df['Irrigation']


Xtrain, Xtest, Ytrain, Ytest = train_test_split(
    Features, Target, test_size=0.2, random_state=2)
scaler = MinMaxScaler()
Xtrain = scaler.fit_transform(Xtrain)
Xtest = scaler.transform(Xtest)


RF = RandomForestClassifier(n_estimators=20, random_state=0)
RF.fit(Xtrain, Ytrain)

predicted_values = RF.predict(Xtest)


# Dump the trained Naive Bayes classifier with Pickle
RF_pkl_filename = 'RandomForest.pkl'
# Open the file to save as pkl file
RF_Model_pkl = open(RF_pkl_filename, 'wb')
pickle.dump(RF, RF_Model_pkl)
# Close the pickle instances
RF_Model_pkl.close()

model = pickle.load(open(RF_pkl_filename, 'rb'))
data = np.array([[2, 3, 189, 24, 50]])
prediction = model.predict(data)
print(prediction)
