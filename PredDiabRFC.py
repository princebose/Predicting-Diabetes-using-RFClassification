import numpy as np
import time
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import Imputer
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics
"""
Here we import the database into a datadrame using Pandas

"""
df = pd.read_csv("Data/pima-database.csv")
"""

In order to see the levels of correlation between the data we use matrix show feature of matplotlib

"""
def plot_correlation(df,size=11): 
    corr = df.corr() # calling the correlation function
    figure, axis = plt.subplots(figsize=(size,size))
    axis.matshow(corr) # color code the rectangles 
    plt.xticks(range(len(corr.columns)),corr.columns) # draw x ticks
    plt.yticks(range(len(corr.columns)),corr.columns) # draw y ticks

"""

In order to start analysis on the data we need to see how many observations are available, how many are true (detected positive) and false (detected negetive),
We map the feature names as per the database

"""	

num_observations = len(df)
numtrue = len(df.loc[df['Outcome'] == 1])
numfalse = len(df.loc[df['Outcome'] == 0])
print("Number of True cases:  {0} ({1:2.2f}%)".format(numtrue, (numtrue/num_observations) * 100))
print("Number of False cases: {0} ({1:2.2f}%)".format(numfalse, (numfalse/num_observations) * 100))

feature_names = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age']
predicted_class = ['Outcome']

X = df[feature_names].values # these are factors for the prediction
y = df[predicted_class].values # this is what we want to predict


#DATA SPLITTING HAPPENS HERE - Explained in the report

split_test_size = 0.4

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size = split_test_size,random_state=42)
"""
Uncomment the following lines to see details and analysis of the original dataset

"""
#print("{0:0.2f}% in training set".format((len(X_train)/len(df.index)) * 100))
#print("{0:0.2f}% in test set".format((len(X_test)/len(df.index)) * 100))
#
#print("Original True  : {0} ({1:0.2f}%)".format(len(df.loc[df['Outcome'] == 1]), (len(df.loc[df['Outcome'] == 1])/len(df.index)) * 100.0))
#print("Original False : {0} ({1:0.2f}%)".format(len(df.loc[df['Outcome'] == 0]), (len(df.loc[df['Outcome'] == 0])/len(df.index)) * 100.0))
#print("")
#print("Training True  : {0} ({1:0.2f}%)".format(len(y_train[y_train[:] == 1]), (len(y_train[y_train[:] == 1])/len(y_train) * 100.0)))
#print("Training False : {0} ({1:0.2f}%)".format(len(y_train[y_train[:] == 0]), (len(y_train[y_train[:] == 0])/len(y_train) * 100.0)))
#print("")
#print("Test True      : {0} ({1:0.2f}%)".format(len(y_test[y_test[:] == 1]), (len(y_test[y_test[:] == 1])/len(y_test) * 100.0)))
#print("Test False     : {0} ({1:0.2f}%)".format(len(y_test[y_test[:] == 0]), (len(y_test[y_test[:] == 0])/len(y_test) * 100.0)))
#print("# rows in dataframe {0}".format(len(df)))
#print("# rows missing glucose_conc: {0}".format(len(df.loc[df['Glucose'] == 0])))
#print("# rows missing diastolic_bp: {0}".format(len(df.loc[df['BloodPressure'] == 0])))
#print("# rows missing thickness: {0}".format(len(df.loc[df['SkinThickness'] == 0])))
#print("# rows missing insulin: {0}".format(len(df.loc[df['Insulin'] == 0])))
#print("# rows missing bmi: {0}".format(len(df.loc[df['BMI'] == 0])))
#print("# rows missing diab_pred: {0}".format(len(df.loc[df['DiabetesPedigreeFunction'] == 0])))
#print("# rows missing age: {0}".format(len(df.loc[df['Age'] == 0])))
#def run_algo(X_train,X_test):

"""

This is where we Impute the missing values, using the mean method

"""


fill_missing_values = Imputer(missing_values=0,strategy="mean",axis=0)

X_train= fill_missing_values.fit_transform(X_train)
X_test = fill_missing_values.fit_transform(X_test)
CR = []


"""

The following lines were used to generate data for analysis

"""
#for randomn in range(1,101,1):
#print("\n\n random_seed = ",randomn,"\n")
#ACC_EST = []

""" 

In the code below, we try to measure the effect of change of N_Estimators on the creadtion time and the prediction time.


MCT_Time_vary_EST = []
PT_Time_vary_EST = []

for n_est in range(1,101,1):
    
    rand_forest_model = RandomForestClassifier(random_state = 42, n_estimators = n_est)
    start = time.time()
    rand_forest_model.fit(X_train,y_train.ravel())
    end = time.time()
    MCT_Time_vary_EST.append(end-start)
    
#    rf_predict_train = rand_forest_model.predict(X_train)
    start = time.time()
    rf_predict_test = rand_forest_model.predict(X_test)
    end = time.time()
    PT_Time_vary_EST.append(end-start)

print("Model Creation \n"+ str(MCT_Time_vary_EST ))
print("Prediction \n"+ str(PT_Time_vary_EST ))

"""

###MODEL CREATION
rand_forest_model = RandomForestClassifier(random_state = 42, n_estimators = n_est)
rand_forest_model.fit(X_train,y_train.ravel())
###PREDICTION ON TRAINING DATA
rf_predict_train = rand_forest_model.predict(X_train)
###PREDICTION ON TESTING DATA
rf_predict_test = rand_forest_model.predict(X_test)


#ACC_EST.append(metrics.accuracy_score(y_train,rf_predict_train))
#print(ACC_EST)
print("Accuracy on Training Data: {0:.4f}".format(metrics.accuracy_score(y_train,rf_predict_train)))
print()

#pred_start = time.time()
#rf_predict_test = rand_forest_model.predict(X_test)
#pred_end = time.time()

print("Accuracy on Testing Data: {0:.4f}".format(metrics.accuracy_score(y_test,rf_predict_test)))
#accuracyDB.append(metrics.accuracy_score(y_test,rf_predict_test))
print()

"""

Below we generate the classification report. This gives us an insight on the Precision, Recall and the Average accuracy of the model.

"""


print("Classification Report")
CR.append(metrics.classification_report(y_test, rf_predict_test))
print(metrics.classification_report(y_test, rf_predict_test))
#    return accuracyDB
#DB = run_algo(X_train,X_test)

#
#print("Time taken for RF Model Creation and Training for split size = "+ str(split_test_size) + " is "+str(mct_end-mct_start) + " s")
#print("Time taken for RF Model Prediction for split size = "+ str(split_test_size) + " is "+str(pred_end-pred_start) + " s")