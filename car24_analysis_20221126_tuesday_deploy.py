# -*- coding: utf-8 -*-
# Web Scraping from cars24.com

import requests
import pandas as pd
import streamlit as st
import pandas as pd

url = 'https://github.com/uma0shubh/cars24_data/blob/main/cars24_20221126.csv?raw=true'
df = pd.read_csv(url)
df.head()

duplicate = df[df.duplicated()]
df1 = df.loc[df['city'].isin(['New Delhi', 'Mumbai', 'Jaipur', 'Chennai', 'Lucknow', 'Bangalore', 'Indore', 'Hyderabad', 'Kochi', 'Pune', 'Kolkata', 'Ahmedabad', 'Gurgaon', 'Noida', 'Ghaziabad'])]

import seaborn as sns
sns.set()
import matplotlib.pyplot as plt
import numpy as np
import sklearn as skt
import plotly
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
# preprocessing
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV, StratifiedKFold
import pandas_profiling as pp
# models
from sklearn.linear_model import LinearRegression, SGDRegressor, RidgeCV
from sklearn.svm import SVR, LinearSVR
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, ExtraTreesRegressor 
from sklearn.ensemble import BaggingRegressor, AdaBoostRegressor, VotingRegressor 
from sklearn.neural_network import MLPRegressor
from sklearn.tree import DecisionTreeRegressor
import sklearn.model_selection
from sklearn.model_selection import cross_val_predict as cvp
from sklearn import metrics
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import xgboost as xgb
import lightgbm as lgb
# model tuning
from hyperopt import STATUS_OK, Trials, fmin, hp, tpe, space_eval
import warnings
warnings.filterwarnings("ignore")


# Data Pre-processing
### Check for missing values & treatment
sns.heatmap(df1.isnull(),cbar=False,cmap='viridis')
df1.dropna(inplace=True)
sns.heatmap(df1.isnull(),cbar=False,cmap='viridis')
df1.reset_index(inplace=True)
df1.info()
df1.drop(["index"],axis=1,inplace=True)

# Descriptive statistics"""
df1.describe(include = 'all')

"""# Data Visualization"""
"""### Univariant plots"""

# Price
fig21 = plt.figure(figsize=(15,8))
sns.distplot(df1['price'])
print("skewness: %f" % df1['price'].skew())
print("kurtosis: %f" % df1['price'].kurt())
st.write(fig21)

fig22 = plt.figure(figsize=(15,4))
sns.boxplot(x='price',data=df1)
st.write(fig22)

plt.subplot(231)
df1['year'].value_counts().plot(kind='bar', title='Year',figsize=(30,15))
plt.xticks(rotation=0)

plt.subplot(232)
df1['fueltype'].value_counts().plot(kind='pie', title='Fuel type')
plt.xticks(rotation=0)

plt.subplot(233)
df1['ownernumber'].value_counts().plot(kind='pie', title='ownernumber')
plt.xticks(rotation=0)

plt.subplot(234)
df1['bodytype'].value_counts().plot(kind='pie', title='bodytype')
plt.xticks(rotation=0)

plt.subplot(235)
df1['transmission'].value_counts().plot(kind='pie', title='transmission')
plt.xticks(rotation=0)

plt.subplot(236)
df1['registrationstate'].value_counts().plot(kind='bar', title='registrationstate')
plt.xticks(rotation=90)

plt.show()

plt.subplot()
df1['city'].value_counts().plot(kind='bar', title='City name', figsize=(15,4))
plt.xticks(rotation=0)
plt.show()

plt.subplot()
df1['make'].value_counts().plot(kind='bar', title='Make',figsize=(15,4))
plt.xticks(rotation=90)
plt.show()

plt.subplot()
df1['model'].value_counts().plot(kind='bar', title='model',figsize=(24,9))
plt.xticks(rotation=90)
plt.show()

"""
### Bi-variant plots
"""

# Fuel Type
f, ax = plt.subplots(figsize=(15, 8))
fig31 = sns.boxplot(x="fueltype", y="price", data=df1)
st.write(fig31)

# Year
f, ax = plt.subplots(figsize=(15, 5))
fig = sns.boxplot(x="year", y="price", data=df1)
fig;
plt.xticks(rotation=90);

# Owner_Type
fig, ax = plt.subplots()
fig
sns.stripplot(x = "ownernumber", y ='price', data = df1)

# City vs Price
sns.catplot(y='price',x="city",data=df1.sort_values('price',ascending=False),kind="boxen",height=5, aspect=3)
plt.show

# Transmission vs Price
sns.catplot(y='price',x="transmission",data=df1.sort_values('price',ascending=False),kind="boxen",height=5, aspect=2)
plt.show

# Make vs Price
sns.catplot(y='price',x="make",data=df1.sort_values('price',ascending=False),kind="boxen",height=5, aspect=3)
plt.show

fig3 = px.sunburst(df1, path=['city', 'fueltype'], color='city',height=800)
fig3.update_layout(title_text="Fuel Type for City (Two-level Sunburst Diagram)", font_size=10)
st.write(fig3)

fig4 = px.treemap(df1, path=['city', 'make'], color='city',height=800,width=1500)
fig4.update_layout(title_text="Manufacture Distribution for City", font_size=10)
st.write(fig4)

fig5 = px.histogram(df1, x="year", y="price",color='city', barmode='group',height=400,width=1500)
fig.update_layout(title_text="Yearly City Growth", font_size=10)
st.write(fig5)


# Label Encoding
df7 = df1.copy(deep=True)

numerics = ['int8', 'int16', 'int32', 'int64', 'float16', 'float32', 'float64']
categorical_columns = []
features = df7.columns.values.tolist()
for col in features:
    if df7[col].dtype in numerics: continue
    categorical_columns.append(col)
# Encoding categorical features
for col in categorical_columns:
    if col in df7.columns:
        le = LabelEncoder()
        le.fit(list(df7[col].astype(str).values))
        df7[col] = le.transform(list(df7[col].astype(str).values))

df7['year'] = (df7['year']-1900).astype(int)
df8 = df7.drop(['name','model', 'storename','isc24assured','registrationcity','discountprice','url','registrationstate','benefits','createdDate'], axis = 1)
df8.corr()

plt.figure(figsize=(15,10))
sns.heatmap(df8.corr(),annot=True,cmap='RdYlGn')
plt.show()

# Train-Test split
target_name = 'price'
train_target0 = df8[target_name]
df8 = df8.drop([target_name], axis=1)

# Train and Test Split
train0, test0, train_target0, test_target0 = train_test_split(df8, train_target0, test_size=0.2, random_state=0)

valid_part = 0.3
pd.set_option('max_columns',100)

# For boosting model
train0b = train0
train_target0b = train_target0
# Synthesis valid as test for selection models
trainb, testb, targetb, target_testb = train_test_split(train0b, train_target0b, test_size=valid_part, random_state=0)

# For models from Sklearn
scaler = StandardScaler()
train0 = pd.DataFrame(scaler.fit_transform(train0), columns = train0.columns)

# getting test from train data (validation)
train, test, target, target_test = train_test_split(train0, train_target0, test_size=valid_part, random_state=0)

# Accuracy List
acc_train_r2 = []
acc_test_r2 = []
acc_train_d = []
acc_test_d = []
acc_train_rmse = []
acc_test_rmse = []

def acc_d(y_meas, y_pred):
    # Relative error between predicted y_pred and measured y_meas values (relative error also known as % error)
    return mean_absolute_error(y_meas, y_pred)*len(y_meas)/sum(abs(y_meas))

def acc_rmse(y_meas, y_pred):
    # RMSE between predicted y_pred and measured y_meas values
    return (mean_squared_error(y_meas, y_pred))**0.5

def acc_boosting_model(num,model,train,test,num_iteration=0):
    # Calculation of accuracy of boosting model by different metrics
    
    global acc_train_r2, acc_test_r2, acc_train_d, acc_test_d, acc_train_rmse, acc_test_rmse
    
    if num_iteration > 0:
        ytrain = model.predict(train, num_iteration = num_iteration)  
        ytest = model.predict(test, num_iteration = num_iteration)
    else:
        ytrain = model.predict(train)  
        ytest = model.predict(test)

    print('target = ', targetb[:5].values)
    print('ytrain = ', ytrain[:5])

    acc_train_r2_num = round(r2_score(targetb, ytrain) * 100, 2)
    print('acc(r2_score) for train =', acc_train_r2_num)   
    acc_train_r2.insert(num, acc_train_r2_num)

    acc_train_d_num = round(acc_d(targetb, ytrain) * 100, 2)
    print('acc(relative error) for train =', acc_train_d_num)   
    acc_train_d.insert(num, acc_train_d_num)

    acc_train_rmse_num = round(acc_rmse(targetb, ytrain) * 100, 2)
    print('acc(rmse) for train =', acc_train_rmse_num)   
    acc_train_rmse.insert(num, acc_train_rmse_num)

    print('target_test =', target_testb[:5].values)
    print('ytest =', ytest[:5])
    
    acc_test_r2_num = round(r2_score(target_testb, ytest) * 100, 2)
    print('acc(r2_score) for test =', acc_test_r2_num)
    acc_test_r2.insert(num, acc_test_r2_num)
    
    acc_test_d_num = round(acc_d(target_testb, ytest) * 100, 2)
    print('acc(relative error) for test =', acc_test_d_num)
    acc_test_d.insert(num, acc_test_d_num)
    
    acc_test_rmse_num = round(acc_rmse(target_testb, ytest) * 100, 2)
    print('acc(rmse) for test =', acc_test_rmse_num)
    acc_test_rmse.insert(num, acc_test_rmse_num)

def acc_model(num,model,train,test):
    # Calculation of accuracy of model Sklearn by different metrics   
  
    global acc_train_r2, acc_test_r2, acc_train_d, acc_test_d, acc_train_rmse, acc_test_rmse
    
    ytrain = model.predict(train)  
    ytest = model.predict(test)

    print('target = ', target[:5].values)
    print('ytrain = ', ytrain[:5])

    acc_train_r2_num = round(r2_score(target, ytrain) * 100, 2)
    print('acc(r2_score) for train =', acc_train_r2_num)   
    acc_train_r2.insert(num, acc_train_r2_num)

    acc_train_d_num = round(acc_d(target, ytrain) * 100, 2)
    print('acc(relative error) for train =', acc_train_d_num)   
    acc_train_d.insert(num, acc_train_d_num)

    acc_train_rmse_num = round(acc_rmse(target, ytrain) * 100, 2)
    print('acc(rmse) for train =', acc_train_rmse_num)   
    acc_train_rmse.insert(num, acc_train_rmse_num)

    print('target_test =', target_test[:5].values)
    print('ytest =', ytest[:5])
    
    acc_test_r2_num = round(r2_score(target_test, ytest) * 100, 2)
    print('acc(r2_score) for test =', acc_test_r2_num)
    acc_test_r2.insert(num, acc_test_r2_num)
    
    acc_test_d_num = round(acc_d(target_test, ytest) * 100, 2)
    print('acc(relative error) for test =', acc_test_d_num)
    acc_test_d.insert(num, acc_test_d_num)
    
    acc_test_rmse_num = round(acc_rmse(target_test, ytest) * 100, 2)
    print('acc(rmse) for test =', acc_test_rmse_num)
    acc_test_rmse.insert(num, acc_test_rmse_num)

"""# Model Building

- Linear Regression
- Support vector machine 
- Decision Tree Regressor
- Ridge Regressor
- AdaBoost Regressor
- Voting Regressor
"""

linreg = LinearRegression()
linreg.fit(train, target)
acc_model(0,linreg,train,test)

# Support vector machine

svr = SVR()
svr.fit(train, target)
acc_model(1,svr,train,test)

# Decision Tree Regressor

decision_tree = DecisionTreeRegressor()
decision_tree.fit(train, target)
acc_model(5,decision_tree,train,test)

# Ridge Regressor

ridge = RidgeCV(cv=5)
ridge.fit(train, target)
acc_model(10,ridge,train,test)

# AdaBoost Regressor

Ada_Boost = AdaBoostRegressor()
Ada_Boost.fit(train, target)
acc_model(13,Ada_Boost,train,test)

# Voting Regressor

Voting_Reg = VotingRegressor(estimators=[('lin', linreg), ('ridge', ridge)])
Voting_Reg.fit(train, target)
acc_model(14,Voting_Reg,train,test)

"""# Models comparison"""

models = pd.DataFrame({
    'Model': ['Linear Regression', 'Support Vector Machines', 'Decision Tree Regressor', 
              'RidgeRegressor', 'AdaBoostRegressor', 'VotingRegressor'],
    
    'r2_train': acc_train_r2,
    'r2_test': acc_test_r2,
    'd_train': acc_train_d,
    'd_test': acc_test_d,
    'rmse_train': acc_train_rmse,
    'rmse_test': acc_test_rmse
                     })

pd.options.display.float_format = '{:,.2f}'.format

print('Prediction accuracy for models by R2 criterion - r2_test')
print11 = models.sort_values(by=['r2_test', 'r2_train'], ascending=False)
st.write(print11)

print('Prediction accuracy for models by RMSE - rmse_test')
print12 = models.sort_values(by=['rmse_test', 'rmse_train'], ascending=True)
st.write(print12)

"""# Model Output - Visualization"""

# Plot
fig11 = plt.figure(figsize=[15,6])
xx = models['Model']
plt.tick_params(labelsize=14)
plt.plot(xx, models['r2_train'], label = 'r2_train')
plt.plot(xx, models['r2_test'], label = 'r2_test')
plt.legend()
plt.title('R2-criterion for 15 popular models for train and test datasets')
plt.xlabel('Models')
plt.ylabel('R2-criterion, %')
plt.xticks(xx, rotation='vertical')
plt.savefig('graph.png')
st.write(fig11)

# Plot
fig12 = plt.figure(figsize=[15,6])
xx = models['Model']
plt.tick_params(labelsize=14)
plt.plot(xx, models['d_train'], label = 'd_train')
plt.plot(xx, models['d_test'], label = 'd_test')
plt.legend()
plt.title('Relative errors for 15 popular models for train and test datasets')
plt.xlabel('Models')
plt.ylabel('Relative error, %')
plt.xticks(xx, rotation='vertical')
plt.savefig('graph.png')
st.write(fig12)

# Plot
fig13 = plt.figure(figsize=[15,6])
xx = models['Model']
plt.tick_params(labelsize=14)
plt.plot(xx, models['rmse_train'], label = 'rmse_train')
plt.plot(xx, models['rmse_test'], label = 'rmse_test')
plt.legend()
plt.title('RMSE for 15 popular models for train and test datasets')
plt.xlabel('Models')
plt.ylabel('RMSE, %')
plt.xticks(xx, rotation='vertical')
plt.savefig('graph.png')
st.write(fig13)

"""
# Prediction
"""
#For models from Sklearn
testn = pd.DataFrame(scaler.transform(test0), columns = test0.columns)

#Ridge Regressor model for basic train
ridge.fit(train0, train_target0)
ridge.predict(testn)[:3]

#Ada_Boost  model for basic train
Ada_Boost.fit(train0, train_target0)
Ada_Boost.predict(testn)[:3]

#Voting Regressor model for basic train
Voting_Reg.fit(train0, train_target0)
Voting_Reg.predict(testn)[:3]

"""# Creating Dashboard"""

# import pywedge as pw

# mc = pw.Pywedge_Charts(df1, c = None, y = 'price')

# st.write(mc.make_charts())
