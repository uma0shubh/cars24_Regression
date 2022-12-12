# Import Libraries *************************************************************
import seaborn as sns
sns.set()
import matplotlib.pyplot as plt
import numpy as np
import sklearn as skt
import plotly
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV, StratifiedKFold
import pandas_profiling as pp
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
from hyperopt import STATUS_OK, Trials, fmin, hp, tpe, space_eval
import warnings
warnings.filterwarnings("ignore")
import requests
import pandas as pd
import streamlit as st
from PIL import Image


# Load Dataset *************************************************************
url = 'https://github.com/uma0shubh/cars24_data/blob/main/cars24_20221210.csv?raw=true'
df = pd.read_csv(url)


# Page Setup *************************************************************
st.set_page_config(layout="wide")

# def add_bg_from_url():
#     st.markdown(
#          f"""
#          <style>
#          .stApp {{
#              background-image: url("https://images8.alphacoders.com/107/1079397.png");
#              background-attachment: fixed;
# 	     background-position: 25% 75%;
#              background-size: cover
#          }}
#          </style>
#          """,
#          unsafe_allow_html=True
#      )

# add_bg_from_url()

# st.markdown("""<style> body {background: #ff0099; background: -webkit-linear-gradient(to right, #ff0099, #493240); background: linear-gradient(to right, #ff0099, #493240);} </style>""", unsafe_allow_html=True)

col1, mid, col2 = st.columns([2,8,2])
with col1:
    st.image("data:image/jpeg;base64,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AI6EDwntEQEREBERAREQEREBERAREQEREBERAREQEREBERAREQEREBERAREQEREBERAREQEREBERAREQETU4dqTbTXaQF7ytLCBzA3KGxn75twEREBERAREQERMQMxEQERMQMxEQEREBERAREQETEzAREQEREBERAREQOfwD/dNP+gp/6azoTn8A/wB00/6Cn/prOhAREQEREBERA1tZv7tu7O1tp2EDdhscjtJGfhI1ZVrxutr7wO1aIQ4rO4hLvXPqjBDbOXIetzBktMCBFdUmrsZ3DaqsEMK1QVjCrYjDIKn1iu7HPnjBmxUNczkM7qptA5ImUqy/NWOQxKhM5HIk4x0kimBAjFg12Qwa4sqahR6tYV2IrasldvIDDgc+qjJIPPz1Sa2yu1GFpDVutGVry5y3O/AGD7OAuMg8+fSWTEDi2WanukBFquG/DmsIWx6/4sEFSNwXqM7SM85pluIFiAXUZcsdteFIFprWvlzU4ryTk8+o54k8QOXoRqdpWxvWFi4Z1X1q8IzDCYAPN1B8MDOfHT4tTabW7sXndUqthiEwroxVOeFYqLBnrkjn0kgEGBGKNK5sQNXqVALtuZmZe7LWBKMBuXJgST4BBzPs+X0HUK9bKti5ZbHVWLBCzgsmS2F2oAOQO7n55krEGBGNPoGsrCmu5N9oyLGbdRUFG719xy7bACQeRc46ZPyatSUbvFuJ7+i38HlSStmXTk/rVqir5BvEEk5lUQOLoKnGpd9liIykHeeRfeSCvrHIKnyG0AD4dufM+oCIiAiIgIiICIiB/9k=", width=100)
with mid:
    st.markdown("<h1 style='text-align: center; color: white;'>Car Price Prediction</h1>", unsafe_allow_html=True)
with col2:
    st.image("https://pbs.twimg.com/profile_images/1439939214162882569/6OPFxXo3_400x400.jpg", width=80)

Prediction, Graphical, Appendix, AboutUs, ContactUs = st.tabs(["Prediction","Graphical Interface","Appendix","About Us","Contact Us"])


# Data Filter *****************************************************************
cars = {
    'Audi': ['A3', 'A6', 'Q3'],
    'Bmw': ['3 Series', '5 Series', 'X3'],
    'Datsun': ['Go', 'Go Plus', 'Redi Go'],
    'Fiat': ['Urban Cross'],
    'Ford': ['Ecosport', 'Endeavour', 'Figo', 'Figo Aspire', 'Freestyle', 'New Figo'],
    'Honda': ['Accord', 'Amaze', 'Brio', 'Br-V', 'City', 'Civic', 'Crv', 'Jazz', 'Wr-V'],
    'Hyundai': ['Accent', 'Alcazar', 'Aura', 'Creta', 'Elite I20', 'Eon', 'Grand I10', 'Grand I10 Nios', 'Grand I10 Prime', 'I10', 'I20', 'I20 Active', 'New Elantra', 'New I20', 'New I20 N Line', 'New Santro', 'Santro Xing', 'Tucson New', 'Venue', 'Verna', 'Xcent'],
    'Jaguar': ['Xf'],
    'Jeep': ['Compass'],
    'Kia': ['Carens', 'Seltos', 'Sonet'],
    'Mahindra': ['Bolero', 'Bolero Neo', 'Kuv 100 Nxt', 'Kuv100', 'Marazzo', 'Scorpio', 'Thar', 'Tuv300', 'Xuv 3Oo', 'Xuv500', 'Xuv700'],
    'Maruti': ['A Star', 'Alto', 'Alto 800', 'Alto K10', 'Baleno', 'Brezza', 'Celerio', 'Celerio X', 'Ciaz', 'Dzire', 'Eeco', 'Ertiga', 'Ignis', 'New  Wagon-R', 'Omni E', 'Ritz', 'S Cross', 'S Presso', 'Swift', 'Swift Dzire', 'Vitara Brezza', 'Wagon R', 'Wagon R 1.0', 'Wagon R Stingray', 'Xl6', 'Zen Estilo'],
    'Mercedes Benz': ['C Class', 'E Class'],
    'Mg': ['Astor', 'Hector', 'Hector Plus'],
    'Nissan': ['Kicks', 'Magnite', 'Micra', 'Micra Active', 'Sunny', 'Terrano'],
    'Renault': ['Captur', 'Duster', 'Kiger', 'Kwid', 'Pulse', 'Triber'],
    'Skoda': ['Kushaq', 'Octavia', 'Rapid', 'Slavia'],
    'Tata': ['Altroz', 'Harrier', 'Hexa', 'Nexon', 'Punch', 'Safari', 'Tiago', 'Tiago Nrg', 'Tigor', 'Zest'],
    'Toyota': ['Camry', 'Corolla Altis', 'Etios', 'Etios Liva', 'Glanza', 'Innova', 'Innova Crysta', 'Urban Cruiser', 'Yaris'],
    'Volkswagen': ['Ameo', 'Jetta', 'Polo', 'Taigun', 'Tiguan', 'Vento']
}


# *****************************************************************
# Checking duplicates *************************************************************
duplicate = df[df.duplicated()]
df['city'].value_counts()

# Considering Top 15 Cities
df1 = df.loc[df['city'].isin(['New Delhi', 'Mumbai', 'Jaipur', 'Chennai', 'Lucknow', 'Bangalore', 'Indore', 'Hyderabad', 'Kochi', 'Pune', 'Kolkata', 'Ahmedabad', 'Gurgaon', 'Noida', 'Ghaziabad'])]
print(df1.shape)
df1['city'].value_counts()

# Data Pre-processing
# Missing Values
#df1.isnull().sum()
#sns.heatmap(df1.isnull(),cbar=False,cmap='viridis')
df1.dropna(inplace=True)
fig101 = df1.isnull().sum()
# fig101 = np.array(fig101).reshape(1, -1)

# fig101 = plt.figure(figsize=(8,4))
# fig101 = sns.heatmap(df1.isnull(),cbar=False,cmap='viridis')

df1.reset_index(inplace=True)
df1.info()
df1.drop(["index"],axis=1,inplace=True)
df1= df1.drop(['name','storename','isc24assured','registrationcity','url','registrationstate','createdDate'], axis = 1)

# *************************************************************
# Descriptive statistics
fig102 = df1.describe(include = 'all')


with Prediction:
    left_column,mid1_column,mid2_column,right_column = st.columns(4)
    def user_input():
        with left_column:
            year = st.selectbox("Year:",options = sorted(df1["year"].unique()))
            fueltype = st.selectbox("Fuel Type:", options=sorted(df1["fueltype"].unique()))
            bodytype = st.selectbox("Body Type:", options=sorted(df1["bodytype"].unique()))

        with mid1_column:
            brand = st.selectbox("Car Name:",options = sorted(df1["make"].unique()))
            kilometerdriven = st.number_input("Enter Km Driven:")
            discountprice = st.number_input("Enter Discount Price:")

        model_name = ""
        if brand == "Audi":
            model_name = cars.get("Audi")
        elif brand == "Bmw":
            model_name = cars.get("Bmw")
        elif brand == "Datsun":
            model_name = cars.get("Datsun")
        elif brand == "Fiat":
            model_name = cars.get("Fiat")
        elif brand == "Ford":
            model_name = cars.get("Ford")
        elif brand == "Honda":
            model_name = cars.get("Honda")
        elif brand == "Hyundai":
            model_name = cars.get("Hyundai")
        elif brand == "Jaguar":
            model_name = cars.get("Jaguar")
        elif brand == "Jeep":
            model_name = cars.get("Jeep")
        elif brand == "Kia":
            model_name = cars.get("Kia")
        elif brand == "Mahindra":
            model_name = cars.get("Mahindra")
        elif brand == "Maruti":
            model_name = cars.get("Maruti")
        elif brand == "Mercedes Benz":
            model_name = cars.get("Mercedes Benz")
        elif brand == "Mg":
            model_name = cars.get("Mg")
        elif brand == "Nissan":
            model_name = cars.get("Nissan")
        elif brand == "Renault":
            model_name = cars.get("Renault")
        elif brand == "Skoda":
            model_name = cars.get("Skoda")
        elif brand == "Tata":
            model_name = cars.get("Tata")
        elif brand == "Toyota":
            model_name = cars.get("Toyota")
        elif brand == "Volkswagen":
            model_name = cars.get("Volkswagen")        

        with mid2_column:
            model = st.selectbox("Model Name:", options=model_name)
            transmission = st.selectbox("Transmission:", options=sorted(df1["transmission"].unique()))
            benefits = st.number_input("Enter Benefits:")

        with right_column:
            city = st.selectbox("City:", options=sorted(df1["city"].unique()))
            ownernumber = st.selectbox("Owner Number:", options=sorted(df1["ownernumber"].unique()))

        new_data = {"make":brand,
                "model":model,
                "city":city,
                "year":year,
                "fueltype":fueltype,
                "kilometerdriven":kilometerdriven,
                "ownernumber":ownernumber,
                "transmission":transmission,
                "bodytype":bodytype,
                "benefits":benefits,
                "discountprice":discountprice
                }

        features = pd.DataFrame(new_data,index = [0])
        return features

    button = st.button("Predict Price")
    pred = user_input()
    
#     fig400 = pred.head()
#     st.write(fig400)


# *************************************************************
df1 = df1.append(pred, ignore_index = True)

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

pred_final = df7.iloc[-1]
pred_final = pred_final.iloc[:-1]
pred_final = np.array(pred_final).reshape(1, -1)
df8 = df7.iloc[:-1]
df9 = df8.copy(deep=True)

# *************************************************************
# Train-Test split
# target_name = 'price'
# train_target0 = df8[target_name]
# df8 = df8.drop([target_name], axis=1)

# # Train and Test Split
# train0, test0, train_target0, test_target0 = train_test_split(df8, train_target0, test_size=0.2, random_state=0)

# valid_part = 0.3
# pd.set_option('max_columns',100)

# # For boosting model
# train0b = train0
# train_target0b = train_target0
# # Synthesis valid as test for selection models
# trainb, testb, targetb, target_testb = train_test_split(train0b, train_target0b, test_size=valid_part, random_state=0)

# # getting test from train data (validation)
# train, test, target, target_test = train_test_split(train0, train_target0, test_size=valid_part, random_state=0)

# # *************************************************************
# # Accuracy List
# acc_train_r2 = []
# acc_test_r2 = []
# acc_train_d = []
# acc_test_d = []
# acc_train_rmse = []
# acc_test_rmse = []

# def acc_d(y_meas, y_pred):
#     # Relative error between predicted y_pred and measured y_meas values (relative error also known as % error)
#     return mean_absolute_error(y_meas, y_pred)*len(y_meas)/sum(abs(y_meas))

# def acc_rmse(y_meas, y_pred):
#     # RMSE between predicted y_pred and measured y_meas values
#     return (mean_squared_error(y_meas, y_pred))**0.5

# def acc_boosting_model(num,model,train,test,num_iteration=0):
#     # Calculation of accuracy of boosting model by different metrics

#     global acc_train_r2, acc_test_r2, acc_train_d, acc_test_d, acc_train_rmse, acc_test_rmse

#     if num_iteration > 0:
#         ytrain = model.predict(train, num_iteration = num_iteration)  
#         ytest = model.predict(test, num_iteration = num_iteration)
#     else:
#         ytrain = model.predict(train)  
#         ytest = model.predict(test)

#     print('target = ', targetb[:5].values)
#     print('ytrain = ', ytrain[:5])

#     acc_train_r2_num = round(r2_score(targetb, ytrain) * 100, 2)
#     print('acc(r2_score) for train =', acc_train_r2_num)   
#     acc_train_r2.insert(num, acc_train_r2_num)

#     acc_train_d_num = round(acc_d(targetb, ytrain) * 100, 2)
#     print('acc(relative error) for train =', acc_train_d_num)   
#     acc_train_d.insert(num, acc_train_d_num)

#     acc_train_rmse_num = round(acc_rmse(targetb, ytrain) * 100, 2)
#     print('acc(rmse) for train =', acc_train_rmse_num)   
#     acc_train_rmse.insert(num, acc_train_rmse_num)

#     print('target_test =', target_testb[:5].values)
#     print('ytest =', ytest[:5])

#     acc_test_r2_num = round(r2_score(target_testb, ytest) * 100, 2)
#     print('acc(r2_score) for test =', acc_test_r2_num)
#     acc_test_r2.insert(num, acc_test_r2_num)

#     acc_test_d_num = round(acc_d(target_testb, ytest) * 100, 2)
#     print('acc(relative error) for test =', acc_test_d_num)
#     acc_test_d.insert(num, acc_test_d_num)

#     acc_test_rmse_num = round(acc_rmse(target_testb, ytest) * 100, 2)
#     print('acc(rmse) for test =', acc_test_rmse_num)
#     acc_test_rmse.insert(num, acc_test_rmse_num)

# def acc_model(num,model,train,test):
#     # Calculation of accuracy of model Sklearn by different metrics   

#     global acc_train_r2, acc_test_r2, acc_train_d, acc_test_d, acc_train_rmse, acc_test_rmse

#     ytrain = model.predict(train)  
#     ytest = model.predict(test)

#     print('target = ', target[:5].values)
#     print('ytrain = ', ytrain[:5])

#     acc_train_r2_num = round(r2_score(target, ytrain) * 100, 2)
#     print('acc(r2_score) for train =', acc_train_r2_num)   
#     acc_train_r2.insert(num, acc_train_r2_num)

#     acc_train_d_num = round(acc_d(target, ytrain) * 100, 2)
#     print('acc(relative error) for train =', acc_train_d_num)   
#     acc_train_d.insert(num, acc_train_d_num)

#     acc_train_rmse_num = round(acc_rmse(target, ytrain) * 100, 2)
#     print('acc(rmse) for train =', acc_train_rmse_num)   
#     acc_train_rmse.insert(num, acc_train_rmse_num)

#     print('target_test =', target_test[:5].values)
#     print('ytest =', ytest[:5])

#     acc_test_r2_num = round(r2_score(target_test, ytest) * 100, 2)
#     print('acc(r2_score) for test =', acc_test_r2_num)
#     acc_test_r2.insert(num, acc_test_r2_num)

#     acc_test_d_num = round(acc_d(target_test, ytest) * 100, 2)
#     print('acc(relative error) for test =', acc_test_d_num)
#     acc_test_d.insert(num, acc_test_d_num)

#     acc_test_rmse_num = round(acc_rmse(target_test, ytest) * 100, 2)
#     print('acc(rmse) for test =', acc_test_rmse_num)
#     acc_test_rmse.insert(num, acc_test_rmse_num)
    
# # Linear Regression
# linreg = LinearRegression()
# linreg.fit(train, target)
# acc_model(0,linreg,train,test)

# # Support vector machine
# svr = SVR()
# svr.fit(train, target)
# acc_model(1,svr,train,test)

# # Linear SVR
# linear_svr = LinearSVR()
# linear_svr.fit(train, target)
# acc_model(2,linear_svr,train,test)

# #     # MLPRegressor
# #     mlp = MLPRegressor()
# #     param_grid = {'hidden_layer_sizes': [i for i in range(2,20)],
# #                   'activation': ['relu'],
# #                   'solver': ['adam'],
# #                   'learning_rate': ['constant'],
# #                   'learning_rate_init': [0.01],
# #                   'power_t': [0.5],
# #                   'alpha': [0.0001],
# #                   'max_iter': [1000],
# #                   'early_stopping': [True],
# #                   'warm_start': [False]}
# #     mlp_GS = GridSearchCV(mlp, param_grid=param_grid, 
# #                        cv=10, verbose=True, pre_dispatch='2*n_jobs')
# #     mlp_GS.fit(train, target)
# #     acc_model(3,mlp_GS,train,test)

# # Stochastic Gradient Descent
# sgd = SGDRegressor()
# sgd.fit(train, target)
# acc_model(4,sgd,train,test)

# # Decision Tree Regressor
# decision_tree = DecisionTreeRegressor()
# decision_tree.fit(train, target)
# acc_model(5,decision_tree,train,test)

# # Random Forest
# random_forest = RandomForestRegressor()
# random_forest.fit(train, target)
# acc_model(6,random_forest,train,test)

# # XGB
# xgb_clf = xgb.XGBRegressor(objective ='reg:squarederror', verbosity = 0, silent=True, random_state=42) 
# parameters = {'n_estimators': [60, 100, 120, 140], 
#               'learning_rate': [0.01, 0.1],
#               'max_depth': [5, 7],
#               'reg_lambda': [0.5]}
# xgb_reg = GridSearchCV(estimator=xgb_clf, param_grid=parameters, cv=5, n_jobs=-1).fit(trainb, targetb)
# print("Best score: %0.3f" % xgb_reg.best_score_)
# print("Best parameters set:", xgb_reg.best_params_)
# acc_boosting_model(7,xgb_reg,trainb,testb)

# # LGBM
# Xtrain, Xval, Ztrain, Zval = train_test_split(trainb, targetb, test_size=0.2, random_state=0)
# train_set = lgb.Dataset(Xtrain, Ztrain, silent=False)
# valid_set = lgb.Dataset(Xval, Zval, silent=False)
# params = {
#         'boosting_type':'gbdt',
#         'objective': 'regression',
#         'num_leaves': 31,
#         'learning_rate': 0.01,
#         'max_depth': -1,
#         'subsample': 0.8,
#         'bagging_fraction' : 1,
#         'max_bin' : 5000 ,
#         'bagging_freq': 20,
#         'colsample_bytree': 0.6,
#         'metric': 'rmse',
#         'min_split_gain': 0.5,
#         'min_child_weight': 1,
#         'min_child_samples': 10,
#         'scale_pos_weight':1,
#         'zero_as_missing': False,
#         'seed':0,        
#     }
# modelL = lgb.train(params, train_set = train_set, num_boost_round=10000,
#                    early_stopping_rounds=8000,verbose_eval=500, valid_sets=valid_set)

# acc_boosting_model(8,modelL,trainb,testb,modelL.best_iteration)

# fig =  plt.figure(figsize = (5,5))
# axes = fig.add_subplot(111)
# lgb.plot_importance(modelL,ax = axes,height = 0.5)
# plt.show();

# #     # GradientBoostingRegressor with HyperOpt
# #     def hyperopt_gb_score(params):
# #         clf = GradientBoostingRegressor(**params)
# #         current_score = cross_val_score(clf, train, target, cv=10).mean()
# #         print(current_score, params)
# #         return current_score 

# #     space_gb = {
# #                 'n_estimators': hp.choice('n_estimators', range(100, 1000)),
# #                 'max_depth': hp.choice('max_depth', np.arange(2, 10, dtype=int))            
# #             }

# #     best = fmin(fn=hyperopt_gb_score, space=space_gb, algo=tpe.suggest, max_evals=10)
# #     print('best:')
# #     print(best)

# #     params = space_eval(space_gb, best)

# #     # Gradient Boosting Regression
# #     gradient_boosting = GradientBoostingRegressor(**params)
# #     gradient_boosting.fit(train, target)
# #     acc_model(9,gradient_boosting,train,test)

# # Ridge Regressor
# ridge = RidgeCV(cv=5)
# ridge.fit(train, target)
# acc_model(10,ridge,train,test)

# # Bagging Regressor
# bagging = BaggingRegressor()
# bagging.fit(train, target)
# acc_model(11,bagging,train,test)

# # Extra Trees Regressor
# etr = ExtraTreesRegressor()
# etr.fit(train, target)
# acc_model(12,etr,train,test)

# # AdaBoost Regressor
# Ada_Boost = AdaBoostRegressor()
# Ada_Boost.fit(train, target)
# acc_model(13,Ada_Boost,train,test)

# # Voting Regressor
# Voting_Reg = VotingRegressor(estimators=[('lin', linreg), ('ridge', ridge), ('sgd', sgd)])
# Voting_Reg.fit(train, target)
# acc_model(14,Voting_Reg,train,test)

# # *************************************************************
# # Models comparison
# models = pd.DataFrame({
#     'Model': ['Linear Regression', 'Support Vector Machines', 'Linear SVR', 'Stochastic Gradient Decent', 
#               'Decision Tree Regressor', 'Random Forest',  'XGB', 'LGBM','RidgeRegressor', 'BaggingRegressor', 'ExtraTreesRegressor',
#               'AdaBoostRegressor', 'VotingRegressor'],

# #         'Model': ['Linear Regression', 'Support Vector Machines', 'Linear SVR', 
# #                   'MLPRegressor', 'Stochastic Gradient Decent', 
# #                   'Decision Tree Regressor', 'Random Forest',  'XGB', 'LGBM',
# #                   'GradientBoostingRegressor', 'RidgeRegressor', 'BaggingRegressor', 'ExtraTreesRegressor', 
# #                   'AdaBoostRegressor', 'VotingRegressor'],

#     'r2_train': acc_train_r2,
#     'r2_test': acc_test_r2,
#     'd_train': acc_train_d,
#     'd_test': acc_test_d,
#     'rmse_train': acc_train_rmse,
#     'rmse_test': acc_test_rmse
#                      })

# pd.options.display.float_format = '{:,.2f}'.format

# print('Prediction accuracy for models by R2 criterion - r2_test')
# models.sort_values(by=['r2_test', 'r2_train'], ascending=False)

# print('Prediction accuracy for models by RMSE - rmse_test')
# models.sort_values(by=['rmse_test', 'rmse_train'], ascending=True)

# # Model Output - Visualization
# # Plot
# fig200 = plt.figure(figsize=[20,8])
# xx = models['Model']
# plt.tick_params(labelsize=14)
# plt.plot(xx, models['r2_train'], label = 'r2_train')
# plt.plot(xx, models['r2_test'], label = 'r2_test')
# plt.legend()
# plt.title('R2-criterion for 15 popular models for train and test datasets')
# plt.xlabel('Models')
# plt.ylabel('R2-criterion, %')
# plt.xticks(xx, rotation='vertical')
# plt.savefig('graph.png')
# plt.show()
# #     st.write(fig200)

# # Plot
# fig201 = plt.figure(figsize=[20,8])
# xx = models['Model']
# plt.tick_params(labelsize=14)
# plt.plot(xx, models['rmse_train'], label = 'rmse_train')
# plt.plot(xx, models['rmse_test'], label = 'rmse_test')
# plt.legend()
# plt.title('RMSE for 15 popular models for train and test datasets')
# plt.xlabel('Models')
# plt.ylabel('RMSE, %')
# plt.xticks(xx, rotation='vertical')
# plt.savefig('graph.png')
# plt.show()
# #     st.write(fig201)

    


with Appendix:
    """# Missing Values"""
    st.write(fig101)
    
    """# Descriptive statistics"""
    st.write(fig102)
    
    """# Model Building

    - Linear Regression
    - Support vector machine 
    - Linear SVR
    - MLPRegressor (Deep Learning)
    - Stochastic Gradient Descent
    - Decision Tree Regressor
    - Random Forest
    - XGB
    - LGBM
    - Gradient Boosting Regressor
    - Ridge Regressor
    - Bagging Regressor
    - ExtraTreesRegressor 
    - AdaBoost Regressor
    - Voting Regressor
    """
    
    """# Models comparison"""
    
#     st.write(fig200)
    
#     st.write(fig201)
    
    """Thus, the best model is Linear Regression."""
    

    

with Graphical:
    # *************************************************************
    """# Data Overview """
    fig300 = df1.head()
    st.write(fig300)
    
    """# Univariant plots """
    """### Numerical Analysis """
    """##### Density Plot """
    x_axis = st.selectbox("Select Variable", options=("Price", "Km", "Benefits", "Discount Price"))
    if x_axis == "Price":
        x = df1["price"]
    elif x_axis == "Km":
        x = df1["kilometerdriven"]
    elif x_axis == "benefits":
        x = df1["Benefits"]
    elif x_axis == "Discount Price":
        x = df1["discountprice"]
    
    fig301 = plt.figure(figsize=(15,8))
    sns.distplot(x)
    st.write(fig301)
    st.write("Skewness: %f" % x.skew())
    st.write("Kurtosis: %f" % x.kurt())
    
    """##### Box Plot """
    fig302 = plt.figure(figsize=(15,4))
    #sns.boxplot(x=x,data=df1)
    sns.boxplot(x=x)
    st.write(fig302)
    
    """### Categorical Analysis """
    """##### Bar Plot """
    x_axis_2 = st.selectbox("Select Variable", options=("Brand", "City", "Year", "Owner Number", "FuelType", "Transmission", "BodyType", "RegistrationState"))
    if x_axis_2 == "Brand":
        x2 = df1["make"]
    elif x_axis_2 == "City":
        x2 = df1["city"]
    elif x_axis_2 == "Year":
        x2 = df1["year"]
    elif x_axis_2 == "Owner Number":
        x2 = df1["ownernumber"]
    elif x_axis_2 == "FuelType":
        x2 = df1["fueltype"]
    elif x_axis_2 == "Transmission":
        x2 = df1["transmission"]
    elif x_axis_2 == "BodyType":
        x2 = df1["bodytype"]
    elif x_axis_2 == "RegistrationState":
        x2 = df1["registrationstate"]
        
#     fig303 = df1.groupby('city').size().plot(kind='bar')
    fig303 = sns.countplot(df1['city'], color='gray')
    
#     fig303 = plt.figure(figsize = (15,4))
#     #plt.bar(x.value_counts())
#     plt.bar(x, df1["price"])
#     plt.xticks(rotation=90)

#     fig303 = df1.plot.bar(x=, rot=0)
    st.write(fig303)
    
#     fig303 = sns.set(font_scale=1.4)
#     x.value_counts().plot(kind='bar', figsize=(15, 4), rot=90)
#     st.write(fig303)
    
#     fig304 = x.value_counts().plot(kind='barh', figsize=(15, 20))
#     st.write(fig304)
    
    
    
    # ************************************************************* 
    """# Bivariate plots """
    x_axis_1 = st.selectbox("Select Variable", options=("Brand", "City", "Year", "Owner Number", "FuelType", "Transmission", "BodyType"))
    if x_axis_1 == "Brand":
        x1 = "make"
    elif x_axis_1 == "City":
        x1 = "city"
    elif x_axis_1 == "Year":
        x1 = "year"
    elif x_axis_1 == "Owner Number":
        x1 = "ownernumber"
    elif x_axis_1 == "FuelType":
        x1 = "fueltype"
    elif x_axis_1 == "Transmission":
        x1 = "transmission"
    elif x_axis_1 == "BodyType":
        x1 = "bodytype"
    
    fig304 = sns.catplot(y='price', x=x1, data=df1.sort_values('price',ascending=False),kind="boxen",height=5, aspect=3)
    st.pyplot(fig304)
    
    # ******
    x_axis_3 = st.selectbox("Select Variable", options=("Brand", "Owner Number", "FuelType", "Transmission", "BodyType"))
    if x_axis_3 == "Brand":
        x3 = "make"
    elif x_axis_3 == "Owner Number":
        x3 = "ownernumber"
    elif x_axis_3 == "FuelType":
        x3 = "fueltype"
    elif x_axis_3 == "Transmission":
        x3 = "transmission"
    elif x_axis_3 == "BodyType":
        x3 = "bodytype"
    
#     fig305 = px.sunburst(df1, path=['city', x3], color='city',height=600)
#     fig305.update_layout(title_text="Two-level Sunburst Diagram", font_size=10)
#     st.write(fig305)
    
    fig306 = px.treemap(df1, path=['city', x3], color='city',height=600,width=1000)
    fig306.update_layout(title_text="Distribution within City", font_size=10)
    st.write(fig306)
    
    
    # ************************************************************* 
    """# Yearly City Growth """
    fig307 = px.histogram(df1, x="year", y="price",color='city', barmode='group',height=500,width=1200)
    fig307.update_layout(title_text="Yearly City Growth", font_size=10)
    st.write(fig307)
    
    # ************************************************************* 
    """# Correlation """
    fig308 = plt.figure(figsize=(15,10))
    sns.heatmap(df8.corr(),annot=True,cmap='RdYlGn')
    st.write(fig308)

    
    
with Prediction:
    # Linear Regression
    linreg_1 = LinearRegression()
    linreg_1.fit(df9.iloc[:,:-1], df9['price'])
    
    fig600 = linreg_1.predict(pred_final)
    st.write(fig600)

                                                    
                                                    
with AboutUs:
    """ 
    ### History of CarsDeal:
    CarsDeal was started by FabFurnish founders Vikram Chopra and Mehul Agrawal in August, 2015. Their aim is to help the car owners to sell their cars instantly without any trouble. And also help the dealers and people who are looking for a second hand car with no trouble and issues in a very less time and minimum paperwork and legal formalities. And apart from that you’ll also get expert assistance and from start to finish.
    
    ### Services offered by CarsDeal
    CarsDeal offers buy and sell service of old cars and also free RC transfer. They help us to sell the cars with the best price and also provide instant payment with zero trouble in paper work or delay in the payments.
    They also provide financial loans to the businesses to help them to grow faster and also to buy the dream car with a very fast process with their AI and machine learning. Currently they are having 3 product in their financial menu as well,
    
    ### Business Model Of CarsDeal: How CarsDeal earns?
    CarsDeal follows asset-heavy customers to business (C2B) model, where it buys used cars from individuals and dealers and sells them to other dealerships and individuals as well.
    There’s no public information about it’s charges but according to Economic Times, unlike a listings-based classifieds platform, Cars24 enables the end-to-end transaction itself, charging a commission of about 4-5% for each transaction and also a small fee from the buyers as an registration
    
    ### CarsDeal Competitors:
    - Droom
    - Cardekho
    - Cartrade
    - Carwale
    - Spinny
    - Cars24
    - Carnation
    - Dealer Direct
    - Checkgaadi
    - Mahindra First Choice Wheels
    
    website : https://www.carsdeal.com/
    """
    
with ContactUs:
    """
    ### Contact Us
    Data Scientist & Developer Team
    - Arpita Pyne : pynearpita06@gmail.com
    - Asmit Pawar : asmitpawar98@gmail.com
    - Sagar Chore : sagar.chore@gmail.com
    - Uma Prajapati : umaprajapati35@gmail.com
    - Vrushali Patil : pvrushali27@gmail.com
    
    ### Study Material
    https://github.com/uma0shubh/cars24_data
    
    https://github.com/uma0shubh/cars24_Regression
    """

# #     model=pickle.load(open('model_linear.pkl','rb'))

# #     prediction = model.predict(df)
# #     pred = np.round(prediction,2)
# #     predic = abs(pred)
# #     predict = str(predic).lstrip('[').rstrip(']')
# #     button_clicked = False

# #     if button:
# #         button_clicked = True
# #         st.subheader(f"Car Price: ₹{predict}")








