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

#https://github.com/uma0shubh/cars24_data/blob/main/background1.jpeg

def add_bg_from_url():
    st.markdown(
         f"""
         <style>
         .stApp {{
             background-image: url("https://wallpapercave.com/download/forex-wallpapers-wp2116598");
             background-attachment: fixed;
	     background-position: 25% 75%;
             background-size: cover
         }}
         </style>
         """,
         unsafe_allow_html=True
     )

add_bg_from_url()

st.title("Car Price Prediction")
# options = st.sidebar.selectbox("Select Analyzing options:", options= ("Prediction","Data Analysis","Graphical Interface"))
Prediction, Graphical, Appendix, AboutUs, ContactUs = st.tabs(["Prediction","Graphical Interface","Appendix","About Us","Contact Us"])
# st.header(options)

# def add_logo(logo_path, width, height):
#     """Read and return a resized logo"""
#     logo = Image.open(logo_path)
#     modified_logo = logo.resize((width, height))
#     return modified_logo
# #st.sidebar.image(add_logo(logo_path="https://www.cars24.com/js/28776e9c38260ac3339c3babe6171dd0.svg", width=50, height=60))

# my_logo = add_logo(logo_path="https://github.com/uma0shubh/cars24_data/blob/main/excelr_logo.jpg?raw=true", width=50, height=60)
# st.sidebar.image(my_logo)
    
# add_logo()


with Appendix:
    # Checking duplicates *************************************************************
    duplicate = df[df.duplicated()]
    df['city'].value_counts()

    # Considering Top 15 Cities
    df1 = df.loc[df['city'].isin(['New Delhi', 'Mumbai', 'Jaipur', 'Chennai', 'Lucknow', 'Bangalore', 'Indore', 'Hyderabad', 'Kochi', 'Pune', 'Kolkata', 'Ahmedabad', 'Gurgaon', 'Noida', 'Ghaziabad'])]
    print(df1.shape)
    df1['city'].value_counts()

    # Data Pre-processing
    """# Missing Values"""
    df1.isnull().sum()
    sns.heatmap(df1.isnull(),cbar=False,cmap='viridis')
    df1.dropna(inplace=True)
    df1.isnull().sum()

    fig101 = plt.figure(figsize=(8,4))
    sns.heatmap(df1.isnull(),cbar=False,cmap='viridis')
    st.pyplot(fig101)

    df1.reset_index(inplace=True)
    df1.info()
    df1.drop(["index"],axis=1,inplace=True)
    
    # *************************************************************
    """# Descriptive statistics"""
    fig102 = df1.describe(include = 'all')
    st.write(fig102)
    
    # *************************************************************
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

    df8= df7.drop(['name','storename','isc24assured','registrationcity','url','registrationstate','createdDate'], axis = 1)

    # *************************************************************
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

    train0.head(3)

    # getting test from train data (validation)
    train, test, target, target_test = train_test_split(train0, train_target0, test_size=valid_part, random_state=0)
    train.head(3)

    # *************************************************************
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

    # *************************************************************
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

    # Linear Regression
    linreg = LinearRegression()
    linreg.fit(train, target)
    acc_model(0,linreg,train,test)

#     # Support vector machine
#     svr = SVR()
#     svr.fit(train, target)
#     acc_model(1,svr,train,test)

#     # Linear SVR
#     linear_svr = LinearSVR()
#     linear_svr.fit(train, target)
#     acc_model(2,linear_svr,train,test)

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

#     # Stochastic Gradient Descent
#     sgd = SGDRegressor()
#     sgd.fit(train, target)
#     acc_model(4,sgd,train,test)

#     # Decision Tree Regressor
#     decision_tree = DecisionTreeRegressor()
#     decision_tree.fit(train, target)
#     acc_model(5,decision_tree,train,test)

#     # Random Forest
#     random_forest = RandomForestRegressor()
#     random_forest.fit(train, target)
#     acc_model(6,random_forest,train,test)

#     # XGB
#     xgb_clf = xgb.XGBRegressor(objective ='reg:squarederror', verbosity = 0, silent=True, random_state=42) 
#     parameters = {'n_estimators': [60, 100, 120, 140], 
#                   'learning_rate': [0.01, 0.1],
#                   'max_depth': [5, 7],
#                   'reg_lambda': [0.5]}
#     xgb_reg = GridSearchCV(estimator=xgb_clf, param_grid=parameters, cv=5, n_jobs=-1).fit(trainb, targetb)
#     print("Best score: %0.3f" % xgb_reg.best_score_)
#     print("Best parameters set:", xgb_reg.best_params_)
#     acc_boosting_model(7,xgb_reg,trainb,testb)

#     # LGBM
#     Xtrain, Xval, Ztrain, Zval = train_test_split(trainb, targetb, test_size=0.2, random_state=0)
#     train_set = lgb.Dataset(Xtrain, Ztrain, silent=False)
#     valid_set = lgb.Dataset(Xval, Zval, silent=False)
#     params = {
#             'boosting_type':'gbdt',
#             'objective': 'regression',
#             'num_leaves': 31,
#             'learning_rate': 0.01,
#             'max_depth': -1,
#             'subsample': 0.8,
#             'bagging_fraction' : 1,
#             'max_bin' : 5000 ,
#             'bagging_freq': 20,
#             'colsample_bytree': 0.6,
#             'metric': 'rmse',
#             'min_split_gain': 0.5,
#             'min_child_weight': 1,
#             'min_child_samples': 10,
#             'scale_pos_weight':1,
#             'zero_as_missing': False,
#             'seed':0,        
#         }
#     modelL = lgb.train(params, train_set = train_set, num_boost_round=10000,
#                        early_stopping_rounds=8000,verbose_eval=500, valid_sets=valid_set)

#     acc_boosting_model(8,modelL,trainb,testb,modelL.best_iteration)

#     fig =  plt.figure(figsize = (5,5))
#     axes = fig.add_subplot(111)
#     lgb.plot_importance(modelL,ax = axes,height = 0.5)
#     plt.show();

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

#     # Ridge Regressor
#     ridge = RidgeCV(cv=5)
#     ridge.fit(train, target)
#     acc_model(10,ridge,train,test)

#     # Bagging Regressor
#     bagging = BaggingRegressor()
#     bagging.fit(train, target)
#     acc_model(11,bagging,train,test)

#     # Extra Trees Regressor
#     etr = ExtraTreesRegressor()
#     etr.fit(train, target)
#     acc_model(12,etr,train,test)

#     # AdaBoost Regressor
#     Ada_Boost = AdaBoostRegressor()
#     Ada_Boost.fit(train, target)
#     acc_model(13,Ada_Boost,train,test)

#     # Voting Regressor
#     Voting_Reg = VotingRegressor(estimators=[('lin', linreg), ('ridge', ridge), ('sgd', sgd)])
#     Voting_Reg.fit(train, target)
#     acc_model(14,Voting_Reg,train,test)

#     # *************************************************************
#     """# Models comparison"""
#     models = pd.DataFrame({
#         'Model': ['Linear Regression', 'Support Vector Machines', 'Linear SVR', 'Stochastic Gradient Decent', 
#                   'Decision Tree Regressor', 'Random Forest',  'XGB', 'LGBM','RidgeRegressor', 'BaggingRegressor', 'ExtraTreesRegressor',
#                   'AdaBoostRegressor', 'VotingRegressor'],
        
# #         'Model': ['Linear Regression', 'Support Vector Machines', 'Linear SVR', 
# #                   'MLPRegressor', 'Stochastic Gradient Decent', 
# #                   'Decision Tree Regressor', 'Random Forest',  'XGB', 'LGBM',
# #                   'GradientBoostingRegressor', 'RidgeRegressor', 'BaggingRegressor', 'ExtraTreesRegressor', 
# #                   'AdaBoostRegressor', 'VotingRegressor'],
        
#         'r2_train': acc_train_r2,
#         'r2_test': acc_test_r2,
#         'd_train': acc_train_d,
#         'd_test': acc_test_d,
#         'rmse_train': acc_train_rmse,
#         'rmse_test': acc_test_rmse
#                          })

#     pd.options.display.float_format = '{:,.2f}'.format

#     print('Prediction accuracy for models by R2 criterion - r2_test')
#     models.sort_values(by=['r2_test', 'r2_train'], ascending=False)

#     print('Prediction accuracy for models by RMSE - rmse_test')
#     models.sort_values(by=['rmse_test', 'rmse_train'], ascending=True)

#     # Model Output - Visualization
#     # Plot
#     fig200 = plt.figure(figsize=[20,8])
#     xx = models['Model']
#     plt.tick_params(labelsize=14)
#     plt.plot(xx, models['r2_train'], label = 'r2_train')
#     plt.plot(xx, models['r2_test'], label = 'r2_test')
#     plt.legend()
#     plt.title('R2-criterion for 15 popular models for train and test datasets')
#     plt.xlabel('Models')
#     plt.ylabel('R2-criterion, %')
#     plt.xticks(xx, rotation='vertical')
#     plt.savefig('graph.png')
#     plt.show()
#     st.write(fig200)

#     # Plot
#     fig201 = plt.figure(figsize=[20,8])
#     xx = models['Model']
#     plt.tick_params(labelsize=14)
#     plt.plot(xx, models['rmse_train'], label = 'rmse_train')
#     plt.plot(xx, models['rmse_test'], label = 'rmse_test')
#     plt.legend()
#     plt.title('RMSE for 15 popular models for train and test datasets')
#     plt.xlabel('Models')
#     plt.ylabel('RMSE, %')
#     plt.xticks(xx, rotation='vertical')
#     plt.savefig('graph.png')
#     plt.show()
#     st.write(fig201)

#     """Thus, the best model is Linear Regression."""

#     # *************************************************************
#     # Prediction
#     #For models from Sklearn
#     testn = pd.DataFrame(scaler.transform(test0), columns = test0.columns)

#     #Ridge Regressor model for basic train
#     ridge.fit(train0, train_target0)
#     #ridge.predict(testn)[:3]

#     #xgb model for basic train
#     xgb_reg.fit(train0, train_target0)
#     #xgb_reg.predict(testn)[:3]

#     #Ada_Boost  model for basic train
#     Ada_Boost.fit(train0, train_target0)
#     #Ada_Boost.predict(testn)[:3]

#     #Voting Regressor model for basic train
#     Voting_Reg.fit(train0, train_target0)
#     #Voting_Reg.predict(testn)[:3]

#     #svr model for basic train
#     svr.fit(train0, train_target0)
#     #svr.predict(testn)[:3]


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
    
#     fig303 = plt.figure(figsize = (15,4))
#     #plt.bar(x.value_counts())
#     plt.bar(x, df1["price"])
#     plt.xticks(rotation=90)
#     st.pyplot(fig303)
    
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
                                                    
                                                    
with AboutUs:
    """ 
    ### History of Cars24:
    Cars24 was started by FabFurnish founders Vikram Chopra and Mehul Agrawal in August, 2015. Their aim is to help the car owners to sell their cars instantly without any trouble. And also help the dealers and people who are looking for a second hand car with no trouble and issues in a very less time and minimum paperwork and legal formalities. And apart from that you’ll also get expert assistance and from start to finish.
    
    ### Services offered by Cars24
    Cars24 offers buy and sell service of old cars and also free RC transfer. They help us to sell the cars with the best price and also provide instant payment with zero trouble in paper work or delay in the payments.
    They also provide financial loans to the businesses to help them to grow faster and also to buy the dream car with a very fast process with their AI and machine learning. Currently they are having 3 product in their financial menu as well,
    
    ### Business Model Of Cars24: How Cars24 earns?
    Cars24 follows asset-heavy customers to business (C2B) model, where it buys used cars from individuals and dealers and sells them to other dealerships and individuals as well.
    There’s no public information about it’s charges but according to Economic Times, unlike a listings-based classifieds platform, Cars24 enables the end-to-end transaction itself, charging a commission of about 4-5% for each transaction and also a small fee from the buyers as an registration
    
    ### Cars24 Competitors:
    - Droom
    - Cardekho
    - Cartrade
    - Carwale
    - Spinny
    - Carnation
    - Dealer Direct
    - Checkgaadi
    - Mahindra First Choice Wheels
    
    website : https://www.cars24.com/
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
    
    fig400 = pred.head()
    st.write(fig400)
    
    numerics = ['int8', 'int16', 'int32', 'int64', 'float16', 'float32', 'float64']
    categorical_columns = []
    features = pred.columns.values.tolist()
    for col in features:
        if pred[col].dtype in numerics: continue
        categorical_columns.append(col)
    # Encoding categorical features
    for col in categorical_columns:
        if col in pred.columns:
            le = LabelEncoder()
            le.fit(list(pred[col].astype(str).values))
            pred[col] = le.transform(list(pred[col].astype(str).values))
    
    pred['year'] = (pred['year']-1900).astype(int)
    fig401 = pred.head()
    st.write(fig401)
    
#     scaler = StandardScaler()
#     pred_trans = pd.DataFrame(scaler.transform(pred), columns = pred.columns)
#     fig402 = pred_trans.head()
#     st.write(fig402)

#     fig403 = linreg.predict(pred)    
#     st.write(fig403)

#     model=pickle.load(open('model_linear.pkl','rb'))

#     prediction = model.predict(df)
#     pred = np.round(prediction,2)
#     predic = abs(pred)
#     predict = str(predic).lstrip('[').rstrip(']')
#     button_clicked = False

#     if button:
#         button_clicked = True
#         st.subheader(f"Car Price: ₹{predict}")








