# -*- coding: utf-8 -*-

# Import Libraries *************************************************************
import seaborn as sns
sns.set()
import matplotlib.pyplot as plt
# %matplotlib inline
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

import requests
import pandas as pd
import streamlit as st

# Load Dataset *************************************************************
url = 'https://github.com/uma0shubh/cars24_data/blob/main/cars24_20221210.csv?raw=true'
df = pd.read_csv(url)

# Page Setup *************************************************************
st.set_page_config(layout="wide")
st.title("Car Price Prediction")
options = st.sidebar.selectbox("Select Analyzing options:", options= ("Prediction","Data Analysis","Graphical Interface"))
# st.header(options)

# Data Filter *****************************************************************
cars = {
    'Maruti': ['Maruti Celerio VXI AMT', 'Maruti Baleno ZETA 1.2 K12 CVT',
       'Maruti Swift VXI', 'Maruti Celerio VXI CNG',
       'Maruti Baleno DELTA 1.2 K12', 'Maruti New  Wagon-R VXI 1.0',
       'Maruti Celerio ZXI', 'Maruti Alto 800 VXI', 'Maruti Dzire VXI',
       'Maruti S PRESSO VXI PLUS', 'Maruti New  Wagon-R VXI 1.2L',
       'Maruti S PRESSO VXI', 'Maruti Alto 800 LXI',
       'Maruti Celerio X Zxi AMT', 'Maruti Vitara Brezza ZXI+ AT SHVS',
       'Maruti XL6 ALPHA SHVS  MT', 'Maruti Vitara Brezza VXI',
       'Maruti Ciaz ALPHA 1.5 MT VTVT SHVS', 'Maruti Wagon R 1.0 VXI',
       'Maruti Alto LXI CNG', 'Maruti Ciaz ZETA 1.4 VVT',
       'Maruti Baleno ZETA 1.2 K12', 'Maruti IGNIS DELTA 1.2 K12',
       'Maruti S PRESSO VXI OPT AGS', 'Maruti Alto LXI',
       'Maruti Dzire ZXI AMT', 'Maruti Ciaz ZETA 1.4 VVT AMT',
       'Maruti Alto VXI', 'Maruti S Cross ALPHA 1.5 SMART HYBRID',
       'Maruti Swift LXI', 'Maruti Eeco 5 STR WITH AC PLUSHTR',
       'Maruti Swift ZXI AMT', 'Maruti Dzire ZDI AMT',
       'Maruti Alto K10 VXI', 'Maruti Swift ZXI',
       'Maruti Swift Dzire ZDI', 'Maruti Swift Dzire VDI',
       'Maruti S PRESSO VXI (O) CNG', 'Maruti Swift ZXI + AMT',
       'Maruti Ritz VXI', 'Maruti Swift Dzire ZDI AMT',
       'Maruti New  Wagon-R VXI (O) 1.2L', 'Maruti Celerio VXI',
       'Maruti New  Wagon-R ZXI 1.2', 'Maruti Swift Dzire VXI',
       'Maruti Alto VXI PLUS', 'Maruti Ritz ZXI',
       'Maruti A Star VXI ABS AT', 'Maruti Swift Dzire ZXI 1.2 BS IV',
       'Maruti Ciaz S 1.3 SHVS', 'Maruti Wagon R LXI', 'Maruti Swift VDI',
       'Maruti Ritz ZDI', 'Maruti Swift Dzire VDI ABS',
       'Maruti Celerio ZXI AMT', 'Maruti Eeco 7 STR',
       'Maruti Swift ZDI plus AMT', 'Maruti Swift VXI AMT',
       'Maruti Vitara Brezza ZDI', 'Maruti Swift ZDI AMT',
       'Maruti Swift ZDI', 'Maruti Vitara Brezza ZDI + AMT',
       'Maruti IGNIS SIGMA 1.2 K12', 'Maruti Swift VXI OPT',
       'Maruti Swift VDI ABS', 'Maruti Wagon R 1.0 LXI',
       'Maruti Dzire ZDI Plus AMT',
       'Maruti Vitara Brezza ZDI PLUS DUAL TONE',
       'Maruti Celerio VXI (O) AMT', 'Maruti Ciaz ZDI+ SHVS',
       'Maruti Baleno ALPHA DDIS 190', 'Maruti Celerio ZXI OPT',
       'Maruti Wagon R Stingray VXI', 'Maruti S Cross ZETA 1.3 SHVS',
       'Maruti Ciaz SIGMA 1.4 VVT', 'Maruti Ciaz DELTA 1.5 SHVS VVT MT',
       'Maruti Zen Estilo LXI', 'Maruti Swift ZDI +',
       'Maruti Baleno DELTA 1.2 K12 AT', 'Maruti S Cross DELTA SHVS',
       'Maruti Vitara Brezza ZDI PLUS', 'Maruti Ciaz ZXI AT',
       'Maruti Vitara Brezza LDI OPT',
       'Maruti Vitara Brezza ZDI+ DUAL TONE AMT', 'Maruti Dzire VXI AMT',
       'Maruti S Cross ALPHA AT SMART HYBRID',
       'Maruti Dzire ZXI Plus AMT', 'Maruti Alto LXI OPT',
       'Maruti S Cross SIGMA 1.3', 'Maruti Alto K10 VXI OPT',
       'Maruti Dzire ZXI Plus', 'Maruti S Cross ZETA SHVS PETROL',
       'Maruti S PRESSO VXI PLUS AGS', 'Maruti Dzire ZDI Plus',
       'Maruti S Cross ZETA 1.3', 'Maruti Ciaz ALPHA 1.4 VVT',
       'Maruti Wagon R 1.0 VXI AMT', 'Maruti Alto 800 VXI (O)',
       'Maruti Ciaz ALPHA 1.3 DDIS SHVS', 'Maruti IGNIS ZETA 1.2 K12 AMT',
       'Maruti Baleno ZETA DDIS 190', 'Maruti Baleno ZETA 1.2 K12 AMT',
       'Maruti S PRESSO VXI AGS', 'Maruti Baleno ALPHA 1.2K12  AT',
       'Maruti IGNIS ALPHA 1.2 K12', 'Maruti OMNI E STD',
       'Maruti Baleno RS 1.0 PETROL', 'Maruti Alto K10 VXI AMT',
       'Maruti Wagon R 1.0 VXI OPT', 'Maruti Alto K10 VXI (O) AMT',
       'Maruti IGNIS ZETA 1.2 K12',
       'Maruti Ertiga VDI Anniversary Edition', 'Maruti Ciaz ZXI AMT',
       'Maruti Swift Dzire VXI 1.3', 'Maruti A Star VXI',
       'Maruti Ertiga ZXI', 'Maruti Vitara Brezza VDI OPT',
       'Maruti Vitara Brezza ZDI AMT', 'Maruti Alto LX',
       'Maruti Swift LXI OPT', 'Maruti Baleno ALPHA 1.2 K12',
       'Maruti IGNIS DELTA 1.2 K12 AMT', 'Maruti Zen Estilo VXI',
       'Maruti Ertiga ZDI SHVS', 'Maruti Celerio ZXI+',
       'Maruti Celerio X Zxi', 'Maruti Ciaz DELTA 1.4 VVT AT',
       'Maruti XL6 ZETA SHVS', 'Maruti Swift ZXI+', 'Maruti Dzire VDI',
       'Maruti Vitara Brezza VDI', 'Maruti Dzire VDI AMT',
       'Maruti Wagon R 1.0 VXI OPT AMT', 'Maruti OMNI E 8 STR',
       'Maruti Celerio ZXI OPT AMT', 'Maruti Ciaz ALPHA 1.5 AT SHVS',
       'Maruti Alto 800 LXI OPT', 'Maruti Ciaz ALPHA 1.4 VVT AMT',
       'Maruti Wagon R 1.0 VXI PLUS AMT', 'Maruti Ertiga VXI',
       'Maruti S Cross SIGMA SHVS', 'Maruti New  Wagon-R LXI 1.0 L',
       'Maruti New  Wagon-R 1.0 LXI (O)', 'Maruti Ertiga ZXI Plus SHVS',
       'Maruti Baleno SIGMA 1.2 K12', 'Maruti Vitara Brezza LXI',
       'Maruti Ertiga VXI SMART HYBRID',
       'Maruti New  Wagon-R 1.0 Lxi (o) cng',
       'Maruti New  Wagon-R LXI CNG 1.0 L',
       'Maruti New  Wagon-R 1.0 VXI (O)',
       'Maruti New  Wagon-R ZXI 1.2L AGS', 'Maruti Swift Dzire ZXI 1.3',
       'Maruti Vitara Brezza ZXI +', 'Maruti Alto K10 LXI',
       'Maruti Celerio VXI CNG OPT', 'Maruti Ciaz VXI PLUS',
       'Maruti Ciaz ZXI PLUS', 'Maruti S PRESSO VXI CNG',
       'Maruti Ciaz ZXI', 'Maruti S Cross SIGMA 1.5 SHVS',
       'Maruti Swift Dzire LXI 1.2 BS IV', 'Maruti Wagon R 1.0 LXI CNG',
       'Maruti Alto K10 LXI CNG (O)',
       'Maruti Vitara Brezza ZXI+ DUAL TONE', 'Maruti Alto K10 LXI CNG',
       'Maruti S Cross ALPHA SHVS', 'Maruti IGNIS ALPHA 1.2 K12 AMT',
       'Maruti Vitara Brezza VDI AMT', 'Maruti Ritz LXI',
       'Maruti Dzire LXI', 'Maruti Eeco 5 STR CNG WITH AC PLUSHTR',
       'Maruti Alto 800 LXI CNG', 'Maruti Swift Dzire LXI OPT',
       'Maruti Swift LDI', 'Maruti Ertiga VXI SMART HYBRID AT',
       'Maruti Ertiga ZDI', 'Maruti Swift Dzire VDI OPT',
       'Maruti Celerio VXI (O)', 'Maruti Vitara Brezza ZXI',
       'Maruti Baleno DELTA DDIS 190', 'Maruti Ciaz VDI+ SHVS',
       'Maruti Dzire ZXI', 'Maruti IGNIS ALPHA 1.2 K12 DUAL TONE',
       'Maruti Ciaz ZETA 1.5 SHVS VVT MT', 'Maruti S PRESSO LXI (O) CNG',
       'Maruti IGNIS ZETA 1.2 K12 DUAL TONE', 'Maruti XL6 ALPHA AT SHVS',
       'Maruti Swift Dzire VXI OPT', 'Maruti Swift VDI OPT',
       'Maruti Ertiga ZDI PLUS SHVS', 'Maruti S Cross ALPHA 1.3',
       'Maruti Swift Dzire VXI 1.2 BS IV', 'Maruti Swift VXI ABS',
       'Maruti Eeco 7 STR STD(O)', 'Maruti Ertiga VXI ABS',
       'Maruti Celerio X ZXI OPT AMT', 'Maruti Ciaz 1.5 SIGMA SHVS',
       'Maruti Swift Dzire VXI AMT', 'Maruti Ertiga VXI CNG',
       'Maruti Wagon R Stingray VXI OPT', 'Maruti Ciaz VXI',
       'Maruti Ciaz 1.5 ZETA VVT SHVS AT', 'Maruti Ciaz ZXI+ AT',
       'Maruti S PRESSO LXI CNG', 'Maruti Celerio X VXI OPT',
       'Maruti Ciaz ZDI SHVS', 'Maruti Alto LXI CNG (O)',
       'Maruti Wagon R 1.0 VXI PLUS', 'Maruti IGNIS ZETA 1.3 DDIS',
       'Maruti IGNIS ZETA 1.2 K12 AMT DUAL TONE',
       'Maruti Ertiga ZXI PLUS', 'Maruti Swift VDI AMT',
       'Maruti Ciaz DELTA 1.4 VVT', 'Maruti Wagon R 1.0 VXI ABS AIR BAG',
       'Maruti Wagon R 1.0 VXI+ OPT. AMT', 'Maruti S Cross ZETA SHVS',
       'Maruti Ertiga ZXI SMART HYBRID AT'],
 'Hyundai': ['Hyundai Creta EX MT', 'Hyundai Elite i20 Magna Executive 1.2',
       'Hyundai Creta 1.4 E PLUS CRDI', 'Hyundai i10 SPORTZ 1.1 IRDE2',
       'Hyundai i10 ERA 1.1 IRDE', 'Hyundai i20 SPORTZ 1.2 VTVT',
       'Hyundai AURA SX+ AT 1.2CRDI', 'Hyundai i10 MAGNA 1.2 KAPPA2',
       'Hyundai VENUE S MT 1.2 KAPPA',
       'Hyundai Grand i10 MAGNA 1.2 KAPPA VTVT',
       'Hyundai Elite i20 MAGNA 1.2', 'Hyundai NEW SANTRO 1.1 SPORTS AMT',
       'Hyundai VENUE S Plus MT 1.2 Kappa', 'Hyundai Creta 1.5 SX (O)',
       'Hyundai NEW SANTRO SPORTZ 1.1', 'Hyundai Xcent S 1.2',
       'Hyundai Elite i20 SPORTZ 1.2', 'Hyundai i20 MAGNA O 1.2',
       'Hyundai Grand i10 SPORTZ 1.2 KAPPA VTVT',
       'Hyundai VENUE SX 1.0 GDI IMT', 'Hyundai Eon ERA PLUS',
       'Hyundai Elite i20 ASTA 1.2 DUAL TONE', 'Hyundai Xcent S 1.2 OPT',
       'Hyundai i20 Active 1.2 S', 'Hyundai Verna 1.6 SX (O) CRDI MT',
       'Hyundai Grand i10 MAGNA 1.2 VTVT', 'Hyundai AURA S CNG',
       'Hyundai VENUE 1.0L Turbo GDI SX MT',
       'Hyundai GRAND I10 NIOS MAGNA 1.2 MT',
       'Hyundai Creta SX PETROL MT', 'Hyundai New Elantra 2.0 SX (O) MT',
       'Hyundai Elite i20 MAGNA+ VTVT',
       'Hyundai New Elantra SX PETROL AT',
       'Hyundai Elite i20 1.2 SPORTS PLUS VTVT CVT',
       'Hyundai Verna 1.6 SX VTVT', 'Hyundai Elite i20 ASTA 1.2 (O)',
       'Hyundai Xcent BASE 1.2', 'Hyundai New Elantra 2.0 SX AT',
       'Hyundai Elite i20 1.2 SPORTS PLUS VTVT',
       'Hyundai Elite i20 ASTA 1.2', 'Hyundai Grand i10 SPORTZ O 1.2',
       'Hyundai Creta 1.5 SX (O) AT',
       'Hyundai GRAND I10 NIOS SPORTZ PETROL',
       'Hyundai Grand i10 1.2 KAPPA SPORTZ (O)',
       'Hyundai Grand i10 Magna 1.2 AT  VTVT',
       'Hyundai Grand i10 SPORTZ AT 1.2 KAPPA VTVT',
       'Hyundai Grand i10 SPORTZ (O) 1.2 AT VTVT',
       'Hyundai Verna SX (O) TURBO GDI AT',
       'Hyundai Grand i10 Sportz(O) 1.2 MT',
       'Hyundai New Elantra 1.6 SX AT O', 'Hyundai Creta S PETROL MT',
       'Hyundai i10 SPORTZ 1.2', 'Hyundai i10 MAGNA 1.1 IRDE2',
       'Hyundai Elite i20 Sportz 1.2 VTVT MT',
       'Hyundai i10 SPORTZ 1.2 KAPPA2',
       'Hyundai i10 SPORTZ 1.2 AT KAPPA2', 'Hyundai Santro Xing GLS',
       'Hyundai i10 MAGNA 1.2', 'Hyundai Eon D LITE PLUS',
       'Hyundai GRAND I10 NIOS SPORTZ 1.2 AT',
       'Hyundai VENUE S MT Turbo GDI 1.0L', 'Hyundai Creta E',
       'Hyundai Verna FLUIDIC 1.6 SX VTVT', 'Hyundai Creta 1.6 SX CRDI',
       'Hyundai Creta 1.6 CRDI SX PLUS AUTO', 'Hyundai i20 SPORTZ 1.2 O',
       'Hyundai Eon MAGNA PLUS', 'Hyundai Creta 1.6 SX (O) VTVT',
       'Hyundai i10 SPORTZ 1.2 AT', 'Hyundai Creta 1.6 E + VTVT',
       'Hyundai Creta 1.6 SX VTVT (O) EXECUTIVE',
       'Hyundai Verna FLUIDIC 1.6 VTVT S', 'Hyundai Eon SPORTZ',
       'Hyundai Eon MAGNA', 'Hyundai Verna FLUIDIC 1.4 VTVT',
       'Hyundai Creta 1.6 VTVT SX AUTO', 'Hyundai Creta 1.4 S CRDI',
       'Hyundai Creta 1.6 SX PLUS AUTO PETROL',
       'Hyundai NEW SANTRO 1.1 ERA EXICUTIVE',
       'Hyundai Grand i10 ASTA 1.2 VTVT', 'Hyundai New Elantra 1.6 SX MT',
       'Hyundai Creta 1.6 SX PLUS PETROL',
       'Hyundai Grand i10 ASTA 1.1 (O) CRDI',
       'Hyundai Verna 1.6 SX VTVT AT (O)',
       'Hyundai NEW SANTRO 1.1 MAGNA MT',
       'Hyundai Creta 1.6 SX PLUS VTVT', 'Hyundai Creta 1.6 S',
       'Hyundai Verna FLUIDIC 1.6 SX CRDI',
       'Hyundai Grand i10 SPORTZ DUAL TONE 1.2 VTVT',
       'Hyundai Elite i20 ASTA 1.4 CRDI',
       'Hyundai Grand i10 ASTA 1.2 (O) VTVT',
       'Hyundai VENUE 1.0 TURBO GDI SX+ AT',
       'Hyundai Grand i10 ASTA 1.2 AT VTVT',
       'Hyundai Creta 1.6 SX PLUS DIESEL',
       'Hyundai NEW I20 ASTA (O) 1.0 TURBO GDI AT',
       'Hyundai Elite i20 Sportz(O) 1.4 CRDi MT',
       'Hyundai Verna FLUIDIC 1.6 SX CRDI OPT',
       'Hyundai Elite i20 1.2 SPORTZ PLUS DUAL TONE VTVT',
       'Hyundai Verna FLUIDIC 1.6 SX VTVT OPT',
       'Hyundai Verna FLUIDIC 1.6 CRDI S', 'Hyundai Elite i20 SPORTZ 1.4',
       'Hyundai GRAND I10 NIOS MAGNA 1.2 AT',
       'Hyundai Verna FLUIDIC 1.6 VTVT S (O)  MT',
       'Hyundai NEW SANTRO ASTA 1.1 MT',
       'Hyundai Grand i10 magna 1.2 crdi',
       'Hyundai GRAND I10 NIOS Asta Petrol',
       'Hyundai Grand i10 ASTA 1.2 KAPPA VTVT', 'Hyundai i20 ASTA 1.2',
       'Hyundai GRAND I10 NIOS SPORTZ MT DUAL TONE',
       'Hyundai Verna FLUIDIC 1.6 VTVT SX AT',
       'Hyundai Grand i10 SPORTS 1.2 VTVT',
       'Hyundai Elite i20 1.4 CRDI ASTA (O)', 'Hyundai i20 Active 1.2 SX',
       'Hyundai Verna 1.6 EX VTVT', 'Hyundai Xcent SX 1.2',
       'Hyundai Elite i20 SPORTZ (O) 1.2',
       'Hyundai Elite i20 Asta 1.2 CVT', 'Hyundai Verna 1.6 CRDI SX + AT',
       'Hyundai Verna 1.6 SX VTVT (O)', 'Hyundai Eon ERA PLUS (O)',
       'Hyundai Santro Xing GL PLUS', 'Hyundai Xcent SX 1.1 CRDI OPT',
       'Hyundai i20 MAGNA 1.2 VTVT', 'Hyundai Grand i10 MAGNA 1.1 CRDI',
       'Hyundai i10 SPORTZ 1.2 KAPPA2 O', 'Hyundai AURA S 1.2 Kappa MT',
       'Hyundai Elite i20 1.2 MAGNA PLUS VTVT',
       'Hyundai Xcent E PLUS CRDI', 'Hyundai Grand i10 ASTA 1.1 CRDI',
       'Hyundai NEW SANTRO ERA 1.1', 'Hyundai Grand i10 Sportz1.2 CRDI',
       'Hyundai Creta 1.6 SX (O) CRDI', 'Hyundai Xcent E PLUS',
       'Hyundai GRAND I10 NIOS 1.0 TURBO SPORTZ', 'Hyundai i20 ERA 1.2',
       'Hyundai Creta 1.6 EX CRDI', 'Hyundai NEW I20 SPORTZ 1.0T GDI IMT',
       'Hyundai Xcent S AT 1.2', 'Hyundai VENUE 1.0L Turbo GDI SX(O) MT',
       'Hyundai Elite i20 Magna Executive Diesel',
       'Hyundai Creta 1.6 E VTVT',
       'Hyundai Grand i10 ASTA 1.2 KAPPA VTVT OPT',
       'Hyundai Eon MAGNA PLUS OPTIONAL', 'Hyundai VENUE SX(O) CRDi',
       'Hyundai Grand i10 SPORTZ 1.1 CRDI',
       'Hyundai Grand i10 ASTA AT 1.2 KAPPA VTVT',
       'Hyundai Tucson New 2WD AT GLS PETROL', 'Hyundai Creta E PETROL',
       'Hyundai Creta 1.6 SX VTVT', 'Hyundai VENUE 1.5 SX CRDI MT',
       'Hyundai Verna SX(O) PETROL MT',
       'Hyundai NEW SANTRO 1.1 MAGNA CNG MT',
       'Hyundai NEW SANTRO SPORTZ EXECUTIVE CNG',
       'Hyundai VENUE 1.0 turbo GDI IMT SX O',
       'Hyundai Elite i20 ERA 1.2', 'Hyundai Tucson New 2WD MT PETROL',
       'Hyundai Verna FLUIDIC 1.6 VTVT S AT',
       'Hyundai Creta 1.6 S AT CRDI', 'Hyundai Creta 1.6 SX AT CRDI',
       'Hyundai VENUE 1.4 CRDI S MT',
       'Hyundai GRAND I10 NIOS SPORTZ CNG MT',
       'Hyundai GRAND I10 NIOS MAGNA 1.2 KAPPA VTVT CNG',
       'Hyundai Verna 1.6 SX VTVT AT', 'Hyundai Creta S CRDI',
       'Hyundai Verna S+ DIESEL MT', 'Hyundai Elite i20 Magna + CRDI',
       'Hyundai Verna FLUIDIC 1.6 EX CRDI',
       'Hyundai NEW SANTRO 1.1 MAGNA AMT',
       'Hyundai Verna SX (O) DIESEL MT',
       'Hyundai Grand i10 1.2 ASTA (O) AT', 'Hyundai Xcent S 1.1 CRDI',
       'Hyundai Creta SX PLUS 1.6 DUAL TONE VTVT',
       'Hyundai Eon MAGNA + 1.0 LITRE KAPPA',
       'Hyundai Elite i20 1.2  ASTA (O) CVT',
       'Hyundai Verna SX(O) PETROL AT', 'Hyundai i20 SPORTZ 1.4 CRDI',
       'Hyundai Xcent SX 1.2 OPT', 'Hyundai Accent EXECUTIVE',
       'Hyundai Eon ERA', 'Hyundai Creta 1.4 SX(O) TURBO GDI DCT',
       'Hyundai Santro Xing GL', 'Hyundai Verna 1.4 VTVT E',
       'Hyundai Eon MAGNA O', 'Hyundai i20 ASTA 1.4 CRDI',
       'Hyundai NEW SANTRO MAGNA 1.1 CORPORATE EDITION',
       'Hyundai GRAND I10 NIOS ASTA PETROL AMT',
       'Hyundai New Elantra S 1.8 MT',
       'Hyundai GRAND I10 NIOS ERA PETROL', 'Hyundai Verna 1.6 CRDI SX',
       'Hyundai NEW SANTRO SPORTZ AMT ANNIVERSARY EDITION',
       'Hyundai Elite i20 ASTA 1.2 AT',
       'Hyundai Tucson New 2WD AT GLS DIESEL',
       'Hyundai Tucson New 2WD AT GL DIESEL',
       'Hyundai Verna FLUIDIC 1.6 EX VTVT',
       'Hyundai Grand i10 ERA 1.2 VTVT',
       'Hyundai i10 ASTA 1.2 AT KAPPA2 WITH SUNROOF',
       'Hyundai AURA SX PETROL', 'Hyundai Verna EX 1.6 VTVT AT',
       'Hyundai VENUE 1.0 Turbo GDI DCT SX+DT',
       'Hyundai Elite i20 MAGNA 1.4 AT'],
 'Datsun': ['Datsun Redi Go T (O)', 'Datsun Redi Go 1.0 T(O) AT',
       'Datsun Go T', 'Datsun Go T(O)', 'Datsun Redi Go S',
       'Datsun Redi Go T', 'Datsun Redi Go A', 'Datsun Go Plus T',
       'Datsun Redi Go 1.0 S AT', 'Datsun Redi Go D',
       'Datsun Redi Go 1.0 T(O)', 'Datsun Go Plus T (O) CVT'],
 'Toyota': ['Toyota Innova 2.5 GX 8 STR BS IV', 'Toyota Glanza G MT PETROL',
       'Toyota YARIS VX CVT', 'Toyota Etios Liva G',
       'Toyota Corolla Altis J', 'Toyota Etios Liva G SP',
       'Toyota Etios Liva VX DUAL TONE', 'Toyota Corolla Altis GL',
       'Toyota Etios CROSS G', 'Toyota YARIS V CVT',
       'Toyota Glanza G CVT', 'Toyota YARIS G CVT', 'Toyota YARIS J MT',
       'Toyota Glanza V CVT', 'Toyota Etios V',
       'Toyota Fortuner 3.0 MT 4X2', 'Toyota YARIS J CVT',
       'Toyota YARIS G MT', 'Toyota YARIS V MT',
       'Toyota Innova Crysta 2.7 ZX AT 7 STR',
       'Toyota Corolla Altis G AT', 'Toyota Etios Liva V',
       'Toyota Etios G', 'Toyota Corolla Altis VL AT',
       'Toyota Innova Crysta 2.8 ZX AT 7 STR', 'Toyota Etios V SP',
       'Toyota Innova Crysta 2.4 ZX AT', 'Toyota Fortuner 2.8 4x2 AT',
       'Toyota Etios Liva D-4D VXD', 'Toyota Etios Liva V DUAL TONE',
       'Toyota Corolla Altis G', 'Toyota Etios CROSS 1.5 V',
       'Toyota Innova Crysta 2.4 GX 7 STR', 'Toyota Camry 2.5 AT',
       'Toyota Etios Liva GD', 'Toyota Etios Liva V  LIMITED',
       'Toyota Camry HYBRID'],
 'Mahindra': ['Mahindra Thar LX PETROL AT 4WD', 'Mahindra XUV500 W11 AT',
       'Mahindra XUV 3OO W8(O)', 'Mahindra Thar LX  4WD AT CONVERTIBLE',
       'Mahindra Thar LX D AT 4WD HT', 'Mahindra XUV500 W11',
       'Mahindra XUV500 W10 FWD', 'Mahindra XUV500 W7 FWD',
       'Mahindra XUV500 W11 (O)', 'Mahindra XUV500 W8 FWD AT',
       'Mahindra XUV500 W6 4X2', 'Mahindra XUV500 W9',
       'Mahindra Scorpio S10', 'Mahindra XUV500 W11 FWD',
       'Mahindra TUV300 T8', 'Mahindra XUV500 W10 AT',
       'Mahindra MARAZZO M8 7 STR', 'Mahindra KUV 100 NXT K4+ 6S',
       'Mahindra Kuv100 K8 5 STR', 'Mahindra Kuv100 K4 D 5 STR',
       'Mahindra Thar CRDE 4X4 BS IV', 'Mahindra XUV500 W8 FWD',
       'Mahindra Scorpio S11', 'Mahindra Kuv100 K8 6 STR',
       'Mahindra Bolero Power+ ZLX', 'Mahindra KUV 100 NXT K8 6 STR',
       'Mahindra KUV 100 NXT K2+6 STR', 'Mahindra XUV500 W5',
       'Mahindra XUV500 W9 AT', 'Mahindra Thar LX D 4*4 MT HT',
       'Mahindra Scorpio S9', 'Mahindra Scorpio S7 120 BHP 2WD',
       'Mahindra KUV 100 NXT K2 D 6S',
       'Mahindra Scorpio S10 Intelli Hybrid 2WD',
       'Mahindra Scorpio S3 2WD', 'Mahindra Kuv100 K4 6 STR',
       'Mahindra Thar LX D 4WD AT CONVERTIBLE', 'Mahindra TUV300 T8 AT',
       'Mahindra Bolero POWER+ SLE', 'Mahindra Thar CRDI 4x4 6str AC',
       'Mahindra TUV300 T 10 MT Dual Tone',
       'Mahindra KUV 100 NXT K6+ 6 STR', 'Mahindra Bolero B4',
       'Mahindra Bolero B6 (O)', 'Mahindra XUV500 W11 (O) AT',
       'Mahindra XUV500 W10 AT FWD', 'Mahindra XUV 3OO W8 (O) DIESEL  AT',
       'Mahindra Kuv100 K6 D 6 STR', 'Mahindra Kuv100 K8 D 5 STR',
       'Mahindra XUV500 W6 AT'],
 'KIA': ['KIA SELTOS HTX 1.5 PETROL MT', 'KIA SELTOS HTK PLUS 1.5 DIESEL',
       'KIA SELTOS 1.4 GTX+ TURBO GDI PETROL AT',
       'KIA SELTOS HTK 1.5 PETROL', 'KIA SELTOS GTX 1.4 GDI AT PETROL',
       'KIA SELTOS GTX+ 1.4 MT', 'KIA SELTOS 1.5 GTX+ AT',
       'KIA SELTOS HTX 1.5 DIESEL', 'KIA SELTOS HTK PLUS 1.5 PETROL',
       'KIA SELTOS GTX 1.4 GDI PETROL', 'KIA SELTOS GTX + AT PETROL',
       'KIA SELTOS GTK 1.4 GDI PETROL', 'KIA SELTOS HTX+ AT 1.5 DIESEL',
       'KIA SELTOS HTK PLUS AT 1.5 DIESEL',
       'KIA SELTOS HTE 1.5 MT DIESEL', 'KIA SELTOS HTX+ MT 1.5 DIESEL'],
 'Volkswagen': ['Volkswagen Polo HIGHLINE1.2L PETROL',
       'Volkswagen Polo GT TSI 1.2 PETROL AT',
       'Volkswagen Vento HIGHLINE PETROL',
       'Volkswagen Vento HIGHLINE DIESEL',
       'Volkswagen Polo HIGH LINE PLUS 1.0',
       'Volkswagen Vento COMFORTLINE 1.0 TSI MT',
       'Volkswagen Vento HIGHLINE TDI AT',
       'Volkswagen Vento 1.2 TSI HIGHLINE PLUS AT',
       'Volkswagen Vento HIGHLINE PLUS TDI AT',
       'Volkswagen Beetle 2.0 AT',
       'Volkswagen Polo COMFORTLINE 1.2L PETROL',
       'Volkswagen Polo Trendline 1.0 L Petrol',
       'Volkswagen Vento HIGHLINE 1.2 TSI AT',
       'Volkswagen Polo COMFORTLINE 1.0 PETROL',
       'Volkswagen Ameo HIGHLINE 1.5L AT (D)',
       'Volkswagen Ameo COMFORTLINE 1.2',
       'Volkswagen Vento HIGHLINE PETROL AT',
       'Volkswagen Vento HIGHLINE PLUS 1.0 TSI AT',
       'Volkswagen TIGUAN HIGHLINE A/T',
       'Volkswagen Ameo HIGHLINE PLUS DSG 1.5',
       'Volkswagen Polo COMFORTLINE  CUP EDITION',
       'Volkswagen Polo HIGHLINE1.2L DIESEL',
       'Volkswagen Polo HIGHLINE1.5L DIESEL',
       'Volkswagen Vento COMFORTLINE MT PETROL',
       'Volkswagen Polo GT TSI AT 1.0', 'Volkswagen Ameo COMFORTLINE 1.0',
       'Volkswagen Vento COMFORTLINE TDI AT',
       'Volkswagen Ameo HIGHLINE 1.2',
       'Volkswagen Vento Highline 1.0 TSI MT',
       'Volkswagen Polo COMFORTLINE 1.2L DIESEL',
       'Volkswagen Polo COMFORTLINE PLUS 1.0'],
 'Renault': ['Renault Kwid RXL', 'Renault Kwid 1.0 CLIMBER OPT',
       'Renault Duster RXL PETROL 104', 'Renault Kwid 1.0 RXT Opt',
       'Renault Duster RXZ 85 PS DIESEL', 'Renault TRIBER 1.0 RXZ',
       'Renault Captur RXT PETROL', 'Renault Duster RXS 106 PS MT',
       'Renault Pulse RX Z DIESEL', 'Renault Duster RXS 85 PS',
       'Renault Pulse RX L PETROL', 'Renault Duster 85 PS RXL',
       'Renault TRIBER RXZ AT', 'Renault TRIBER 1.0 RXT',
       'Renault Kwid RXT Opt', 'Renault Kwid 1.0 RXT Opt AT',
       'Renault Duster RXS CVT 106 PS', 'Renault Kwid RXT 1.0 EASY-R  AT',
       'Renault Duster RXZ DIESEL 110',
       'Renault Duster RXL PLUS DIESEL 85', 'Renault Kwid CLIMBER 1.0',
       'Renault Duster RXZ AMT 110 PS', 'Renault Duster RXZ 110 4WD',
       'Renault Kwid CLIMBER 1.0 AT', 'Renault Duster RXL AMT 110 PS',
       'Renault Captur RXT Diesel Dual Tone', 'Renault Captur 1.5 RXL MT',
       'Renault Captur RXE diesel (MT)',
       'Renault Captur PLATINE PETROL DUAL TONE', 'Renault Duster RXZ',
       'Renault Kwid 1.0 RXT', 'Renault Kwid RXT',
       'Renault Kwid 1.0 RXT AT', 'Renault Kwid 1.0 CLIMBER OPT AMT',
       'Renault Kwid RXT 1.0 EASY-R AT OPTION',
       'Renault Captur RXE PETROL MT', 'Renault Duster RXZ CVT 1.3 TURBO',
       'Renault TRIBER RXE MT PETROL', 'Renault Duster RXZ 1.3 TURBO MT',
       'Renault Duster 85 PS RXE DIESEL ADVENTURE',
       'Renault TRIBER 1.0 RXL PETROL',
       'Renault Captur PLATINE DUAL TONE',
       'Renault Kwid 1.0 RXT 02 Anniversary Edition',
       'Renault Pulse RX L DIESEL',
       'Renault Kwid RXT 02 Anniversary Edition',
       'Renault Kwid Neotech 1.0 EasyR'],
 'Ford': ['Ford Ecosport 1.5TITANIUM TDCI',
       'Ford Ecosport 1.0 ECOBOOST TITANIUM',
       'Ford Ecosport 1.0 TREND+ (ECOBOOST)',
       'Ford Ecosport 1.5 TITANIUM TI VCT',
       'Ford Endeavour 2.2l 4X2 MT Titanium',
       'Ford Figo Aspire 1.2 TITANIUM PETROL',
       'Ford FREESTYLE TITANIUM 1.2 TI-VCT MT',
       'Ford Ecosport 1.5 TITANIUM PLUS TI VCT AT',
       'Ford Ecosport 1.5 TDCI TITANIUM PLUS',
       'Ford Ecosport 1.5 TITANIUM TI VCT AT',
       'Ford Ecosport 1.5 TREND TI VCT', 'Ford Ecosport TITANIUM+ SPORT',
       'Ford Ecosport 1.5 TREND+ TDCI',
       'Ford Ecosport 1.5 TITANIUM SIGNATURE TI VCT (SUNROOF)',
       'Ford Ecosport 1.5 TITANIUM THUNDER EDITION TI VCT',
       'Ford FREESTYLE TITANIUM Plus 1.5 TDCI MT',
       'Ford Figo 1.2 ZXI DURATEC',
       'Ford Ecosport 1.5  TITANIUM SPORTS(SUNROOF)',
       'Ford Ecosport 1.0 ECOBOOST TITANIUM SPORTS(SUNROOF)',
       'Ford Ecosport 1.5AMBIENTE TI VCT', 'Ford Ecosport 1.5 TREND TDCI',
       'Ford Ecosport 1.5 TITANIUMTDCI OPT', 'Ford New Figo 1.2 TITANIUM',
       'Ford Figo Aspire 1.2 TITANIUM+ PETROL',
       'Ford FREESTYLE TITANIUM + 1.2 TI-VCT',
       'Ford Ecosport 1.5 AMBIENTE TDCI',
       'Ford New Figo 1.5 TITANIUM AT PETROL',
       'Ford Ecosport 1.0 ECOBOOST TITANIUM OPT',
       'Ford FREESTYLE TITANIUM 1.5 TDCI',
       'Ford Ecosport TITANIUM PLUS 1.5 TI VCT MT',
       'Ford Ecosport TITANIUM PLUS SE1.5 TDCI MT',
       'Ford Ecosport TREND + 1.5 TI VCT AT',
       'Ford Endeavour 3.2l 4X4 AT Titanium',
       'Ford Figo Aspire 1.2 Trend+ Petrol',
       'Ford Figo Aspire TREND PLUS CNG'],
 'MG': ['MG HECTOR SHARP 2.0 DIESEL', 'MG HECTOR SHARP DCT PETROL',
       'MG HECTOR SHARP HYBIRD PETROL MT',
       'MG HECTOR SUPER HYBRID PETROL MT', 'MG HECTOR SHARP CVT',
       'MG HECTOR PLUS SHARP HYBRID PETROL', 'MG HECTOR SUPER DIESEL',
       'MG HECTOR SMART DCT PETROL'],
 'Jeep': ['Jeep Compass SPORT PLUS PETROL', 'Jeep Compass 2.0 LONGITUDE',
       'Jeep Compass 2.0 LONGITUDE (O)', 'Jeep Compass 2.0 SPORT',
       'Jeep Compass 2.0 LIMITED 4*2', 'Jeep Compass 1.4 LIMITED PLUS AT',
       'Jeep Compass LIMITED 2.0 4*4',
       'Jeep Compass LIMITED PLUS 2.0 4*4', 'Jeep Compass 2.0 LIMITED',
       'Jeep Compass LIMITED O 1.4 AT',
       'Jeep Compass 1.4  LONGITUDE (O) AT',
       'Jeep Compass LIMITED 1.4 AT', 'Jeep Compass LIMITED (O) 2.0',
       'Jeep Compass 2.0 SPORT PLUS', 'Jeep Compass LIMITED (O) 2.0 4*4'],
 'Honda': ['Honda Brio 1.2 S MT I VTEC', 'Honda City ZX MT PETROL',
       'Honda Jazz 1.2 V MT', 'Honda City SV MT EDGE EDITION PETROL',
       'Honda City VX CVT PETROL', 'Honda Civic VX CVT i-VTEC',
       'Honda City V MT PETROL', 'Honda Amaze 1.2 V MT I-VTEC',
       'Honda Amaze 1.2 SMT I VTEC', 'Honda City ZX CVT',
       'Honda Jazz VX 1.2 CVT', 'Honda City V CVT',
       'Honda Brio 1.2 VX AT I VTEC', 'Honda WR-V 1.2 i-VTEC VX MT',
       'Honda WR-V 1.2 i-VTEC S MT', 'Honda Civic ZX CVT PETROL',
       'Honda Amaze 1.2 EMT I VTEC', 'Honda Brio 1.2 S (O) MT IVTEC',
       'Honda Amaze 1.2 VX CVT I VTEC', 'Honda Amaze 1.2 VXMT I VTEC',
       'Honda Amaze 1.2 V CVT I VTEC', 'Honda Jazz 1.2 VX I-VTECH',
       'Honda Jazz 1.2 BASE I VTEC', 'Honda Brio 1.2 VX MT I VTEC',
       'Honda WR-V 1.5 i-DTEC VX MT', 'Honda Jazz 1.2 VX AT',
       'Honda City SV MT PETROL', 'Honda Jazz VX  1.2',
       'Honda Amaze 1.2 SX MT I VTEC', 'Honda Brio 1.2 V MT I VTEC',
       'Honda Jazz 1.2 SV MT', 'Honda Amaze 1.2 EXMT I VTEC',
       'Honda City VX MT PETROL', 'Honda City V MT DIESEL',
       'Honda Jazz 1.2 V AT', 'Honda Civic 1.8V MT SUN ROOF',
       'Honda City S MT PETROL', 'Honda Amaze 1.5 EMT I DTEC',
       'Honda City SV MT DIESEL', 'Honda Amaze 1.5 V I-D TECH',
       'Honda Amaze 1.5 SMT I DTEC', 'Honda City VX (O) PETROL',
       'Honda Jazz 1.2 ZX MT', 'Honda BR-V 1.5 i-VTEC V CVT',
       'Honda City V AT', 'Honda Jazz 1.2 E MT', 'Honda Jazz 1.2 S MT',
       'Honda Amaze 1.2 SAT I VTEC', 'Honda Amaze 1.5 V CVT I-DTEC',
       'Honda BR-V 1.5 i- DTEC S', 'Honda City CORPORATE MT',
       'Honda Amaze 1.2 S (O) MT I VTEC', 'Honda Amaze 1.5 SMT O I DTEC',
       'Honda City ZX CVT ANNIVERSARY EDITION',
       'Honda Jazz VX EXCLUSIVE CVT', 'Honda Jazz 1.5 VX I DTEC',
       'Honda City SV CVT PETROL', 'Honda WR-V 1.2 SV MT',
       'Honda CRV 2.4 AT 4WD AVN', 'Honda City 1.5 E MT PETROL',
       'Honda Jazz 1.5 V I DTEC', 'Honda Jazz ZX CVT',
       'Honda Jazz 1.2 SELECT I VTEC', 'Honda Brio 1.2 E MT I VTEC',
       'Honda Civic 1.8V AT', 'Honda City ZX MT I-DTEC',
       'Honda CRV 2.0 2WD AT', 'Honda Amaze VX AT I DTEC'],
 'Tata': ['Tata TIGOR XZ 1.2 REVOTRON', 'Tata Harrier XM 2.0L Kryotec',
       'Tata NEXON XM 1.2', 'Tata Harrier XZ 2.0L Kryotec',
       'Tata NEXON XZ+ SUNROOF 1.2 RTN', 'Tata Harrier XZA+ DARK EDITION',
       'Tata NEXON XZ+ OPT PETROL', 'Tata Safari XT 2.0 KRYOTEC',
       'Tata Tiago XZA 1.2 REVOTRON', 'Tata Tiago 1.2 XT (O)',
       'Tata NEXON KRAZ AT PETROL', 'Tata Hexa Varicor 400 XTA',
       'Tata Tiago XZ+ 1.2 Revotron', 'Tata NEXON XZ+ 1.5',
       'Tata Tiago XZ 1.2 REVOTRON', 'Tata Tiago XT 1.2 REVOTRON',
       'Tata ALTROZ XZ 1.2', 'Tata NEXON XZ+ 1.5 Dual Tone',
       'Tata Harrier XZ+ DARK EDITION', 'Tata Tiago XM 1.2 REVOTRON',
       'Tata NEXON XZ+ SUNROOF 1.5 RTQ', 'Tata NEXON XZA+ 1.2 Dual Tone',
       'Tata NEXON XZA + 1.2 PETROL A/T', 'Tata NEXON XZA+ 1.5',
       'Tata NEXON XZ+ 1.2 DUAL TONE', 'Tata TIGOR XZ 1.2 REVOTRON OPT',
       'Tata NEXON XZA+ 1.5 Dual Tone', 'Tata NEXON XZ+ 1.2',
       'Tata Hexa Varicor 400 XT', 'Tata NEXON XM SUNROOF 1.2 RTN',
       'Tata Safari 4X2 GX DICOR 2.2 VTT', 'Tata Zest XT RT',
       'Tata Harrier XZ +', 'Tata Tiago XE 1.05 REVOTORQ',
       'Tata TIGOR 1.2 XZA+ RTN', 'Tata Harrier XT 2.0L Kryotec',
       'Tata ALTROZ XE 1.2', 'Tata Tiago XE 1.2 REVOTRON',
       'Tata NEXON XMA SUNROOF 1.2 RTN', 'Tata Safari XT+2.0 KRYOTEC',
       'Tata Safari XM 2.0 KRYOTEC', 'Tata Harrier XT+ 2.0 KRYOTEC',
       'Tata Harrier XZA  DUAL TONE', 'Tata Harrier XZ DARK EDITION',
       'Tata NEXON XMA 1.5', 'Tata Hexa Varicor 400 XMA',
       'Tata Hexa Varicor 400 XT 4X4', 'Tata NEXON XT1.5',
       'Tata TIGOR XZ PLUS 1.2 REVOTRON', 'Tata Harrier XZ DUAL TONE',
       'Tata NEXON XZA+ (O) 1.5  RTQ', 'Tata Harrier XZA',
       'Tata NEXON XM 1.5', 'Tata TIGOR Revotron XE',
       'Tata Tiago XZA+ 1.2 RTN', 'Tata Tiago XT 1.05 REVOTORQ',
       'Tata NEXON XMA 1.2', 'Tata NEXON XZA+ (O) 1.2 RTN',
       'Tata Harrier XZA + DUAL TONE'],
 'Skoda': ['Skoda Octavia LK 1.8 TSI AT', 'Skoda Rapid AMBITION 1.6 MPFI MT',
       'Skoda Rapid ACTIVE 1.0 TSI', 'Skoda Rapid ELEGANCE 1.6 MPFI MT',
       'Skoda Rapid ELEGANCE 1.6 TDI MT', 'Skoda Rapid STYLE 1.6 MPI MT',
       'Skoda Rapid 1.6 MPI STYLE AT', 'Skoda Rapid STYLE 1.0TSI MT',
       'Skoda Rapid AMBITION 1.6 MPFI AT',
       'Skoda Octavia STYLE 2.0 TDI MT', 'Skoda Rapid 1.6 MPI MT ACTIVE',
       'Skoda Rapid Style 1.5 TDI AT', 'Skoda Rapid 1.0 AMBITION TSI AT',
       'Skoda Rapid ELEGANCE 1.6 MPFI AT',
       'Skoda Rapid MONTE CARLO 1.0 MT', 'Skoda Octavia 2.0 L&K AT TDI',
       'Skoda Rapid 1.5 TDI MT AMBITION',
       'Skoda Rapid 1.5 TDI AT AMBITION',
       'Skoda Octavia 2.0 TDI STYLE PLUS AT', 'Skoda Rapid STYLE 1.0 AT',
       'Skoda Rapid 1.0 AMBITION TSI MT'],
 'Nissan': ['Nissan Terrano XL 85 PS DEISEL', 'Nissan Terrano XL P',
       'Nissan Micra XV CVT', 'Nissan Micra Active XV',
       'Nissan Micra Active XV S', 'Nissan Micra XL CVT (PETROL)',
       'Nissan Terrano XV D THP PREMIUM 110 PS',
       'Nissan Terrano XL 110 DIESEL'],
 'Mercedes': ['Mercedes Benz C Class C 200 AVANTGARDE',
       'Mercedes Benz E Class E 220 CDI ELEGANCE',
       'Mercedes Benz E Class E 250 CDI ELEGANCE',
       'Mercedes Benz C Class 250 CDI ELEGANCE AT'],
 'Jaguar': ['Jaguar XF 2.2 DIESEL'],
 'Audi': ['Audi A6 35 TDI S LINE', 'Audi Q5 30 TDI PREMIUM PLUS',
       'Audi A6 2.0 TDI TECHNOLOGY', 'Audi Q3 35TDI PREMIUM',
       'Audi A3 35TDI PREMIUM', 'Audi Q3 30 TDI MT S EDITION'],
 'BMW': ['BMW 5 Series 520D 2.0']
}

left_column,mid_column,right_column= st.columns(3)
if options == "Prediction":
    def user_input():
        with left_column:
            year = st.selectbox("year:",options = sorted(df["year"].unique()))
            city = st.selectbox("city:", options=sorted(df["city"].unique()))
            ownernumber = st.selectbox("ownernumber:", options=sorted(df["ownernumber"].unique()))
            discountprice = st.number_input("Select Discount Price:")
            
        with mid_column:
            brand = st.selectbox("Car Name:",options = sorted(df["make"].unique()))
            fueltype = st.selectbox("Fuel Type:", options=sorted(df["fueltype"].unique()))
            transmission = st.selectbox("Car Transmission:", options=sorted(df["transmission"].unique()))
            benefits = st.number_input("Select Benefits:")
        
        model_name = ""
        if brand == "Maruti":
            model_name = cars.get("Maruti")
        elif brand == "Hyundai":
            model_name = cars.get("Hyundai")
        elif brand == "Datsun":
            model_name = cars.get("Datsun")
        elif brand == "Toyota":
            model_name = cars.get("Toyota")
        elif brand == "Mahindra":
            model_name = cars.get("Mahindra")
        elif brand == "KIA":
            model_name = cars.get("KIA")
        elif brand == "Volkswagen":
            model_name = cars.get("Volkswagen")
        elif brand == "Renault":
            model_name = cars.get("Renault")
        elif brand == "Ford":
            model_name = cars.get("Ford")
        elif brand == "MG":
            model_name = cars.get("MG")
        elif brand == "Jeep":
            model_name = cars.get("Jeep")
        elif brand == "Honda":
            model_name = cars.get("Honda")
        elif brand == "Tata":
            model_name = cars.get("Tata")
        elif brand == "Skoda":
            model_name = cars.get("Skoda")
        elif brand == "Nissan":
            model_name = cars.get("Nissan")
        elif brand == "Mercedes":
            model_name = cars.get("Mercedes")
        elif brand == "Jaguar":
            model_name = cars.get("Jaguar")
        elif brand == "Audi":
            model_name = cars.get("Audi")
        elif brand == "BMW":
            model_name = cars.get("BMW")

        with right_column:
            model = st.selectbox("Model Name:", options=model_name)
            kilometerdriven = st.number_input("Select Km Driven:")
            bodytype = st.selectbox("Body Type:", options=sorted(data["bodytype"].unique()))
        
        new_data = {"year":year,
                "brand":brand,
                "model":model,
                "city":city,
                "fueltype":fueltype,
                "kilometerdriven":kilometerdriven,
                "ownernumber":ownernumber,
                "transmission":transmission,
                "bodytype":bodytype,
                "discountprice":discountprice,
                "benefits":benefits
                }
        features = pd.DataFrame(new_data,index = [0])
        return features

    button = st.button("Predict Price")
    df = user_input()

    model=pickle.load(open('model_linear.pkl','rb'))

    prediction = model.predict(df)
    pred = np.round(prediction,2)
    predic = abs(pred)
    predict = str(predic).lstrip('[').rstrip(']')
    button_clicked = False

    if button:
        button_clicked = True
        st.subheader(f"Car Price: ₹{predict}")




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

# Data Visualization"""
for col in df1.columns:
    print('{} : {}'.format(col,df1[col].unique()))

# *************************************************************
# Univariant plots
# Price
plt.figure(figsize=(15,8))
sns.distplot(df1['price'])
print("skewness: %f" % df1['price'].skew())
print("kurtosis: %f" % df1['price'].kurt())

plt.figure(figsize=(15,4))
sns.boxplot(x='price',data=df1)

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

# *************************************************************
# Bi-variant plots
# Fuel Type
f, ax = plt.subplots(figsize=(15, 8))
fig = sns.boxplot(x="fueltype", y="price", data=df1)
#fig

# Year
f, ax = plt.subplots(figsize=(15, 5))
fig = sns.boxplot(x="year", y="price", data=df1)
#fig;
plt.xticks(rotation=90);

# Owner_Type
fig, ax = plt.subplots()
#fig
sns.stripplot(x = "ownernumber", y ='price', data = df1)

# City vs Price
sns.catplot(y='price',x="city",data=df1.sort_values('price',ascending=False),kind="boxen",height=5, aspect=3)
#plt.show

# Transmission vs Price
sns.catplot(y='price',x="transmission",data=df1.sort_values('price',ascending=False),kind="boxen",height=5, aspect=2)
#plt.show

# Make vs Price
sns.catplot(y='price',x="make",data=df1.sort_values('price',ascending=False),kind="boxen",height=5, aspect=3)
#plt.show

fig3 = px.sunburst(df1, path=['city', 'fueltype'], color='city',height=800)
fig3.update_layout(title_text="Fuel Type for City (Two-level Sunburst Diagram)", font_size=10)
fig3.show()

fig4 = px.treemap(df1, path=['city', 'make'], color='city',height=800,width=1500)
fig4.update_layout(title_text="Manufacture Distribution for City", font_size=10)
fig4.show()

fig = px.histogram(df1, x="year", y="price",color='city', barmode='group',height=400,width=1500)
fig.update_layout(title_text="Yearly City Growth", font_size=10)
fig.show()

# *************************************************************
# Label Encoding
df1.head()

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

df8.corr()

plt.figure(figsize=(15,10))
sns.heatmap(df8.corr(),annot=True,cmap='RdYlGn')
plt.show()

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

# Support vector machine
svr = SVR()
svr.fit(train, target)
acc_model(1,svr,train,test)

# Linear SVR
linear_svr = LinearSVR()
linear_svr.fit(train, target)
acc_model(2,linear_svr,train,test)

# MLPRegressor
mlp = MLPRegressor()
param_grid = {'hidden_layer_sizes': [i for i in range(2,20)],
              'activation': ['relu'],
              'solver': ['adam'],
              'learning_rate': ['constant'],
              'learning_rate_init': [0.01],
              'power_t': [0.5],
              'alpha': [0.0001],
              'max_iter': [1000],
              'early_stopping': [True],
              'warm_start': [False]}
mlp_GS = GridSearchCV(mlp, param_grid=param_grid, 
                   cv=10, verbose=True, pre_dispatch='2*n_jobs')
mlp_GS.fit(train, target)
acc_model(3,mlp_GS,train,test)

# Stochastic Gradient Descent
sgd = SGDRegressor()
sgd.fit(train, target)
acc_model(4,sgd,train,test)

# Decision Tree Regressor
decision_tree = DecisionTreeRegressor()
decision_tree.fit(train, target)
acc_model(5,decision_tree,train,test)

# Random Forest
random_forest = RandomForestRegressor()
random_forest.fit(train, target)
acc_model(6,random_forest,train,test)

# XGB
xgb_clf = xgb.XGBRegressor(objective ='reg:squarederror', verbosity = 0, silent=True, random_state=42) 
parameters = {'n_estimators': [60, 100, 120, 140], 
              'learning_rate': [0.01, 0.1],
              'max_depth': [5, 7],
              'reg_lambda': [0.5]}
xgb_reg = GridSearchCV(estimator=xgb_clf, param_grid=parameters, cv=5, n_jobs=-1).fit(trainb, targetb)
print("Best score: %0.3f" % xgb_reg.best_score_)
print("Best parameters set:", xgb_reg.best_params_)
acc_boosting_model(7,xgb_reg,trainb,testb)

# LGBM
Xtrain, Xval, Ztrain, Zval = train_test_split(trainb, targetb, test_size=0.2, random_state=0)
train_set = lgb.Dataset(Xtrain, Ztrain, silent=False)
valid_set = lgb.Dataset(Xval, Zval, silent=False)
params = {
        'boosting_type':'gbdt',
        'objective': 'regression',
        'num_leaves': 31,
        'learning_rate': 0.01,
        'max_depth': -1,
        'subsample': 0.8,
        'bagging_fraction' : 1,
        'max_bin' : 5000 ,
        'bagging_freq': 20,
        'colsample_bytree': 0.6,
        'metric': 'rmse',
        'min_split_gain': 0.5,
        'min_child_weight': 1,
        'min_child_samples': 10,
        'scale_pos_weight':1,
        'zero_as_missing': False,
        'seed':0,        
    }
modelL = lgb.train(params, train_set = train_set, num_boost_round=10000,
                   early_stopping_rounds=8000,verbose_eval=500, valid_sets=valid_set)

acc_boosting_model(8,modelL,trainb,testb,modelL.best_iteration)

fig =  plt.figure(figsize = (5,5))
axes = fig.add_subplot(111)
lgb.plot_importance(modelL,ax = axes,height = 0.5)
plt.show();

# GradientBoostingRegressor with HyperOpt
def hyperopt_gb_score(params):
    clf = GradientBoostingRegressor(**params)
    current_score = cross_val_score(clf, train, target, cv=10).mean()
    print(current_score, params)
    return current_score 
 
space_gb = {
            'n_estimators': hp.choice('n_estimators', range(100, 1000)),
            'max_depth': hp.choice('max_depth', np.arange(2, 10, dtype=int))            
        }
 
best = fmin(fn=hyperopt_gb_score, space=space_gb, algo=tpe.suggest, max_evals=10)
print('best:')
print(best)

params = space_eval(space_gb, best)

# Gradient Boosting Regression
gradient_boosting = GradientBoostingRegressor(**params)
gradient_boosting.fit(train, target)
acc_model(9,gradient_boosting,train,test)

# Ridge Regressor
ridge = RidgeCV(cv=5)
ridge.fit(train, target)
acc_model(10,ridge,train,test)

# Bagging Regressor
bagging = BaggingRegressor()
bagging.fit(train, target)
acc_model(11,bagging,train,test)

# Extra Trees Regressor
etr = ExtraTreesRegressor()
etr.fit(train, target)
acc_model(12,etr,train,test)

# AdaBoost Regressor
Ada_Boost = AdaBoostRegressor()
Ada_Boost.fit(train, target)
acc_model(13,Ada_Boost,train,test)

# Voting Regressor
Voting_Reg = VotingRegressor(estimators=[('lin', linreg), ('ridge', ridge), ('sgd', sgd)])
Voting_Reg.fit(train, target)
acc_model(14,Voting_Reg,train,test)

# *************************************************************
"""# Models comparison"""
models = pd.DataFrame({
    'Model': ['Linear Regression', 'Support Vector Machines', 'Linear SVR', 
              'MLPRegressor', 'Stochastic Gradient Decent', 
              'Decision Tree Regressor', 'Random Forest',  'XGB', 'LGBM',
              'GradientBoostingRegressor', 'RidgeRegressor', 'BaggingRegressor', 'ExtraTreesRegressor', 
              'AdaBoostRegressor', 'VotingRegressor'],
    
    'r2_train': acc_train_r2,
    'r2_test': acc_test_r2,
    'd_train': acc_train_d,
    'd_test': acc_test_d,
    'rmse_train': acc_train_rmse,
    'rmse_test': acc_test_rmse
                     })

pd.options.display.float_format = '{:,.2f}'.format

print('Prediction accuracy for models by R2 criterion - r2_test')
models.sort_values(by=['r2_test', 'r2_train'], ascending=False)

print('Prediction accuracy for models by RMSE - rmse_test')
models.sort_values(by=['rmse_test', 'rmse_train'], ascending=True)

# Model Output - Visualization
# Plot
fig200 = plt.figure(figsize=[20,8])
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
plt.show()
st.write(fig200)

# Plot
fig201 = plt.figure(figsize=[20,8])
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
plt.show()
st.write(fig201)

"""Thus, the best model is Linear Regression."""

# *************************************************************
# Prediction
#For models from Sklearn
testn = pd.DataFrame(scaler.transform(test0), columns = test0.columns)

#Ridge Regressor model for basic train
ridge.fit(train0, train_target0)
#ridge.predict(testn)[:3]

#xgb model for basic train
xgb_reg.fit(train0, train_target0)
#xgb_reg.predict(testn)[:3]

#Ada_Boost  model for basic train
Ada_Boost.fit(train0, train_target0)
#Ada_Boost.predict(testn)[:3]

#Voting Regressor model for basic train
Voting_Reg.fit(train0, train_target0)
#Voting_Reg.predict(testn)[:3]

#svr model for basic train
svr.fit(train0, train_target0)
#svr.predict(testn)[:3]

# *************************************************************
# Creating Dashboard
# import pywedge as pw
# mc = pw.Pywedge_Charts(df1, c = None, y = 'price') # use c if you want to remove any column
# charts = mc.make_charts()

