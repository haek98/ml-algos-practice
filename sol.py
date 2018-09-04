# -*- coding: utf-8 -*-
"""
Created on Sat May 26 18:56:03 2018

@author: ASUS
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from xgboost.sklearn import XGBRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.cross_validation import train_test_split
from catboost import CatBoostRegressor
dataset1=pd.read_csv('train.csv')
dataset2=pd.read_csv('test.csv')
dataset3=pd.read_csv('test.csv')
dataset1.BsmtCond.unique()
dataset2.isnull().sum()  
droplist=['Alley','Id']
dataset1.drop(droplist,axis=1,inplace=True)
dataset2.drop(droplist,axis=1,inplace=True)
null_columns=dataset1.columns[dataset1.any(axis=0)]
print(dataset1[null_columns].select_dtypes(include='object'))
cat_fill=['PoolQC','MiscFeature','BsmtQual','BsmtCond', 'BsmtExposure', 'BsmtFinType1','BsmtFinType2'\
          ,'FireplaceQu', 'GarageType', 'GarageFinish', 'GarageQual', 'GarageCond','Fence']
dataset1.loc[:,cat_fill]=dataset1[cat_fill].fillna('None')
dataset2.loc[:,cat_fill]=dataset2[cat_fill].fillna('None')
null_columns=dataset1.columns[dataset1.isnull().any(axis=0)]
print(dataset2[null_columns].select_dtypes(include=['float64','int64']))
num_fill=['BsmtFinSF1','BsmtFinSF2','BsmtUnfSF','TotalBsmtSF','BsmtFullBath','BsmtHalfBath',\
          'GarageCars','GarageArea']
dataset2.loc[:,num_fill]=dataset2[num_fill].fillna(0)
dataset2.loc[:,'GarageYrBlt']=dataset2['GarageYrBlt'].fillna(dataset2['YearBuilt'])
print(dataset2[null_columns].isnull().sum())
cat_fill_m=['MasVnrType', 'Electrical']
cat_fill_m=['MSZoning','Utilities','Exterior1st','Exterior2nd'\
            ,'MasVnrType','KitchenQual','Functional','SaleType']
dataset2.loc[:,cat_fill_m]=dataset2[cat_fill_m].fillna(dataset2[cat_fill_m].mode().iloc[0])
num_fill_m=['LotFrontage','MasVnrArea']
dataset1.loc[:,num_fill_m]=dataset1[num_fill_m].fillna(dataset1[num_fill_m].mean())
dataset1.loc[:,'GarageYrBlt']=dataset1['GarageYrBlt'].fillna(dataset1['YearBuilt'])
y_train=dataset3.iloc[:,80].values
droplist=['SalePrice']
dataset1.drop(droplist,axis=1,inplace=True)
"""one_hot_encoded_training_predictors = pd.get_dummies(dataset1,drop_first=True)
one_hot_encoded_test_predictors = pd.get_dummies(dataset2,drop_first=True)
final_train, final_test = one_hot_encoded_training_predictors.align(one_hot_encoded_test_predictors,
                                                                   join='left', 
                                                                   axis=1)"""
final_train=dataset1
final_test=dataset2
X=dataset1
y=dataset3.SalePrice
null_columns=final_test.columns[final_test.isnull().any(axis=0)]
print(final_test[null_columns].isnull().sum())
"""zero_col=['Utilities_NoSeWa','Condition2_RRAe','Condition2_RRAn','Condition2_RRNn','HouseStyle_2.5Fin',\
          'RoofMatl_CompShg','RoofMatl_Membran','RoofMatl_Metal','RoofMatl_Roll','Exterior1st_ImStucc'\
          ,'Exterior1st_Stone','Exterior2nd_Other','Heating_GasA','Heating_OthW','Electrical_Mix','GarageQual_Fa'\
          ,'PoolQC_Fa','MiscFeature_TenC']
final_test.loc[:,zero_col]=final_test[zero_col].fillna(0)"""
X_train=final_train.iloc[:, :].values
#X_test[:,268].fill(0)
X_test=final_test.iloc[:, :].values
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)
# Applying PCA
from sklearn.decomposition import PCA
pca = PCA(n_components = 150)
X_train = pca.fit_transform(X_train)
X_test = pca.transform(X_test)
explained_variance = pca.explained_variance_ratio_

# Applying LDA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
lda = LDA(n_components = 200)
X_train = lda.fit_transform(X_train, y_train)
X_test = lda.transform(X_test)
X_train, X_validation, y_train, y_validation = train_test_split(X, y, train_size=0.7, random_state=0)
categorical_features_indices = np.where(X.dtypes ==np.object)[0]

model=CatBoostRegressor(iterations=1500,learning_rate=0.06)
#model=model.fit(X, y,cat_features=categorical_features_indices)
model.fit(X_train, y_train,cat_features=categorical_features_indices,eval_set=(X_validation, y_validation),plot=True)
regressor=CatBoostRegressor()
parameters1={
          'iterations':[1100],
          'learning_rate':[0.01,0.06],
          'loss_function':['RMSE']
        }
model.score(X,y)
"""
regressor=regressor.fit(X,y,cat_features=categorical_features_indices)
pool=catboost.Pool(X,y)"""
cr_grid=GridSearchCV(regressor,\
                        parameters1,\
                        cv = 2,\
                        n_jobs = 5,\
                        verbose=True)
cr_grid=cr_grid.fit(X,y,cat_features=categorical_features_indices)
best=cr_grid.best_score_
best_params=cr_grid.best_params_
y_pred=model.predict(final_test)
xgb1 = XGBRegressor(booster='gbtree',colsample_bytree=0.7,gamma=0,learning_rate=0.03\
                    ,max_depth=6,min_child_weight=3,n_estimators=1200,objective='reg:linear'\
                    ,subsample=0.9,colsample_bylevel=0.5,base_score=0.7)
xgb1=XGBRegressor()
parameters = {#when use hyperthread, xgboost may become slower
              'objective':['reg:linear'],
              'booster':['gbtree'],
              'gamma':[0],
              'learning_rate': [0.0305], #so called `eta` value
              'max_depth': [6],
              'min_child_weight': [2,3,4],
              'subsample': [0.85,0.9,0.95],
              'colsample_bytree': [0.65,0.7,0.77],
              'colsample_bylevel': [0.45,0.5,0.55],
              'n_estimators': [1190],
              'base_score':[0.65,0.7,0.75]}
parameters1={}
xgb_grid = GridSearchCV(xgb1,
                        parameters,
                        cv = 5,
                        n_jobs =15,
                        scoring='mean_squared_error',
                        verbose=True)
xgb_grid=xgb_grid.fit(X_train,y_train)
best=xgb_grid.best_score_
best_params=xgb_grid.best_params_

xgb1.fit(X_train,y_train)


from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, y_train)

# Fitting Logistic Regression to the Training set
from sklearn.linear_model import LogisticRegression
classifier = LogisticRegression(random_state = 0)
classifier.fit(X_train, y_train)

# Predicting the Test set results
y_pred = xgb1.predict(X_test)

y_pred=y_pred.reshape((1459,1))
y_pred=np.append(arr=np.zeros((1459,1)).astype(int),values=y_pred,axis=1)
X_temp=dataset3.iloc[:, :].values
y_pred[:,0]=X_temp[:,0]
df = pd.DataFrame(y_pred,columns=['Id','SalePrice'])
df.to_csv('out1.csv', sep='\t')