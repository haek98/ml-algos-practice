# -*- coding: utf-8 -*-
"""
Created on Mon Aug 20 22:41:56 2018

@author: ASUS
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from xgboost.sklearn import XGBRegressor
from xgboost.sklearn import XGBClassifier
import math
training_set=pd.read_csv('train.csv')
test_set=pd.read_csv('test.csv')
y=training_set.iloc[:,[1]]
training_set.isnull().sum()
final_set=pd.concat([training_set,test_set],ignore_index=True)
names=final_set.Name.str.split('[,.]')
names=[str.strip(abc[1]) for abc in names.values]
final_set["Title"]=names
final_set=final_set.drop(columns=['Survived','PassengerId','Name','Ticket'],axis=1)
final_set.count()
final_set.Title.unique()
final_set['Fare'] = final_set['Fare'].fillna(final_set['Fare'].mode()[0])
final_set['Embarked'] = final_set['Embarked'].fillna(final_set['Embarked'].mode()[0])
age=final_set.Age
cabin=final_set.Cabin
final_set=final_set.drop(columns=['Age','Cabin'],axis=1)
final_set=pd.get_dummies(final_set,drop_first=True)
final_set['Age']=age
prediction_set=final_set.dropna()
age_train_set=prediction_set.loc[:,['Age']]
prediction_set=prediction_set.drop(columns=['Age'],axis=1)
model=XGBRegressor()
model.fit(prediction_set,age_train_set)
temp=final_set
for i in range(final_set.shape[0]):
    if(math.isnan((final_set.loc[[i],['Age']]).values)):
        temp1=final_set.iloc[[i],:-1]
        final_set.loc[[i],['Age']]=model.predict(temp1)
temp['Age']=final_set['Age']
final_set=temp
final_set['FamilySize']=final_set["Parch"]+final_set['SibSp']
final_set=final_set.drop(columns=['Parch','SibSp'],axis=1)
final_set['FamilySize']=final_set['FamilySize']+1
final_set['FamilySize'].unique()
temp=training_set
temp['Size']=final_set['FamilySize']
print (training_set[['Fare', 'Survived']].groupby(['Fare'], as_index=False).mean())
sns.barplot(x='Fare',y='Survived',data=training_set)
train=final_set.iloc[:(training_set.shape[0]),:]
test=final_set.iloc[(training_set.shape[0]):,:]
model=XGBClassifier()
model.fit(train,y)
predicted=model.predict(test)
answer=test_set.loc[:,['PassengerId']]
answer['Survived']=predicted
answer.to_csv('out.csv', sep='\t')
sns.factorplot(x='Cabin',y='Survived',data=temp)
plt.show()
