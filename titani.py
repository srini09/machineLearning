# -*- coding: utf-8 -*-
"""
Created on Mon Jan 14 15:34:30 2019

@author: ms186162
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import Imputer


#import the dataset
traindata = pd.read_csv('train.csv')
testdata = pd.read_csv('test.csv')

#X = traindata.iloc[:,:-1].values    
traindata.info()
testdata.info()
#sns.countplot(x='Survived', data=traindata)

#find age and sex of survived people
#traindata.groupby(['Survived','Pclass','Sex'])['Survived'].count()

#print("% of survivals in") 
#print("Pclass=1 : ", traindata.Survived[traindata.Pclass == 1].sum()/traindata[traindata.Pclass == 1].Survived.count())
#print("Pclass=2 : ", traindata.Survived[traindata.Pclass == 2].sum()/traindata[traindata.Pclass == 2].Survived.count())
#print("Pclass=3 : ", traindata.Survived[traindata.Pclass == 3].sum()/traindata[traindata.Pclass == 3].Survived.count())

traindata['Embarked'] = traindata['Embarked'].fillna("S")

traindata = traindata.drop(['PassengerId','Cabin','Name','Ticket'], axis=1)
traindata.info()

temp = traindata.select_dtypes(include=[object])
temp.head()

def one_hot(df, cols):
    """
    @param df pandas DataFrame
    @param cols a list of columns to encode 
    @return a DataFrame with one-hot encoding
    """
    for each in cols:
        dummies = pd.get_dummies(df[each], prefix=each, drop_first=False)
        df = pd.concat([df, dummies], axis=1)
    return df

temp= one_hot(temp, temp.columns)
temp.columns

traindata = pd.concat([traindata,temp],axis=1)

testdata['Embarked'] = testdata['Embarked'].fillna("S")
testdata = testdata.drop(['PassengerId','Cabin','Name','Ticket'], axis=1)
testdata.info()

temp = testdata.select_dtypes(include=[object])
temp= one_hot(temp, temp.columns)
temp.columns
testdata = pd.concat([testdata,temp],axis=1)
temp.info()
temp.head()
testdata.info()
testdata =  testdata.select_dtypes(['number'])
traindata = traindata.select_dtypes(['number'])
testdata.info()

traindata['Age'] = traindata['Age'].fillna(value=0)
traindata.isna().sum()
testdata['Age'] = testdata['Age'].fillna(value=0)
testdata['Fare'] = testdata['Fare'].fillna(value=0)
testdata.isna().sum()
#find age and sex of survived people
traindata.groupby(['Survived','Pclass'])['Survived'].count()

print("% of survivals in") 
print("Pclass=1 : ", traindata.Survived[traindata.Pclass == 1].sum()/traindata[traindata.Pclass == 1].Survived.count())
print("Pclass=2 : ", traindata.Survived[traindata.Pclass == 2].sum()/traindata[traindata.Pclass == 2].Survived.count())
print("Pclass=3 : ", traindata.Survived[traindata.Pclass == 3].sum()/traindata[traindata.Pclass == 3].Survived.count())

#actual training
traindata.head()
X_trainTest = traindata.loc[:,traindata.columns != 'Survived']
Y_trainTest = traindata['Survived']
from sklearn.linear_model import LogisticRegression
logReg = LogisticRegression()
logReg.fit(X_trainTest,Y_trainTest)
acc = logReg.score(X_trainTest,Y_trainTest)
Y_pred = logReg.predict(testdata)




testdatacopy = pd.read_csv('test.csv')
submission = pd.DataFrame({"PassengerID":testdatacopy['PassengerId'], "Survived":Y_pred})
submission.to_csv('submission.csv', index=False)

from sklearn.tree import DecisionTreeClassifier
decTree = DecisionTreeClassifier()
decTree.fit(X_trainTest,Y_trainTest)
acc = decTree.score(X_trainTest,Y_trainTest)
Y_pred = decTree.predict(testdata)