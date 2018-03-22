
# coding: utf-8

# Titanic Challenge
# Alphonse Doutriaux - March 2018

# 0. Preliminaries

# 0.1. Imports

import pandas as pd
import numpy as np
import xgboost 
import re
import datetime
import warnings

from sklearn.model_selection import GridSearchCV, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from datetime import date, datetime
warnings.filterwarnings('ignore')

# 0.2. Import des données

train = pd.read_csv("./data/train.csv", index_col=0)
test = pd.read_csv("./data/test.csv", index_col=0)
# here we concatenate train & test in order to do the preprocessing once only
data = pd.concat([train, test])

# 1. Preprocessing

X = data.copy()
y_train = train[['Survived']]
X = X[['Pclass', 'Name', 'Cabin', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked']]

# We extract the title from the name column

i=1
for name in X['Name']:
    title = re.findall(r"[,][ ][A-Za-z]*", name)[0][2:] #Ici, on extrait les caractères compris entre ", " et "."
    X.loc[i, 'Name'] = title
    i=i+1
X.rename(columns = {'Name':'Title'}, inplace = True)

# handling with sparse values
X['Title'] = X['Title'].replace(('Mme', 'Ms'), 'Mrs') 
X['Title'] = X['Title'].replace('Mlle', 'Miss') 
X['Title'] = X['Title'].replace(('Col', 'Major', 'Capt'), 'Military') 
X['Title'] = X['Title'].replace(('Dooley, Mr. Patrick', 'Jonkheer'), 'Mr')
X['Title'] = X['Title'].replace('Don', 'Sir')
X['Title'] = X['Title'].replace(('the', 'Dona'), 'Lady')


# Input missing data

X['Age'].fillna(X['Age'].median(), inplace=True)
X['Fare'].fillna(X['Fare'].median(), inplace=True)
X['Embarked'].fillna('S', inplace=True)

# Handling with the cabins
# my hypothesis is that the Cabin number is carying information about the deck and the position (front/back)

X['Cabin'].fillna('Z0', inplace=True)

# cleaning the non-conform cabin numbers('D' et 'T')
X.loc[340, 'Cabin'] = 'T0'
X.loc[1001, 'Cabin'] = 'F0' # cleaning des numéros de cabines non conformes('F')
X.loc[1193, 'Cabin'] = 'D0' # cleaning des numéros de cabines non conformes('D')
X.loc[949, 'Cabin'] = 'G63' # cleaning des numéros de cabines non conformes('F G63')
X.loc[1180, 'Cabin'] = 'E46' # cleaning des numéros de cabines non conformes('F E46')
X.loc[1213, 'Cabin'] = 'E57' # cleaning des numéros de cabines non conformes('F E57')
for i in (293, 328, 474):
    X.loc[i, 'Cabin'] = 'D0'

# split the cabin between cabin_number and cabin_deck (the letter)
i=1

for cabin_id in X['Cabin']:
    cabin_num = int(re.findall(r"\d+", cabin_id)[0])
    cabin_deck = re.findall(r"[A-Z]", cabin_id)[0]
    X.loc[i, 'CabinNum'] = cabin_num
    X.loc[i, 'CabinDeck'] = cabin_deck
    i=i+1
    
X = X.drop(['Cabin'], axis=1)

# We transforme alphabetical deck values into numerical deck values (knowing that there is a hierarchy : deck A aka deck 7 is the highest)
X['CabinDeck'] = X['CabinDeck'].replace(['T', 'A', 'B', 'C', 'D', 'E', 'F', 'G', 'Z'],[7,7,6,5,4,3,2,1, np.NaN])
X['CabinDeck'] = X['CabinDeck'].fillna(X['CabinDeck'].median())

# We replace missing values in the CabinNum column with the mean
X['CabinNum'] = X['CabinNum'].replace(0, np.NaN)
X['CabinNum'].fillna(X['CabinNum'].median(), inplace=True)

# One hot encoding
columns_to_encode = ['Title', 'Sex', 'Embarked']
X = pd.get_dummies(X, columns=columns_to_encode, prefix=columns_to_encode)


# FamilySize & IsChild columns
X['FamilySize'] = X['SibSp'] + X['Parch'] + 1
X = X.drop(['Parch', 'SibSp'], axis=1)

#  Sex x Class column
X['FirstClassMale'] = pd.Series(0, index=X.index)
X['SecondClassMale'] = pd.Series(0, index=X.index)
X['ThirdClassMale'] = pd.Series(0, index=X.index)

i=1
for passenger in X.index:
    if i in X[X['Pclass'] == 1][X['Sex_male']==1].index.tolist():
        X.loc[i, 'FirstClassMale'] = 1
    elif i in X[X['Pclass'] == 2][X['Sex_male']==1].index.tolist():
        X.loc[i, 'SecondClassMale'] = 1
    elif i in X[X['Pclass'] == 3][X['Sex_male']==1].index.tolist():
        X.loc[i, 'ThirdClassMale'] = 1
    i += 1

# Train / test split
X_train = X[:len(train)]

# 2. Random Forest
# A GridSearch was performed to tune hyperparameters, not shown here
rf = RandomForestClassifier(max_depth = 5,
                            max_features = 0.3,
                            n_estimators = 10000,
                            n_jobs=-1)

#rf_scores = cross_val_score(rf, X_train, y_train, cv=3, scoring="accuracy", n_jobs=-1)
#print("RandomForests")
#print("Average accuracy: {:.2%}".format(rf_scores.mean()))
#print("Interval: [", round(rf_scores.mean()-3*rf_scores.std(),4), ";", round(rf_scores.mean()+3*rf_scores.std(),4),"]")

rf.fit(X_train, y_train)

# 4. Predictions on testset

test = X[891:]
preds = rf.predict(test)
preds = pd.DataFrame({"Survived":preds}, index=test.index)
preds.to_csv(path_or_buf= './submission_files/preds_' + datetime.now().strftime("%d%m%Y-%H%M%S") + '.csv')

