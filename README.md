# Titanic - Kaggle challenge
#### Author : Alphonse Doutriaux - March 2018

Original train and tests sets can be found [here] (https://www.kaggle.com/c/titanic/data).

#### 1. Concatenation of the train and test to combine preprocessing

#### 2. Preprocessing
2.1. Extraction of the title from the name column & handling of the sparse values
2.2. Handling of missing data
2.3. Cabin column : separation between Cabin_deck and Cabin_num using regular expressions. The alphabetical values are transformed into numerical values.
2.4. One hot encoding
2.5. Creation of Sex x Class columns for males
2.6. Creation of Family Size & IsChild columns

#### 3. Random Forest : a gridsearch was performed to tune hyperparameters

#### 4. Results :
Best score on kaggle.com (private score) : 0.78950


