# Import the needed referances
import pandas as pd
import numpy as np
import csv as csv

from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC, LinearSVC
from sklearn.ensemble import RandomForestClassifier

#Shuffle the datasets
from sklearn.utils import shuffle

#Learning curve
import matplotlib.pyplot as plt
from sklearn.model_selection import learning_curve
from sklearn.model_selection import ShuffleSplit

#import seaborn as sns
#Output plots in notebook
#%matplotlib inline 
# 

addpoly = True
plot_lc = 0   # 1--display learning curve/ 0 -- don't display

#loading the data sets from the csv files
print('--------load train & test file------')
train_dataset = pd.read_csv('train.csv')
test_dataset = pd.read_csv('test.csv')

print('train dataset: %s, test dataset %s' %(str(train_dataset.shape), str(test_dataset.shape)) )
train_dataset.head()

print('Id is unique.') if train_dataset.PassengerId.nunique() == train_dataset.shape[0] else print('oops')
print('Train and test sets are distinct.') if len(np.intersect1d(train_dataset.PassengerId.values, test_dataset.PassengerId.values))== 0 else print('oops')
# print('We do not need to worry about missing values.') if train_dataset.count().min() == train_dataset.shape[0] and test_dataset.count().min() == test_dataset.shape[0] else print('oops we have nan')

datasetHasNan = False
if train_dataset.count().min() == train_dataset.shape[0] and test_dataset.count().min() == test_dataset.shape[0] :
    print('We do not need to worry about missing values.') 
else:
    datasetHasNan = True
    print('oops we have nan')

print('----train dataset column types information-------')
dtype_df = train_dataset.dtypes.reset_index()
dtype_df.columns = ["Count", "Column Type"]
dtype_df.groupby("Column Type").aggregate('count').reset_index()

#print(dtype_df.groupby("Column Type").aggregate('count').reset_index())

print('----train dataset information-------')
#print(dtype_df)

#Check for missing data & list them 
if datasetHasNan == True:
    nas = pd.concat([train_dataset.isnull().sum(), test_dataset.isnull().sum()], axis=1, keys=['Train Dataset', 'Test Dataset']) 
    print('Nan in the data sets')
    print(nas[nas.sum(axis=1) > 0])

# Class vs Survived
print(train_dataset[['Pclass', 'Survived']].groupby(['Pclass'], as_index=False).mean().sort_values(by='Survived', ascending=False))

# sex vs Survived
print(train_dataset[["Sex", "Survived"]].groupby(['Sex'], as_index=False).mean().sort_values(by='Survived', ascending=False))

# SibSp vs Survived
#Sibling = brother, sister, stepbrother, stepsister
#Spouse = husband, wife (mistresses and fiancés were ignored)
print(train_dataset[["SibSp", "Survived"]].groupby(['SibSp'], as_index=False).mean().sort_values(by='Survived', ascending=False))

# Parch vs Survived
#Parent = mother, father
#Child = daughter, son, stepdaughter, stepson
#Some children travelled only with a nanny, therefore parch=0 for them.
print(train_dataset[["Parch", "Survived"]].groupby(['Parch'], as_index=False).mean().sort_values(by='Survived', ascending=False))


# Data sets cleaing, fill nan (null) where needed and delete uneeded columns
print('----Start data cleaning ------------')

#train_dataset['IsMinor'] = 0
#train_dataset.loc[(train_dataset['Age'] < 14) & ((train_dataset['Pclass'] == 1) | (train_dataset['Pclass'] == 2) ), 'IsMinor'] = 1

#test_dataset['IsMinor'] = 0
#test_dataset.loc[(test_dataset['Age'] < 14) & ((test_dataset['Pclass'] == 1 ) | (test_dataset['Pclass'] == 2 )), 'IsMinor'] = 1


#manage Age
# randint gives random integer between (start, end)
train_random_ages = np.random.randint(train_dataset["Age"].mean() - train_dataset["Age"].std(),
                                          train_dataset["Age"].mean() + train_dataset["Age"].std(),
                                          size = train_dataset["Age"].isnull().sum())
print(train_random_ages)

test_random_ages = np.random.randint(test_dataset["Age"].mean() - test_dataset["Age"].std(),
                                          test_dataset["Age"].mean() + test_dataset["Age"].std(),
                                          size = test_dataset["Age"].isnull().sum())

train_dataset["Age"][np.isnan(train_dataset["Age"])] = train_random_ages
test_dataset["Age"][np.isnan(test_dataset["Age"])] = test_random_ages
train_dataset['Age'] = train_dataset['Age'].astype(int)
test_dataset['Age'] = test_dataset['Age'].astype(int)

# Embarked 
train_dataset["Embarked"].fillna('S', inplace=True)
test_dataset["Embarked"].fillna('S', inplace=True)

# replace characters with int
train_dataset['Port'] = train_dataset['Embarked'].map( {'S': 0, 'C': 1, 'Q': 2} ).astype(int) 
test_dataset['Port'] = test_dataset['Embarked'].map({'S': 0, 'C': 1, 'Q': 2}).astype(int)
del train_dataset['Embarked']
del test_dataset['Embarked']

# Fare
test_dataset["Fare"].fillna(test_dataset["Fare"].median(), inplace=True)

print(train_dataset.Cabin.unique())