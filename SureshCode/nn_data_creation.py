# -*- coding: utf-8 -*-
"""
Created on Fri Jan 20 14:23:40 2017

@author: JTay
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
#from fast_ml.model_development import train_valid_test_split

df = pd.read_csv('E:\CS7641\Assignment2\SureshCode\data\winequality-white.csv',';')
df['quality'] = np.where(df['quality'] > 5, 1, 0)
X = df.drop(['quality'], axis=1)
#X = df.drop(columns = ['quality'])
scaler = StandardScaler()
# transform data
X = scaler.fit_transform(X)
y=df['quality']
y = y.astype(int)
train_size=0.8

# In the first step we will split the data in training and remaining dataset
X_train, X_rem, y_train, y_rem = train_test_split(X,y, train_size=0.8)

y_train = np.atleast_2d(y_train).T
y_rem = np.atleast_2d(y_rem).T
# Now since we want the valid and test size to be equal (10% each of overall data).
# we have to define valid_size=0.5 (that is 50% of remaining data)
test_size = 0.5
X_valid, X_test, y_valid, y_test = train_test_split(X_rem,y_rem, test_size=0.5)


#X_train, y_train, X_valid, y_valid, X_test, y_test = train_valid_test_split(df, target = 'SalePrice',
#                                                                            train_size=0.8, valid_size=0.1, test_size=0.1)

tst = pd.DataFrame(np.hstack((X_test,y_test)))
trg = pd.DataFrame(np.hstack((X_train,y_train)))
val = pd.DataFrame(np.hstack((X_valid,y_valid)))
tst.to_csv('E:\CS7641\Assignment2\SureshCode\data\wine_test.csv',index=False,header=False)
trg.to_csv('E:\CS7641\Assignment2\SureshCode\data\wine_trg.csv',index=False,header=False)
val.to_csv('E:\CS7641\Assignment2\SureshCode\data\wine_val.csv',index=False,header=False)
