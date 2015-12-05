#!/usr/bin/env python
#-*- coding: utf-8 -*-

import pandas as pd 
import numpy as np
from sklearn.cross_validation import train_test_split
from sklearn.ensemble import RandomForestClassifier 


def nomalizing(df):
    if df != 0:
        df = 1
    return df 


df_train = pd.read_csv('d:/PythonProject/kaggle/dataset/digit_recognizer/train.csv')
df_test = pd.read_csv('d:/PythonProject/kaggle/dataset/digit_recognizer/test.csv')
print df_train.info()
df_train = df_train.ix[1:]
df_train_data = df_train.ix[:, 1:]
df_train_label = df_train.ix[:, 0:1]

df_train_data = df_train_data.applymap(nomalizing)
df_test_data = df_test.applymap(nomalizing)

# df_train_data_train, df_train_data_test, df_train_label_train, df_train_label_test = train_test_split(df_train_data, df_train_label, test_size = 0.2, random_state = 42)

clf = RandomForestClassifier()
s = clf.fit(df_train_data, df_train_label)
print s 

df_test_label = pd.DataFrame(clf.predict(df_test_data))
# print df_test_label.head()

df_test_label.to_csv('d:/PythonProject/kaggle/dataset/digit_recognizer/result.csv')

