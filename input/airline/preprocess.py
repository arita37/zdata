import pandas as pd
import random
import os
import sys
import numpy as np

# Read File
path_repo_root = os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__)))) + "/"
print("path_repo_root", path_repo_root)
sys.path.append(path_repo_root)
folder = 'raw/'
train = pd.read_csv(folder+'train.csv')
test = pd.read_csv(folder+'test.csv')
train.drop(['Unnamed: 0'], axis=1, inplace=True)
train.drop(['Arrival Delay in Minutes'], axis=1, inplace=True)
test.drop(['Unnamed: 0'], axis=1, inplace=True)
test.drop(['Arrival Delay in Minutes'], axis=1, inplace=True)
train.satisfaction = [1 if each ==
                      "satisfied" else 0 for each in train.satisfaction]
test.satisfaction = [1 if each ==
                     "satisfied" else 0 for each in test.satisfaction]


# saving train
feature_tr = train.drop(["satisfaction"], axis=1)
target_tr = train[["satisfaction", "id"]]
feature_tr.to_csv("train/features.csv", index=False)
target_tr.to_csv("train/target.csv", index=False)

features = dict(method='zip', archive_name='features.csv')
target = dict(method='zip', archive_name='target.csv')
feature_tr.to_csv('train/features.zip', index=False, compression=features)
target_tr.to_csv('train/target.zip', index=False, compression=target)

# saving test
feature_test = test.drop(["satisfaction"], axis=1)
target_test = test[["satisfaction", "id"]]
feature_test.to_csv('test/features.zip', index=False, compression=features)
target_test.to_csv('test/target.zip', index=False, compression=target)
