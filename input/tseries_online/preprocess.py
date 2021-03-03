import pandas as pd
import random, os, sys
import numpy as np
from source.prepro import *


#### Add path for python import  #######################################
path_repo_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))) + "/"
print("path_repo_root", path_repo_root)
sys.path.append( path_repo_root)
########################################################################


quant_cols = ["Quan_1", "Quan_2", "Quan_3", "Quan_4", "Quan_5", "Quan_6", "Quan_7", "Quan_8", "Quan_9", "Quan_10", "Quan_11", "Quan_12", "Quan_13", "Quan_14"]


folder     = 'raw/'
df         = pd.read_csv(folder+'TrainingDataset.csv', delimiter=',')
df_test     = pd.read_csv(folder+'TestDataset.csv', delimiter=',')

# Replacing the NA in the quant columns with median of those columns

pd_colnum_fill_na_median(df, quant_cols, None)
pd_colnum_fill_na_median(df_test, quant_cols, None)

pd_colnum(df, quant_cols, None)
pd_colnum(df_test, quant_cols, None)

pd_colnum_normalize(df, quant_cols, None)
pd_colnum_normalize(df_test, quant_cols, None)


# Cat columns are fine

feature_tr = df.drop(["Outcome_M1", "Outcome_M2", "Outcome_M3", "Outcome_M4", "Outcome_M5", "Outcome_M6", "Outcome_M7", "Outcome_M8", "Outcome_M9", "Outcome_M10", "Outcome_M11", "Outcome_M12"],axis=1)
target_tr  = df[["Outcome_M1", "Outcome_M2", "Outcome_M3", "Outcome_M4", "Outcome_M5", "Outcome_M6", "Outcome_M7", "Outcome_M8", "Outcome_M9", "Outcome_M10", "Outcome_M11", "Outcome_M12"]]
feature_tr.to_csv( "train/features.csv", index=False)
target_tr.to_csv(  "train/target.csv",index=False)

features = dict(method='zip',archive_name='features.csv')
target = dict(method='zip',archive_name='target.csv')
feature_tr.to_csv('train/features.zip', index=False, compression=features)
target_tr.to_csv('train/target.zip', index=False,compression=target)

feature_test = df_test.drop(["id"],axis=1)
feature_test.to_csv('test/features.zip', index=False, compression=features)
