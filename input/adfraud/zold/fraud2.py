import pandas as pd
import numpy as np
import random
import copy
random.seed(100)



########################################################################################################
def pd_colall_preprocess(df, col=None, pars=None):
   df = copy.deepcopy(df)

   df = df.drop(columns=["attributed_time"])
   ## Let's see on which hour the click was happend
   df["click_time"]   = pd.to_datetime(df["click_time"])

   df["hour"]         = df["click_time"].dt.hour.astype("uint8")
   df["minute"]       = df["click_time"].dt.minute.astype("uint8")
   df["second"]       = df["click_time"].dt.second.astype("uint8")
   df["day"]          = df["click_time"].dt.day.astype("uint8")
   df["day_of_week"]  = df["click_time"].dt.dayofweek.astype("uint8")


   ##Let's divide the day in four section ,See in which section click has happend ")
   day_section = 0
   for start_time, end_time in zip([0, 6, 12, 18], [6, 12, 18, 24]):
              df.loc[(df['hour'] >= start_time) & (df['hour'] < end_time), 'day_section'] = day_section
              day_section += 1

   print( "Let's see new clicks count features")
   df["n_ip_clicks"]  = df[['ip', 'channel']].groupby(by=["ip"])[["channel"]].transform("count").astype("uint8")



   ##Computing the number of clicks associated with a given app per hour...')
   df["n_app_clicks"] = df[['app', 'day', 'hour', 'channel']].groupby(by=['app', 'day', 'hour'])[['channel']].transform("count").astype("uint8")

   ##Computing the number of channels associated with a given IP address within each hour...')
   df["n_channels"]   = df[['ip', 'day', 'hour', 'channel']].groupby(by=['ip', 'day', 'hour'])[['channel']].transform("count").astype("uint8")


   ##Let's divide the day in four section ,See in which section click has happend ")
   day_section              = 0
   for start_time, end_time in zip([0, 6, 12, 18], [6, 12, 18, 24]):
     df.loc[(df['hour'] >= start_time) & (df['hour'] < end_time), 'day_section'] = day_section
     day_section             += 1

   ##Computing the number of channels associated with ')
   df['ip_app_count']       = df[['ip', 'app', 'channel']].groupby(by=['ip', 'app'])[['channel']].transform("count").astype("uint8")
   print( "Let's see new clicks count features")
   df["n_ip_clicks"]        = df[['ip', 'channel']].groupby(by=["ip"])[["channel"]].transform("count").astype("uint8")

   ##Computing the number of channels associated with ')
   df["ip_app_os_count"]    = df[['ip', 'app', 'os', 'channel']].groupby(by=['ip', 'app', 'os'])[['channel']].transform("count").astype("uint8")

   df['n_ip_os_day_hh']     = df[['ip', 'os', 'day', 'hour', 'channel']].groupby(by=['ip', 'os', 'day', 'hour'])[['channel']].transform("count").astype("uint8")
   ##Computing the number of clicks associated with a given app per hour...')
   df["n_app_clicks"]       = df[['app', 'day', 'hour', 'channel']].groupby(by=['app', 'day', 'hour'])[['channel']].transform("count").astype("uint8")

   df['n_ip_app_day_hh']    = df[['ip', 'app', 'day', 'hour', 'channel']].groupby(by=['ip', 'app', 'day', 'hour'])[['channel']].transform("count").astype("uint8")
   ##Computing the number of channels associated with a given IP address within each hour...')
   df["n_channels"]         = df[['ip', 'day', 'hour', 'channel']].groupby(by=['ip', 'day', 'hour'])[['channel']].transform("count").astype("uint8")

   df['n_ip_app_os_day_hh'] = df[['ip', 'app', 'os', 'day', 'hour', 'channel']].groupby(by=['ip', 'app', 'os', 'day', 'hour'])[['channel']].transform("count").astype("uint8")
   ##Computing the number of channels associated with ')
   df['ip_app_count']       = df[['ip', 'app', 'channel']].groupby(by=['ip', 'app'])[['channel']].transform("count").astype("uint8")

   df['n_ip_app_dev_os']    = df[['ip', 'app', 'device', 'os', 'channel']].groupby(by=['ip', 'app', 'device', 'os'])[['channel']].transform("count").astype("uint8")
   ##Computing the number of channels associated with ')
   df["ip_app_os_count"]    = df[['ip', 'app', 'os', 'channel']].groupby(by=['ip', 'app', 'os'])[['channel']].transform("count").astype("uint8")

   df['n_ip_dev_os']        = df[['ip', 'device', 'os', 'channel']].groupby(by=['ip', 'device', 'os'])[['channel']].transform("count").astype("uint8")


   GROUPBY_AGGREGATIONS = [
        # Count, for ip-day-hour
        {'groupby': ['ip','day','hour'], 'select': 'channel', 'agg': 'count'},
        # Count, for ip-app
        {'groupby': ['ip', 'app'], 'select': 'channel', 'agg': 'count'},
        # Count, for ip-app-os
        {'groupby': ['ip', 'app', 'os'], 'select': 'channel', 'agg': 'count'},
        # Count, for ip-app-day-hour
        {'groupby': ['ip','app','day','hour'], 'select': 'channel', 'agg': 'count'},
        # Mean hour, for ip-app-channel
        {'groupby': ['ip','app','channel'], 'select': 'hour', 'agg': 'mean'},

        ################### V2 - GroupBy Features #
        # Average clicks on app by distinct users; is it an app they return to?
        {'groupby': ['app'],
         'select': 'ip',
         'agg': lambda x: float(len(x)) / len(x.unique()),
         'agg_name': 'AvgViewPerDistinct'
        },
        # How popular is the app or channel?
        {'groupby': ['app'], 'select': 'channel', 'agg': 'count'},
        {'groupby': ['channel'], 'select': 'app', 'agg': 'count'},

        #################### V3 - GroupBy Features                                              #
        # https://www.kaggle.com/bk0000/non-blending-lightgbm-model-lb-0-977 #
        {'groupby': ['ip'], 'select': 'channel', 'agg': 'nunique'},
        {'groupby': ['ip'], 'select': 'app', 'agg': 'nunique'},
        {'groupby': ['ip','day'], 'select': 'hour', 'agg': 'nunique'},
        {'groupby': ['ip','app'], 'select': 'os', 'agg': 'nunique'},
        {'groupby': ['ip'], 'select': 'device', 'agg': 'nunique'},
        {'groupby': ['app'], 'select': 'channel', 'agg': 'nunique'},
        {'groupby': ['ip', 'device', 'os'], 'select': 'app', 'agg': 'nunique'},
        {'groupby': ['ip','device','os'], 'select': 'app', 'agg': 'cumcount'},
        {'groupby': ['ip'], 'select': 'app', 'agg': 'cumcount'},
        {'groupby': ['ip'], 'select': 'os', 'agg': 'cumcount'}
  ]

   # Apply all the groupby transformations
   for spec in GROUPBY_AGGREGATIONS:
      # Name of the aggregation we're applying
      agg_name = spec['agg_name'] if 'agg_name' in spec else spec['agg']

      # Name of new feature
      new_feature = '{}_{}_{}'.format('_'.join(spec['groupby']), agg_name, spec['select'])

      print("Grouping by {}, and aggregating {} with {}".format( spec['groupby'], spec['select'], agg_name ))

      # Unique list of features to select
      all_features = list(set(spec['groupby'] + [spec['select']]))

      # Perform the groupby
      gp = df[all_features]. \
          groupby(spec['groupby'])[spec['select']]. \
          agg(spec['agg']). \
          reset_index(). \
          rename(index=str, columns={spec['select']: new_feature})

      # Merge back to df
      if 'cumcount' == spec['agg']:
          df[new_feature] = gp[0].values
      else:
          df = df.merge(gp, on=spec['groupby'], how='left')

   del gp

   df['n_ip_os_day_hh']     = df[['ip', 'os', 'day', 'hour', 'channel']].groupby(by=['ip', 'os', 'day', 'hour'])[['channel']].transform("count").astype("uint8")
   df['n_ip_app_day_hh']    = df[['ip', 'app', 'day', 'hour', 'channel']].groupby(by=['ip', 'app', 'day', 'hour'])[['channel']].transform("count").astype("uint8")
   df['n_ip_app_os_day_hh'] = df[['ip', 'app', 'os', 'day', 'hour', 'channel']].groupby(by=['ip', 'app', 'os', 'day', 'hour'])[['channel']].transform("count").astype("uint8")
   df['n_ip_app_dev_os']    = df[['ip', 'app', 'device', 'os', 'channel']].groupby(by=['ip', 'app', 'device', 'os'])[['channel']].transform("count").astype("uint8")
   df['n_ip_dev_os']        = df[['ip', 'device', 'os', 'channel']].groupby(by=['ip', 'device', 'os'])[['channel']].transform("count").astype("uint8")


   df.drop(columns=["click_time"], axis=1, inplace=True)


   #### coly  genereate
   df['is_attributed'].value_counts(normalize=True)  ### ?
   df["is_attributed"]=df["is_attributed"].astype("uint8")

   dfnew    = df
   col_pars = {}
   return dfnew, col_pars
########################################################################################################################



########################################################################################################################
########################################################################################################################
# coly = "is_attributed"


try :
    df = pd.read_parquet("train_100k/full.parquet")
    df_X = df.drop("is_attributed", axis=1)
    df_y = df["is_attributed"]

except :
    df           = pd.read_csv("raw/train_100k.zip")
    df, col_pars = pd_colall_preprocess(df)

    df.to_parquet("train_100k/full.parquet")
    df_X = df.drop("is_attributed", axis=1)
    df_y = df["is_attributed"]


# df, col_pars = pd_colall_preprocess(df)
# df_X = df.drop("is_attributed", axis=1)
# df_y = df["is_attributed"]


########################################################################################################################
########################################################################################################################
from sklearn.model_selection import train_test_split
train_X,test_X,train_y,test_y = train_test_split(df_X,df_y,stratify=df_y,test_size=0.15)
train_X,val_X,train_y,val_y   = train_test_split(train_X,train_y,stratify=train_y,test_size=0.1)



import lightgbm as lgb
dtrain = lgb.Dataset(train_X, train_y)
dvalid = lgb.Dataset(val_X, val_y)
dtest = lgb.Dataset(test_X,test_y)

param = {'num_leaves': 64, 'objective': 'binary',"seed":1,'boosting_type': 'dart',  # Use boosting_type="gbrt" for large dataset
    'objective': 'binary',
    'metric': 'auc',
    'learning_rate': 0.1,
    'num_leaves': 11,
    'max_depth': -1,
    'min_child_samples': 100,
    'max_bin': 100,
    'subsample': 0.9,  # Was 0.7
    'subsample_freq': 1,
    'colsample_bytree': 0.7,
    'min_child_weight': 0,
         'min_split_gain': 0,
    'reg_alpha': 0,
    'reg_lambda': 0,}
num_round = 100
bst = lgb.train(param, dtrain, num_round, valid_sets=[dvalid], early_stopping_rounds=20)

from sklearn import metrics
ypred = bst.predict(test_X)
score = metrics.roc_auc_score(test_y, ypred)
print(f"Test score: {score}")





########################################################################################################################
########################################################################################################################
from gplearn.genetic import SymbolicTransformer

def pd_col_genetic_transform(df=None, col=None, pars=None):
    num_gen=20
    num_comp=10
    function_set = ['add', 'sub', 'mul', 'div',
           'sqrt', 'log', 'abs', 'neg', 'inv','tan']

    gp = SymbolicTransformer(generations=num_gen, population_size=200,
                            hall_of_fame=100, n_components=num_comp,
                            function_set=function_set,
                            parsimony_coefficient=0.0005,
                            max_samples=0.9, verbose=1,
                            random_state=0, n_jobs=6)

    gen_feats = gp.fit_transform(train_X, train_y)
    gen_feats = pd.DataFrame(gen_feats, columns=["gen_"+str(a) for a in range(gen_feats.shape[1])])
    gen_feats.index = train_X.index
    train_X_all=pd.concat((train_X,gen_feats),axis=1)

    gen_feats = gp.transform(test_X)
    gen_feats = pd.DataFrame(gen_feats, columns=["gen_"+str(a) for a in range(gen_feats.shape[1])])
    gen_feats.index = test_X.index
    test_X_all=pd.concat((test_X,gen_feats),axis=1)

    gen_feats = gp.transform(val_X)
    gen_feats = pd.DataFrame(gen_feats, columns=["gen_"+str(a) for a in range(gen_feats.shape[1])])
    gen_feats.index = val_X.index
    val_X_all=pd.concat((val_X,gen_feats),axis=1)
    return train_X_all,test_X_all,val_X_all


train_X_all,test_X_all,val_X_all = pd_col_genetic_transform(df, col= list(df.columns),
                                                            pars= {'mode': 'transform'})



import lightgbm as lgb
dtrain = lgb.Dataset(train_X_all, train_y)
dvalid = lgb.Dataset(val_X_all, val_y)
dtest = lgb.Dataset(test_X_all,test_y)

param = {'num_leaves': 64, 'objective': 'binary',"seed":1,'boosting_type': 'dart',  # Use boosting_type="gbrt" for large dataset
    'objective': 'binary',
    'metric': 'auc',
    'learning_rate': 0.1,
    'num_leaves': 11,
    'max_depth': -1,
    'min_child_samples': 100,
    'max_bin': 100,
    'subsample': 0.9,  # Was 0.7
    'subsample_freq': 1,
    'colsample_bytree': 0.7,
    'min_child_weight': 0,
         'min_split_gain': 0,
    'reg_alpha': 0,
    'reg_lambda': 0,}
num_round = 100
bst = lgb.train(param, dtrain, num_round, valid_sets=[dvalid], early_stopping_rounds=20)

from sklearn import metrics
ypred = bst.predict(test_X_all)
score = metrics.roc_auc_score(test_y, ypred)
print(f"Test score: {score}")




########################################################################################################################
########################################################################################################################
"""### Encoding the categorical features:"""
#### Let's Apply Target Encoding
train_X.columns
cat_features=['ip', 'app', 'device', 'os', 'channel']
import category_encoders as ce
target_enc = ce.TargetEncoder(cols = cat_features)
def category_encoding(encoder):
       encoder.fit(train_X[cat_features], train_y)
       train_encode_tar =encoder.transform(train_X[cat_features]).add_suffix("_target")
       train_X_all = train_X.join( train_encode_tar )
       val_encode_tar = encoder.transform(val_X[cat_features]).add_suffix("_target")
       val_X_all = val_X.join( val_encode_tar )
       test_encoded_tar = encoder.transform(val_X[cat_features]).add_suffix("_target")
       test_X_all=test_X.join(test_encoded_tar)
       return train_X_all,test_X_all,val_X_all
train_X_all,test_X_all,val_X_all=category_encoding(target_enc)
import lightgbm as lgb
dtrain = lgb.Dataset(train_X_all, train_y)
dvalid = lgb.Dataset(val_X_all, val_y)
dtest = lgb.Dataset(test_X_all,test_y)
param = {'num_leaves': 64, 'objective': 'binary',"seed":1,'boosting_type': 'dart',  # Use boosting_type="gbrt" for large dataset
    'objective': 'binary',
    'metric': 'auc',
    'learning_rate': 0.1,
    'num_leaves': 11,
    'max_depth': -1,
    'min_child_samples': 100,
    'max_bin': 100,
    'subsample': 0.9,  # Was 0.7
    'subsample_freq': 1,
    'colsample_bytree': 0.7,
    'min_child_weight': 0,
         'min_split_gain': 0,
    'reg_alpha': 0,
    'reg_lambda': 0,}
num_round = 100
bst = lgb.train(param, dtrain, num_round, valid_sets=[dvalid], early_stopping_rounds=20)
##############################################

from sklearn import metrics
ypred = bst.predict(test_X_all)
score = metrics.roc_auc_score(test_y, ypred)
print(f"Test score: {score}")
#############################

"""### Let's Apply catboost Encoding"""

import category_encoders as ce

cat_enc = ce.CatBoostEncoder(cols = cat_features,random_state=100)

train_X_all,test_X_all,val_X_all=category_encoding(cat_enc)

import lightgbm as lgb
dtrain = lgb.Dataset(train_X_all, train_y)
dvalid = lgb.Dataset(val_X_all, val_y)
dtest = lgb.Dataset(test_X_all,test_y)

param = {'num_leaves': 64, 'objective': 'binary',"seed":1,'boosting_type': 'dart',  # Use boosting_type="gbrt" for large dataset
    'objective': 'binary',
    'metric': 'auc',
    'learning_rate': 0.1,
    'num_leaves': 11,
    'max_depth': -1,
    'min_child_samples': 100,
    'max_bin': 100,
    'subsample': 0.9,  # Was 0.7
    'subsample_freq': 1,
    'colsample_bytree': 0.7,
    'min_child_weight': 0,
         'min_split_gain': 0,
    'reg_alpha': 0,
    'reg_lambda': 0,}
num_round = 100
bst = lgb.train(param, dtrain, num_round, valid_sets=[dvalid], early_stopping_rounds=20)




from sklearn import metrics
ypred = bst.predict(test_X_all)
score = metrics.roc_auc_score(test_y, ypred)
print(f"Test score: {score}")


