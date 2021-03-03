#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 25 18:46:02 2021

"""
import umap
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.cluster import DBSCAN
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import numpy as np
import hdbscan
import seaborn as sns

from sklearn import preprocessing
print(pd.__version__)


######################################################################################
"""
Expected Format : userid, time, pagecategory, pageaction

Now the Feature engineering required :
    Timestamp Features : server_timestamp_epoch_ms
        Prepared Features : day of week, hour of day, hour, minutes, seconds, weekend, weekday

    Category Features : product_action, product_skus_hash
        Prepared Features : Dummy Hot encoding - product_skus_hash , Dummy Hot encoding - product_action

"""

# Clustering
# read df
#path = "/Users/yashwanthkumar/Work/upwork/clustering/online_shoppers_intention.csv"
dir_path    = "/Users/yashwanthkumar/Work/upwork/clustering/shopper_intent_prediction"
path        = dir_path+"/shopper_intent_prediction-df.csv"
result_path = "/result_v7"
df_full     = pd.read_csv(path)
print(len(df_full))



#########     df PRE - PROCESSING      ############################################

# sampling
df_sample = df_full.sample(n=5000,random_state=1)
print(len(df_sample))
df_sample.head()
df_sample.info()
df = df_sample[df_sample[colid].notna()]

# USE THE BELOW CODE TO DROP NA
# check how much uique products skus are there
# print(len(df[["product_skus_hash"]].dropna()))
# d = df[["product_skus_hash"]].dropna().drop_duplicates(["product_skus_hash"])
# print(len(d))


#################################################################################
# drop the unused columns  ######################################################
df = df.drop(['hashed_url','event_type'],axis=1) 


colid   = ['session_id_hash']




############################################################################################
############ Feature engineering   #########################################################

############ time related transformation
colt              = 'timestamp'
coltime           = ['hour_of_day','day_of_week','weekday']

df[colt]          = pd.to_datetime(df['server_timestamp_epoch_ms'],unit='ms')
df['hour_of_day'] = df[colt].dt.hour.astype("uint8")
df['day_of_week'] = df[colt].dt.dayofweek.astype("uint8")
df['weekday']     = df[colt].dt.weekday.astype("uint8").apply(lambda x: 0 if x<5 else 1)
#df['minutes'] = df[colt].dt.minute.astype("uint8")
#df['second'] = df[colt].dt.second.astype("uint8")
#df = df.drop(['server_timestamp_epoch_ms'],axis=1) 
df.info()

df_time_dummies = pd.get_dummies(df[colid+coltime], columns=coltime).groupby(colid).sum()


##### Agg by time
df_dt = df.groupby(colid).agg(avgtime =('server_timestamp_epoch_ms','mean'),
                	                      maxtime =('server_timestamp_epoch_ms','max'),
                	                      mintime =('server_timestamp_epoch_ms','min'))

df_dt['duration'] = df_dt['maxtime']-df_dt['mintime']
#df_dt['duration_mins'] = (df_dt['duration']/(1000*60))
df_dt = df_dt.drop(['avgtime','maxtime','mintime'],axis=1)



###########################################################################################
#colcat  = ['product_action','product_skus_hash'] 
colcat  = ['product_action' ] 


# category related transformation
# df['product_sku_action'] = df['product_action'].astype('str')+"|"+df['product_skus_hash'].astype('str')
# can include is_purchase also as a feature. But since vil be using event action not needed
# df['is_purchase'] = df['product_action'].apply(lambda x: 1 if x == 'purchase' else 0)


df_cat_dummies = pd.get_dummies(df[colid+colcat], columns= colcat).groupby(colid).sum()

# concat both df. since they have same index session id. they will be merged properly
df_model = df_cat_dummies.merge(df_time_dummies, how='left',left_index=True,right_index=True)
df_model = df_model.merge(df_dt,                 how='left',left_index=True,right_index=True)


#####################################################################################
#### Ratio features  ################################################################
def div(numerator, denominator):
  return lambda row: 0.0 if row[denominator] == 0 else float(row[numerator]/row[denominator])

df_model['buy_view_ratio'] = df_model.apply(div('product_action_purchase','product_action_detail'), axis = 1)
df_model['view_dur_ratio'] = df_model.apply(div('product_action_detail','duration'), axis = 1)
df_model['buy_dur_ratio']  = df_model.apply(div('product_action_purchase','duration'), axis = 1)



#################################################################################################
# minmax scaling ################################################################################
df_model.fillna(0, inplace=True)
min_max_scaler = preprocessing.MinMaxScaler()
df_model       = pd.dfFrame(min_max_scaler.fit_transform(df_model.values), columns=df_model.columns, index=df_model.index)
df_model.info()



#################################################################################################
# to check total columns  #######################################################################
total_cnt=0
cnt=0
for col in colcat+coltime:
    cnt = len(df[col].unique())
    total_cnt +=cnt
    print(" Total unique values of - {} is {} | Total columns = {}".format(col,cnt,total_cnt))


# the count matches (remmeber for dummy subtract 2 from total cnt)
df_model = df_model.reset_index()
print(len(df_model[colid]))
# 31928 sessions
print(df_model.duplicated([colid]).any())





#####################################################################################################
#########     MODEL TRAINING      ############

df_model = df_model[df_model.columns[~df_model.columns.isin([colid,'cluster_id'])]]


# UMAP embedding
sns.set(style='white', rc={'figure.figsize':(10,8)})


# UMAP_HDBCAN
embedding = umap.UMAP(n_neighbors=30, min_dist=0.0, 
	                  n_components=2, random_state=42, ).fit_transform(df_model)

labels    = hdbscan.HDBSCAN(min_samples=10, min_cluster_size=500, ).fit_predict(clusterable_embedding)
clustered = (labels >= 0)


plt.scatter(embedding[~clustered, 0], embedding[~clustered, 1], c=(0.5, 0.5, 0.5), s=0.1, alpha=0.5)
plt.scatter(embedding[clustered, 0],  embedding[clustered, 1], c=labels[clustered], s=0.1, cmap='Spectral');

plt.title('UMAP_HDBCAN')
plt.savefig(dir_path+result_path+'/UMAP_HDBCAN.png')
plt.savefig(dir_path+result_path+'/UMAP_HDBCAN.svg', format='svg', dpi=1200)
df_model['ClusterID_UMAP_HDBCAN'] = labels






"""
Tried K-Means . Not working well

Code:
# clustering
# finding the right K using elbow method
Sum_of_squared_distances = []
K = range(1,20)
for k in K:
    km = KMeans(n_clusters=k)
    km = km.fit(df_model.loc[:, df_model.columns != colid])
    # evaluating clusters
    Sum_of_squared_distances.append(km.inertia_)

plt.plot(K, Sum_of_squared_distances, 'bx-')
plt.xlabel('k')
plt.ylabel('Sum_of_squared_distances')
plt.title('Elbow Method For Optimal k')
plt.show()
# from the plot choose K at the elbow point


# predict clusters
#k = 4
#km = KMeans(n_clusters=k)
#model = km.fit(df_model.loc[:, df_model.columns != colid])
df_model = df_model[df_model.columns[~df_model.columns.isin([colid,'cluster_id'])]]


"""



# DBS


##########  SAVING PREDICTION #############
# save predictions
#cluster_ids = ['ClusterID_PCA_DBCAN','ClusterID_PCA_HDBCAN','ClusterID_UMAP_HDBCAN']
cluster_ids = ['ClusterID_UMAP_HDBCAN']

df_model[[colid]+cluster_ids].to_csv(dir_path+result_path+'/prediction_output_sid_cid.csv')
df_model.to_csv(dir_path+result_path+'/feature_outputs.csv')





# =============================================================================
# 
# # PCA_DBCAN
# lowd_df = PCA(n_components=50).fit_transform(df_model)
# #hdbscan_labels = hdbscan.HDBSCAN(min_samples=10, min_cluster_size=100).fit_predict(lowd_df)
# model = DBSCAN(eps=3, min_samples=500).fit(lowd_df)
# labels = model.labels_
# clustered = (labels >= 0)
# plt.scatter(embedding[~clustered, 0],
#             embedding[~clustered, 1],
#             c=(0.5, 0.5, 0.5),
#             s=0.1,
#             alpha=0.5)
# plt.scatter(embedding[clustered, 0],
#             embedding[clustered, 1],
#             c=labels[clustered],
#             s=0.1,
#             cmap='Spectral');
# 
# plt.title('PCA_DBCAN')
# plt.savefig(dir_path+result_path+'/PCA_DBCAN.png')
# plt.savefig(dir_path+result_path+'/PCA_DBCAN.svg', format='svg', dpi=1200)
# df_model['ClusterID_PCA_DBCAN'] = labels
# 
# 
# 
# # PCA_HDBCAN
# lowd_df = PCA(n_components=50).fit_transform(df_model)
# labels = hdbscan.HDBSCAN(min_samples=10, min_cluster_size=500).fit_predict(lowd_df)
# clustered = (labels >= 0)
# plt.scatter(embedding[~clustered, 0],
#             embedding[~clustered, 1],
#             c=(0.5, 0.5, 0.5),
#             s=0.1,
#             alpha=0.5)
# plt.scatter(embedding[clustered, 0],
#             embedding[clustered, 1],
#             c=labels[clustered],
#             s=0.1,
#             cmap='Spectral');
# 
# plt.title('PCA_HDBCAN')
# plt.savefig(dir_path+result_path+'/PCA_HDBCAN.png')
# plt.savefig(dir_path+result_path+'/PCA_HDBCAN.svg', format='svg', dpi=1200)
# df_model['ClusterID_PCA_HDBCAN'] = labels 
# =============================================================================



#CAN
#from sklearn.manifold import TSNE
# Project the df: this step will take several seconds
#tsne = TSNE(n_components=2, init='random', random_state=0)
#embedding = tsne.fit_transform(df_model)
# =============================================================================
# model = DBSCAN(eps=3, min_samples=500).fit(lowd_df)
# labels = model.labels_
# clustered = (labels >= 0)
# plt.scatter(embedding[~clustered, 0],
#             embedding[~clustered, 1],
#             c=(0.5, 0.5, 0.5),
#             s=0.1,
#             alpha=0.5)
# plt.scatter(embedding[clustered, 0],
#             embedding[clustered, 1],
#             c=labels[clustered],
#             s=0.1,
#             cmap='Spectral');
# 
# 
# plt.title('DBSCAN')
# plt.savefig(dir_path+result_path+'/DBSCAN.png')
# plt.savefig(dir_path+result_path+'/DBSCAN.svg', format='svg', dpi=1200)
# df_model['ClusterID_DBCAN'] = labels
# =============================================================================







# Extracting duration per session
"""
when analysed about whether to take duration in hours or mins or seconds .
For Hours - 98.48 % of sessions was less than 1 hour. So taking hour would be skewed dist. 
For Minutes = 86.72% of sessions was less than 1 minute
For seconds = 85.33% of sessions was less than 1 second
for milliseconds = 85.025 %     ,,           ,,
So took mins as as a measure as representing as seconds/milliseconds offers than 2% exttra coverage

CODE TO CHECK :
s[['maxtime','mintime','duration','duration_mins']].head()
d_under1min = len(s[s['duration_mins']<1])
d = len(s)
print(d_under1min)
print(d)
print("percentage of session less than 1 min")
print(d_under1min/d*100)

"""


# one hot encoding of categorical variables
"""
For Categorical lets use:
    1. Just the actions alone to know how was his overall activity
    2. His product sku-actions combination to know action at product sku level
    
For Time related:
    1. Hour of Day
    2. Day of week
    3. Weekday or not

"""




"""
Check for session id per row
df_model = pd.read_csv(dir_path+result_path+'/feature_outputs.csv')
print(len(df_model[colid]))
# 31928 sessions
print(df_model.duplicated([colid]).any())

"""