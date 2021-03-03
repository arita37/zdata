import pandas as pd, numpy as np, random, copy, os, sys
random.seed(100)


########################################################################################################
def generate_train(df, col=None, pars=None):
   """
     By IP Apress
           channel                                  1011
      os                                        544
      hour                                      472
      app                                       468
      ip_app_os_device_day_click_time_next_1     320
      app_channel_os_mean_is_attributed         189
      ip_app_mean_is_attributed                 124
      ip_app_os_device_day_click_time_next_2     120
      ip_os_device_count_click_id               113
      ip_var_hour                                94
      ip_day_hour_count_click_id                 91
      ip_mean_is_attributed                      74
      ip_count_click_id                          73
      ip_app_os_device_day_click_time_lag1       67
      app_mean_is_attributed                     67
      ip_nunique_os_device                       65
      ip_nunique_app                             63
      ip_nunique_os                              51
      ip_nunique_app_channel                     49
      ip_os_device_mean_is_attributed            46
      device                                     41
      app_channel_os_count_click_id              37
      ip_hour_mean_is_attributed                 21


   """
   df = copy.deepcopy(df)

   df = df.drop(columns=["attributed_time"])
   print(df.columns)

   ## Let's see on which hour the click was happend
   df["click_time"]   = pd.to_datetime(df["click_time"])
   df["hour"]         = df["click_time"].dt.hour.astype("uint8")
   #df["minute"]       = df["click_time"].dt.minute.astype("uint8")
   df["day"]          = df["click_time"].dt.day.astype("uint8")
   df["dayweek"]      = df["click_time"].dt.dayofweek.astype("uint8")


   #### By IP Address
   df1 = df.groupby(['ip', 'app', 'os', 'device', 'channel']).agg( {  'is_attributed' : 'max' }).reset_index()     #   'hour' : {'min', 'max'} } )
   # print(df1.head(3).T )

   GROUPBY_AGGREGATIONS = [

        {'groupby': ['ip'],          'select': 'channel', 'agg': 'nunique'},
        {'groupby': ['ip'],          'select': 'app',     'agg': 'nunique'},
        {'groupby': ['ip'],          'select': 'device',  'agg': 'nunique'},

        {'groupby': ['ip', 'app'],             'select': 'channel', 'agg': 'count'},
        {'groupby': ['ip', 'app', 'os'],       'select': 'channel', 'agg': 'count'},

        {'groupby': ['ip'], 'select': 'app',     'agg': lambda x: float(len(x)) / len(x.unique()), 'agg_name': 'AvgViewPerDistinct'},
        {'groupby': ['ip'], 'select': 'channel', 'agg': lambda x: float(len(x)) / len(x.unique()), 'agg_name': 'AvgViewPerDistinct'},

        {'groupby': ['ip',   'app'], 'select': 'os', 'agg': 'nunique'},
        {'groupby': ['ip',  'device', 'os'], 'select': 'app', 'agg': 'nunique'},


        ##### Time Related
        {'groupby': ['ip'],    'select': 'hour', 'agg': 'max'},
        {'groupby': ['ip'],    'select': 'hour', 'agg': 'min'},

        {'groupby': ['ip'],    'select': 'dayweek', 'agg': 'max'},
        {'groupby': ['ip'],    'select': 'dayweek', 'agg': 'min'},

        {'groupby': ['ip'], 'select': 'hour',    'agg': lambda x: float(len(x)) / len(x.unique()), 'agg_name': 'AvgViewPerDistinct'},
        {'groupby': ['ip'], 'select': 'dayweek', 'agg': lambda x: float(len(x)) / len(x.unique()), 'agg_name': 'AvgViewPerDistinct'},


        {'groupby': ['ip','channel'],    'select': 'hour', 'agg': 'max'},
        {'groupby': ['ip','channel'],    'select': 'hour', 'agg': 'min'},

        {'groupby': ['ip','channel'],    'select': 'dayweek', 'agg': 'max'},
        {'groupby': ['ip','channel'],    'select': 'dayweek', 'agg': 'min'},


        ################### V2 - GroupBy Features #
        # Average clicks on app by distinct users; is it an app they return to?
        {'groupby': ['app'], 'select': 'ip', 'agg': lambda x: float(len(x)) / len(x.unique()), 'agg_name': 'AvgViewPerDistinct'},
        {'groupby': ['app'],         'select': 'channel', 'agg': 'count'},
        {'groupby': ['app'],         'select': 'channel', 'agg': 'nunique'},


        {'groupby': ['channel'],     'select': 'app',     'agg': 'count'},

        # {'groupby': ['ip',  'device','os'],  'select': 'app', 'agg': 'cumcount'},
        # {'groupby': ['ip'], 'select': 'app', 'agg': 'cumcount'},
        # {'groupby': ['ip'], 'select': 'os',  'agg': 'cumcount'}
  ]

   # Apply all the groupby transformations
   for spec in GROUPBY_AGGREGATIONS:
      agg_name    = spec['agg_name'] if 'agg_name' in spec else spec['agg']
      new_feature = '{}_{}_{}'.format('_'.join(spec['groupby']), agg_name, spec['select'])

      print("Grouping by {}, and aggregating {} with {}".format( spec['groupby'], spec['select'], agg_name ))

      all_features = list(set(spec['groupby'] + [spec['select']]))

      # Perform the groupby
      gp = df[all_features].groupby(spec['groupby'])[spec['select']]. \
          agg(spec['agg']).reset_index(). \
          rename(index=str, columns={spec['select']: new_feature})
      
      if 'cumcount' == spec['agg']:
          df1[new_feature] = gp[0].values
      else:
          df1 = df1.merge(gp, on=spec['groupby'], how='left')
          # print( df1.head(3).T )
   del gp



   dfnew    = df1
   col_pars = {}
   return dfnew, col_pars




def generatePastClickFeatures(df):
    # Time between past clicks
    df2 = pd.DataFrame([], index= df.index)  
    pastClickAggregateFeatures = [
        {'groupBy': ['ip', 'channel']},
        {'groupBy': ['ip', 'os']}
    ]
    for spec in pastClickAggregateFeatures:
        feature_name = '{}-past-click'.format('_'.join(spec['groupBy']))   
        df[feature_name] = df[spec['groupBy'] + ['click_time']].groupby(['ip']).click_time.transform(lambda x: x.diff().shift(1)).dt.seconds
    return df2




# ### 2) Number of events in the past six hours 
# The first feature you'll be creating is the number of events from the same IP in the last six hours. It's likely that someone who is visiting often will download the app.
# Implement a function `count_past_events` that takes a Series of click times (timestamps) and returns another Series with the number of events in the last six hours. **Tip:** The `rolling` method is useful for this.


def count_past_events(series):
    new_series = pd.Series(index=series, data=series.index, name="count_6_hours").sort_index()
    print(new_series.head())
    count_6_hours = new_series.rolling('6h').count() - 1
    return count_6_hours



# Because this can take a while to calculate on the full data, we'll load pre-calculated versions in the cell below to test model performance.
# Loading in from saved Parquet file
# past_events = pd.read_parquet('../input/feature-engineering-data/past_6hr_events.pqt')
# clicks['ip_past_6hr_counts'] = past_events


#train, valid, test = get_data_splits(clicks)
#_ = train_model(train, valid)
valid_name_model='V11 FIN ip_past_6hr_counts'
my_own_train_plot_model(clicks, valid_name_model, my_own_metrics)


# ### 3) Features from future information 
# In the last exercise you created a feature that looked at past events. You could also make features that use information from events in the future. Should you use future events or not? 


# ### 4) Time since last event
# Implement a function `time_diff` that calculates the time since the last event in seconds from a Series of timestamps. This will be ran like so:
# ```python
# timedeltas = clicks.groupby('ip')['click_time'].transform(time_diff)
# ```


def time_diff(series):
    """Returns a series with the time since the last timestamp in seconds."""
    time_since_last_event = series.diff().dt.total_seconds()
    return time_since_last_event


# We'll again load pre-computed versions of the data, which match what your function would return
# Loading in from saved Parquet file
past_events = pd.read_parquet('../input/feature-engineering-data/time_deltas.pqt')
clicks['past_events_6hr'] = past_events

#train, valid, test = get_data_splits(clicks.join(past_events))
#_ = train_model(train, valid)





"""
New features
parallelizable_feature_map = {
    'ip'                            : Ip,
    'ip_for_filtering'              : IpForFiltering,
    'app'                           : App,
    'os'                            : Os,
    'device'                        : Device,
    'channel'                       : Channel,
    'hour'                          : ClickHour,
    'click_time'                    : ClickTime,
    'minute'                        : ClickSecond,
    'second'                        : ClickMinute,
    'count'                         : BasicCount,
    'is_attributed'                 : IsAttributed,
    'zero-minute'                   : ZeroMinute,
    'future_click_count_1'          : features.time_series_click.generate_future_click_count(60),
    'future_click_count_10'         : features.time_series_click.generate_future_click_count(600),
    'past_click_count_10'           : features.time_series_click.generate_past_click_count(600),
    'future_click_count_80'         : features.time_series_click.generate_future_click_count(4800),
    'past_click_count_80'           : features.time_series_click.generate_past_click_count(4800),
    'future_click_ratio_10'         : features.time_series_click.generate_future_click_ratio(600),
    'past_click_ratio_10'           : features.time_series_click.generate_future_click_ratio(600),
    'future_click_ratio_80'         : features.time_series_click.generate_future_click_ratio(4800),
    'past_click_ratio_80'           : features.time_series_click.generate_future_click_ratio(4800),
    'next_click_time_delta'         : features.time_series_click.NextClickTimeDelta,
    'prev_click_time_delta'         : features.time_series_click.PrevClickTimeDelta,
    'exact_same_click'              : features.time_series_click.ExactSameClick,  # It will be duplicated with all id counts
    'exact_same_click_id'           : features.time_series_click.ExactSameClickId,
    'all_click_count'               : features.time_series_click.AllClickCount,
    'hourly_click_count'            : features.time_series_click.HourlyClickCount,
    'average_attributed_ratio'      : features.time_series_click.AverageAttributedRatio,
    'cumulative_click_count'        : features.time_series_click.CumulativeClickCount,
    'cumulative_click_count_future' : features.time_series_click.CumulativeClickCountFuture,
    'median_attribute_time'         : features.time_series_click.MedianAttributeTime,
    'median_attribute_time_past'    : features.time_series_click.MedianAttributeTimePast,
    'median_attribute_time_past_v2' : features.time_series_click.MedianAttributeTimePastV2,
    'duplicated_row_index_diff'     : DuplicatedRowIndexDiff
}

unparallelizable_feature_map = {
    'komaki_lda_5'                    : features.category_vector.KomakiLDA5,
    'komaki_lda_10_ip'                : features.category_vector.KomakiLDA10_Ip,
    'komaki_lda_5_no_device'          : features.category_vector.KomakiLDA5NoDevice,
    'komaki_lda_10_no_device_1'       : features.category_vector.KomakiLDA10NoDevice_1,
    'komaki_lda_10_no_device_2'       : features.category_vector.KomakiLDA10NoDevice_2,
    'komaki_lda_20_ip'                : features.category_vector.KomakiLDA20_Ip,
    'komaki_lda_20_no_device_ip'      : features.category_vector.KomakiLDA20NoDevice_Ip,
    'komaki_lda_20_no_device_os'      : features.category_vector.KomakiLDA20NoDevice_Os,
    'komaki_lda_20_no_device_channel' : features.category_vector.KomakiLDA20NoDevice_Channel,
    'komaki_lda_20_no_device_app'     : features.category_vector.KomakiLDA20NoDevice_App,
    'komaki_lda_30_ip'                : features.category_vector.KomakiLDA30_Ip,
    'komaki_pca_5'                    : features.category_vector.KomakiPCA5,
    'komaki_pca_5_no_device'          : features.category_vector.KomakiPCA5NoDevice,
    'komaki_nmf_5'                    : features.category_vector.KomakiNMF5,
    'komaki_nmf_5_no_device'          : features.category_vector.KomakiNMF5NoDevice,
    'single_pca_count'                : features.category_vector.SinglePCACount,
    'single_pca_tfidf'                : features.category_vector.SinglePCATfIdf,
    'komaki_lda_5_mindf_1'            : features.category_vector.KomakiLDA5MinDF1,
    "user_item_lda_30"                : features.category_vector.UserItemLDA,
    "item_user_lda_30"                : features.category_vector.ItemUserLDA,
}


"""



################################################################################################
##### Train sample data ########################################################################
coly = "is_attributed"


def a10m():
  ##### Test Data  #############################################
  df = pd.read_csv("raw/raw_10m.zip")
  df, col_pars = generate_train(df)
  df_X = df.drop(coly, axis=1)
  df_y = df[[coly]]


  path = "train_10m/"
  os.makedirs(path, exist_ok=True)
  df_X.to_parquet( f"{path}/features.parquet")
  df_y.to_parquet( f"{path}/target.parquet")



def a100k():
  df   = pd.read_csv("raw/train_100k.csv")

  df, col_pars = generate_train2(df)
  df_X         = df.drop(coly, axis=1)
  df_y         = df[[coly]]
  print( np.sum(df_y))
  print( df_X.shape )
  print(df_X.columns)

  path = "train_100k/"
  os.makedirs(path, exist_ok=True)
  df_X.to_parquet( f"{path}features.parquet")
  df_y.to_parquet( f"{path}/target.parquet")

  df_X.to_csv( f"{path}features.csv")
  df_y.to_csv( f"{path}/target.csv")




def a200m():
  ##### Train data  ############################################
  df           = pd.read_csv("raw/train_200m.zip")
  df, col_pars = generate_train(df)
  df_X         = df.drop(coly, axis=1)
  df_y         = df[[coly]]


  path = "train_200m/"
  os.makedirs(path, exist_ok=True)
  df_X.to_parquet( f"{path}/features.parquet")
  df_y.to_parquet( f"{path}/target.parquet")




###########################################################################################################
###########################################################################################################
if __name__ == "__main__":

    import fire
    fire.Fire()
    








###### Load data  #############################################################
"""
dtypes = {'ip': np.uint32, 'app': np.uint16, 'device': np.uint8, 'os': np.uint8, 'channel': np.uint8, 'is_attributed': np.bool}
df     = pd.read_csv('raw/train_100k.csv', sep=',', dtype=dtypes, parse_dates=['click_time', 'attributed_time'])


df2 = generateAggregateFeatures(df)
df3 = generatePastClickFeatures(df)


df  = df.join(  df2, how='left' )
df  = df.join(  df3, how='left' )


"""





########################################################################################################
def generate_train2(df, col=None, pars=None):
   df = copy.deepcopy(df)

   df = df.drop(columns=["attributed_time"])

   ## Let's see on which hour the click was happend
   df["click_time"]   = pd.to_datetime(df["click_time"])

   df["hour"]         = df["click_time"].dt.hour.astype("uint8")
   df["minute"]       = df["click_time"].dt.minute.astype("uint8")
   # df["second"]       = df["click_time"].dt.second.astype("uint8")
   df["day"]          = df["click_time"].dt.day.astype("uint8")
   df["day_of_week"]  = df["click_time"].dt.dayofweek.astype("uint8")


   ##Let's divide the day in four section ,See in which section click has happend ")
   day_section = 0
   for start_time, end_time in zip([0, 6, 12, 18], [6, 12, 18, 24]):
              df.loc[(df['hour'] >= start_time) & (df['hour'] < end_time), 'day_section'] = day_section
              day_section += 1

   print( "clicks count features")
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
   print( "new clicks count features")
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
        {'groupby': ['ip'],  'select': 'channel', 'agg': 'nunique'},
        {'groupby': ['ip'],  'select': 'app', 'agg': 'nunique'},
        {'groupby': ['ip',   'day'], 'select': 'hour', 'agg': 'nunique'},
        {'groupby': ['ip',   'app'], 'select': 'os', 'agg': 'nunique'},
        {'groupby': ['ip'],  'select': 'device', 'agg': 'nunique'},
        {'groupby': ['app'], 'select': 'channel', 'agg': 'nunique'},
        {'groupby': ['ip',  'device', 'os'], 'select': 'app', 'agg': 'nunique'},
        {'groupby': ['ip',  'device','os'], 'select': 'app', 'agg': 'cumcount'},
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





   ###############################################################################
   #### coly  genereate  #########################################################
   df['is_attributed'].value_counts(normalize=True)  ### ?
   df["is_attributed"]=df["is_attributed"].astype("uint8")

   dfnew    = df
   col_pars = {}
   return dfnew, col_pars


def generateAggregateFeatures(df):
    ### New Agg features
    df2 = pd.DataFrame([], index= df.index)
    aggregateFeatures = [
        # How popular is the app in channel?
        {'name': 'app-popularity', 'groupBy': ['app'], 'select': 'channel', 'agg': 'count'},
        # How popular is the channel in app?
        {'name': 'channel-popularity', 'groupBy': ['channel'], 'select': 'app', 'agg': 'count'},
        # Average clicks on app by distinct users; is it an app they return to?
        {'name': 'avg-clicks-on-app', 'groupBy': ['app'], 'select': 'ip', 'agg': lambda x: float(len(x)) / len(x.unique())}
    ]

    for spec in aggregateFeatures:
        print("Generating aggregate feature {} group by {}, and aggregating {} with {}".format(spec['name'], spec['groupBy'], spec['select'], spec['agg']))
        gp = df[spec['groupBy'] + [spec['select']]] \
            .groupby(by=spec['groupBy'])[spec['select']] \
            .agg(spec['agg']) \
            .reset_index() \
            .rename(index=str, columns={spec['select']: spec['name']})
        df2 = df2.merge(gp, on=spec['groupBy'], how='left')
        del gp
        gc.collect()

    return df2