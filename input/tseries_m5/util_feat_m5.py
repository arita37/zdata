
"""

Packages to use :

tsfresh

tsfel https://tsfel.readthedocs.io/en/latest/

sktime

feature tools : https://docs.featuretools.com/en/stable/automated_feature_engineering/handling_time.html

Cesium http://cesium-ml.org/docs/feature_table.html

Feature Tools for advacned fewatures `https://github.com/Featuretools/predict-remaining-useful-life/blob/master/Advanced%20Featuretools%20RUL.ipynb


"""

import pandas as pd
import tsfresh
from tsfresh import extract_relevant_features, extract_features
import numpy as np
import pdb
import re



def features_time_basic(df, input_raw_path = None, dir_out = None, features_group_name = None, auxiliary_csv_path = None, drop_cols = None, index_cols = None, merge_cols_mapping = None, cat_cols = None, id_cols = None, dep_col = None, max_rows = 10):
    df['date_t'] = pd.to_datetime(df['date'])
    df['year'] = df['date_t'].dt.year
    df['month'] = df['date_t'].dt.month
    df['week'] = df['date_t'].dt.week
    df['day'] = df['date_t'].dt.day
    df['dayofweek'] = df['date_t'].dt.dayofweek
    cat_cols = []
    return df[['year', 'month', 'week', 'day', 'dayofweek'] + id_cols], cat_cols




def features_lag(df, fname):
    out_df = df[['item_id', 'dept_id', 'cat_id', 'store_id', 'state_id']]
    ###############################################################################
    # day lag 29~57 day and last year's day lag 1~28 day
    day_lag = df.iloc[:,-28:]
    day_year_lag = df.iloc[:,-393:-365]
    day_lag.columns = [str("lag_{}_day".format(i)) for i in range(29,57)] # Rename columns
    day_year_lag.columns = [str("lag_{}_day_of_last_year".format(i)) for i in range(1,29)]

    # Rolling mean(3) and (7) and (28) and (84) 29~57 day and last year's day lag 1~28 day
    rolling_3 = df.iloc[:,-730:].T.rolling(3).mean().T.iloc[:,-28:]
    rolling_3.columns = [str("rolling3_lag_{}_day".format(i)) for i in range(29,57)] # Rename columns
    rolling_3_year = df.iloc[:,-730:].T.rolling(3).mean().T.iloc[:,-393:-365]
    rolling_3_year.columns = [str("rolling3_lag_{}_day_of_last_year".format(i)) for i in range(1,29)]

    rolling_7 = df.iloc[:,-730:].T.rolling(7).mean().T.iloc[:,-28:]
    rolling_7.columns = [str("rolling7_lag_{}_day".format(i)) for i in range(29,57)] # Rename columns
    rolling_7_year = df.iloc[:,-730:].T.rolling(7).mean().T.iloc[:,-393:-365]
    rolling_7_year.columns = [str("rolling7_lag_{}_day_of_last_year".format(i)) for i in range(1,29)]

    rolling_28 = df.iloc[:,-730:].T.rolling(28).mean().T.iloc[:,-28:]
    rolling_28.columns = [str("rolling28_lag_{}_day".format(i)) for i in range(29,57)]
    rolling_28_year = df.iloc[:,-730:].T.rolling(28).mean().T.iloc[:,-393:-365]
    rolling_28_year.columns = [str("rolling28_lag_{}_day_of_last_year".format(i)) for i in range(1,29)]

    rolling_84 = df.iloc[:,-730:].T.rolling(84).mean().T.iloc[:,-28:]
    rolling_84.columns = [str("rolling84_lag_{}_day".format(i)) for i in range(29,57)]
    rolling_84_year = df.iloc[:,-730:].T.rolling(84).mean().T.iloc[:,-393:-365]
    rolling_84_year.columns = [str("rolling84_lag_{}_day_of_last_year".format(i)) for i in range(1,29)]

    # monthly lag 1~18 month
    month_lag = pd.DataFrame({})
    for i in range(1,19):
        if i == 1:
            monthly = df.iloc[:,-28*i:].T.sum().T
            month_lag["monthly_lag_{}_month".format(i)] = monthly
        else:
            monthly = df.iloc[:, -28*i:-28*(i-1)].T.sum().T
            month_lag["monthly_lag_{}_month".format(i)] = monthly

    # combine day lag and monthly lag
    out_df = pd.concat([out_df, day_lag], axis=1)
    out_df = pd.concat([out_df, day_year_lag], axis=1)
    out_df = pd.concat([out_df, rolling_3], axis=1)
    out_df = pd.concat([out_df, rolling_3_year], axis=1)
    out_df = pd.concat([out_df, rolling_7], axis=1)
    out_df = pd.concat([out_df, rolling_7_year], axis=1)
    out_df = pd.concat([out_df, rolling_28], axis=1)
    out_df = pd.concat([out_df, rolling_28_year], axis=1)
    out_df = pd.concat([out_df, rolling_84], axis=1)
    out_df = pd.concat([out_df, rolling_84_year], axis=1)
    out_df = pd.concat([out_df, month_lag], axis=1)

    ###############################################################################
    # dept_id
    group_dept = df.groupby("dept_id").sum()
    # day lag 29~57 day and last year's day lag 1~28 day
    dept_day_lag = group_dept.iloc[:,-28:]
    dept_day_year_lag = group_dept.iloc[:,-393:-365]
    dept_day_lag.columns = [str("dept_lag_{}_day".format(i)) for i in range(29,57)]
    dept_day_year_lag.columns = [str("dept_lag_{}_day_of_last_year".format(i)) for i in range(1,29)]
    # monthly lag 1~18 month
    month_dept_lag = pd.DataFrame({})
    for i in range(1,19):
        if i == 1:
            monthly_dept = group_dept.iloc[:,-28*i:].T.sum().T
            month_dept_lag["dept_monthly_lag_{}_month".format(i)] = monthly_dept
        elif i >= 7 and i < 13:
            continue
        else:
            monthly = group_dept.iloc[:, -28*i:-28*(i-1)].T.sum().T
            month_dept_lag["dept_monthly_lag_{}_month".format(i)] = monthly_dept
    # combine out df
    out_df = pd.merge(out_df, dept_day_lag, left_on="dept_id", right_index=True, how="left")
    out_df = pd.merge(out_df, dept_day_year_lag, left_on="dept_id", right_index=True, how="left")
    out_df = pd.merge(out_df, month_dept_lag, left_on="dept_id", right_index=True, how="left")

    ###############################################################################
    # cat_id
    group_cat = df.groupby("cat_id").sum()
    # day lag 29~57 day and last year's day lag 1~28 day
    cat_day_lag = group_cat.iloc[:,-28:]
    cat_day_year_lag = group_cat.iloc[:,-393:-365]
    cat_day_lag.columns = [str("cat_lag_{}_day".format(i)) for i in range(29,57)]
    cat_day_year_lag.columns = [str("cat_lag_{}_day_of_last_year".format(i)) for i in range(1,29)]
    # monthly lag 1~18 month
    month_cat_lag = pd.DataFrame({})
    for i in range(1,19):
        if i == 1:
            monthly_cat = group_cat.iloc[:,-28*i:].T.sum().T
            month_cat_lag["cat_monthly_lag_{}_month".format(i)] = monthly_cat
        elif i >= 7 and i < 13:
            continue
        else:
            monthly_cat = group_cat.iloc[:, -28*i:-28*(i-1)].T.sum().T
            month_cat_lag["dept_monthly_lag_{}_month".format(i)] = monthly_cat

    # combine out df
    out_df = pd.merge(out_df, cat_day_lag, left_on="cat_id", right_index=True, how="left")
    out_df = pd.merge(out_df, cat_day_year_lag, left_on="cat_id", right_index=True, how="left")
    out_df = pd.merge(out_df, month_cat_lag, left_on="cat_id", right_index=True, how="left")

    ###############################################################################
    # store_id
    group_store = df.groupby("store_id").sum()
    # day lag 29~57 day and last year's day lag 1~28 day
    store_day_lag = group_store.iloc[:,-28:]
    store_day_year_lag = group_store.iloc[:,-393:-365]
    store_day_lag.columns = [str("store_lag_{}_day".format(i)) for i in range(29,57)]
    store_day_year_lag.columns = [str("store_lag_{}_day_of_last_year".format(i)) for i in range(1,29)]
    # monthly lag 1~18 month
    month_store_lag = pd.DataFrame({})
    for i in range(1,19):
        if i == 1:
            monthly_store = group_store.iloc[:,-28*i:].T.sum().T
            month_store_lag["store_monthly_lag_{}_month".format(i)] = monthly_store
        elif i >= 7 and i <13:
            continue
        else:
            monthly_store = group_store.iloc[:, -28*i:-28*(i-1)].T.sum().T
            month_store_lag["store_monthly_lag_{}_month".format(i)] = monthly_store

    # combine out df
    out_df = pd.merge(out_df, store_day_lag, left_on="store_id", right_index=True, how="left")
    out_df = pd.merge(out_df, store_day_year_lag, left_on="store_id", right_index=True, how="left")
    out_df = pd.merge(out_df, month_store_lag, left_on="store_id", right_index=True, how="left")

    ###############################################################################
    # state_id
    group_state = df.groupby("state_id").sum()
    # day lag 29~57 day and last year's day lag 1~28 day
    state_day_lag = group_state.iloc[:,-28:]
    state_day_year_lag = group_state.iloc[:,-393:-365]
    state_day_lag.columns = [str("state_lag_{}_day".format(i)) for i in range(29,57)]
    state_day_year_lag.columns = [str("state_lag_{}_day_of_last_year".format(i)) for i in range(1,29)]
    # monthly lag 1~18 month
    month_state_lag = pd.DataFrame({})
    for i in range(1,13):
        if i == 1:
            monthly_state = group_state.iloc[:,-28*i:].T.sum().T
            month_state_lag["state_monthly_lag_{}_month".format(i)] = monthly_state
        elif i >= 7 and i < 13:
            continue
        else:
            monthly_state = group_state.iloc[:, -28*i:-28*(i-1)].T.sum().T
            month_state_lag["state_monthly_lag_{}_month".format(i)] = monthly_state

    # combine out df
    out_df = pd.merge(out_df, state_day_lag, left_on="state_id", right_index=True, how="left")
    out_df = pd.merge(out_df, state_day_year_lag, left_on="state_id", right_index=True, how="left")
    out_df = pd.merge(out_df, month_state_lag, left_on="state_id", right_index=True, how="left")

    ###############################################################################
    # category flag
    col_list = ['dept_id', 'cat_id', 'store_id', 'state_id']

    df_cate_oh = pd.DataFrame({})
    for i in col_list:
        df_oh = pd.get_dummies(df[i])
        df_cate_oh = pd.concat([df_cate_oh, df_oh], axis=1)

    out_df = pd.concat([out_df, df_cate_oh], axis=1)

    out_df.to_parquet(fname)
    # return out_df


def _get_tsfresh_melted_features_single_row(single_row_df, index_cols):
    df_cols = single_row_df.columns.tolist()
    selected_cols = [x for x in df_cols if re.match("d_[0-9]",x)]
    single_row_df_T = single_row_df[selected_cols].T
    single_row_df_T["time"] = range(0, len(single_row_df_T.index))
    single_row_df_T["id"] = range(0, len(single_row_df_T.index))
    single_row_df_T.rename(columns={ single_row_df_T.columns[0]: "val" }, inplace = True)

    X_feat = extract_features(single_row_df_T, column_id='id', column_sort='time')

    feat_col_names = X_feat.columns.tolist()
    feat_col_names_mapping = {}
    for feat_col_name in feat_col_names:
        feat_col_names_mapping[feat_col_name] = feat_col_name.replace('"','').replace(',','')

    X_feat = X_feat.rename(columns = feat_col_names_mapping)
    X_feat_T = X_feat.T

    for col in index_cols:
        X_feat_T[col] = np.repeat(single_row_df[col].tolist()[0], len(X_feat_T.index))
    # X_feat_T["item_id"] = np.repeat(df_single_row["item_id"].tolist()[0], len(X_feat_T.index))
    # X_feat_T["id"] = np.repeat(df_single_row["id"].tolist()[0], len(X_feat_T.index))
    # X_feat_T["cat_id"] = np.repeat(df_single_row["cat_id"].tolist()[0], len(X_feat_T.index))
    # X_feat_T["dept_id"] = np.repeat(df_single_row["dept_id"].tolist()[0], len(X_feat_T.index))
    # X_feat_T["store_id"] = np.repeat(df_single_row["store_id"].tolist()[0], len(X_feat_T.index))
    # X_feat_T["state_id"] = np.repeat(df_single_row["state_id"].tolist()[0], len(X_feat_T.index))
    X_feat_T["variable"] = X_feat_T.index

    single_row_df["variable"] = pd.Series(["demand"])
    X_feat_T = X_feat_T.append(single_row_df, ignore_index= True)
    return X_feat_T.set_index(index_cols + ['variable']).rename_axis(['day'], axis=1).stack().unstack('variable').reset_index()


def _get_tsfresh_df_sales_melt(df_sales, dir_out, features_group_name, drop_cols, df_calendar, index_cols, merge_cols_mapping, id_cols):
    # X_feat = pd.DataFrame()
    auxiliary_dropped_cols = [x for x in df_calendar.columns.tolist() if x in drop_cols]
    df_calendar.drop(auxiliary_dropped_cols, inplace = True, axis = 1)

    for i in range(len(df_sales.index)):
        single_row_df = df_sales.loc[[i]]
        X_feat_single_row_df = _get_tsfresh_melted_features_single_row(single_row_df, index_cols)
        if i % 5 ==0:
            X_feat = X_feat_single_row_df
        if (i+1) % 5 == 0 :
            merged_df = pd.merge(X_feat, df_calendar, how = 'left', left_on = [merge_cols_mapping["left"]], right_on = [merge_cols_mapping["right"]])

            # merged_df = pd.concat([df_sales_val_melt, df_submi_val, df_submi_eval], axis = 0)
            # selected_cols = [x for x in merged_df.columns.tolist() if x not in drop_cols]
            selected_cols = [x for x in merged_df.columns.tolist() if x in id_cols or str(x).startswith("val__")]
            merged_df_selected_cols = merged_df[selected_cols]
            merged_df_selected_cols.columns = merged_df_selected_cols.columns.astype(str)
            merged_df_selected_cols.to_parquet(f'{dir_out}/{features_group_name}_{i}.parquet')
        else:
            X_feat.append(X_feat_single_row_df, ignore_index = True)
    return merged_df_selected_cols



def features_tsfresh(df, input_raw_path, dir_out, features_group_name, auxiliary_csv_path, drop_cols, index_cols, merge_cols_mapping, cat_cols = None, id_cols = None, dep_col = None, max_rows = 5):
    # df is taken as an argument to make it work in the existing pipeline of saving features in meta_csv
    df_sales_val              = pd.read_csv(input_raw_path)
    df_calendar               = pd.read_csv(auxiliary_csv_path)

    merged_df         = _get_tsfresh_df_sales_melt(df_sales_val[0:max_rows], dir_out, features_group_name, drop_cols, df_calendar, index_cols, merge_cols_mapping, id_cols)
    # df_calendar.drop(['weekday', 'wday', 'month', 'year'], inplace = True, axis = 1)
    # merged_df = pd.merge(df_sales_val_melt, df_calendar, how = 'left', left_on = ['day'], right_on = ['d'])
    # merged_df = pd.merge(df_sales_val_melt, df_calendar, how = 'left', left_on = [merge_cols_mapping["left"]], right_on = [merge_cols_mapping["right"]])

    # merged_df = pd.concat([df_sales_val_melt, df_submi_val, df_submi_eval], axis = 0)
    # selected_cols = [x for x in merged_df.columns.tolist() if x not in [ 'id', 'cat_id', 'dept_id', 'store_id', 'variable', 'day', 'demand', 'state_id']]
    selected_cols = [x for x in merged_df.columns.tolist() if x not in drop_cols]
    return merged_df[selected_cols], []


def features_tsfresh_select(df):
    df = df[['snap_CA', 'snap_TX', 'snap_WI', 'sell_price', 'item_id', 'date', 'store_id', 'id']]
    df = roll_time_series(df, column_id="item_id", column_sort="date")
    existing_cols = df.columns.tolist()
    y = df['demand']
    X_cols = [x for x in existing_cols if not x == "demand"]
    X = df[X_cols]
    X = X.fillna(value = {'sell_price' : X['sell_price'].mean(skipna = True)})
    X = X[['snap_CA', 'snap_TX', 'snap_WI', 'sell_price', 'item_id', 'date']]
    X_filtered = extract_relevant_features(X, y, column_id='item_id', column_sort='date')

    filtered_col_names = X_filtered.columns.tolist()

    filtered_col_names_mapping = {}

    for filtered_col_name in filtered_col_names:
        filtered_col_names_mapping[filtered_col_name] = filtered_col_name.replace('"','').replace(',','')

    X_filtered = X_filtered.rename(columns = filtered_col_names_mapping)
    # This is done because lightgbm can not have features with " in the feature name

    feature_df = pd.concat([X[['item_id', 'date']], X_filtered])

    return feature_df, []


def basic_time_features(df, input_raw_path = None, dir_out = None, features_group_name = None, auxiliary_csv_path = None, drop_cols = None, index_cols = None, merge_cols_mapping = None, cat_cols = None, id_cols = None, dep_col = None, max_rows = 10):

    df['date_t'] = pd.to_datetime(df['date'])
    df['year'] = df['date_t'].dt.year
    df['month'] = df['date_t'].dt.month
    df['week'] = df['date_t'].dt.week
    df['day'] = df['date_t'].dt.day
    df['dayofweek'] = df['date_t'].dt.dayofweek
    cat_cols = []
    return df[['year', 'month', 'week', 'day', 'dayofweek'] + id_cols], cat_cols


def features_mean(df, input_raw_path = None, max_rows = 10):
    pass


def identity_features(df, input_raw_path = None, dir_out = None, features_group_name = None, auxiliary_csv_path = None, drop_cols = None, index_cols = None, merge_cols_mapping = None, cat_cols = None, id_cols = None, dep_col = None, max_rows = 10):
    df_drop_cols = [x for x in df.columns.tolist() if x in drop_cols]
    df = df.drop(df_drop_cols, axis = 1)
    return df, cat_cols


def features_rolling(df, input_raw_path = None, dir_out = None, features_group_name = None, auxiliary_csv_path = None, drop_cols = None, index_cols = None, merge_cols_mapping = None, cat_cols = None, id_cols = None, dep_col = None, max_rows = 10):
    cat_cols = []
    created_cols = []

    len_shift = 28
    for i in [7,14,30,60,180]:
        print('Rolling period:', i)
        df['rolling_mean_'+str(i)] = df.groupby(['id'])[dep_col].transform(lambda x: x.shift(len_shift).rolling(i).mean())
        df['rolling_std_'+str(i)]  = df.groupby(['id'])[dep_col].transform(lambda x: x.shift(len_shift).rolling(i).std())
        created_cols.append('rolling_mean_'+str(i))
        created_cols.append('rolling_std_'+str(i))

    # Rollings
    # with sliding shift
    for len_shift in [1,7,14]:
        print('Shifting period:', len_shift)
        for len_window in [7,14,30,60]:
            col_name = 'rolling_mean_tmp_'+str(len_shift)+'_'+str(len_window)
            df[col_name] = df.groupby(['id'])[dep_col].transform(lambda x: x.shift(len_shift).rolling(len_window).mean())
            created_cols.append(col_name)


    for col_name in id_cols:
        created_cols.append(col_name)

    return df[created_cols], cat_cols



def features_lag(df, input_raw_path = None, dir_out = None, features_group_name = None, auxiliary_csv_path = None, drop_cols = None, index_cols = None, merge_cols_mapping = None, cat_cols = None, id_cols = None, dep_col = None, max_rows = 10):
    created_cols = []
    cat_cols = []

    lag_days = [col for col in range(28, 28+15)]
    for lag_day in lag_days:
        created_cols.append('lag_' + str(lag_day))
        df['lag_' + str(lag_day)] = df.groupby(['id'])[dep_col].transform(lambda x: x.shift(lag_day))


    for col_name in id_cols:
        created_cols.append(col_name)

    return df[created_cols], cat_cols


''''
import pandas as pd
import numpy as np
from sklearn import preprocessing
import datetime
from sklearn.preprocessing import MinMaxScaler
from keras.preprocessing.text import one_hot
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Flatten
from keras.layers.embeddings import Embedding
def date_transform:
	df=pd.read_csv('calendar.csv')
	df['date'] = pd.to_datetime(df['date'])
	df['year'] = df['date'].dt.year
	df['month'] = df['date'].dt.month
	df['week'] = df['date'].dt.week
	df['day'] = df['date'].dt.day
	df['dayofweek'] = df['date'].dt.dayofweek
	date_cols=df[['wday','month','year','week','day']]
	scaler = MinMaxScaler(feature_range=(-0.5,0.5))
	scaler.fit(date_cols)
	transformed=scaler.transform(date_cols)
	new_df=pd.DataFrame(transformed,columns=date_cols.columns)
	embeded=df[['event_name_1','event_type_1','event_name_2','event_type_2']]
	unique=embeded['event_name_1'].unique()
	unique_vals=np.append(unique,embeded['event_name_2'].unique())
	event=pd.DataFrame(unique_vals,)
	array_nm=event['event'].unique()
	list_array=str(list(array_nm))
	vocab_size = 100
	encoded_docs = [one_hot(d, vocab_size) for d in list_array]
	max_length = 4
	padded_docs1 = pad_sequences(encoded_docs, maxlen=max_length, padding='post')
	unique=embeded['event_type_1'].unique()
	unique_vals=np.append(unique,embeded['event_type_2'].unique())
	event=pd.DataFrame(unique_vals,)
	array_nm=event['event'].unique()
	list_array=str(list(array_nm))
	vocab_size = 100
	encoded_docs = [one_hot(d, vocab_size) for d in list_array]
	max_length = 4
	padded_docs2 = pad_sequences(encoded_docs, maxlen=max_length, padding='post')
	values=df[df['snap_CA','snap_TX']]
	return new_df,padded_docs1,padded_docs2,values
def sales:
	sales=pd.read_csv('sell_prices.csv')
	sales['dept_id']=str(sales['item_id'])
	def remove(x):
    	value=x.split('_')
   	    dept=value[0]+"_"+value[1]
    	return dept
    sales['dept']=sales['item_id'].apply(remove)
    price_sum=pd.DataFrame(sales.groupby('wm_yr_wk')['sell_price'].sum())
    scaler = MinMaxScaler()
    scaler.fit(price_sum)
    price_sum['normalized']=scaler.transform(price_sum)
    x=pd.DataFrame(sales.groupby('dept')['sell_price'].unique())
	x.reset_index(inplace=True)
	def recive(v):
    	sum=0
    	for i in v:
        sum+=i
    	return round(sum,2)

	x['summed']=x['sell_price'].apply(recive)
	scaler = MinMaxScaler()
	x.set_index(keys='dept',inplace=True)
	x.drop('sell_price',inplace=True,axis=1)
	scaler.fit(x)
	x['transformed']=scaler.transform(x)
	return price_sum,x
def sales_validation:
	sales_eval=pd.read_csv('sales_train_validation.csv')
	TARGET='sales'
	index_columns = ['id','item_id','dept_id','cat_id','store_id','state_id']
	sales_eval = pd.melt(sales_eval,id_vars = index_columns,var_name = 'd',value_name = TARGET)
	temp_df = sales_eval[['id','d',TARGET]]
	#lag=1
	i=1
	print('Shifting:', i)
	temp_df['lag_'+str(i)] = temp_df.groupby(['id'])[TARGET].transform(lambda x: x.shift(i))
	#Moving avg=(7,28)
	temp_df1 = sales_eval[['id','d','sales']]
	for i in [7,28]:
    	print('Rolling period:', i)
    	temp_df['rolling_mean_'+str(i)] = temp_df1.groupby(['id'])[TARGET].transform(lambda x: x.shift(1).rolling(i).mean())
    	temp_df['rolling_std_'+str(i)]  = temp_df1.groupby(['id'])[TARGET].transform(lambda x: x.shift(1).rolling(i).std())
    	return temp_df
'''
