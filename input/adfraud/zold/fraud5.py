
# ## Please select an option before submitting results to the competition


submit_flag = True #False #True
print(submit_flag)


# # TalkingData AdTracking Fraud Detection Challenge
# # Can you detect fraudulent click traffic for mobile app ads?
# # https://www.kaggle.com/c/talkingdata-adtracking-fraud-detection


# **This notebook is inspired by an exercise in the [Feature Engineering](https://www.kaggle.com/learn/feature-engineering) course**  
# **You can reference the tutorial at [this link](https://www.kaggle.com/matleonard/feature-generation)**  
# **You can reference my notebook at [this link](https://www.kaggle.com/georgezoto/feature-engineering-feature-generation)**  
# 
# ---
# 


# <center><a href="https://www.kaggle.com/c/talkingdata-adtracking-fraud-detection"><img src="https://i.imgur.com/srKxEkD.png" width=600px></a></center>


# # Introduction
# 
# In this set of exercises, you'll create new features from the existing data. Again you'll compare the score lift for each new feature compared to a baseline model. First off, run the cells below to set up a baseline dataset and model.


import numpy as np
import pandas as pd
from sklearn import preprocessing, metrics
import lightgbm as lgb

import matplotlib.pyplot as plt
plt.rcParams["figure.figsize"] = (16,9)




# ## Helpful content packed methods used throughout the notebook 😀
def get_data_splits(dataframe, valid_fraction=0.1):

    dataframe = dataframe.sort_values('click_time')
    valid_rows = int(len(dataframe) * valid_fraction)
    train = dataframe[:-valid_rows * 2]
    # valid size == test size, last two sections of the data
    valid = dataframe[-valid_rows * 2:-valid_rows]
    test = dataframe[-valid_rows:]
    
    return train, valid, test

def train_model(train, valid, test=None, feature_cols=None, valid_name_model='Baseline Model'):
    if feature_cols is None:
        feature_cols = train.columns.drop(['click_time', 'attributed_time',
                                           'is_attributed'])
    dtrain = lgb.Dataset(train[feature_cols], label=train['is_attributed'])
    dvalid = lgb.Dataset(valid[feature_cols], label=valid['is_attributed'])
    
    param = {'num_leaves': 64, 'objective': 'binary', 
             'metric': 'auc', 'seed': 7}
    num_round = 1000
    
    #Record eval results for plotting
    validation_metrics = {} 
    
    print("Training model. Hold on a minute to see the validation score")
    bst = lgb.train(param, dtrain, num_round, valid_sets=[dvalid], valid_names=valid_name_model,
                    early_stopping_rounds=20, evals_result=validation_metrics, verbose_eval=False)
    
    valid_pred = bst.predict(valid[feature_cols])
    valid_score = metrics.roc_auc_score(valid['is_attributed'], valid_pred)
    print(f"Validation AUC score: {valid_score}")
    
    if test is not None: 
        test_pred = bst.predict(test[feature_cols])
        test_score = metrics.roc_auc_score(test['is_attributed'], test_pred)
        return bst, valid_score, test_score, validation_metrics
    else:
        return bst, valid_score, validation_metrics


def my_own_train_plot_model(clicks, valid_name_model, my_own_metrics):
    #valid_name_model='V11 FI Numerical ip_past_6hr_counts Model'
    print(valid_name_model+' score')

    train, valid, test = get_data_splits(clicks)
    bst, valid_score, validation_metrics = train_model(train, valid, valid_name_model=valid_name_model)

    my_own_metrics[valid_name_model] = valid_score
    print(my_own_metrics)
    plot_model_information(bst, validation_metrics, my_own_metrics)
    
    return bst


# ## Model information


def plot_model_information(bst, validation_metrics, my_own_metrics):
    print('Number of trees:', bst.num_trees())
    
    print('Plot model performance')
    ax = lgb.plot_metric(validation_metrics, metric='auc');
    plt.show()
    
    print('Plot feature importances...')
    ax = lgb.plot_importance(bst, max_num_features=15)
    plt.show()
    
    def plot_my_own_metrics(my_own_metrics):
        x=list(my_own_metrics.keys())
        y=list(my_own_metrics.values())
        plt.barh(x, y);

        for index, value in enumerate(y):
            plt.text(value, index, str(value))

    print('plot_my_own_metrics')    
    plot_my_own_metrics(my_own_metrics)
    plt.show()
    
    tree_index = 0
    print('Plot '+str(tree_index)+'th tree...')  # one tree use categorical feature to split
    ax = lgb.plot_tree(bst, tree_index=tree_index, figsize=(64, 36), show_info=['split_gain'])
    plt.show()






# Create features from   timestamps
click_data = pd.read_csv('../input/feature-engineering-data/train_100k.csv',
                         parse_dates=['click_time'])
click_times = click_data['click_time']
clicks = click_data.assign(day=click_times.dt.day.astype('uint8'),
                           hour=click_times.dt.hour.astype('uint8'),
                           minute=click_times.dt.minute.astype('uint8'),
                           second=click_times.dt.second.astype('uint8'))

# Label encoding for categorical features
cat_features = ['ip', 'app', 'device', 'os', 'channel']
for feature in cat_features:
    label_encoder = preprocessing.LabelEncoder()
    clicks[feature] = label_encoder.fit_transform(clicks[feature])


clicks.shape


clicks.head()


clicks['is_attributed'].value_counts()


clicks['is_attributed'].value_counts(normalize=True)


# ## Competition data


#Read only first limit rows
limit = 20_000_000

#Read only these columns - skip attributed_time 
usecols = ['ip', 'app', 'device', 'os', 'channel', 'click_time', 'is_attributed']


df = pd.read_csv('../input/talkingdata-adtracking-fraud-detection/train.csv', 
                               nrows=limit, 
                               usecols=usecols, 
                               parse_dates=['click_time'])


df['is_attributed'].value_counts()


df['is_attributed'].value_counts(normalize=True)


df_test = pd.read_csv('../input/talkingdata-adtracking-fraud-detection/test.csv', 
                                    parse_dates=['click_time'])



# Add new columns for timestamp features day, hour, minute, and second
df_test = df_test.copy()
df_test['day'] = df_test['click_time'].dt.day.astype('uint8')
# Fill in the rest
df_test['hour'] = df_test['click_time'].dt.hour.astype('uint8')
df_test['minute'] = df_test['click_time'].dt.minute.astype('uint8')
df_test['second'] = df_test['click_time'].dt.second.astype('uint8')


df_test.shape
df_test.head()


my_own_metrics = {}
valid_name_model='Baseline LightGBM Model'
bst = my_own_train_plot_model(clicks, valid_name_model, my_own_metrics)


# ### 1) Add interaction features
# Here you'll add interaction features for each pair of categorical features (ip, app, device, os, channel). The easiest way to iterate through the pairs of features is with `itertools.combinations`. For each new column, join the values as strings with an underscore, so 13 and 47 would become `"13_47"`. As you add the new columns to the dataset, be sure to label encode the values.


# ### Data leakage using clicks/entire dataset to LabelEncode ???
# Not the best solution to ValueError: y contains previously unseen labels: [0, 1, 2,...
unknown_value = -1 #Make sure this is int (as other labels) or you will not be able to predict in the end ⚠️


import itertools

cat_features = ['ip', 'app', 'device', 'os', 'channel']
interactions = pd.DataFrame(index=clicks.index)

# Iterate through each pair of features, combine them into interaction features
for interaction_feature_tuple in itertools.combinations(cat_features,2):
    #New feature name as concatination of 2 categorical features
    interaction_feature  = '_'.join(list(interaction_feature_tuple))
    print(interaction_feature_tuple, interaction_feature)
    
    #New interaction as concatination of the values of each combination of cateforical features
    interactions_values = clicks[interaction_feature_tuple[0]].astype(str) + '_' + clicks[interaction_feature_tuple[1]].astype(str)
    
    #New label encoder for each interaction_feature 
    label_enc = preprocessing.LabelEncoder()
    #interactions = interactions.assign(interaction_feature=label_enc.fit_transform(interactions_values)) ??? uses the string interaction_feature as the column name ???
    #interactions[interaction_feature] = label_enc.fit_transform(interactions_values)                     #??? index values and how do they relate to the full dataset clicks ???

    #Fit on all possible values of this feature
    label_enc.fit(interactions_values)
    #Create LabelEncoder of input to output
    le_dict = dict(zip(label_enc.classes_, label_enc.transform(label_enc.classes_)))
    #Encode unseen values to the unknown_value label
    encoded = interactions_values.apply(lambda x: le_dict.get(x, unknown_value))
    clicks[interaction_feature] = encoded
    
    print('clicks.head()')
    print(clicks.head())

    #Competition submission
    # Apply encoding to the competition test dataset
    comp_interactions_values = df_test[interaction_feature_tuple[0]].astype(str) + '_' + df_test[interaction_feature_tuple[1]].astype(str)
    #df_test[interaction_feature] = label_enc.transform(comp_interactions_values)  #??? ValueError: y contains previously unseen labels: '119901_9' ???
    
    competition_encoded = comp_interactions_values.apply(lambda x: le_dict.get(x, unknown_value))
    df_test[interaction_feature] = competition_encoded
    print('df_test.head()')
    print(df_test.head())


# ## How many unknown_value did we get in the test dataset?


train_ip_labels_unknowns = sum(clicks['ip_app'] == unknown_value)
train_ip_labels_unknowns


compet_test_ip_labels_unknowns = sum(df_test['ip_app'] == unknown_value)
compet_test_ip_labels_unknowns





valid_name_model='V10 FI Categorical Model'
my_own_train_plot_model(clicks, valid_name_model, my_own_metrics)


# # Generating numerical features 
# Adding interactions is a quick way to create more categorical features from the data. It's also effective to create new numerical features, you'll typically get a lot of improvement in the model. This takes a bit of brainstorming and experimentation to find features that work well.
# For these exercises I'm going to have you implement functions that operate on Pandas Series. It can take multiple minutes to run these functions on the entire data set so instead I'll provide feedback by running your function on a smaller dataset.


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
past_events = pd.read_parquet('../input/feature-engineering-data/past_6hr_events.pqt')
clicks['ip_past_6hr_counts'] = past_events


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


valid_name_model='V12 FIN time_since_last_event'
my_own_train_plot_model(clicks, valid_name_model, my_own_metrics)




# ### 5) Number of previous app downloads
# It's likely that if a visitor downloaded an app previously, 
# it'll affect the likelihood they'll download one again. Implement a function `previous_attributions` that returns a Series with the number of times an app has been downloaded (`'is_attributed' == 1`) before the current event.
def previous_attributions(series):
    """Returns a series with the number of times an app has been downloaded."""
    print(series)
    print(series.expanding(min_periods=2).sum())
    sums = series.expanding(min_periods=2).sum() - series
    return sums


# Again loading pre-computed data.
# Loading in from saved Parquet file
past_events = pd.read_parquet('../input/feature-engineering-data/downloads.pqt')
#clicks['ip_past_6hr_counts'] = past_events ??? Typo to overwrite ???
clicks['prev_app_downloads'] = past_events 
       
#train, valid, test = get_data_splits(clicks)
#_ = train_model(train, valid)


valid_name_model = 'V13 FIN prev_app_downloads'
my_own_train_plot_model(clicks, valid_name_model, my_own_metrics)


# ### 6) Tree-based vs Neural Network Models
# So far we've been using LightGBM, a tree-based model. Would these features we've generated work well for neural networks as well as tree-based models?
# Now that you've generated a bunch of different features, you'll typically want to remove some of them to reduce the size of the model and potentially improve the performance. Next, I'll show you how to do feature selection using a few different methods such as L1 regression and Boruta.
# You know how to generate a lot of features. In practice, you'll frequently want to pare them down for modeling. Learn to do that in the **[Feature Selection lesson](https://www.kaggle.com/matleonard/feature-selection)**.
# *Have questions or comments? Visit the [Learn Discussion forum](https://www.kaggle.com/learn-forum/161443) to chat with other Learners.*



# ## Constrain additional training features to just categorical feature engineering features
clicks.head()
clicks.columns
df_test.columns
df_test.head()

valid_name_model = 'V10 Feature Eng Categorical Model'
bst              = my_own_train_plot_model(clicks, valid_name_model, my_own_metrics)


feature_cols = clicks.columns.drop(['click_time', 'attributed_time','is_attributed'])
feature_cols


# ## Submit test predictions to TalkingData AdTracking Fraud Detection Challenge competition using the limited train.csv records from this notebook
bst

df_pred    = bst.predict(df_test[feature_cols])
df_pred_df = pd.DataFrame(df_pred, columns=['is_attributed'])
df_pred_df

df_pred_df['click_id'] = df_test['click_id']
df_pred_df             = df_pred_df[['click_id', 'is_attributed']]
df_pred_df


pd.cut(df_pred_df['is_attributed'], bins=10).value_counts()
pd.cut(df_pred_df['is_attributed'], bins=10).value_counts().plot(kind='bar', rot=45);


if submit_flag == True:
    df_pred_df.to_csv('submission.csv', index=False)
    print('submission.csv generated successfully :)')





