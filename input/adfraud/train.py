"""

python train.py  train_lgb_class_imbalance



"""
import copy
import random
import time

import lightgbm as lgb
import numpy as np
import pandas as pd

from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import CondensedNearestNeighbour

from sklearn.model_selection import train_test_split
from sklearn import metrics
from matplotlib import pyplot as plt

random.seed(100)

## load df
df = pd.read_csv("raw/raw_10m.zip")


############  Feature Engineering  ###################################################################
from generate_train import generate_train

df, col_pars = generate_train(df)

df["is_attributed"] = df["is_attributed"].astype("uint8")
df_X                = df.drop("is_attributed", axis=1)
df_y                = df["is_attributed"]


##### Split  
train_X, test_X, train_y, test_y = train_test_split(df_X, df_y, stratify=df_y, test_size=0.15)
train_X, val_X, train_y, val_y   = train_test_split(train_X, train_y, stratify=train_y, test_size=0.1)


####################################################################################################
####################################################################################################



# ####################################################################################################
# ## Now we will test two methods to handle imbalance in the dataset first we use:
# ## 1) scale_pos_weight': 99  # because training df is extremely unbalanced
# ### Since the df is highly imbalanced we use lightgbm scale_pos_weight
#
def train_lgb_class_imbalance():
    dtrain = lgb.Dataset(train_X, train_y)
    dvalid = lgb.Dataset(val_X, val_y)

    def lgb_modelfit_nocv(params, dtrain, dvalid, objective='binary', metrics='auc',
                          feval=None, early_stopping_rounds=20, num_boost_round=3000, verbose_eval=10):
        lgb_params = {
            'boosting_type': 'gbdt',
            'objective': objective,
            'metric': metrics,
            'learning_rate': 0.01,
            'num_leaves': 31,  # we should let it be smaller than 2^(max_depth)
            'max_depth': -1,  # -1 means no limit
            'min_child_samples': 20,  # Minimum number of df need in a child(min_data_in_leaf)
            'max_bin': 255,  # Number of bucketed bin for feature values
            'subsample': 0.6,  # Subsample ratio of the training instance.
            'subsample_freq': 0,  # frequence of subsample, <=0 means no enable
            'colsample_bytree': 0.3,  # Subsample ratio of columns when constructing each tree.
            'min_child_weight': 5,  # Minimum sum of instance weight(hessian) needed in a child(leaf)
            'subsample_for_bin': 200000,  # Number of samples for constructing bin
            'min_split_gain': 0,  # lambda_l1, lambda_l2 and min_gain_to_split to regularization
            'reg_alpha': 0,  # L1 regularization term on weights
            'reg_lambda': 0,  # L2 regularization term on weights
            'nthread': 4,
            'verbose': 0,
        }

        lgb_params.update(params)
        print("preparing validation datasets")
        xgtrain = dtrain  # we're using the feature engineered dataset not the genetic one
        xgvalid = dvalid

        evals_results = {}

        bst1 = lgb.train(lgb_params,
                         xgtrain,
                         valid_sets=[xgtrain, xgvalid],
                         valid_names=['train', 'valid'],
                         evals_result=evals_results,
                         num_boost_round=num_boost_round,
                         early_stopping_rounds=early_stopping_rounds,
                         verbose_eval=10,
                         feval=feval)

        n_estimators = bst1.best_iteration
        print("\nModel Report")
        print("n_estimators : ", n_estimators)
        print(metrics + ":", evals_results['valid'][metrics][n_estimators - 1])

        return bst1

    print("Starting the Training og LightGBM with class imbalance mitigation...")
    start_time = time.time()

    params = {
        'learning_rate': 0.15,
        # 'is_unbalance': 'true', # replaced with scale_pos_weight argument
        'num_leaves': 7,  # 2^max_depth - 1
        'max_depth': 3,  # -1 means no limit
        'min_child_samples': 100,  # Minimum number of df need in a child(min_data_in_leaf)
        'max_bin': 100,  # Number of bucketed bin for feature values
        'subsample': 0.7,  # Subsample ratio of the training instance.
        'subsample_freq': 1,  # frequence of subsample, <=0 means no enable
        'colsample_bytree': 0.9,  # Subsample ratio of columns when constructing each tree.
        'min_child_weight': 0,  # Minimum sum of instance weight(hessian) needed in a child(leaf)
        'scale_pos_weight': 99  # because training df is extremely unbalanced
    }
    bst = lgb_modelfit_nocv(params,
                            dtrain,
                            dvalid,
                            objective='binary',
                            metrics='auc',
                            early_stopping_rounds=30,
                            verbose_eval=True,
                            num_boost_round=500,
                            )

    print('[{}]: model training time'.format(time.time() - start_time))
    from sklearn import metrics

    ypred = bst.predict(test_X)
    score = metrics.f1_score(test_y, ypred)
    print(f"Test score: {score}")


# ####################################################################################################
# ## Now we will test two methods to handle imbalance in the dataset second we use:
# ## 2) Synthetic Minority Oversampling Technique (SMOTE) for Over-Sampling

def train_model_with_smote_oversampling():
    from imblearn.under_sampling import CondensedNearestNeighbour
    X_SMOTE_resampled, y_SMOTE_resampled = SMOTE().fit_resample(train_X, train_y)

    dtrain = lgb.Dataset(X_SMOTE_resampled, y_SMOTE_resampled)
    dvalid = lgb.Dataset(val_X, val_y)

    param = {'num_leaves': 63, 'objective': 'binary', "seed": 1, 'boosting_type': 'dart',
             # Use boosting_type="gbrt" for large dataset
             'metric': 'auc',
             'learning_rate': 0.1,
             'max_depth': -1,
             'min_child_samples': 100,
             'max_bin': 100,
             'subsample': 0.9,
             'subsample_freq': 1,
             'colsample_bytree': 0.7,
             'min_child_weight': 0,
             'min_split_gain': 0,
             'reg_alpha': 0,
             'reg_lambda': 0, }
    num_round = 1000
    bst = lgb.train(param, dtrain, num_round, valid_sets=[dvalid], early_stopping_rounds=20)

    from sklearn import metrics

    ypred = bst.predict(test_X)
    score = metrics.f1_score(test_y, ypred)
    print(f"Test score: {score}")


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




####################################################################################################
# Train a baseline RandomForest

def train_RF():
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import f1_score
    # Create a Gaussian Classifier
    clf = RandomForestClassifier(n_estimators=1000)

    # Train the model using the training sets y_pred=clf.predict(X_test)
    clf.fit(train_X, train_y)

    y_pred = clf.predict(test_X)
    y_prob = clf.predict_proba(test_X)

    score = f1_score(test_y, y_prob[:, 1])
    print(f"Test score for RandomForst Model: {score}")



####################################################################################################
####################################################################################################
## Now we will use the GeFS Model on our df
#3 from gefs import RandomForest


# Auxiliary functions for GeFS
def get_dummies(df):
    df = df.copy()
    if isinstance(df, pd.Series):
        df = pd.factorize(df)[0]
        return df
    for col in df.columns:
        df.loc[:, col] = pd.factorize(df[col])[0]
    return df


## Now we will use the GeFS Model on our df

# Auxiliary functions for GeFS
def get_dummies(df):
    df = df.copy()
    if isinstance(df, pd.Series):
        df = pd.factorize(df)[0]
        return df
    for col in df.columns:
        df.loc[:, col] = pd.factorize(df[col])[0]
    return df


def pd_colcat_get_catcount(df, classcol=None, continuous_ids=[]):
    """
        Learns the number of categories in each variable and standardizes the df.
        Parameters
        -------
        ncat: numpy m
            The number of categories of each variable. One if the variable is
            continuous.
    """
    df = df.copy()
    ncat = np.ones(df.shape[1])
    if not classcol:
        classcol = df.shape[1] - 1
    for i in range(df.shape[1]):
        if i != classcol and (i in continuous_ids or is_continuous(df[:, i])):
            continue
        else:
            df[:, i] = df[:, i].astype(int)
            ncat[i] = max(df[:, i]) + 1
    return ncat


def pd_colnum_stats_univariate(df, ncat=None):
    """
        mean, std: numpy m
            The mean and standard deviation of the variable. Zero and one, resp.
            if the variable is categorical.
    """
    df = df.copy()
    maxv = np.ones(df.shape[1])
    minv = np.zeros(df.shape[1])
    mean = np.zeros(df.shape[1])
    std = np.zeros(df.shape[1])
    if ncat is not None:
        for i in range(df.shape[1]):
            if ncat[i] == 1:
                maxv[i] = np.max(df[:, i])
                minv[i] = np.min(df[:, i])
                mean[i] = np.mean(df[:, i])
                std[i] = np.std(df[:, i])
                assert maxv[i] != minv[i], 'Cannot have constant continuous variable in the df'
                df[:, i] = (df[:, i] - minv[i]) / (maxv[i] - minv[i])
    else:
        for i in range(df.shape[1]):
            if is_continuous(df[:, i]):
                maxv[i] = np.max(df[:, i])
                minv[i] = np.min(df[:, i])
                mean[i] = np.mean(df[:, i])
                std[i] = np.std(df[:, i])
                assert maxv[i] != minv[i], 'Cannot have constant continuous variable in the df'
                df[:, i] = (df[:, i] - minv[i]) / (maxv[i] - minv[i])
    return df, maxv, minv, mean, std


def normalize_data(df, maxv, minv):
    df = df.copy()
    for v in range(df.shape[1]):
        if maxv[v] != minv[v]:
            df[:, v] = (df[:, v] - minv[v]) / (maxv[v] - minv[v])
    return df


def standardize_data(df, mean, std):
    df = df.copy()
    for v in range(df.shape[1]):
        if std[v] > 0:
            df[:, v] = (df[:, v] - mean[v]) / (std[v])
            #  Clip values more than 6 standard deviations from the mean
            df[:, v] = np.clip(df[:, v], -6, 6)
    return df


def is_continuous(df):
    observed = df[~np.isnan(df)]  # not consider missing values for this.
    rules = [np.min(observed) < 0,
             np.sum((observed) != np.round(observed)) > 0,
             len(np.unique(observed)) > min(30, len(observed) / 3)]
    if any(rules):
        return True
    else:
        return False


def train_test_split_gefs(df, ncat, train_ratio=0.7, prep='std'):
    assert train_ratio >= 0
    assert train_ratio <= 1
    shuffle = np.random.choice(range(df.shape[0]), df.shape[0], replace=False)
    data_train = df[shuffle[:int(train_ratio * df.shape[0])], :]
    data_test = df[shuffle[int(train_ratio * df.shape[0]):], :]
    if prep == 'norm':
        data_train, maxv, minv, _, _, = pd_colnum_stats_univariate(data_train, ncat)
        data_test = normalize_data(data_test, maxv, minv)
    elif prep == 'std':
        _, maxv, minv, mean, std = pd_colnum_stats_univariate(data_train, ncat)
        data_train = standardize_data(data_train, mean, std)
        data_test = standardize_data(data_test, mean, std)

    X_train, y_train = data_train[:, :-1], data_train[:, -1]
    X_test, y_test = data_test[:, :-1], data_test[:, -1]

    return X_train, X_test, y_train, y_test, data_train, data_test


def load_dataset(df):
    cat_cols = ['ip', 'app', 'device', 'os', 'channel', 'hour', 'minute',
                'second', 'day', 'day_of_week', 'day_section', 'is_attributed']
    cont_cols = [x for x in df.columns if x not in cat_cols]
    # IMPORTANT! Move target attribute to last column (assumed in the prediction code below)
    df.insert(len(df.columns) - 1, 'is_attributed', df.pop('is_attributed'))
    df.loc[:, cat_cols] = get_dummies(df[cat_cols])
    ncat = pd_colcat_get_catcount(df.values, classcol=-1, continuous_ids=[df.columns.get_loc(c) for c in cont_cols])
    return df.values.astype(float), ncat


def train_gefs_model():
    from     source.bin.model_gefs.gefs import RandomForest

    print("Preparing df for GeFs Random forest model")
    df, ncat = load_dataset(df)  # Preprocess the df
    # ncat is the number of categories of each variable in the df
    X_train, X_test, y_train, y_test, data_train, data_test = train_test_split_gefs(df, ncat)
    rf = RandomForest(n_estimators=30, ncat=ncat)  # Train a Random Forest
    print('Starting the GeFs Random Forest Training')
    rf.fit(X_train, y_train)
    print('Converting Random Forest to GeF')
    gef = rf.topc()  # Convert to a GeF

    ## Classification is performed either by averaging the prediction of each tree (`classify_avg` method)
    #  or by defining a mixture over them (`classify` method).
    print('Making predictions on test df')
    y_pred_avg = gef.classify_avg(X_test, classcol=df.shape[1] - 1)
    y_pred_mixture = gef.classify(X_test, classcol=df.shape[1] - 1)

    _, y_prob = gef.classify(X_test, classcol=df.shape[1] - 1, return_prob=True)
    y_prob = np.max(y_prob, axis=1)
    from sklearn import metrics
    score = metrics.f1_score(y_test, y_prob)
    print(f"Test score for GeFs Model: {score}")



# in order to get rid of the ../lib/python3.6/site-packages/numba/np/ufunc/parallel.py:355: NumbaWarning: The TBB threading layer requires TBB version 2019.5 or later i.e., TBB_INTERFACE_VERSION >= 11005. Found TBB_INTERFACE_VERSION = 9107. The TBB threading layer is disabled.
#   warnings.warn(problem) run `conda install tbb`
if __name__ == '__main__':
    import fire
    fire.Fire()

    """
    train_baseline_model()
    # train_lgb_class_imbalance()
    train_model_with_smote_oversampling()
    # train_gefs_model()
    # train_RF()
    """

























































####################################################################################################
## Let's train a baseline model with the features we created in the above code
def train_baseline_model():
    print('Training a baseline model with the features we engineered')
    dtrain = lgb.Dataset(train_X, train_y)
    dvalid = lgb.Dataset(val_X, val_y)

    param = {'num_leaves': 33, 'objective': 'binary', "seed": 1, 'boosting_type': 'dart',
             # Use boosting_type="gbrt" for large dataset
             'metric': 'auc',
             'learning_rate': 0.1,
             'max_depth': -1,
             'min_child_samples': 100,
             'max_bin': 100,
             'subsample': 0.9,  # Was 0.7
             'subsample_freq': 1,
             'colsample_bytree': 0.7,
             'min_child_weight': 0,
             'min_split_gain': 0,
             'reg_alpha': 0,
             'reg_lambda': 0, }
    num_round = 1000
    bst = lgb.train(param, dtrain, num_round, valid_sets=[dvalid], early_stopping_rounds=20)

    from sklearn import metrics

    ypred = bst.predict(test_X)
    score = metrics.f1_score(test_y, ypred)
    print(
        f"Test score: {score}")



















"""


#############################################################################################
#############################################################################################
print("Preparing df for GeFs Random forest model")
df, ncat = load_dataset(df)  # Preprocess the df

# ncat is the number of categories of each variable in the df
X_train, X_test, y_train, y_test, data_train, data_test = train_test_split_gefs(df, ncat)
rf = RandomForest(n_estimators=30, ncat=ncat)  # Train a Random Forest

print('Starting the GeFs Random Forest Training')
rf.fit(X_train, y_train)

def pd_colcat_get_catcount(df, classcol=None, continuous_ids=[]):

    df = df.copy()
    ncat = np.ones(df.shape[1])
    if not classcol:
        classcol = df.shape[1] - 1
    for i in range(df.shape[1]):
        if i != classcol and (i in continuous_ids or is_continuous(df[:, i])):
            continue
        else:
            df[:, i] = df[:, i].astype(int)
            ncat[i] = max(df[:, i]) + 1
    return ncat


def pd_colnum_stats_univariate(df, ncat=None):

    df = df.copy()
    maxv = np.ones(df.shape[1])
    minv = np.zeros(df.shape[1])
    mean = np.zeros(df.shape[1])
    std = np.zeros(df.shape[1])
    if ncat is not None:
        for i in range(df.shape[1]):
            if ncat[i] == 1:
                maxv[i] = np.max(df[:, i])
                minv[i] = np.min(df[:, i])
                mean[i] = np.mean(df[:, i])
                std[i] = np.std(df[:, i])
                assert maxv[i] != minv[i], 'Cannot have constant continuous variable in the df'
                df[:, i] = (df[:, i] - minv[i]) / (maxv[i] - minv[i])
    else:
        for i in range(df.shape[1]):
            if is_continuous(df[:, i]):
                maxv[i] = np.max(df[:, i])
                minv[i] = np.min(df[:, i])
                mean[i] = np.mean(df[:, i])
                std[i] = np.std(df[:, i])
                assert maxv[i] != minv[i], 'Cannot have constant continuous variable in the df'
                df[:, i] = (df[:, i] - minv[i]) / (maxv[i] - minv[i])
    return df, maxv, minv, mean, std


def normalize_data(df, maxv, minv):
    df = df.copy()
    for v in range(df.shape[1]):
        if maxv[v] != minv[v]:
            df[:, v] = (df[:, v] - minv[v]) / (maxv[v] - minv[v])
    return df


def standardize_data(df, mean, std):
    df = df.copy()
    for v in range(df.shape[1]):
        if std[v] > 0:
            df[:, v] = (df[:, v] - mean[v]) / (std[v])
            #  Clip values more than 6 standard deviations from the mean
            df[:, v] = np.clip(df[:, v], -6, 6)
    return df


def is_continuous(df):
    observed = df[~np.isnan(df)]  # not consider missing values for this.
    rules = [np.min(observed) < 0,
             np.sum((observed) != np.round(observed)) > 0,
             len(np.unique(observed)) > min(30, len(observed) / 3)]
    if any(rules):
        return True
    else:
        return False


def train_test_split_gefs(data, ncat, train_ratio=0.7, prep='std'):
    assert train_ratio >= 0
    assert train_ratio <= 1
    shuffle    = np.random.choice(range(data.shape[0]), data.shape[0], replace=False)
    data_train = data[shuffle[:int(train_ratio * data.shape[0])], :]
    data_test  = data[shuffle[int(train_ratio * data.shape[0]):], :]

    if prep == 'norm':
        data_train, maxv, minv, _, _, = pd_colnum_stats_univariate(data_train, ncat)
        data_test = normalize_data(data_test, maxv, minv)

    elif prep == 'std':
        _, maxv, minv, mean, std = pd_colnum_stats_univariate(data_train, ncat)
        data_train = standardize_data(data_train, mean, std)
        data_test = standardize_data(data_test, mean, std)

    X_train, y_train = data_train[:, :-1], data_train[:, -1]
    X_test, y_test   = data_test[:, :-1], data_test[:, -1]

    return X_train, X_test, y_train, y_test, data_train, data_test



def load_dataset(df):
    cat_cols = ['ip', 'app', 'device', 'os', 'channel', 'hour', 'minute',
                'second', 'day', 'day_of_week', 'day_section', 'is_attributed']
    cont_cols = [x for x in df.columns if x not in cat_cols]


    # IMPORTANT! Move target attribute to last column (assumed in the prediction code below)
    df.insert(len(df.columns) - 1, 'is_attributed', df.pop('is_attributed'))
    df.loc[:, cat_cols] = get_dummies(df[cat_cols])
    ncat                = pd_colcat_get_catcount(df.values, classcol=-1, continuous_ids=[df.columns.get_loc(c) for c in cont_cols])
    return df.values.astype(float), ncat





print('Converting Random Forest to GeF')
gef = rf.topc()  # Convert to a GeF


from sklearn.ensemble import RandomForestClassifier as rfsk
rfsk.fit(X_train, y_train)
y_pred_avg_sk = rfsk.predict(X_test)




## Classification is performed either by averaging the prediction of each tree (`classify_avg` method)
#  or by defining a mixture over them (`classify` method).
print('Making predictions on test df')
y_pred_avg = gef.classify_avg(X_test, classcol=df.shape[1]-1)
y_pred_mixture = gef.classify(X_test, classcol=df.shape[1]-1)

from sklearn import metrics
score = metrics.f1_score(y_test, y_pred_avg)
print(f"Test score for GeFs Model: {score}")

### Computing Robustness Values
##  Robustness values can be computed with the `compute_rob_class` function.
from gefs import compute_rob_class
#pred, rob = compute_rob_class(gef.root, X_test, df.shape[1]-1, int(ncat[-1]))



"""
