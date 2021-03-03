import numpy as np
import pandas
import pickle
import gzip
import datetime
import json
import pandas as pd, numpy as np, random, copy, os, sys
from sklearn.model_selection import train_test_split
random.seed(100)

root = os.path.abspath(os.getcwd()).replace("\\", "/") + "/"
print(root)



colnums = ["Quan_4", "Quan_5", "Quan_6", "Quan_7", "Quan_8", "Quan_9", "Quan_10", "Quan_11", "Quan_12", "Quan_13", "Quan_14", "Quan_15", "Quan_16", "Quan_17", "Quan_18", "Quan_19", "Quan_21", "Quan_22", "Quan_27", "Quan_28", "Quan_29", "Quant_22", "Quant_24", "Quant_25"]
def split_train_test(df):
    """
    fill null values with median of the column
    Dates are inserted multiple times: number of days, year, month, day
    and binary vectors for year, month and day.
    finally redundant columns are removed
    """
    df['Date_3']   = df.Date_1 - df.Date_2
    train_size     = 600
    X_cat  = []
    X_nums = []
    X_date         = []
    X_id           = []
    ys             = np.zeros((train_size,12), dtype=np.int)
    columns        = []


    for col in df.columns:
        if col.startswith('Cat_'):
            columns.append(col)
            uni = np.unique(df[col])
            if len(uni) > 1:
                # Quick smart way to binarize categorical variables:
                X_cat.append(uni==df[col].values[:,None])


        elif col.startswith('Quan_') or col.startswith('Quant_'):
            columns.append(col)
            # Use logscale when needed:
            #if col in colnums:
            #    df[col] = np.log(df[col])

            # if the column is not just full of NaN:
            #if (pd.isnull(df[col])).sum() > 1:
            #    tmp = df[col].copy()


                # illing missing values with median BAD
                #tmp = tmp.fillna(tmp.median())
            tmp = df[col].copy()                
            X_nums.append(tmp.values)


        elif col.startswith('Date_'):
            columns.append(col)
            # if the column is not just full of NaN
            tmp = df[col].copy()
            if (pd.isnull(tmp)).sum() > 1:
                # median imputation:
                tmp = tmp.fillna(tmp.median())
            X_date.append(tmp.values[:,None])
            # extract day/month/year for seasonal info
            year = np.zeros((tmp.size,1))
            month = np.zeros((tmp.size,1))
            day = np.zeros((tmp.size,1))
            for i, date_number in enumerate(tmp):
                date = datetime.date.fromordinal(int(date_number))
                year[i,0] = date.year
                month[i,0] = date.month
                day[i,0] = date.day
            X_date.append(year)
            X_date.append(month)
            X_date.append(day)
            # consider year, month day as categorical
            X_date.append((np.unique(year)==year).astype(np.int))
            X_date.append((np.unique(month)==month).astype(np.int))
            X_date.append((np.unique(day)==day).astype(np.int))

        elif col=='id':
            pass # X_id.append(df[col].values)


        elif col.startswith('Outcome_'):
            outcome_col_number = int(col.split('M')[1]) - 1
            tmp = df[col][:train_size].copy()
            # median imputation:
            tmp = tmp.fillna(tmp.median())
            ys[:,outcome_col_number] = tmp.values

        else:
            continue

    X_cat  = np.hstack(X_cat).astype(np.float)
    X_nums = np.vstack(X_nums).astype(np.float).T
    X_date         = np.hstack(X_date).astype(np.float)

    X = np.hstack([X_cat, X_nums, X_date])
    
    X_train = X[:train_size,:]
    X_test = X[train_size:,:]
    return X_train, X_test, ys, columns


def save_data(data_info, base_path):
    data_info['train_features'].to_csv(f"{base_path}/train/features.zip",compression = 'gzip')
    data_info['test_features'].to_csv(f"{base_path}/test/features.zip",compression = 'gzip')
    data_info['targets'].to_csv(f"{base_path}/train/target.zip",compression = 'gzip')
    
    cols_group = {"colcat":data_info["colcat"],"colnum":data_info["colnum"]}
    with open('cols_group.json', 'w') as fp:
        json.dump(cols_group, fp)


def main():
   path = 'raw/TrainingDataset.zip'
    df = pd.read_csv(path)

    print ("df:", df)
    ids = df.values[:,0].astype(np.int)

    X_train, X_test, targets, columns = split_train_test(df)
    
    #adding column names
    cat_cols = ["cat_"+str(x) for x in range(1,1742+1)]
    num_cols = ["num_"+str(x) for x in range(1743,1933+1)]
    
    combined_cols = cat_cols +num_cols
    
    X_train = pd.DataFrame(X_train)
    X       = np.vstack([X_train, X_test])
    X_train = pd.DataFrame( X[:X_train.shape[0], :])
    X_test  = pd.DataFrame(X[X_train.shape[0]:, :])
    targets = pd.DataFrame(targets)

    print ("********Data Cleaned****************")
    d = {i:j for i,j in zip(X_train.columns,combined_cols)}

    X_train.rename(columns = d,inplace = True )
    X_test.rename(columns = d,inplace = True )
    
    dataset_info = {"train_features": X_train,
                "test_features": X_test,
                "colcat": cat_cols,
                "colnum": num_cols,
                "targets": targets}
    save_data(dataset_info,root)



if __name__ == '__main__':
     import fire
     fire.Fire()



 