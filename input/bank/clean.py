"""
python  clean.py
python  clean.py  profile     #### Data Profile
python  clean.py  train_test_split
"""
import pandas as pd, numpy as np
import os
import pandas_profiling
from sklearn.model_selection import train_test_split

#######################################################################################


##### Load from samples   ##################
df = pd.read_csv('raw/bank_additional_full.csv', nrows= 9280)
print(df.head(5).T)
print(df.tail(5).T)
print(df.dtypes)

#######################################################################################
colid=""
colcat,colnum,coltext = [],[],[]
coly = "y"
def categorize_cols():
  global colid,colcat,colnum,coltext
  Num = ['int','int32','int64','float','float32','float64']
  df_size = len(df)
  for cols in df.columns:
    if cols != coly:
      col_size = df[cols].unique().size
      if col_size == df_size and colid == "":
        colid = str(cols)
      elif df[cols].dtype in Num:
        colnum.append(cols)
      elif col_size > (df_size/2):
        coltext.append(cols)
      else:
        colcat.append(cols)

  print(colid,colcat,colnum,coltext,coly)



###########################################################################################



"""
Put manually column by data type :
"""

colid = "id"

coly = "y"  # "PassengerId"

colcat = ["job","marital","education","default","housing","loan","contact","month","day_of_week","poutcome"]

colnum = ["age","duration","campaign","pdays","previous","emp_var_rate","cons_price_idx","cons_conf_idx","euribor3m","nr_employed"]

##########################################################################################

colsX = colcat + colnum

print('coly', coly)
print('colsX', colsX)
#######################################################################################
#######################################################################################



#######################################################################################
#######################################################################################

def profile():
    os.makedirs("profile/", exist_ok=True)
    for x in colcat:
        df[x] = df[x].factorize()[0]

    ##### Pandas Profile   ###################################
    profile = pandas_profiling.ProfileReport(df)
    profile.to_file(output_file="profile/raw_report.html")


print("profile/raw_report.html")


#######################################################################################
#######################################################################################
def create_features(df):
    return df


def train_test_split():
    os.makedirs("train/", exist_ok=True)
    os.makedirs("test/", exist_ok=True)
    df1 = df.dropna()
    df1_train = pd.DataFrame()
    df1_test = pd.DataFrame()
    icol = int(0.8 * len(df1))
    df1[colsX].iloc[:icol, :].to_parquet("train/features.parquet")
    df1[[coly]].iloc[:icol, :].to_parquet("train/target.parquet")
    df1_train[colsX] = df[colsX].iloc[:icol, :]
    df1_train[[coly]] = df[[coly]].iloc[:icol, :]
    df1_train.to_csv("train/train.csv")
    df1[colsX].iloc[icol:, :].to_parquet("test/features.parquet")
    df1[[coly]].iloc[icol:, :].to_parquet("test/target.parquet")
    df1_test[colsX] = df[colsX].iloc[icol:, :]
    df1_test[[coly]] = df[[coly]].iloc[icol:, :]
    df1_test.to_csv("test/test.csv")

########################################################################################

def save_features(df, name, path):
    if path is not None :
       os.makedirs( f"{path}/{name}" , exist_ok=True)
       if isinstance(df, pd.Series):
           df0=df.to_frame()
       else:
           df0=df
       df0.to_parquet( f"{path}/{name}/features.parquet")



"""
python  clean.py
python  clean.py  profile
python  clean.py  to_train
python  clean.py  to_test
"""
if __name__ == "__main__":
    import fire

    fire.Fire()
