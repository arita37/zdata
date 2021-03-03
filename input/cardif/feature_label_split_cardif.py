import pandas as pd
import random

training = pd.read_csv('train.csv')

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)



train_X_raw=training.drop(columns=['target']) 
train_y_raw=training[['ID','target']]


num=[]
cat=[]


for i in training.columns:
	datatype=training[i].dtypes
	if datatype=='int64' or datatype=='float64':
		num.append(i)
	else:
		cat.append(i)


train_X_raw.to_csv('cardif_Features_raw.csv',index=False)
train_y_raw.to_csv('cardif_Labels_raw.csv',index=False)