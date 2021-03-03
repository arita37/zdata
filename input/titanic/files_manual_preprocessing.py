import pandas as pd

training = pd.read_csv('train.csv')
#test=pd.read_csv('test.csv')
#primer=pd.read_csv('gender_submission.csv')

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)

print (training.columns)
print (training.head(10))

for i in training.columns:
	datatype=training[i].dtypes
	print ('Type of column', i, 'is: ', datatype)


for i in training.columns:
	length=len(training[i][training[i].isna()])
	print ('Number of Nan values in column' ,i,'is:', length)

embarked=training[['Pclass','Fare','Embarked']]
embarked_with_nan=embarked[embarked.isna().any(axis=1)]
print (embarked_with_nan.head())
embarked_without_nan=embarked.dropna(how='any')

training.loc[training['Fare']==80.0,'Embarked']='S'


opt_1=round(training.loc[(training['Parch']!=0) & (training['SibSp']!=0)].mean()['Age'],1)
opt_2=round(training.loc[(training['Parch']==0) & (training['SibSp']!=0)].mean()['Age'],1)
opt_3=round(training.loc[(training['Parch']!=0) & (training['SibSp']==0)].mean()['Age'],1)
opt_4=round(training.loc[(training['Parch']==0) & (training['SibSp']==0)].mean()['Age'],1)

df1=training[['PassengerId','SibSp','Parch','Age']]
df2=df1[df1.isna().any(axis=1)]

df2.loc[(df2['Parch']!=0) & (df2['SibSp']!=0),'Age']=opt_1
df2.loc[(df2['Parch']==0) & (df2['SibSp']!=0),'Age']=opt_2
df2.loc[(df2['Parch']!=0) & (df2['SibSp']==0),'Age']=opt_3
df2.loc[(df2['Parch']==0) & (df2['SibSp']==0),'Age']=opt_4
print(df2)

training.loc[training['PassengerId'].isin(df2['PassengerId']), ['Age']] = df2['Age']
print (training.head(10))

training_set=training.drop(columns=['Cabin'])

train_X=training_set[['PassengerId', 'Pclass', 'Name', 'Sex', 'Age', 'SibSp','Parch', 'Ticket', 'Fare', 'Embarked']]
train_y=training_set[['PassengerId','Survived']]

print(train_X.head(20))
print(train_y.head(20))
train_X.to_csv('Titanic_Featues.csv',index=False)
train_y.to_csv('Titanic_Labels.csv',index=False)
