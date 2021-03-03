import pandas as pd

training = pd.read_csv('train.csv')

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)

print(training.columns)

train_X_raw=training[['PassengerId', 'Pclass', 'Name', 'Sex', 'Age', 'SibSp','Parch', 'Ticket', 'Fare','Cabin', 'Embarked']]
train_y_raw=training[['PassengerId','Survived']]

train_X_raw.to_csv('Titanic_Features_raw.csv',index=False)
train_y_raw.to_csv('Titanic_Labels_raw.csv',index=False)