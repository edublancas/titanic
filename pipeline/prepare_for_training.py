import pandas as pd

columns_to_drop = ['ticket', 'source', 'name', 'embarked'] #'cabin'

df = pd.read_csv('data/combined_with_features.csv', index_col='id')

#Split  in train and testing
train = df[df.source=='train']
test = df[df.source=='test']

#Drop columns
train.drop(columns_to_drop, axis=1, inplace=True)
test.drop(columns_to_drop, axis=1, inplace=True)

#Convert every object variable to dummy variable
train = pd.get_dummies(train)
test = pd.get_dummies(test)

train.drop('cabin_T', axis=1, inplace=True)

#Drop survived column in tets
test.drop('survived', axis=1, inplace=True)

#Save to csv
train.to_csv("data/train.csv")
test.to_csv("data/test.csv")