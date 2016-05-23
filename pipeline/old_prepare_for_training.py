import pandas as pd

df = pd.read_csv('data/combined_with_features.csv', index_col='id')

#Convert sex to binary variable
df.sex = df.sex.map({'male':0, 'female':1})
#df.drop('sex', axis=1, inplace=True)

#df.drop('p_class', axis=1, inplace=True)

#Name dummies
name_dummies = pd.get_dummies(df.name, prefix='name').astype(int)
df = df.join(name_dummies)
df.drop('name', axis=1, inplace=True)

#Social status
#social_dummies = pd.get_dummies(df.social_status, prefix='social_status').astype(int)
#df = df.join(social_dummies)
#df.drop('social_status', axis=1, inplace=True)

#Cabin
#df['cabin'] =  df.cabin.map(lambda x: x[0])
#cabin_dummies = pd.get_dummies(df.cabin, prefix='cabin').astype(int)
#df = df.join(cabin_dummies)
df.drop('cabin', axis=1, inplace=True)

#Drop unused variables
df.drop('embarked', axis=1, inplace=True)
df.drop('ticket', axis=1, inplace=True)

#Split  in train and testing
train = df[df.source=='train']
#train.survived = train.survived.astype(int)
test = df[df.source=='test']
train.drop('source', axis=1, inplace=True)
test.drop(['source','survived'], axis=1, inplace=True)
#Save to csv
train.to_csv("data/train.csv")
test.to_csv("data/test.csv")