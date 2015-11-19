import pandas as pd

renaming = {
            'Pclass': 'p_class',
            'Name': 'name',
            'Sex': 'sex',
            'Age': 'age',
            'SibSp': 'siblings_and_spouses',
            'Parch': 'parents_and_children',
            'Fare': 'fare',
            'Ticket': 'ticket',
            'Embarked': 'embarked',
            'Cabin': 'cabin',
            'Survived': 'survived'
            }

train = pd.read_csv('raw_data/train.csv', index_col='PassengerId')
train['source'] = 'train'
 
test = pd.read_csv('raw_data/test.csv', index_col='PassengerId')
test['source'] = 'test'

combined = train.append(test)
combined.index.rename('id', inplace=True)
combined.rename(columns=renaming,inplace=True)
combined.to_csv('data/combined.csv')
