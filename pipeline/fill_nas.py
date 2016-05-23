import pandas as pd

df = pd.read_csv('data/combined.csv', index_col='id')

#Fill missing fare
df.loc[1044,'fare'] = df[df.p_class==3]['fare'].median()

#Fill missing age
median_ages = df[['p_class','age','sex']].groupby(['p_class','sex']).median()

def estimate_age(row):
    if pd.isnull(row.age):
        return float(median_ages.ix[row.p_class].ix[row.sex])
    return row.age

df['age'] = df.apply(estimate_age, axis=1)

#Fill missing embarked
df.loc[62,'embarked'] = 'S'
df.loc[830,'embarked'] = 'S'

df.cabin[df.cabin.isnull()] = 'U'

#Write to csv
df.to_csv('data/combined_clean.csv')