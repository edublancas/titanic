import pandas as pd
import re

df = pd.read_csv('data/combined_clean.csv', index_col='id')

#Replace name wit title
regex = '.*,{1}\s{1}([a-zA-Z\s]+)\.{1}.*'
name_title = df.name.map(lambda name: re.search(regex, name).group(1))
df.name = name_title

#Cabin with first letter
prefix_cabin = df.cabin.map(lambda c: c[:1])
df.cabin = prefix_cabin

#Map name title to social status
dic = {'Lady': 'high', 'Sir': 'high', 'the Countess': 'high',
        'Jonkheer': 'high', 'Major': 'high', 'Master': 'high'}
df['social_status'] = df.name.map(lambda name: dic.get(name, 'normal'))

#Round fare
#df.fare = df.fare.astype(int)

#Feature interactions
df['fam_size']  = df.siblings_and_spouses + df.parents_and_children
df['fam_mul_size']  = df.siblings_and_spouses * df.parents_and_children
df['fare_mul_pclass'] = df.fare/df.p_class.astype(float)
df['fare_mul_age'] = df.fare*df.age
df['fare_div_age'] = df.fare/df.age.astype(float)
df['pclass_mul_age'] = df.p_class*df.age.astype(float)
df['pclass_div_age'] = df.p_class/df.age.astype(float)

df.to_csv('data/combined_with_features.csv')