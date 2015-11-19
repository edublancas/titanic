import pandas as pd
import re

df = pd.read_csv('data/combined_clean.csv', index_col='id')

#Replace name wit title
regex = '.*,{1}\s{1}([a-zA-Z\s]+)\.{1}.*'
name_title = df.name.map(lambda name: re.search(regex, name).group(1))
df.name = name_title


#Feature interactions
df['fam_size']  = df.siblings_and_spouses + df.parents_and_children

df['fam_mul_size']  = df.siblings_and_spouses * df.parents_and_children


df.to_csv('data/combined_with_features.csv')