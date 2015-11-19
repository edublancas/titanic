import pandas as pd

df = pd.read_csv('output/Titanic_Evaluate_Tuned_train_train.csv_test_test.csv_RandomForestClassifier.predictions',
                  sep='\t', index_col='id')

df.index.rename('PassengerId', inplace=True)
df.rename(columns={'prediction': 'Survived'},inplace=True)

df.Survived = df.Survived.astype(int)

df.to_csv('kaggle.csv')
