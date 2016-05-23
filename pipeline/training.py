import pandas as pd
from sklearn import grid_search
from sklearn.externals import joblib
import numpy as np
import json
from sklearn_model_eval.Logger import Logger
import os
from sklearn import metrics
from sklearn import ensemble
from sklearn import svm
from sklearn import cross_validation


from sklearn import preprocessing

df = pd.read_csv('data/train.csv', index_col='id')
train_y = df.survived.values.astype(int)
df.drop('survived', axis=1, inplace=True)
train_x = df.values

#Load testing data
test = pd.read_csv('data/test.csv', index_col='id')
test_x = test.values

#Bagging classifier
grid = [{
         'n_estimators': [10, 100, 1000],
         'max_samples' : [0.5, 0.75, 1.0],
         'max_features': [0.5, 0.75, 1.0],
         'bootstrap' : [True, False],
         'bootstrap_features': [True, False]
        }]
model = ensemble.BaggingClassifier(n_jobs=-1)

#SVC
grid = [
  {'C': [1, 10, 100, 1000], 'kernel': ['linear']},
  {'C': [1, 10, 100, 1000], 'gamma': [0.01, 0.001, 0.0001], 'kernel': ['rbf']},
 ]

grid = [
  {'C': [100], 'gamma': [0.01], 'kernel': ['rbf']}, 
  {'C': [1000], 'gamma': [0.001], 'kernel': ['rbf']}
 ]

model = svm.SVC()

#SVC needs feature scaling
scaler = preprocessing.StandardScaler().fit(train_x)
train_x = scaler.transform(train_x)
test_x  = scaler.transform(test_x)


#GradientBoostingClassifier
grid = [{'loss': ['deviance', 'exponential'],
         'n_estimators': [100, 1000],
         'max_depth' : [3, 6, 10],
        }]
model = ensemble.GradientBoostingClassifier()



#Adaboost classifier
grid = [{'algorithm': ['SAMME', 'SAMME.R'],
         'n_estimators': [100, 1000],
         'learning_rate' : [0.5, 1, 2],
        }]
model = ensemble.AdaBoostClassifier()


#RandomForest
grid = [{'criterion': ['gini'],
                 'n_estimators': [100],
                 'max_features' : ['auto'],
                 'bootstrap' : [True],
                }]


grid = [{'criterion': ['gini', 'entropy'],
                 'n_estimators': [100, 1000],
                 'max_features' : ['auto', 'log2'],
                 'bootstrap' : [True],#, False],
                 'oob_score': [False, True],
                }]

model = ensemble.RandomForestClassifier(n_jobs=-1)



#Create grid search object optimized for accuracy
model = grid_search.GridSearchCV(model, grid, scoring='accuracy')

#Perform rf grid search
model.fit(train_x, train_y)
print model.best_params_
best = model.best_estimator_
predicted_y = best.predict(train_x)#.astype(int)
#predicted_scores = best.predict_proba(train_x)

#Predict using the best model and save to csv
predictions = best.predict(test_x)#.astype(int)
predictions = pd.DataFrame(data={'PassengerId':test.index, 'Survived':predictions})
predictions.to_csv("predictions.csv", index=False)

#Pickle model for evaluation
joblib.dump(best, 'model/model.pkl') 

#Save train_y, predicted_y and scores
np.save('model/train_y.npy', train_y)
np.save('model/predicted_y.npy', predicted_y)
#np.save('model/predicted_scores.npy', predicted_scores)

#Calculate some metrics
#acc = metrics.accuracy_score(train_y, predicted_y)

#CV accuracy
cv_scores = cross_validation.cross_val_score(best, train_x, train_y, cv=3)
cv_acc = cv_scores.mean()
cv_std = cv_scores.std()

print("\n\nmodel accuracy: %0.2f (+/- %0.2f)\n\n" % (cv_acc, cv_std * 2))

#Load db credentials
path = os.path.join(os.path.expanduser('~'), '.auth', 'mongo')
with open(path, 'r') as f:
    db_credentials = f.read()

#Log experiment
logger = Logger(db_credentials, 'models', 'titanic')
#Log model along with features, accuracy,
#predicted_y and predicted_scores
logger.log_model(best, features=list(df),
                       #acc=acc,
                       cv_acc = cv_acc,
                       cv_std = cv_std,
                       predicted_y=list(predicted_y))#,
                       #predicted_scores=list(predicted_scores[:,1]))

#Write features list
with open('model/features.json', "w") as f:
    json.dump(list(df), f)