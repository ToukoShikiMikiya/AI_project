# -*- coding: utf-8 -*-
# %% [markdown]

"""
Homework:

The folder '~//data//homework' contains data of Titanic with various features and survivals.

Try to use what you have learnt today to predict whether the passenger shall survive or not.

Evaluate your model.
"""
# %%
# 读取数据
from sklearn.model_selection import cross_val_score
import pandas as pd

data = pd.read_csv('data//train.csv')
df = data.copy()
df.sample(10)
# %%
# 去除无用特征
df.drop(columns=['PassengerId', 'Name', 'Ticket', 'Cabin'], inplace=True)
df.info()
# %%
# 替换/删除空值，这里是删除
print('Is there any NaN in the dataset: {}'.format(df.isnull().values.any()))
df.dropna(inplace=True)
print('Is there any NaN in the dataset: {}'.format(df.isnull().values.any()))
# %%
# 把categorical数据通过one-hot变成数值型数据
# 很简单，比如sex=[male, female]，变成两个特征,sex_male和sex_female，用0, 1表示
df = pd.get_dummies(df, columns=['Sex', 'Embarked'])
# %%
X = df.iloc[:, 1:]
y = df.iloc[:, 0]
# train-test split
# %%
import numpy as np
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)
print('X_train: {}'.format(np.shape(X_train)))
print('y_train: {}'.format(np.shape(y_train)))
print('X_test: {}'.format(np.shape(X_test)))
print('y_test: {}'.format(np.shape(y_test)))
# build model
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

models = dict()

models['SVM'] = SVC(kernel='rbf')  # SVM这里我们搞个最常用的
models['KNeighbor'] = KNeighborsClassifier(n_neighbors=5)  # n_neighbors表示neighbor个数
models['RandomForest'] = RandomForestClassifier(n_estimators=100)  # n_estimators表示树的个数
models['DecisionTree'] = DecisionTreeClassifier()
for model_name in models:
    scores = cross_val_score(models[model_name], X=X_train, y=y_train, verbose=0, cv=5, scoring='f1')
    print(f'{model_name}:', scores.mean())
# predict and evaluate
# %%
from sklearn import ensemble
from sklearn.model_selection import GridSearchCV

n_estimators = [3, 5, 10, 15, 20, 40, 55]
max_depth = [10, 100, 1000]
parameters = {'n_estimators': n_estimators, 'max_depth': max_depth}
model = ensemble.RandomForestClassifier()
clf = GridSearchCV(model, parameters, cv=5)
clf = clf.fit(X_train, y_train)
print(clf.best_estimator_)

from sklearn.model_selection import cross_val_score

model_adjust = ensemble.RandomForestClassifier(max_depth=10, n_estimators=55)
scores = cross_val_score(model_adjust, X=X_test, y=y_test, cv=5, scoring='f1')
scores_mean = scores.mean()
print(scores_mean)
