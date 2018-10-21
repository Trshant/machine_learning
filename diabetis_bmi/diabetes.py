#imports
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
# %matplotlib inline

# Put this when it's called
from sklearn.model_selection import train_test_split
from sklearn.model_selection import learning_curve
from sklearn.model_selection import validation_curve
from sklearn.model_selection import cross_val_score
from sklearn.linear_model    import LinearRegression

# Create table for missing data analysis
def draw_missing_data_table(df):
    total = df.isnull().sum().sort_values(ascending=False)
    percent = (df.isnull().sum()/df.isnull().count()).sort_values(ascending=False)
    missing_data = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])
    return missing_data



# Plot learning curve
def plot_learning_curve(estimator, title, X, y, ylim=None, cv=None, n_jobs=1, train_sizes=np.linspace(.1, 1.0, 5)):
    plt.figure()
    plt.title(title)
    if ylim is not None:
        plt.ylim(*ylim)
    plt.xlabel("Training examples")
    plt.ylabel("Score")
    train_sizes, train_scores, test_scores = learning_curve( estimator, X, y, cv=cv, n_jobs=n_jobs, train_sizes=train_sizes)
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)
    plt.grid()
    plt.fill_between( train_sizes, train_scores_mean - train_scores_std + 0 , train_scores_mean + train_scores_std + 0 , alpha=0.1 , color="r")
    plt.fill_between( train_sizes, test_scores_mean  - test_scores_std  + 0 , test_scores_mean  + test_scores_std  + 0 , alpha=0.1 , color="g")
    plt.plot(train_sizes, train_scores_mean, 'o-', color="r", label="Training score")
    plt.plot(train_sizes, test_scores_mean, 'o-', color="g", label="Validation score")
    plt.legend(loc="best")
    return plt


# Plot validation curve
def plot_validation_curve(estimator, title, X, y, param_name, param_range, ylim=None, cv=None, n_jobs=1, train_sizes=np.linspace(.1, 1.0, 5)):
    train_scores, test_scores = validation_curve(estimator, X, y, param_name, param_range, cv)
    train_mean = np.mean(train_scores, axis=1)
    train_std = np.std(train_scores, axis=1)
    test_mean = np.mean(test_scores, axis=1)
    test_std = np.std(test_scores, axis=1)
    plt.plot(param_range, train_mean, color='r', marker='o', markersize=5, label='Training score')
    plt.fill_between(param_range, train_mean + train_std, train_mean - train_std, alpha=0.15, color='r')
    plt.plot(param_range, test_mean, color='g', linestyle='--', marker='s', markersize=5, label='Validation score')
    plt.fill_between(param_range, test_mean + test_std, test_mean - test_std, alpha=0.15, color='g')
    plt.grid() 
    plt.xscale('log')
    plt.legend(loc='best') 
    plt.xlabel('Parameter') 
    plt.ylabel('Score') 
    plt.ylim(ylim)
    return plt


# Import data
df = pd.read_csv('diabetes.csv', header = None, names = ['X','Y'] )
df_raw = df.copy()  # Save original data set, just in case.

# Overview
df.head()

#draw_missing_data_table(df)


# creating datasets 
X = df[df.loc[:, df.columns != 'Y'].columns]
y = df['Y']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.2, random_state=1)


# logistic regresiion
# logreg = LogisticRegression( solver='liblinear',multi_class='ovr',n_jobs=2 )
# logreg.fit(X_train, y_train)

linreg = LinearRegression().fit( X_train , y_train )
linreg.score( X_test , y_test )

'''

# Plot learning curves
title = "Learning Curves (Logistic Regression)"
cv = 5
v = plot_learning_curve(logreg, title, X_train, y_train, ylim=(0, 0.25), cv=cv, n_jobs=1);
v.show()

# Plot validation curve
title = 'Validation Curve (Logistic Regression)'
param_name = 'C'
param_range = [0.001, 0.01, 0.1, 1.0, 10.0, 100.0] 
cv = 10
#v2 = plot_validation_curve(estimator=logreg, title=title, X=X_train, y=y_train, param_name=param_name, ylim=(0.5, 1.01), param_range=param_range);
#v2.show()

'''
tt = linreg.predict(np.array([27.2]).reshape(-1,1))
print(tt)
