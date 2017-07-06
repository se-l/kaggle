import os
import getpass
import datetime
from mlbox.preprocessing import *
from mlbox.optimisation import *
from mlbox.prediction import *
import numpy as np

if os.name == 'posix':
    projectDir = r'/home/' + getpass.getuser() + r'/repos/kaggle/mercedes'
else:
    projectDir = r'C:\repos\kaggle\mercedes'

paths = [os.path.join(projectDir,"input/train.csv"), os.path.join(projectDir, "input/test.csv")] #to modify
target_name = "y" #to modify "column name"

df = Reader(sep=",").train_test_split(paths, target_name)  #reading
df = Drift_thresholder().fit_transform(df)  #deleting non-stable drift variables

# setting the hyperparameter space
space={'ne__numerical_strategy':{"search":"choice","space":['mean','median']},
'ne__categorical_strategy':{"search":"choice","space":[np.NaN]},
'ce__strategy':{"search":"choice","space":['label_encoding','entity_embedding','random_projection']},
'fs__strategy':{"search":"choice","space":['l1','variance','rf_feature_importance']},
'fs__threshold':{"search":"uniform","space":[0.01, 0.3]},
'est__max_depth':{"search":"choice","space":[3,5,7,9]},
'est__n_estimators':{"search":"choice","space":[250,500,700,1000]}}

# calculating the best hyper-parameter
best=Optimiser(scoring="mean_squared_error",n_folds=5).optimise(space,df,40)

# predicting on the test dataset
Predictor().fit_predict(best,df)