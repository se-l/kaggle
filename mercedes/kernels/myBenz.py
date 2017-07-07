# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np
import argparse
from sklearn.base import BaseEstimator,TransformerMixin, ClassifierMixin
import xgboost as xgb
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn.linear_model import ElasticNetCV, LassoLarsCV
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.pipeline import make_pipeline, make_union
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import train_test_split
from sklearn.utils import check_array
from functools import partial
from hyperopt import STATUS_OK, Trials
from sklearn.metrics import r2_score
import os
import getpass
import datetime
from utils.utils import dotdict, Logger
from utils.utilFunc import getProjectDir
import funcBenz as fbz
import modelsBenz as mbz

#set file Paths
projectDir = getProjectDir()

#define all parameters and seach space
params = dotdict([
    ('leaksToTrain', True),
    ('rm0VarFeats', True),
    ('rmOutliers', True),
    ('engProj', True),
    ('leaksIntoSub', True),
    ('addX0groups', True),
    ('polyFeat', False),
    ('seedRounds', 1),
    ('max_evals', 1),
])

Logger.init_log(os.path.join(projectDir,'log/log-{}'.format(datetime.date.today())))
parser = argparse.ArgumentParser(description='Laala')
parser.add_argument('-srcargs', type=int, action='store', help='', default=1)
parser.add_argument('-gpu', action='store', help='', default='grow_gpu')
if os.name == 'posix':
    args = parser.parse_args()
    if args.srcargs == 1:
        # argsDef = ['-srcargs', '1', '-gpu', '0']
        argsDef = ['-gpu', 'grow_gpu']
        args = parser.parse_args(argsDef)
else:
    argsDef = ['-srcargs', '1', '-gpu', '0']
    args = parser.parse_args(argsDef)

Logger.init_log(os.path.join(projectDir,'log/xgblog-{}'.format(datetime.date.today())))
Logger.debug('Arguments: {}'.format(args))

# any classes here
class StackingEstimator(BaseEstimator, TransformerMixin):
    
    def __init__(self, estimator):
        self.estimator = estimator

    def fit(self, X, y=None, **fit_params):
        self.estimator.fit(X, y, **fit_params)
        return self
    def transform(self, X):
        X = check_array(X)
        X_transformed = np.copy(X)
        # add class probabilities as a synthetic feature
        if issubclass(self.estimator.__class__, ClassifierMixin) and hasattr(self.estimator, 'predict_proba'):
            X_transformed = np.hstack((self.estimator.predict_proba(X), X))

        # add class prodiction as a synthetic feature
        X_transformed = np.hstack((np.reshape(self.estimator.predict(X), (-1, 1)), X_transformed))

        return X_transformed

#get source data
train = pd.read_csv(os.path.join(projectDir, r'input/train.csv'), dtype={'ID': np.int32})
test = pd.read_csv(os.path.join(projectDir, r'input/test.csv'), dtype={'ID': np.int32})

#add leaks to train data
if params.leaksToTrain:
    train = fbz.addLeaksTrain(train, test)

y_train = train['y'].values
train_ids = train['ID'].values
train = train.drop(['ID', 'y'], axis=1)
print('\n Initial Train Set Matrix Dimensions: %d x %d' % (train.shape[0], train.shape[1]))
train_len = len(train)
test_ids = test['ID'].values
test = test.drop(['ID'], axis=1)
print('\n Initial Test Set Matrix Dimensions: %d x %d' % (test.shape[0], test.shape[1]))
all_data = pd.concat((train, test))

#label encode categoricals
train, test = fbz.labelEncode(train, test)

# remove non-unique feats
if params.rm0VarFeats:
    all_data = fbz.rm0VarFeats(all_data)
# Remove identical columns where all data points have the same value
if params.rmIdenticalFeats:
    all_data = fbz.rmIdenticalFeats(all_data)
if params.addX0groups:
    all_data = fbz.addX0groups(all_data)

features = all_data.columns
print('\n Final Matrix Dimensions: %d x %d' % (all_data.shape[0], all_data.shape[1]))
train_data = pd.DataFrame(all_data[ : train_len].values, columns=features)
test_data = pd.DataFrame(all_data[train_len : ].values, columns=features)
train_data.reset_index(drop=True, inplace=True)
test_data.reset_index(drop=True, inplace=True)

# perform parameterized outlier detection and handling
if params.rmOutlier:
    train_OL = fbz.findOutliers(train, train_data, train_ids, test, test_data, test_ids, target=y_train)
    train = fbz.rmOutliers(train, train_OL)

#optional, drop y outliers  by Liam
train = train[train.y < 250] # Optional: Drop y outliers

#FEAT ENGINEERING
all_data = pd.concat((train, test))
# group X0 and label encode
# review the snte again
# switch on poly feat batch processing
# more feat eng from all threads

# PCA / ICA / GRP / tSVD
#save columns list before adding the decomposition components
usable_columns = list(set(train.columns) - set(['y']))
if params.engProj:
    train_proj, test_proj = fbz.getProjections(['svd','pca','ica','grp','srp'], n_comp=12, random_state=420)
    train = pd.concat([train, train_proj], axis=1)
    test = pd.concat([test, test_proj], axis=1)

if params.polyFeat:
    dfPoly = fbz.polyFeat(all_data)
    all_data = pd.concat([all_data, dfPoly], axis=1)

y_mean = np.mean(y_train)
# id_test = test['ID'].values
#finaltrainset and finaltestset are data to be used only the stacked model (does not contain PCA, SVD... arrays) 
finaltrainset = train[usable_columns].values
finaltestset = test[usable_columns].values

# TRAIN 1 for Stacking
# Models - with guessed params ( optimize when hyperopt params come in)
# include lgbm
# include TPOT out of the box results for stacking and ensemble
# perform initial training with pipeline with trees and NN (liam is straight forward)
# add it to train features

# somewhere get in that recursive feature selection
# train -> save Imps -> pick all relevant or as many as fit into memory / alternatively batch model training (new)
# or combo of abvoe
# use sklearn pipeline or automated feat selection packages for that

# TRAIN 2 for Prediction
    #hyperopt wrapper to optimize each model separately on 5 fold local CV results, finally use only 1 space setting

'''Train the xgb model then predict the test data'''

sub = pd.DataFrame()
sub['ID'] = test_ids
sub['y'] = 0
#SEED WRAPPER
for seedRound in range(0, params.seedRounds):
    np.random.seed(seedRound)

    '''1. XGB Model & predict'''
    xgbSpace = {
        'learning_rate': 0.0045,  # hp.quniform('learning_rate', 0.01, 0.2, 0.01), alias: eta
        # A problem with max_depth casted to float instead of int with
        # the hp.quniform method.
        'max_depth': 4,  # hp.choice('max_depth', np.arange(3, 10, dtype=int)),
        # 'min_child_weight': xgbparams.min_child_weight,  # hp.quniform('min_child_weight', 1, 6, 1),
        'subsample': 0.93,  # hp.quniform('subsample', 0.5, 1, 0.1),
        'n_trees': 520,
        # 'gamma': xgbparams.gamma,  # hp.quniform('gamma', 0.1, 1, 0.3),
        # 'colsample_bytree': xgbparams.colsample_bytree,  # hp.quniform('colsample_bytree', 0.5, 0.9, 0.05),
        # 'max_delta_step': xgbparams.max_delta_step,  # 0,
        # colsample_bylevel = 1,
        # reg_alpha = 0,
        # reg_lambda = 1
        # 'scale_pos_weight': xgbparams.scale_pos_weight,
        'objective': 'reg:linear',
        'eval_metric': 'rmse',
        'base_score': y_mean,  # base prediction = mean(y_train)
        'silent': True,
        # 'booster': xgbparams.booster,
        'tree_method': 'exact',
        'seed': seedRound,
        # 'missing': None,
        'xgbArgs': {
            'num_boost_round': 1250,
            # sample(scope.int(hp.quniform('num_boost_round', 100, 1000, 1))),
            # 'early_stopping_rounds': xgbparams.xgbArgs['early_stopping_rounds'],
            'verbose_eval': 50,
        }}
    if args.gpu != '0':
        xgbSpace['updater'] = args.gpu

    hyperOptTrials = Trials()
    params.batch = seedRound
    #replace this with a 5-fold CV if parameters are better known and all is tested for long runs
    x_train, y_train, x_valid, y_valid = train_test_split(train.drop('y', axis=1), y_train, test_size=0.2, stratify=True, random_state=params.seedRounds)
    # NOTE: Make sure that the class is labeled 'class' in the data file
    xgbMTrain = xgb.DMatrix(x_train, label=y_train, feature_names=x_train.columns)
    xgbMValid = xgb.DMatrix(x_valid, label=y_valid, feature_names=x_valid.columns)
    xgbMTest = xgb.DMatrix(test.drop('y', axis=1), label=test['y'], feature_names=test.columns)
    if True:  # run a CV
        # specify validations set to watch performance
        xgbSpace['updater']['xgbArgs']['evals'] = [(xgbMTrain, 'train'), (xgbMValid, 'val')]
        xgbSpace['updater']['xgbArgs']['evals_result'] = {}

    xgbFunc = partial(mbz.xgbTrain, args=args, xgbMTrain=xgbMTrain, xgbMTest=xgbMValid, params=params)
    best_params = mbz.optimize(space=xgbSpace, scoreF=xgbFunc, trials=hyperOptTrials, params=params)
    bestIter = hyperOptTrials.best_trial['result']['bestIter']
    model = hyperOptTrials.best_trial['result']['model']
    xgbPred = model.predict(xgbMTest)
    # r2Train = r2_score(y_train, model.predict())
    #                     sess.run(scaled_out, feed_dict={x: train[ktest], keep_prob: 1.0}))
    # print('Step: %d, Fold: %d, R2 Score: %g' % (i, k, accuracy))
    # CV.append(accuracy)
    # print('Mean R2: %g' % (np.mean(CV)))
    
    '''2. Train stacked models & predict the test data'''
    
    stacked_pipeline = make_pipeline(
        StackingEstimator(estimator=LassoLarsCV(normalize=True)),
        StackingEstimator(estimator=GradientBoostingRegressor(
            learning_rate=0.001,
            loss="huber",
            max_depth=3,
            max_features=0.55,
            min_samples_leaf=18,
            min_samples_split=14,
            subsample=0.7)),
        LassoLarsCV()
    )
    stacked_pipeline.fit(finaltrainset, y_train)
    results = stacked_pipeline.predict(finaltestset)
    Logger.info('Stacked model: R2 on train data: {}'.format(r2_score(y_train,stacked_pipeline.predict(finaltrainset))))

    '''3. Neural networks model'''

    nnSpace = {
        #####################
        # Neural Network
        #####################
        # Training steps
        'STEPS': 500,
        'LEARNING_RATE': 0.0001,
        'BETA': 0.01,
        'DROPOUT': 0.5,
        'RANDOM_SEED': 12345,
        'MAX_Y': 250,
        'RESTORE': True,
        'START': 0,
        # Training variables
        'IN_DIM': 13,
        # Network Parameters - Hidden layers
        'n_hidden_1': 100,
        'n_hidden_2': 50,
        }

    nnFunc = partial(mbz.deepNN(train, y_train))
    best_params = mbz.optimize(space=nnSpace, scoreF=nnFunc, trials=hyperOptTrials, params=params)
    nnModel = hyperOptTrials.best_trial['result']['model']

    '''4. TPOT automated ML approach'''

    '''R2 Score on the entire Train data when averaging'''
    
    print('R2 score on train data:')
    print(r2_score(y_train,stacked_pipeline.predict(finaltrainset)*0.2855 + model.predict(xgbMTrain)*0.7145))
    
    '''Average the preditionon test data  of both models then save it on a csv file'''

    # ENSEMBLE all weighted by public leaderboard score
    # or first majority vote, then weight of local CV
    sub['y'] += xgbPred*0.75 + results*0.25


# PREDICT with each model for each see


#Divide by number of seed runs
sub['y'] /= params.seedRounds + 1

if params.leaksIntoSub:
    sub = fbz.leaksIntoSub(sub)

sub.to_csv(os.path.join(projectDir,r'subm/myBenz-{}.csv'.format(
        datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S'))), index=False)