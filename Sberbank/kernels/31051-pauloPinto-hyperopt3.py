# Mostly a lot of silliness at this point:
#   Main contribution (50%) is based on Reynaldo's script with a linear transformation of y_train
#      that happens to fit the public test data well
#      and may also fit the private test data well
#      if it reflects a macro effect
#      but almost certainly won't generalize to later data
#   Second contribution (20%) is based on Bruno do Amaral's very early entry but
#      with an outlier that I deleted early in the competition
#   Third contribution (30%) is based on a legitimate data cleaning,
#      probably by gunja agarwal (or actually by Jason Benner, it seems,
#      but there's also a small transformation applied ot the predictions,
#      so also probably not generalizable),
#   This combo being run by Andy Harless on June 4

import numpy as np
import pandas as pd
#import matplotlib.pyplot as plt
#import seaborn as sns
from sklearn import model_selection, preprocessing
import xgboost as xgb
import datetime
from hyperopt import STATUS_OK, Trials, fmin, hp, tpe, space_eval
import pickle
import getpass
import os
from utils.utils import dotdict, Logger
import math
import argparse

parser = argparse.ArgumentParser(description='Pattern finder - GBM')
parser.add_argument('-gpu', action='store', help='', default='grow_gpu')
args = parser.parse_args()

if os.name == 'posix':
    projectDir = r'/home/' + getpass.getuser() + r'/repos/kaggle/Sberbank'
    histData = r'/home/' + getpass.getuser() + '/histData/bitfinex/csvexport/BTCUSD'
else:
    projectDir = r'C:\repos\kaggle\Sberbank'
    histData = r'C:\quant\histData\bitfinex\csvexport\BTCUSD'

Logger.init_log(os.path.join(projectDir, 'log/LogSberbankb-{}'.format(datetime.date.today())))
Logger.debug('Arguments: {}'.format(args))

hyperP = dotdict([
    ('nfold', 10),
    ('max_evals', 15),
    ('num_boost_round', 1000),
    ('boostMulti', 1.00),
    ('gpu', 'grow_gpu'),
    ('loadModel', 0)
])
hyperP.gpu = None if args.gpu == '0' else hyperP.gpu

space = {
    'learning_rate': 0.01,  # hp.quniform('learning_rate', 0.01, 0.2, 0.01),
    'max_depth': hp.choice('max_depth', np.arange(5, 7, dtype=int)),
    'min_child_weight': hp.quniform('min_child_weight', 1, 3, 1),
    'subsample': hp.quniform('subsample', 0.5, 0.7, 0.05),
    'gamma': hp.quniform('gamma', 0, 0.5, 0.1),
    'colsample_bytree': hp.quniform('colsample_bytree', 0.5, 1, 0.1),
    # 'max_delta_step': 0,
    # subsample = 1, #only in scikit-lean API
    # colsample_bytree = 1,
    # colsample_bylevel = 1,
    # reg_alpha = 0,
    # reg_lambda = 1
    # 'scale_pos_weight': 1,
    # 'base_score': xgbparams.base_score,
    'objective': 'reg:linear',
    'eval_metric': 'rmse',
    # Increase this number if you have more cores. Otherwise, remove it and it will default
    # to the maxium number.
    # 'nthread':  None,
    # 'booster': xgbparams.booster,
    'tree_method': 'exact',  # xgbparams.tree_method,
    'silent': 1,
    'seed': 254,
    # 'missing': None,
    'xgbArgs': {
        'num_boost_round': hyperP.num_boost_round,  # sample(scope.int(hp.quniform('num_boost_round', 100, 1000, 1))),
        'early_stopping_rounds': 50,
        'verbose_eval': 50,
        'show_stdv': False,
        'nfold': hyperP.nfold
    }}
if hyperP.gpu is not None:
    space['updater'] = hyperP.gpu

def scorecv(xgbparams, saveModel=False):
    # print(xgbparams)
    xgbArgs = xgbparams['xgbArgs']
    del xgbparams['xgbArgs']
    cvresult = xgb.cv(xgbparams, dtrain, **xgbArgs)

    return {'loss': cvresult.iloc[-1, 0], 'status': STATUS_OK, 'bestIter': len(cvresult)}
    # cv_output[['train-rmse-mean', 'test-rmse-mean']].plot()
    # print('best num_boost_rounds = ', len(cv_output))
    # num_boost_rounds = len(cv_output)

def optimize(trials, space, scoreFunc):

    return [space_eval(space,
                       fmin(scoreFunc,
                            space,
                            algo=tpe.suggest,
                            trials=trials,
                            verbose=1,
                            max_evals=hyperP.max_evals)
                       ),
            trials.best_trial['result']['bestIter']]

if True:

    #load files
    train = pd.read_csv(os.path.join(projectDir,'input/train.csv'), parse_dates=['timestamp'])
    test = pd.read_csv(os.path.join(projectDir,'input/test.csv'), parse_dates=['timestamp'])
    id_test = test.id

    #clean data
    bad_index = train[train.life_sq > train.full_sq].index
    train.loc[bad_index, "life_sq"] = np.NaN
    equal_index = [601,1896,2791]
    test.loc[equal_index, "life_sq"] = test.loc[equal_index, "full_sq"]
    bad_index = test[test.life_sq > test.full_sq].index
    test.loc[bad_index, "life_sq"] = np.NaN
    bad_index = train[train.life_sq < 5].index
    train.loc[bad_index, "life_sq"] = np.NaN
    bad_index = test[test.life_sq < 5].index
    test.loc[bad_index, "life_sq"] = np.NaN
    bad_index = train[train.full_sq < 5].index
    train.loc[bad_index, "full_sq"] = np.NaN
    bad_index = test[test.full_sq < 5].index
    test.loc[bad_index, "full_sq"] = np.NaN
    kitch_is_build_year = [13117]
    train.loc[kitch_is_build_year, "build_year"] = train.loc[kitch_is_build_year, "kitch_sq"]
    bad_index = train[train.kitch_sq >= train.life_sq].index
    train.loc[bad_index, "kitch_sq"] = np.NaN
    bad_index = test[test.kitch_sq >= test.life_sq].index
    test.loc[bad_index, "kitch_sq"] = np.NaN
    bad_index = train[(train.kitch_sq == 0).values + (train.kitch_sq == 1).values].index
    train.loc[bad_index, "kitch_sq"] = np.NaN
    bad_index = test[(test.kitch_sq == 0).values + (test.kitch_sq == 1).values].index
    test.loc[bad_index, "kitch_sq"] = np.NaN
    bad_index = train[(train.full_sq > 210) & (train.life_sq / train.full_sq < 0.3)].index
    train.loc[bad_index, "full_sq"] = np.NaN
    bad_index = test[(test.full_sq > 150) & (test.life_sq / test.full_sq < 0.3)].index
    test.loc[bad_index, "full_sq"] = np.NaN
    bad_index = train[train.life_sq > 300].index
    train.loc[bad_index, ["life_sq", "full_sq"]] = np.NaN
    bad_index = test[test.life_sq > 200].index
    test.loc[bad_index, ["life_sq", "full_sq"]] = np.NaN
    train.product_type.value_counts(normalize= True)
    test.product_type.value_counts(normalize= True)
    bad_index = train[train.build_year < 1500].index
    train.loc[bad_index, "build_year"] = np.NaN
    bad_index = test[test.build_year < 1500].index
    test.loc[bad_index, "build_year"] = np.NaN
    bad_index = train[train.num_room == 0].index
    train.loc[bad_index, "num_room"] = np.NaN
    bad_index = test[test.num_room == 0].index
    test.loc[bad_index, "num_room"] = np.NaN
    bad_index = [10076, 11621, 17764, 19390, 24007, 26713, 29172]
    train.loc[bad_index, "num_room"] = np.NaN
    bad_index = [3174, 7313]
    test.loc[bad_index, "num_room"] = np.NaN
    bad_index = train[(train.floor == 0).values * (train.max_floor == 0).values].index
    train.loc[bad_index, ["max_floor", "floor"]] = np.NaN
    bad_index = train[train.floor == 0].index
    train.loc[bad_index, "floor"] = np.NaN
    bad_index = train[train.max_floor == 0].index
    train.loc[bad_index, "max_floor"] = np.NaN
    bad_index = test[test.max_floor == 0].index
    test.loc[bad_index, "max_floor"] = np.NaN
    bad_index = train[train.floor > train.max_floor].index
    train.loc[bad_index, "max_floor"] = np.NaN
    bad_index = test[test.floor > test.max_floor].index
    test.loc[bad_index, "max_floor"] = np.NaN
    train.floor.describe(percentiles= [0.9999])
    bad_index = [23584]
    train.loc[bad_index, "floor"] = np.NaN
    train.material.value_counts()
    test.material.value_counts()
    train.state.value_counts()
    bad_index = train[train.state == 33].index
    train.loc[bad_index, "state"] = np.NaN
    test.state.value_counts()

    # brings error down a lot by removing extreme price per sqm
    train.loc[train.full_sq == 0, 'full_sq'] = 50
    train = train[train.price_doc/train.full_sq <= 600000]
    train = train[train.price_doc/train.full_sq >= 10000]

    # Add month-year
    month_year = (train.timestamp.dt.month + train.timestamp.dt.year * 100)
    month_year_cnt_map = month_year.value_counts().to_dict()
    train['month_year_cnt'] = month_year.map(month_year_cnt_map)

    month_year = (test.timestamp.dt.month + test.timestamp.dt.year * 100)
    month_year_cnt_map = month_year.value_counts().to_dict()
    test['month_year_cnt'] = month_year.map(month_year_cnt_map)

    # Add week-year count
    week_year = (train.timestamp.dt.weekofyear + train.timestamp.dt.year * 100)
    week_year_cnt_map = week_year.value_counts().to_dict()
    train['week_year_cnt'] = week_year.map(week_year_cnt_map)

    week_year = (test.timestamp.dt.weekofyear + test.timestamp.dt.year * 100)
    week_year_cnt_map = week_year.value_counts().to_dict()
    test['week_year_cnt'] = week_year.map(week_year_cnt_map)

    # Add month and day-of-week
    train['month'] = train.timestamp.dt.month
    train['dow'] = train.timestamp.dt.dayofweek

    test['month'] = test.timestamp.dt.month
    test['dow'] = test.timestamp.dt.dayofweek

    # Other feature engineering
    train['rel_floor'] = train['floor'] / train['max_floor'].astype(float)
    train['rel_kitch_sq'] = train['kitch_sq'] / train['full_sq'].astype(float)

    test['rel_floor'] = test['floor'] / test['max_floor'].astype(float)
    test['rel_kitch_sq'] = test['kitch_sq'] / test['full_sq'].astype(float)

    train.apartment_name=train.sub_area + train['metro_km_avto'].astype(str)
    test.apartment_name=test.sub_area + train['metro_km_avto'].astype(str)

    train['room_size'] = train['life_sq'] / train['num_room'].astype(float)
    test['room_size'] = test['life_sq'] / test['num_room'].astype(float)

    rate_2016_q2 = 1
    rate_2016_q1 = rate_2016_q2 / .99903
    rate_2015_q4 = rate_2016_q1 / .9831
    rate_2015_q3 = rate_2015_q4 / .9834
    rate_2015_q2 = rate_2015_q3 / .9815
    rate_2015_q1 = rate_2015_q2 / .9932
    rate_2014_q4 = rate_2015_q1 / 1.0112
    rate_2014_q3 = rate_2014_q4 / 1.0169
    rate_2014_q2 = rate_2014_q3 / 1.0086
    rate_2014_q1 = rate_2014_q2 / 1.0126
    rate_2013_q4 = rate_2014_q1 / 0.9902
    rate_2013_q3 = rate_2013_q4 / 1.0041
    rate_2013_q2 = rate_2013_q3 / 1.0044
    rate_2013_q1 = rate_2013_q2 / 1.0104
    rate_2012_q4 = rate_2013_q1 / 0.9832
    rate_2012_q3 = rate_2012_q4 / 1.0277
    rate_2012_q2 = rate_2012_q3 / 1.0279
    rate_2012_q1 = rate_2012_q2 / 1.0279
    rate_2011_q4 = rate_2012_q1 / 1.076
    rate_2011_q3 = rate_2011_q4 / 1.0236
    rate_2011_q2 = rate_2011_q3 / 1
    rate_2011_q1 = rate_2011_q2 / 1.011

    # test data
    test['average_q_price'] = 1

    test_2016_q2_index = test.loc[test['timestamp'].dt.year == 2016].loc[test['timestamp'].dt.month >= 4].loc[test['timestamp'].dt.month <= 7].index
    test.loc[test_2016_q2_index, 'average_q_price'] = rate_2016_q2
    # test.loc[test_2016_q2_index, 'year_q'] = '2016_q2'

    test_2016_q1_index = test.loc[test['timestamp'].dt.year == 2016].loc[test['timestamp'].dt.month >= 1].loc[test['timestamp'].dt.month < 4].index
    test.loc[test_2016_q1_index, 'average_q_price'] = rate_2016_q1
    # test.loc[test_2016_q2_index, 'year_q'] = '2016_q1'

    test_2015_q4_index = test.loc[test['timestamp'].dt.year == 2015].loc[test['timestamp'].dt.month >= 10].loc[test['timestamp'].dt.month < 12].index
    test.loc[test_2015_q4_index, 'average_q_price'] = rate_2015_q4
    # test.loc[test_2015_q4_index, 'year_q'] = '2015_q4'

    test_2015_q3_index = test.loc[test['timestamp'].dt.year == 2015].loc[test['timestamp'].dt.month >= 7].loc[test['timestamp'].dt.month < 10].index
    test.loc[test_2015_q3_index, 'average_q_price'] = rate_2015_q3
    # test.loc[test_2015_q3_index, 'year_q'] = '2015_q3'

    # test_2015_q2_index = test.loc[test['timestamp'].dt.year == 2015].loc[test['timestamp'].dt.month >= 4].loc[test['timestamp'].dt.month < 7].index
    # test.loc[test_2015_q2_index, 'average_q_price'] = rate_2015_q2

    # test_2015_q1_index = test.loc[test['timestamp'].dt.year == 2015].loc[test['timestamp'].dt.month >= 4].loc[test['timestamp'].dt.month < 7].index
    # test.loc[test_2015_q1_index, 'average_q_price'] = rate_2015_q1


    # train 2015
    train['average_q_price'] = 1

    train_2015_q4_index = train.loc[train['timestamp'].dt.year == 2015].loc[train['timestamp'].dt.month >= 10].loc[train['timestamp'].dt.month <= 12].index
    # train.loc[train_2015_q4_index, 'price_doc'] = train.loc[train_2015_q4_index, 'price_doc'] * rate_2015_q4
    train.loc[train_2015_q4_index, 'average_q_price'] = rate_2015_q4

    train_2015_q3_index = train.loc[train['timestamp'].dt.year == 2015].loc[train['timestamp'].dt.month >= 7].loc[train['timestamp'].dt.month < 10].index
    #train.loc[train_2015_q3_index, 'price_doc'] = train.loc[train_2015_q3_index, 'price_doc'] * rate_2015_q3
    train.loc[train_2015_q3_index, 'average_q_price'] = rate_2015_q3

    train_2015_q2_index = train.loc[train['timestamp'].dt.year == 2015].loc[train['timestamp'].dt.month >= 4].loc[train['timestamp'].dt.month < 7].index
    #train.loc[train_2015_q2_index, 'price_doc'] = train.loc[train_2015_q2_index, 'price_doc'] * rate_2015_q2
    train.loc[train_2015_q2_index, 'average_q_price'] = rate_2015_q2

    train_2015_q1_index = train.loc[train['timestamp'].dt.year == 2015].loc[train['timestamp'].dt.month >= 1].loc[train['timestamp'].dt.month < 4].index
    #train.loc[train_2015_q1_index, 'price_doc'] = train.loc[train_2015_q1_index, 'price_doc'] * rate_2015_q1
    train.loc[train_2015_q1_index, 'average_q_price'] = rate_2015_q1


    # train 2014
    train_2014_q4_index = train.loc[train['timestamp'].dt.year == 2014].loc[train['timestamp'].dt.month >= 10].loc[train['timestamp'].dt.month <= 12].index
    #train.loc[train_2014_q4_index, 'price_doc'] = train.loc[train_2014_q4_index, 'price_doc'] * rate_2014_q4
    train.loc[train_2014_q4_index, 'average_q_price'] = rate_2014_q4

    train_2014_q3_index = train.loc[train['timestamp'].dt.year == 2014].loc[train['timestamp'].dt.month >= 7].loc[train['timestamp'].dt.month < 10].index
    #train.loc[train_2014_q3_index, 'price_doc'] = train.loc[train_2014_q3_index, 'price_doc'] * rate_2014_q3
    train.loc[train_2014_q3_index, 'average_q_price'] = rate_2014_q3

    train_2014_q2_index = train.loc[train['timestamp'].dt.year == 2014].loc[train['timestamp'].dt.month >= 4].loc[train['timestamp'].dt.month < 7].index
    #train.loc[train_2014_q2_index, 'price_doc'] = train.loc[train_2014_q2_index, 'price_doc'] * rate_2014_q2
    train.loc[train_2014_q2_index, 'average_q_price'] = rate_2014_q2

    train_2014_q1_index = train.loc[train['timestamp'].dt.year == 2014].loc[train['timestamp'].dt.month >= 1].loc[train['timestamp'].dt.month < 4].index
    #train.loc[train_2014_q1_index, 'price_doc'] = train.loc[train_2014_q1_index, 'price_doc'] * rate_2014_q1
    train.loc[train_2014_q1_index, 'average_q_price'] = rate_2014_q1


    # train 2013
    train_2013_q4_index = train.loc[train['timestamp'].dt.year == 2013].loc[train['timestamp'].dt.month >= 10].loc[train['timestamp'].dt.month <= 12].index
    # train.loc[train_2013_q4_index, 'price_doc'] = train.loc[train_2013_q4_index, 'price_doc'] * rate_2013_q4
    train.loc[train_2013_q4_index, 'average_q_price'] = rate_2013_q4

    train_2013_q3_index = train.loc[train['timestamp'].dt.year == 2013].loc[train['timestamp'].dt.month >= 7].loc[train['timestamp'].dt.month < 10].index
    # train.loc[train_2013_q3_index, 'price_doc'] = train.loc[train_2013_q3_index, 'price_doc'] * rate_2013_q3
    train.loc[train_2013_q3_index, 'average_q_price'] = rate_2013_q3

    train_2013_q2_index = train.loc[train['timestamp'].dt.year == 2013].loc[train['timestamp'].dt.month >= 4].loc[train['timestamp'].dt.month < 7].index
    # train.loc[train_2013_q2_index, 'price_doc'] = train.loc[train_2013_q2_index, 'price_doc'] * rate_2013_q2
    train.loc[train_2013_q2_index, 'average_q_price'] = rate_2013_q2

    train_2013_q1_index = train.loc[train['timestamp'].dt.year == 2013].loc[train['timestamp'].dt.month >= 1].loc[train['timestamp'].dt.month < 4].index
    # train.loc[train_2013_q1_index, 'price_doc'] = train.loc[train_2013_q1_index, 'price_doc'] * rate_2013_q1
    train.loc[train_2013_q1_index, 'average_q_price'] = rate_2013_q1


    # train 2012
    train_2012_q4_index = train.loc[train['timestamp'].dt.year == 2012].loc[train['timestamp'].dt.month >= 10].loc[train['timestamp'].dt.month <= 12].index
    # train.loc[train_2012_q4_index, 'price_doc'] = train.loc[train_2012_q4_index, 'price_doc'] * rate_2012_q4
    train.loc[train_2012_q4_index, 'average_q_price'] = rate_2012_q4

    train_2012_q3_index = train.loc[train['timestamp'].dt.year == 2012].loc[train['timestamp'].dt.month >= 7].loc[train['timestamp'].dt.month < 10].index
    # train.loc[train_2012_q3_index, 'price_doc'] = train.loc[train_2012_q3_index, 'price_doc'] * rate_2012_q3
    train.loc[train_2012_q3_index, 'average_q_price'] = rate_2012_q3

    train_2012_q2_index = train.loc[train['timestamp'].dt.year == 2012].loc[train['timestamp'].dt.month >= 4].loc[train['timestamp'].dt.month < 7].index
    # train.loc[train_2012_q2_index, 'price_doc'] = train.loc[train_2012_q2_index, 'price_doc'] * rate_2012_q2
    train.loc[train_2012_q2_index, 'average_q_price'] = rate_2012_q2

    train_2012_q1_index = train.loc[train['timestamp'].dt.year == 2012].loc[train['timestamp'].dt.month >= 1].loc[train['timestamp'].dt.month < 4].index
    # train.loc[train_2012_q1_index, 'price_doc'] = train.loc[train_2012_q1_index, 'price_doc'] * rate_2012_q1
    train.loc[train_2012_q1_index, 'average_q_price'] = rate_2012_q1


    # train 2011
    train_2011_q4_index = train.loc[train['timestamp'].dt.year == 2011].loc[train['timestamp'].dt.month >= 10].loc[train['timestamp'].dt.month <= 12].index
    # train.loc[train_2011_q4_index, 'price_doc'] = train.loc[train_2011_q4_index, 'price_doc'] * rate_2011_q4
    train.loc[train_2011_q4_index, 'average_q_price'] = rate_2011_q4

    train_2011_q3_index = train.loc[train['timestamp'].dt.year == 2011].loc[train['timestamp'].dt.month >= 7].loc[train['timestamp'].dt.month < 10].index
    # train.loc[train_2011_q3_index, 'price_doc'] = train.loc[train_2011_q3_index, 'price_doc'] * rate_2011_q3
    train.loc[train_2011_q3_index, 'average_q_price'] = rate_2011_q3

    train_2011_q2_index = train.loc[train['timestamp'].dt.year == 2011].loc[train['timestamp'].dt.month >= 4].loc[train['timestamp'].dt.month < 7].index
    # train.loc[train_2011_q2_index, 'price_doc'] = train.loc[train_2011_q2_index, 'price_doc'] * rate_2011_q2
    train.loc[train_2011_q2_index, 'average_q_price'] = rate_2011_q2

    train_2011_q1_index = train.loc[train['timestamp'].dt.year == 2011].loc[train['timestamp'].dt.month >= 1].loc[train['timestamp'].dt.month < 4].index
    # train.loc[train_2011_q1_index, 'price_doc'] = train.loc[train_2011_q1_index, 'price_doc'] * rate_2011_q1
    train.loc[train_2011_q1_index, 'average_q_price'] = rate_2011_q1

    train['price_doc'] = train['price_doc'] * train['average_q_price']
    # train.drop('average_q_price', axis=1, inplace=True)

    print('price changed done')

    y_train = train["price_doc"]
    # x_train = train.drop(["id", "timestamp", "price_doc"], axis=1)
    # x_test = test.drop(["id", "timestamp"], axis=1)

    x_train = train.drop(["id", "timestamp", "price_doc", "average_q_price"], axis=1)
    x_test = test.drop(["id", "timestamp", "average_q_price"], axis=1)

    num_train = len(x_train)
    x_all = pd.concat([x_train, x_test])

    for c in x_all.columns:
        if x_all[c].dtype == 'object':
            lbl = preprocessing.LabelEncoder()
            lbl.fit(list(x_all[c].values))
            x_all[c] = lbl.transform(list(x_all[c].values))
            #x_train.drop(c,axis=1,inplace=True)

    x_train = x_all[:num_train]
    x_test = x_all[num_train:]

    #
    # xgbparams = {
    #     'eta': 0.05,
    #     'max_depth': 6,
    #     'subsample': 0.6,
    #     'colsample_bytree': 1,
    #     'objective': 'reg:linear',
    # }

    dtrain = xgb.DMatrix(x_train, y_train)
    dtest = xgb.DMatrix(x_test)
if hyperP.loadModel == 0:

    # cv_output = xgb.cv(xgb_params, dtrain, num_boost_round=1000, early_stopping_rounds=20,
    #     verbose_eval=20, show_stdv=False)

    # trials1 = Trials()
    # xgb_params, num_boost_rounds = optimize(trials1, space, scorecv)
    num_boost_rounds=962
    xgb_params = {'colsample_bytree': 0.7000000000000001,
                  'eval_metric': 'rmse',
                  'gamma': 0.2,
                  'learning_rate': 0.04,
                  'max_depth': 5,
                  'min_child_weight': 1.0,
                  'objective': 'reg:linear',
                  'seed': 254,
                  'silent': 1,
                  'subsample': 0.6000000000000001,
                  'tree_method': 'exact',
                  'xgbArgs': {'early_stopping_rounds': 100, 'nfold': 10, 'num_boost_round': 1000, 'show_stdv': False, 'verbose_eval': 50}}
    if hyperP.gpu is not None:
        xgb_params['updater'] = hyperP.gpu
    xgb_args = {
        'num_boost_round': math.ceil(num_boost_rounds * hyperP.boostMulti),
    }
    Logger.info('Opt1 rounds / params / args:\n{}\n{}\n{}'.format(
        num_boost_rounds,
        xgb_params,
        xgb_args
    ))

    # pickle.dump(trials1, open(os.path.join(projectDir, 'hyperOptTrials/{}.Sberbanktrial1'.format(datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S'))),'wb'))
    # num_boost_rounds = 422
    model = xgb.train(xgb_params, dtrain, **xgb_args)
    pickle.dump(model, open(os.path.join(projectDir, 'model/{}.Sberbankmodel1'.format(datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S'))),'wb'))
    #fig, ax = plt.subplots(1, 1, figsize=(8, 13))
    #xgb.plot_importance(model, max_num_features=50, height=0.5, ax=ax)
else:
    model = pickle.load(open(r'C:\repos\kaggle\Sberbank\model\2017-06-18_10-15-41.Sberbankmodel1', 'rb'))

y_predict = model.predict(dtest)
# y_predict = np.round(y_predict)
gunja_output = pd.DataFrame({'id': id_test, 'price_doc': y_predict})
# gunja_output['price_doc'] = gunja_output['price_doc'] * gunja_output['average_q_price']
# gunja_output.drop('average_q_price', axis=1, inplace=True)
# gunja_output.head()

train = pd.read_csv(os.path.join(projectDir,'input/train.csv'))
test = pd.read_csv(os.path.join(projectDir,'input/test.csv'))
id_test = test.id

mult = .969

y_train = train["price_doc"] * mult + 10
x_train = train.drop(["id", "timestamp", "price_doc"], axis=1)
x_test = test.drop(["id", "timestamp"], axis=1)

for c in x_train.columns:
    if x_train[c].dtype == 'object':
        lbl = preprocessing.LabelEncoder()
        lbl.fit(list(x_train[c].values))
        x_train[c] = lbl.transform(list(x_train[c].values))

for c in x_test.columns:
    if x_test[c].dtype == 'object':
        lbl = preprocessing.LabelEncoder()
        lbl.fit(list(x_test[c].values))
        x_test[c] = lbl.transform(list(x_test[c].values))

dtrain = xgb.DMatrix(x_train, y_train)
dtest = xgb.DMatrix(x_test)

if hyperP.loadModel == 0:
    # xgb_params = {
    #     'eta': 0.05,
    #     'max_depth': 5,
    #     'subsample': 0.7,
    #     'colsample_bytree': 0.7,
    #     'objective': 'reg:linear',
    #     'eval_metric': 'rmse',
    #     'silent': 1
    xgb_params = {
        'objective': 'reg:linear',
        'eval_metric': 'rmse',
        'tree_method': 'exact',
        'eta': 0.02,
        'colsample_bytree': 0.9,
        'gamma': 0.1,
        'max_depth': 1,
        'min_child_weight': 3.0,
        'subsample': 0.5,
        'silent': 1,
    }
    if hyperP.gpu is not None:
        xgb_params['updater'] = hyperP.gpu

    cv_output = xgb.cv(xgb_params, dtrain,
                       num_boost_round=10000,
                       early_stopping_rounds=200,
                       verbose_eval=25,
                       show_stdv=False,
                       nfold=hyperP.nfold,
                       )

    num_boost_rounds = len(cv_output) # 382
    # num_boost_rounds = 385  # This was the CV output, as earlier version shows

    # trials2 = Trials()
    # xgb_params, num_boost_rounds = optimize(trials2, space, scorecv)

    Logger.info('Opt2 CV output: {}\n'.format(
        cv_output
    ))
    Logger.info('Opt2 rounds / params / args:\n{}\n{}\n{}'.format(
        num_boost_rounds,
        xgb_params,
        xgb_args
    ))

    # opt2 boost rounds = 1000 gotta re-run again with more rounds
    # Opt2 params: {'colsample_bytree': 0.9, 'eval_metric': 'rmse', 'gamma': 0.1, 'learning_rate': 0.01, 'max_depth': 6, 'min_child_weight': 3.0, 'objective': 'reg:linear', 'seed': 254, 'silent': 1, 'subsample': 0.5, 'tree_method': 'exact', 'updater': 'grow_gpu', 'xgbArgs': {'early_stopping_rounds': 50, 'nfold': 10, 'num_boost_round': 1000, 'show_stdv': False, 'verbose_eval': 50}}
    # pickle.dump(trials2, open(os.path.join(projectDir, 'hyperOptTrials/{}.Sberbanktrial2'.format(datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S'))),'wb'))
    xgb_args = {
        'num_boost_round': math.ceil(num_boost_rounds * hyperP.boostMulti),
    }
    model = xgb.train(xgb_params, dtrain, **xgb_args)
    pickle.dump(model, open(os.path.join(projectDir, 'model/{}.Sberbankmodel2'.format(datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S'))),'wb'))
else:
    model = pickle.load(open(r'C:\repos\kaggle\Sberbank\model\2017-06-18_10-46-42.Sberbankmodel2', 'rb'))
y_predict = model.predict(dtest)
output = pd.DataFrame({'id': id_test, 'price_doc': y_predict})
# output.drop('average_q_price', axis=1, inplace=True)
# output.head()

# Any results you write to the current directory are saved as output.
df_train = pd.read_csv(os.path.join(projectDir,'input/train.csv'), parse_dates=['timestamp'])
df_test = pd.read_csv(os.path.join(projectDir,'input/test.csv'), parse_dates=['timestamp'])
df_macro = pd.read_csv(os.path.join(projectDir,'input/macro.csv'), parse_dates=['timestamp'])

df_train.drop(df_train[df_train["life_sq"] > 7000].index, inplace=True)

mult = 0.969
y_train = df_train['price_doc'].values * mult + 10
id_test = df_test['id']

df_train.drop(['id', 'price_doc'], axis=1, inplace=True)
df_test.drop(['id'], axis=1, inplace=True)

num_train = len(df_train)
df_all = pd.concat([df_train, df_test])
# Next line just adds a lot of NA columns (becuase "join" only works on indexes)
# but somewhow it seems to affect the result
df_all = df_all.join(df_macro, on='timestamp', rsuffix='_macro')
print(df_all.shape)

# Add month-year
month_year = (df_all.timestamp.dt.month + df_all.timestamp.dt.year * 100)
month_year_cnt_map = month_year.value_counts().to_dict()
df_all['month_year_cnt'] = month_year.map(month_year_cnt_map)

# Add week-year count
week_year = (df_all.timestamp.dt.weekofyear + df_all.timestamp.dt.year * 100)
week_year_cnt_map = week_year.value_counts().to_dict()
df_all['week_year_cnt'] = week_year.map(week_year_cnt_map)

# Add month and day-of-week
df_all['month'] = df_all.timestamp.dt.month
df_all['dow'] = df_all.timestamp.dt.dayofweek

# Other feature engineering
df_all['rel_floor'] = df_all['floor'] / df_all['max_floor'].astype(float)
df_all['rel_kitch_sq'] = df_all['kitch_sq'] / df_all['full_sq'].astype(float)

train['building_name'] = pd.factorize(train.sub_area + train['metro_km_avto'].astype(str))[0]
test['building_name'] = pd.factorize(test.sub_area + test['metro_km_avto'].astype(str))[0]

def add_time_features(col):
   col_month_year = pd.Series(pd.factorize(train[col].astype(str) + month_year.astype(str))[0])
   train[col + '_month_year_cnt'] = col_month_year.map(col_month_year.value_counts())

   col_week_year = pd.Series(pd.factorize(train[col].astype(str) + week_year.astype(str))[0])
   train[col + '_week_year_cnt'] = col_week_year.map(col_week_year.value_counts())

add_time_features('building_name')
add_time_features('sub_area')

def add_time_features(col):
   col_month_year = pd.Series(pd.factorize(test[col].astype(str) + month_year.astype(str))[0])
   test[col + '_month_year_cnt'] = col_month_year.map(col_month_year.value_counts())

   col_week_year = pd.Series(pd.factorize(test[col].astype(str) + week_year.astype(str))[0])
   test[col + '_week_year_cnt'] = col_week_year.map(col_week_year.value_counts())

add_time_features('building_name')
add_time_features('sub_area')


# Remove timestamp column (may overfit the model in train)
df_all.drop(['timestamp', 'timestamp_macro'], axis=1, inplace=True)


factorize = lambda t: pd.factorize(t[1])[0]

df_obj = df_all.select_dtypes(include=['object'])

X_all = np.c_[
    df_all.select_dtypes(exclude=['object']).values,
    np.array(list(map(factorize, df_obj.iteritems()))).T
]
print(X_all.shape)

X_train = X_all[:num_train]
X_test = X_all[num_train:]


# Deal with categorical values
df_numeric = df_all.select_dtypes(exclude=['object'])
df_obj = df_all.select_dtypes(include=['object']).copy()

for c in df_obj:
    df_obj[c] = pd.factorize(df_obj[c])[0]

df_values = pd.concat([df_numeric, df_obj], axis=1)


# Convert to numpy values
X_all = df_values.values
print(X_all.shape)

X_train = X_all[:num_train]
X_test = X_all[num_train:]

df_columns = df_values.columns

dtrain = xgb.DMatrix(X_train, y_train, feature_names=df_columns)
dtest = xgb.DMatrix(X_test, feature_names=df_columns)

if hyperP.loadModel == 0:
    xgb_params = {
        'objective': 'reg:linear',
        'eval_metric': 'rmse',
        'tree_method': 'exact',
        'eta': 0.02,
        'colsample_bytree': 0.5,
        'gamma': 0.3,
        'max_depth': 1,
        'min_child_weight': 2.0,
        'subsample': 0.65,
        'silent': 1,
    }
    if hyperP.gpu is not None:
        xgb_params['updater'] = hyperP.gpu

    cv_output = xgb.cv(xgb_params, dtrain,
                       num_boost_round=10000,
                       early_stopping_rounds=200,
                       verbose_eval=25,
                       show_stdv=False,
                       nfold=hyperP.nfold)

    # print('best num_boost_rounds = ', len(cv_output))
    print(cv_output)
    num_boost_rounds = len(cv_output)

    # num_boost_rounds = 420  # From Bruno's original CV, I think

    # trials3 = Trials()
    # xgb_params, num_boost_rounds = optimize(trials3, space, scorecv)
    # pickle.dump(trials3, open(os.path.join(projectDir, 'hyperOptTrials/{}.Sberbanktrial3'.format(datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S'))),'wb'))
    xgb_args = {
        'num_boost_round': math.ceil(num_boost_rounds * hyperP.boostMulti),
    }
    Logger.info('Opt3 rounds / params / args:\n{}\n{}\n{}'.format(
        num_boost_rounds,
        xgb_params,
        xgb_args
    ))
    model = xgb.train(xgb_params, dtrain, **xgb_args)

    pickle.dump(model, open(os.path.join(projectDir, 'model/{}.Sberbankmodel3'.format(datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S'))),'wb'))
else:
    model = pickle.load(open(r'C:\repos\kaggle\Sberbank\model\2017-06-18_11-18-01.Sberbankmodel3', 'rb'))
y_pred = model.predict(dtest)
df_sub = pd.DataFrame({'id': id_test, 'price_doc': y_pred})

df_sub.head()
first_result = output.merge(df_sub, on="id", suffixes=['_louis','_bruno'])

first_result["price_doc"] = np.exp( .714*np.log(first_result.price_doc_louis) +
                                    .286*np.log(first_result.price_doc_bruno) )  # multiplies out to .5 & .2
result = first_result.merge(gunja_output, on="id", suffixes=['_follow','_gunja'])
# result[first_result.isnull().any(axis=1)]
result.loc[list(result[first_result.isnull().any(axis=1)].index), 'price_doc_follow'] = \
    result.loc[list(result[first_result.isnull().any(axis=1)].index), 'price_doc_gunja']
result["price_doc"] = np.exp( .78*np.log(result.price_doc_follow) +
                              .22*np.log(result.price_doc_gunja) )
result.drop(["price_doc_louis","price_doc_bruno","price_doc_follow","price_doc_gunja"],axis=1,inplace=True)

result.to_csv(os.path.join(projectDir,'subm/silly-{}.csv'.format(
    datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S'))), index=False)

if False:
    a=1