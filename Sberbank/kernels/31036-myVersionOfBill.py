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
from scipy.stats import norm
from itertools import combinations
from sklearn.preprocessing import PolynomialFeatures
import gc

parser = argparse.ArgumentParser(description='Pattern finder - GBM')
parser.add_argument('-gpu', action='store', help='', default='grow_gpu')
# parser.add_argument('-gpu', action='store', help='', default='0')
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
    ('nfold', 5),
    ('max_evals', 15),
    ('num_boost_round', 20000),
    ('boostMulti', 1.00),
    ('gpu', 'grow_gpu'),
    ('loadModel', 0),
    ('useKaggleParam', 1)
])
hyperP.gpu = None if args.gpu == '0' else hyperP.gpu

# space = {
#     'learning_rate': 0.03,  # hp.quniform('learning_rate', 0.01, 0.2, 0.01),
#     'max_depth': hp.choice('max_depth', np.arange(5, 7, dtype=int)),
#     'min_child_weight': hp.quniform('min_child_weight', 1, 3, 1),
#     'subsample': hp.quniform('subsample', 0.5, 0.7, 0.05),
#     'gamma': hp.quniform('gamma', 0, 0.5, 0.1),
#     'colsample_bytree': hp.quniform('colsample_bytree', 0.5, 1, 0.1),
#     # 'max_delta_step': 0,
#     # subsample = 1, #only in scikit-lean API
#     # colsample_bytree = 1,
#     # colsample_bylevel = 1,
#     # reg_alpha = 0,
#     # reg_lambda = 1
#     # 'scale_pos_weight': 1,
#     # 'base_score': xgbparams.base_score,
#     'objective': 'reg:linear',
#     'eval_metric': 'rmse',
#     # Increase this number if you have more cores. Otherwise, remove it and it will default
#     # to the maxium number.
#     # 'nthread':  None,
#     # 'booster': xgbparams.booster,
#     'tree_method': 'exact',  # xgbparams.tree_method,
#     'silent': 1,
#     'seed': 254,
#     # 'missing': None,
#     'xgbArgs': {
#         'num_boost_round': hyperP.num_boost_round,  # sample(scope.int(hp.quniform('num_boost_round', 100, 1000, 1))),
#         'early_stopping_rounds': 50,
#         'verbose_eval': 100,
#         'show_stdv': False,
#         'nfold': hyperP.nfold
#     }}
# if hyperP.gpu is not None:
#     space['updater'] = hyperP.gpu

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

prediction_stderr = 0.0073  #  assumed standard error of predictions
                          #  (smaller values make output closer to input)
train_test_logmean_diff = 0.1  # assumed shift used to adjust frequencies for time trend
probthresh = 90  # minimum probability*frequency to use new price instead of just rounding
rounder = 2  # number of places left of decimal point to zero

polyOn = True
def getPoly(df):
    dfT = df.fillna(value=0)
    poly = PolynomialFeatures(degree=2, interaction_only=True, include_bias=True)
    d = poly.fit_transform(dfT)
    colNames = poly.get_feature_names(dfT.columns)
    print("Total poly features length: {}".format(len(colNames)))
    colNames = [str(x).replace(' ', '-') for x in colNames]
    d = pd.DataFrame(d, columns=colNames)
    return pd.concat([df, d], axis=1)

if True:

    #load files
    train = pd.read_csv(os.path.join(projectDir,'input/train.csv'), parse_dates=['timestamp'])
    test = pd.read_csv(os.path.join(projectDir,'input/test.csv'), parse_dates=['timestamp'])
    id_test = test.id

    # clean data
    print('Data Clean...')
    bad_index = train[train.life_sq > train.full_sq].index
    train.loc[bad_index, "life_sq"] = np.NaN
    equal_index = [601, 1896, 2791]
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
    train.product_type.value_counts(normalize=True)
    test.product_type.value_counts(normalize=True)
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
    train.floor.describe(percentiles=[0.9999])
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
    train = train[train.price_doc / train.full_sq <= 600000]
    train = train[train.price_doc / train.full_sq >= 10000]

    print('Feature Engineering...')
    # Add month-year
    month_year = (train.timestamp.dt.month * 30 + train.timestamp.dt.year * 365)
    month_year_cnt_map = month_year.value_counts().to_dict()
    train['month_year_cnt'] = month_year.map(month_year_cnt_map)

    month_year = (test.timestamp.dt.month * 30 + test.timestamp.dt.year * 365)
    month_year_cnt_map = month_year.value_counts().to_dict()
    test['month_year_cnt'] = month_year.map(month_year_cnt_map)

    # Add week-year count
    week_year = (train.timestamp.dt.weekofyear * 7 + train.timestamp.dt.year * 365)
    week_year_cnt_map = week_year.value_counts().to_dict()
    train['week_year_cnt'] = week_year.map(week_year_cnt_map)

    week_year = (test.timestamp.dt.weekofyear * 7 + test.timestamp.dt.year * 365)
    week_year_cnt_map = week_year.value_counts().to_dict()
    test['week_year_cnt'] = week_year.map(week_year_cnt_map)

    # Add month and day-of-week
    train['month'] = train.timestamp.dt.month
    train['dow'] = train.timestamp.dt.dayofweek

    test['month'] = test.timestamp.dt.month
    test['dow'] = test.timestamp.dt.dayofweek

    # Other feature engineering
    train['rel_floor'] = 0.05 + train['floor'] / train['max_floor'].astype(float)
    train['rel_kitch_sq'] = 0.05 + train['kitch_sq'] / train['full_sq'].astype(float)

    test['rel_floor'] = 0.05 + test['floor'] / test['max_floor'].astype(float)
    test['rel_kitch_sq'] = 0.05 + test['kitch_sq'] / test['full_sq'].astype(float)

    train.apartment_name = train.sub_area + train['metro_km_avto'].astype(str)
    test.apartment_name = test.sub_area + train['metro_km_avto'].astype(str)

    train['room_size'] = train['life_sq'] / train['num_room'].astype(float)
    test['room_size'] = test['life_sq'] / test['num_room'].astype(float)

    train['area_per_room'] = train['life_sq'] / train['num_room'].astype(float)  # rough area per room
    train['livArea_ratio'] = train['life_sq'] / train['full_sq'].astype(float)  # rough living area
    train['yrs_old'] = 2017 - train['build_year'].astype(float)  # years old from 2017
    train['avgfloor_sq'] = train['life_sq'] / train['max_floor'].astype(float)  # living area per floor
    train['pts_floor_ratio'] = train['public_transport_station_km'] / train['max_floor'].astype(float)
    # looking for significance of apartment buildings near public t
    train['room_size'] = train['life_sq'] / train['num_room'].astype(float)
    train['gender_ratio'] = train['male_f'] / train['female_f'].astype(float)
    train['kg_park_ratio'] = train['kindergarten_km'] / train['park_km'].astype(float)  # significance of children?
    train['high_ed_extent'] = train['school_km'] / train['kindergarten_km']  # schooling
    train['pts_x_state'] = train['public_transport_station_km'] * train['state'].astype(
        float)  # public trans * state of listing
    train['lifesq_x_state'] = train['life_sq'] * train['state'].astype(float)  # life_sq times the state of the place
    train['floor_x_state'] = train['floor'] * train['state'].astype(float)  # relative floor * the state of the place

    test['area_per_room'] = test['life_sq'] / test['num_room'].astype(float)
    test['livArea_ratio'] = test['life_sq'] / test['full_sq'].astype(float)
    test['yrs_old'] = 2017 - test['build_year'].astype(float)
    test['avgfloor_sq'] = test['life_sq'] / test['max_floor'].astype(float)  # living area per floor
    test['pts_floor_ratio'] = test['public_transport_station_km'] / test['max_floor'].astype(
        float)  # apartments near public t?
    test['room_size'] = test['life_sq'] / test['num_room'].astype(float)
    test['gender_ratio'] = test['male_f'] / test['female_f'].astype(float)
    test['kg_park_ratio'] = test['kindergarten_km'] / test['park_km'].astype(float)
    test['high_ed_extent'] = test['school_km'] / test['kindergarten_km']
    test['pts_x_state'] = test['public_transport_station_km'] * test['state'].astype(
        float)  # public trans * state of listing
    test['lifesq_x_state'] = test['life_sq'] * test['state'].astype(float)
    test['floor_x_state'] = test['floor'] * test['state'].astype(float)

    #########################################################################
    print('Rate Mults...')
    # Aggreagte house price data derived from
    # http://www.globalpropertyguide.com/real-estate-house-prices/R#russia
    # by luckyzhou
    # See https://www.kaggle.com/luckyzhou/lzhou-test/comments

    rate_2015_q2 = 1
    rate_2015_q1 = rate_2015_q2 / 0.9932
    rate_2014_q4 = rate_2015_q1 / 1.0112
    rate_2014_q3 = rate_2014_q4 / 1.0169
    rate_2014_q2 = rate_2014_q3 / 1.0086
    rate_2014_q1 = rate_2014_q2 / 1.0126
    rate_2013_q4 = rate_2014_q1 / 0.9902
    rate_2013_q3 = rate_2013_q4 / 1.0041
    rate_2013_q2 = rate_2013_q3 / 1.0044
    rate_2013_q1 = rate_2013_q2 / 1.0104  # This is 1.002 (relative to mult), close to 1:
    rate_2012_q4 = rate_2013_q1 / 0.9832  # maybe use 2013q1 as a base quarter and get rid of mult?
    rate_2012_q3 = rate_2012_q4 / 1.0277
    rate_2012_q2 = rate_2012_q3 / 1.0279
    rate_2012_q1 = rate_2012_q2 / 1.0279
    rate_2011_q4 = rate_2012_q1 / 1.076
    rate_2011_q3 = rate_2011_q4 / 1.0236
    rate_2011_q2 = rate_2011_q3 / 1
    rate_2011_q1 = rate_2011_q2 / 1.011

    # train 2015
    train['average_q_price'] = 1

    train_2015_q2_index = train.loc[train['timestamp'].dt.year == 2015].loc[train['timestamp'].dt.month >= 4].loc[
        train['timestamp'].dt.month < 7].index
    train.loc[train_2015_q2_index, 'average_q_price'] = rate_2015_q2

    train_2015_q1_index = train.loc[train['timestamp'].dt.year == 2015].loc[train['timestamp'].dt.month >= 1].loc[
        train['timestamp'].dt.month < 4].index
    train.loc[train_2015_q1_index, 'average_q_price'] = rate_2015_q1

    # train 2014
    train_2014_q4_index = train.loc[train['timestamp'].dt.year == 2014].loc[train['timestamp'].dt.month >= 10].loc[
        train['timestamp'].dt.month <= 12].index
    train.loc[train_2014_q4_index, 'average_q_price'] = rate_2014_q4

    train_2014_q3_index = train.loc[train['timestamp'].dt.year == 2014].loc[train['timestamp'].dt.month >= 7].loc[
        train['timestamp'].dt.month < 10].index
    train.loc[train_2014_q3_index, 'average_q_price'] = rate_2014_q3

    train_2014_q2_index = train.loc[train['timestamp'].dt.year == 2014].loc[train['timestamp'].dt.month >= 4].loc[
        train['timestamp'].dt.month < 7].index
    train.loc[train_2014_q2_index, 'average_q_price'] = rate_2014_q2

    train_2014_q1_index = train.loc[train['timestamp'].dt.year == 2014].loc[train['timestamp'].dt.month >= 1].loc[
        train['timestamp'].dt.month < 4].index
    train.loc[train_2014_q1_index, 'average_q_price'] = rate_2014_q1

    # train 2013
    train_2013_q4_index = train.loc[train['timestamp'].dt.year == 2013].loc[train['timestamp'].dt.month >= 10].loc[
        train['timestamp'].dt.month <= 12].index
    train.loc[train_2013_q4_index, 'average_q_price'] = rate_2013_q4

    train_2013_q3_index = train.loc[train['timestamp'].dt.year == 2013].loc[train['timestamp'].dt.month >= 7].loc[
        train['timestamp'].dt.month < 10].index
    train.loc[train_2013_q3_index, 'average_q_price'] = rate_2013_q3

    train_2013_q2_index = train.loc[train['timestamp'].dt.year == 2013].loc[train['timestamp'].dt.month >= 4].loc[
        train['timestamp'].dt.month < 7].index
    train.loc[train_2013_q2_index, 'average_q_price'] = rate_2013_q2

    train_2013_q1_index = train.loc[train['timestamp'].dt.year == 2013].loc[train['timestamp'].dt.month >= 1].loc[
        train['timestamp'].dt.month < 4].index
    train.loc[train_2013_q1_index, 'average_q_price'] = rate_2013_q1

    # train 2012
    train_2012_q4_index = train.loc[train['timestamp'].dt.year == 2012].loc[train['timestamp'].dt.month >= 10].loc[
        train['timestamp'].dt.month <= 12].index
    train.loc[train_2012_q4_index, 'average_q_price'] = rate_2012_q4

    train_2012_q3_index = train.loc[train['timestamp'].dt.year == 2012].loc[train['timestamp'].dt.month >= 7].loc[
        train['timestamp'].dt.month < 10].index
    train.loc[train_2012_q3_index, 'average_q_price'] = rate_2012_q3

    train_2012_q2_index = train.loc[train['timestamp'].dt.year == 2012].loc[train['timestamp'].dt.month >= 4].loc[
        train['timestamp'].dt.month < 7].index
    train.loc[train_2012_q2_index, 'average_q_price'] = rate_2012_q2

    train_2012_q1_index = train.loc[train['timestamp'].dt.year == 2012].loc[train['timestamp'].dt.month >= 1].loc[
        train['timestamp'].dt.month < 4].index
    train.loc[train_2012_q1_index, 'average_q_price'] = rate_2012_q1

    # train 2011
    train_2011_q4_index = train.loc[train['timestamp'].dt.year == 2011].loc[train['timestamp'].dt.month >= 10].loc[
        train['timestamp'].dt.month <= 12].index
    train.loc[train_2011_q4_index, 'average_q_price'] = rate_2011_q4

    train_2011_q3_index = train.loc[train['timestamp'].dt.year == 2011].loc[train['timestamp'].dt.month >= 7].loc[
        train['timestamp'].dt.month < 10].index
    train.loc[train_2011_q3_index, 'average_q_price'] = rate_2011_q3

    train_2011_q2_index = train.loc[train['timestamp'].dt.year == 2011].loc[train['timestamp'].dt.month >= 4].loc[
        train['timestamp'].dt.month < 7].index
    train.loc[train_2011_q2_index, 'average_q_price'] = rate_2011_q2

    train_2011_q1_index = train.loc[train['timestamp'].dt.year == 2011].loc[train['timestamp'].dt.month >= 1].loc[
        train['timestamp'].dt.month < 4].index
    train.loc[train_2011_q1_index, 'average_q_price'] = rate_2011_q1

    train['price_doc'] = train['price_doc'] * train['average_q_price']

    #########################################################################################################

    mult = 1.054880504
    train['price_doc'] = train['price_doc'] * mult
    y_train = train["price_doc"]

    #########################################################################################################
    print('Running Model 1...')
    x_train = train.drop(["id", "timestamp", "price_doc", "average_q_price"], axis=1)
    # x_test = test.drop(["id", "timestamp", "average_q_price"], axis=1)
    x_test = test.drop(["id", "timestamp"], axis=1)

    num_train = len(x_train)
    x_all = pd.concat([x_train, x_test])

    for c in x_all.columns:
        if x_all[c].dtype == 'object':
            lbl = preprocessing.LabelEncoder()
            lbl.fit(list(x_all[c].values))
            x_all[c] = lbl.transform(list(x_all[c].values))

    ###################### poly interaction features ######################
    if polyOn:
        x_all = getPoly(df=x_all)
        print(x_all.shape)
    ###################### end poly interaction features ######################

    x_train = x_all[:num_train]
    x_test = x_all[num_train:]

    del x_all
    gc.collect()

    dtrain = xgb.DMatrix(x_train, y_train)
    dtest = xgb.DMatrix(x_test)

if hyperP.loadModel == 0:

    if hyperP.useKaggleParam:
        xgb_params = {
            'eta': 0.05,
            'max_depth': 6,
            'subsample': 0.6,
            'colsample_bytree': 1,
            'objective': 'reg:linear',
            'eval_metric': 'rmse',
            'silent': 1
        }
        num_boost_rounds = 739  # 422 from kaggle set
        xgb_args = {
            'num_boost_round': math.ceil(num_boost_rounds * hyperP.boostMulti),
        }
    else:
        # cv_output = xgb.cv(xgb_params, dtrain, num_boost_round=1000, early_stopping_rounds=20,
        #     verbose_eval=20, show_stdv=False)

        num_boost_rounds=962
        xgb_params = {'colsample_bytree': 0.7,
                      'eval_metric': 'rmse',
                      'gamma': 0.2,
                      'learning_rate': 0.04,
                      'max_depth': 5,
                      'min_child_weight': 1.0,
                      'objective': 'reg:linear',
                      'seed': 254,
                      'silent': 1,
                      'subsample': 0.6,
                      'tree_method': 'exact',
                      'xgbArgs': {'early_stopping_rounds': 100,
                                  'nfold': 10,
                                  'num_boost_round': 1000,
                                  'show_stdv': False,
                                  'verbose_eval': 50}}
        xgb_args = {
            'num_boost_round': math.ceil(num_boost_rounds * hyperP.boostMulti),
        }

    if hyperP.gpu is not None:
        xgb_params['updater'] = hyperP.gpu

    Logger.info('Opt1 rounds / params / args:\n{}\n{}\n{}'.format(
        num_boost_rounds,
        xgb_params,
        xgb_args
    ))


    # this cv resulted in best boosting rounds: 739
    # 2017-06-27 12:42:23,500 - INFO Opt1 CV par: {'colsample_bytree': 1, 'eta': 0.05, 'eval_metric': 'rmse', 'max_depth': 6, 'objective': 'reg:linear', 'silent': 1, 'subsample': 0.6, 'updater': 'grow_gpu', 'xgbArgs': {'early_stopping_rounds': 100, 'nfold': 10, 'num_boost_round': 1000, 'show_stdv': False, 'verbose_eval': 50}}  numboost: 739
    #
    xgb_params['xgbArgs'] = {'early_stopping_rounds': 100,
                             'nfold': 5,
                             'num_boost_round': 1000,
                             'show_stdv': False,
                             'verbose_eval': 50}
    trials1 = Trials()
    # xgb_paramsT, num_boost_roundsT = optimize(trials1, space, scorecv)
    xgb_params, num_boost_rounds = optimize(trials1, xgb_params, scorecv)
    Logger.info('Opt1 CV par: {}  numboost: {}'.format(
        xgb_params, num_boost_rounds))
    # pickle.dump(trials1, open(os.path.join(projectDir, 'hyperOptTrials/{}.Sberbanktrial1'.format(datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S'))),'wb'))
    xgb_args = {
        'num_boost_round': math.ceil(num_boost_rounds * hyperP.boostMulti),
    }
    model = xgb.train(xgb_params, dtrain, **xgb_args)
    pickle.dump(model, open(os.path.join(projectDir, 'model/{}.Sberbankmodel1'.format(
        datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S'))), 'wb'))

else:
    model = pickle.load(open(r'C:\repos\kaggle\Sberbank\model\2017-06-18_10-15-41.Sberbankmodel1', 'rb'))

y_predict = model.predict(dtest)
gunja_output = pd.DataFrame({'id': id_test, 'price_doc': y_predict})

######################################################################################################
print('Running Model 2...')
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
    if hyperP.useKaggleParam:
        xgb_params = {
            'eta': 0.05,
            'max_depth': 5,
            'subsample': 0.7,
            'colsample_bytree': 0.7,
            'objective': 'reg:linear',
            'eval_metric': 'rmse',
            'silent': 1
        }
        # num_boost_rounds = len(cv_output) # 382    from kaggel
        num_boost_rounds = 806
        xgb_args = {
            'num_boost_round': math.ceil(num_boost_rounds * hyperP.boostMulti),
        }
    else:
        xgb_params = {
            'objective': 'reg:linear',
            'eval_metric': 'rmse',
            'tree_method': 'exact',
            'eta': 0.03,
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
                       num_boost_round=20000,
                       early_stopping_rounds=100,
                       verbose_eval=100,
                       show_stdv=False,
                       nfold=hyperP.nfold,
                       )

    num_boost_rounds= len(cv_output) # 382
    xgb_args = {
        'num_boost_round': math.ceil(num_boost_rounds * hyperP.boostMulti),
    }
    # num_boost_rounds = 385  # This was the CV output, as earlier version shows
    Logger.info('Opt2 rounds / params / args:\n{}\n{}\n{}'.format(
        num_boost_rounds,
        xgb_params,
        xgb_args
    ))

    model = xgb.train(xgb_params, dtrain, **xgb_args)
    pickle.dump(model, open(os.path.join(projectDir, 'model/{}.Sberbankmodel2'.format(
        datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S'))), 'wb'))
    # xgb_params['xgbArgs'] = {'early_stopping_rounds': 100,
    #                          'nfold': 5,
    #                          'num_boost_round': 1000,
    #                          'show_stdv': False,
    #                          'verbose_eval': 50}
    # trials2 = Trials()
    # xgb_paramsT, num_boost_roundsT = optimize(trials2, xgb_params, scorecv)
    # Logger.info('Opt2 CV output:par {} rounds{}'.format(
    #     xgb_paramsT, num_boost_roundsT
    # ))
    # cv result: boost 806
    # 2017-06-27 15:03:09,679 - INFO Opt2 CV output:par {'colsample_bytree': 0.7, 'eta': 0.05, 'eval_metric': 'rmse', 'max_depth': 5, 'objective': 'reg:linear', 'silent': 1, 'subsample': 0.7, 'updater': 'grow_gpu', 'xgbArgs': {'early_stopping_rounds': 100, 'nfold': 5, 'num_boost_round': 1000, 'show_stdv': False, 'verbose_eval': 50}} rounds806


    # Opt2 params: {'colsample_bytree': 0.9, 'eval_metric': 'rmse', 'gamma': 0.1, 'learning_rate': 0.01, 'max_depth': 6, 'min_child_weight': 3.0, 'objective': 'reg:linear', 'seed': 254, 'silent': 1, 'subsample': 0.5, 'tree_method': 'exact', 'updater': 'grow_gpu', 'xgbArgs': {'early_stopping_rounds': 50, 'nfold': 10, 'num_boost_round': 1000, 'show_stdv': False, 'verbose_eval': 50}}
    # pickle.dump(trials2, open(os.path.join(projectDir, 'hyperOptTrials/{}.Sberbanktrial2'.format(datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S'))),'wb'))
else:
    model = pickle.load(open(r'C:\repos\kaggle\Sberbank\model\2017-06-18_10-46-42.Sberbankmodel2', 'rb'))

y_predict = model.predict(dtest)
output = pd.DataFrame({'id': id_test, 'price_doc': y_predict})
# output.drop('average_q_price', axis=1, inplace=True)
# output.head()

#######################################################################################################
print('Running Model 3...')

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
month_year = (df_all.timestamp.dt.month*30 + df_all.timestamp.dt.year * 365)
month_year_cnt_map = month_year.value_counts().to_dict()
df_all['month_year_cnt'] = month_year.map(month_year_cnt_map)

# Add week-year count
week_year = (df_all.timestamp.dt.weekofyear*7 + df_all.timestamp.dt.year * 365)
week_year_cnt_map = week_year.value_counts().to_dict()
df_all['week_year_cnt'] = week_year.map(week_year_cnt_map)

# Add month and day-of-week
df_all['month'] = df_all.timestamp.dt.month
df_all['dow'] = df_all.timestamp.dt.dayofweek

# Other feature engineering
df_all['rel_floor'] = df_all['floor'] / df_all['max_floor'].astype(float)
df_all['rel_kitch_sq'] = df_all['kitch_sq'] / df_all['full_sq'].astype(float)

## same ones as above
df_all['area_per_room'] = df_all['life_sq'] / df_all['num_room'].astype(float)
df_all['livArea_ratio'] = df_all['life_sq'] / df_all['full_sq'].astype(float)
df_all['yrs_old'] = 2017 - df_all['build_year'].astype(float)
df_all['avgfloor_sq'] = df_all['life_sq']/df_all['max_floor'].astype(float) #living area per floor
df_all['pts_floor_ratio'] = df_all['public_transport_station_km']/df_all['max_floor'].astype(float) #apartments near public t?
df_all['room_size'] = df_all['life_sq'] / df_all['num_room'].astype(float)
df_all['gender_ratio'] = df_all['male_f']/df_all['female_f'].astype(float)
df_all['kg_park_ratio'] = df_all['kindergarten_km']/df_all['park_km'].astype(float)
df_all['high_ed_extent'] = df_all['school_km'] / df_all['kindergarten_km']
df_all['pts_x_state'] = df_all['public_transport_station_km'] * df_all['state'].astype(float) #public trans * state of listing
df_all['lifesq_x_state'] = df_all['life_sq'] * df_all['state'].astype(float)
df_all['floor_x_state'] = df_all['floor'] * df_all['state'].astype(float)

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
###################### poly interaction features ######################
if polyOn:
    X_all = getPoly(df=X_all)
    print(X_all.shape)
###################### end poly interaction features ######################

print(X_all.shape)

X_train = X_all[:num_train]
X_test = X_all[num_train:]
del X_all
gc.collect()

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
    if hyperP.useKaggleParam:
        xgb_params = {
            'eta': 0.05,
            'max_depth': 5,
            'subsample': 0.7,
            'colsample_bytree': 0.7,
            'objective': 'reg:linear',
            'eval_metric': 'rmse',
            'silent': 1
        }
        num_boost_rounds = 957#420  # From Bruno's original CV, I think
        xgb_args = {
            'num_boost_round': math.ceil(num_boost_rounds * hyperP.boostMulti),
        }

    else:
        xgb_params = {
            'objective': 'reg:linear',
            'eval_metric': 'rmse',
            'tree_method': 'exact',
            'eta': 0.03,
            'colsample_bytree': 0.5,
            'gamma': 0.3,
            'max_depth': 1,
            'min_child_weight': 2.0,
            'subsample': 0.65,
            'silent': 1,
        }

    if hyperP.gpu is not None:
        xgb_params['updater'] = hyperP.gpu

    Logger.info('Opt3 rounds / params / args:\n{}\n{}\n{}'.format(
        num_boost_rounds,
        xgb_params,
        xgb_args
    ))


    # print('best num_boost_rounds = ', len(cv_output))
    cv_output = xgb.cv(xgb_params, dtrain,
                       num_boost_round=20000,
                       early_stopping_rounds=200,
                       verbose_eval=100,
                       show_stdv=False,
                       nfold=hyperP.nfold)
    print(cv_output)
    num_boost_roundsT = len(cv_output)
    Logger.info('Opt3 CV rounds / params / args:\n{}\n{}'.format(
        num_boost_roundsT,
        xgb_params,
    ))
    xgb_args = {
        'num_boost_round': math.ceil(num_boost_rounds * hyperP.boostMulti),
    }
    # cv returned this boosting 957 for below params
    # {'eta': 0.05, 'max_depth': 5, 'subsample': 0.7, 'colsample_bytree': 0.7, 'objective': 'reg:linear',
    #  'eval_metric': 'rmse', 'silent': 1, 'updater': 'grow_gpu'}
    model = xgb.train(xgb_params, dtrain, **xgb_args)
    pickle.dump(model, open(os.path.join(projectDir, 'model/{}.Sberbankmodel3'.format(
        datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S'))), 'wb'))

    # trials3 = Trials()
    # xgb_params, num_boost_rounds = optimize(trials3, space, scorecv)
    # pickle.dump(trials3, open(os.path.join(projectDir, 'hyperOptTrials/{}.Sberbanktrial3'.format(datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S'))),'wb'))
else:
    model = pickle.load(open(r'C:\repos\kaggle\Sberbank\model\2017-06-18_11-18-01.Sberbankmodel3', 'rb'))

y_pred = model.predict(dtest)
df_sub = pd.DataFrame({'id': id_test, 'price_doc': y_pred})
df_sub.head()

####################################################################################################3
print('Combining Models....')
first_result = output.merge(df_sub, on="id", suffixes=['_louis', '_bruno'])
first_result["price_doc"] = np.exp(.714 * np.log(first_result.price_doc_louis) +
                                   .286 * np.log(first_result.price_doc_bruno))
result = first_result.merge(gunja_output, on="id", suffixes=['_follow', '_gunja'])

result["price_doc"] = np.exp(.78 * np.log(result.price_doc_follow) +
                             .22 * np.log(result.price_doc_gunja))

result["price_doc"] = result["price_doc"] * 0.9915
result.drop(["price_doc_louis", "price_doc_bruno", "price_doc_follow", "price_doc_gunja"], axis=1, inplace=True)
result.head()
result.to_csv(os.path.join(projectDir,'subm/magicBill-same-{}.csv'.format(datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S'))), index=False)

# APPLY PROBABILISTIC IMPROVEMENTS
preds = result
df_train = pd.read_csv(os.path.join(projectDir,'input/train.csv'))
df_test = pd.read_csv(os.path.join(projectDir,'input/test.csv'))

# Select investment sales from training set and generate frequency distribution
invest = train[train.product_type == "Investment"]
freqs = invest.price_doc.value_counts().sort_index()

# Select investment sales from test set predictions
test_invest_ids = test[test.product_type == "Investment"]["id"]
invest_preds = pd.DataFrame(test_invest_ids).merge(preds, on="id")

# Express X-axis of training set frequency distribution as logarithms,
#    and save standard deviation to help adjust frequencies for time trend.
lnp = np.log(invest.price_doc)
stderr = lnp.std()
lfreqs = lnp.value_counts().sort_index()

# Adjust frequencies for time trend
lnp_diff = train_test_logmean_diff
lnp_mean = lnp.mean()
lnp_newmean = lnp_mean + lnp_diff

def norm_diff(value):
    return norm.pdf((value - lnp_diff) / stderr) / norm.pdf(value / stderr)

newfreqs = lfreqs * (pd.Series(lfreqs.index.values - lnp_newmean).apply(norm_diff).values)

# Logs of model-predicted prices
lnpred = np.log(invest_preds.price_doc)

# Create assumed probability distributions
stderr = prediction_stderr
mat = (np.array(newfreqs.index.values)[:, np.newaxis] - np.array(lnpred)[np.newaxis, :]) / stderr
modelprobs = norm.pdf(mat)

# Multiply by frequency distribution.
freqprobs = pd.DataFrame(np.multiply(np.transpose(modelprobs), newfreqs.values))
freqprobs.index = invest_preds.price_doc.values
freqprobs.columns = freqs.index.values.tolist()

# Find mode for each case.
prices = freqprobs.idxmax(axis=1)

# Apply threshold to exclude low-confidence cases from recoding
priceprobs = freqprobs.max(axis=1)
mask = priceprobs < probthresh
prices[mask] = np.round(prices[mask].index, -rounder)

# Data frame with new predicitons
newpricedf = pd.DataFrame({"id": test_invest_ids.values, "price_doc": prices})

# Merge these new predictions (for just investment properties) back into the full prediction set.
newpreds = preds.merge(newpricedf, on="id", how="left", suffixes=("_old", ""))
newpreds.loc[newpreds.price_doc.isnull(), "price_doc"] = newpreds.price_doc_old
newpreds.drop("price_doc_old", axis=1, inplace=True)
newpreds.head()

newpreds.to_csv(os.path.join(projectDir,'subm/magicBill-different-{}.csv'.format(datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S'))), index=False)

if False:
    import pandas as pd
    import numpy as np
    #merge subs on id
    sub1 = pd.read_csv(r'C:\repos\kaggle\Sberbank\subm\0.31038-different_result.csv')
    sub1 = sub1.set_index('id')
    sub1 = sub1.rename(columns={'price_doc': 'sub1'})
    # sub2 = pd.read_csv(r'C:\repos\kaggle\Sberbank\subm\0.31091-sub-silly.csv')
    sub2 = pd.read_csv(r'C:\repos\kaggle\Sberbank\subm\0.31039-different_result.csv')
    sub2 = sub2.set_index('id')
    sub2 = sub2.rename(columns={'price_doc': 'sub2'})
    subM = pd.concat([sub1,sub2], axis=1)
    # subM['f'] = np.exp( 0.66*np.log(subM.loc[:,'sub1']) +
    #                     0.34*np.log(subM.loc[:,'sub2']) )
    subM['f'] = np.exp(0.5 * np.log(subM.loc[:, 'sub1']) +
                       0.5 * np.log(subM.loc[:, 'sub2']))
    subM.drop(['sub1','sub2'],axis=1,inplace=True)
    subM = subM.rename(columns={'f': 'price_doc'})
    subM['id']=subM.index
    subM.to_csv(os.path.join(projectDir, 'subm/merge2-{}.csv'.format(
        datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S'))), index=False)